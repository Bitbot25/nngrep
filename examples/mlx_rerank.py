"""Prototype of nngrep's reranker on MLX, to measure it against llama.cpp.

Loads the same files as nngrepd: the Qwen3-Embedding-4B GGUF (MLX reads Q8_0,
Q4_0 and Q4_1 GGUFs natively) and the reranker adapter (LoRA, rerank.head and
rerank.suffix). It ports what src/rerank.rs and the llama.cpp patches do: the
prompt prefix is computed once and shared, documents are packed into one
sequence per group with a mask, the suffix goes in as input embeddings, only
the last suffix token of each document is computed in the last layer, there is
no output layer, and the LoRA's product is added in the same kernel (addmm).

Reads PAIRS on stdin in the format of examples/rerank.rs (query, document, each
followed by NUL) and prints the logit of each relevance, one per line, so the
output and timing can be compared with `cargo run --release --example rerank`.

    pip install mlx tokenizers
    curl -LO https://huggingface.co/Qwen/Qwen3-Embedding-4B/resolve/main/tokenizer.json
    python examples/mlx_rerank.py MODEL.gguf ADAPTER.gguf tokenizer.json < PAIRS

On Linux (pip install "mlx[cpu]" numpy), matmuls and attention go through
numpy, as MLX's Linux CPU GEMM is very slow; that path checks the scores only.
Unlike llama.cpp, the tokenizer parses special tokens such as <|im_end|> that
appear in documents.
"""

import os
import sys
import time

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

SYSTEM = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
INSTRUCT = (
    "Given a search query about a codebase, judge whether the code or text "
    "passage is what the query is looking for"
)
CONTEXT_TOKENS = 3072
BATCH_TOKENS = 2048
GROUP = 32  # Q8_0 and Q4_0 blocks hold 32 weights
# MLX_METAL_PATH=1 runs the Metal code path on any device (for testing it)
CPU_FP32 = mx.default_device() == mx.cpu and not os.environ.get("MLX_METAL_PATH")
# Activations in float32 on the CPU (as llama.cpp's CPU backend), float16 on Metal
ACT = mx.float32 if CPU_FP32 else mx.float16


class Linear:
    """A projection whose weight is quantized (GGUF Q8_0/Q4_0) or float, with
    an optional LoRA."""

    def __init__(self, w, name, adapter=None):
        self.w = w[name]
        prefix = name.rsplit(".", 1)[0]
        self.scales = w.get(prefix + ".scales")
        self.biases = w.get(prefix + ".biases")
        if self.scales is not None:
            # 32 values per group: 8 bits packs 4 per uint32, 4 bits packs 8
            self.bits = 32 * self.w.shape[1] // (self.scales.shape[1] * GROUP)
        self.lora = None
        if adapter is not None and name + ".lora_a" in adapter:
            # lora_a is [rank, in] (Q8_0, whose scales mx.load names
            # "<name>.scales"), lora_b is [out, rank] (F16)
            a = adapter[name + ".lora_a"]
            a_scales, a_biases = adapter.get(name + ".scales"), adapter.get(name + ".biases")
            b = adapter[name + ".lora_b"].astype(ACT).T
            self.lora = (a, a_scales, a_biases, b)

    def __call__(self, x, lora=False):
        if CPU_FP32:
            # Validation on Linux only: MLX's Linux CPU GEMM is a slow
            # reference implementation, so multiply with numpy (OpenBLAS).
            # On a Mac, the quantized Metal kernels below are used.
            w = self.w
            if self.scales is not None:
                w = mx.dequantize(w, self.scales, self.biases, GROUP, self.bits)
            y = np_mm(x, w.astype(mx.float32))
            if lora and self.lora is not None:
                a, a_s, a_b, b = self.lora
                if a_s is not None:
                    a = mx.dequantize(a, a_s, a_b, GROUP, 8)
                y = y + np_mm(np_mm(x, a), b.T)
            return y
        if self.scales is not None:
            y = mx.quantized_matmul(
                x, self.w, self.scales, self.biases, True, GROUP, self.bits
            )
        else:
            y = x @ self.w.T
        if lora and self.lora is not None:
            a, a_s, a_b, b = self.lora
            xa = mx.quantized_matmul(x, a, a_s, a_b, True, GROUP, 8) if a_s is not None else x @ a.T
            # y + (x A^T) B^T, adding in the matmul's kernel: the counterpart of
            # the Vulkan MUL_MAT + ADD fusion. The scale is 1 (the adapter's
            # alpha is 0), so there is no scaling op, as in the LoRA patch.
            y = mx.addmm(y, xa, b)
        return y


def np_mm(x, w):
    """x @ w.T through numpy."""
    return mx.array(np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T)


def np_sdpa(q, k, v, scale, mask):
    """Grouped-query attention through numpy, for validation on Linux."""
    q, k, v, mask = (np.asarray(t) for t in (q, k, v, mask))
    rep = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    s = (q @ k.transpose(0, 1, 3, 2)) * scale
    s = np.where(mask[None, None], s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    return mx.array((s @ v).astype(np.float32))


def sdpa(q, k, v, scale, mask):
    if CPU_FP32:
        return np_sdpa(q, k, v, scale, mask)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def rms_norm(x, w, eps):
    return mx.fast.rms_norm(x, w, eps)


class Qwen3:
    def __init__(self, model_path, adapter_path=None):
        w, meta = mx.load(model_path, return_metadata=True)
        arch = str(meta["general.architecture"])
        hp = lambda k: meta[f"{arch}.{k}"].item()
        self.n_layer = int(os.environ.get("MLX_LAYERS", hp("block_count")))
        self.n_head = int(hp("attention.head_count"))
        self.n_kv = int(hp("attention.head_count_kv"))
        self.n_embd = int(hp("embedding_length"))
        self.head_dim = int(hp("attention.key_length"))
        self.eps = float(hp("attention.layer_norm_rms_epsilon"))
        self.theta = float(hp("rope.freq_base"))

        ad = None
        self.head = self.suffix = None
        if adapter_path:
            ad, ameta = mx.load(adapter_path, return_metadata=True)
            self.head = ameta["rerank.head"].astype(mx.float32)
            self.suffix = ameta["rerank.suffix"].reshape(-1, self.n_embd)

        self.tok = w["token_embd.weight"]
        self.tok_s = w.get("token_embd.scales")
        self.tok_b = w.get("token_embd.biases")
        self.out_norm = w["output_norm.weight"].astype(ACT)
        self.layers = []
        for i in range(self.n_layer):
            p = f"blk.{i}."
            L = {k: Linear(w, p + k + ".weight", ad) for k in
                 ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")}
            for k in ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"):
                L[k] = w[p + k + ".weight"].astype(ACT)
            self.layers.append(L)
        self.inv_freq = self.theta ** (
            -mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        mx.eval(w)

    def embed_tokens(self, ids):
        ids = mx.array(ids)
        if self.tok_s is None:
            return self.tok[ids].astype(ACT)
        bits = 32 * self.tok.shape[1] // (self.tok_s.shape[1] * GROUP)
        return mx.dequantize(self.tok[ids], self.tok_s[ids], self.tok_b[ids], GROUP, bits).astype(ACT)

    def rope(self, x, pos):
        # x [1, H, N, D], NEOX layout (Qwen3 GGUF weights are not permuted)
        ang = pos.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        cos, sin = mx.cos(ang).astype(x.dtype), mx.sin(ang).astype(x.dtype)
        h = self.head_dim // 2
        x1, x2 = x[..., :h], x[..., h:]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def forward(self, h, pos, mask, past=None, out_rows=None, lora=False):
        """Runs the layers over h [N, n_embd] at positions pos [N].

        mask [N, P+N] (bool, True = attend) covers past keys then these tokens.
        past is a list of per-layer (k, v) [1, n_kv, P, D] to attend to.
        If out_rows is given, the last layer computes only those rows (as
        llama.cpp does with inp_out_ids), and their normed outputs are
        returned. Otherwise returns (hidden, per-layer kv).
        """
        N = h.shape[0]
        D = self.head_dim
        kvs = []
        for i, L in enumerate(self.layers):
            last = i == self.n_layer - 1
            x = rms_norm(h, L["attn_norm"], self.eps)
            q_in = x
            k = L["attn_k"](x, lora).reshape(1, N, self.n_kv, D).transpose(0, 2, 1, 3)
            v = L["attn_v"](x, lora).reshape(1, N, self.n_kv, D).transpose(0, 2, 1, 3)
            k = self.rope(rms_norm(k, L["attn_k_norm"], self.eps), pos)
            rows_mask, q_pos = mask, pos
            if last and out_rows is not None:
                # Only the output rows need queries, attention and the MLP
                q_in, h = x[out_rows], h[out_rows]
                rows_mask, q_pos = mask[out_rows], pos[out_rows]
            M = q_in.shape[0]
            q = L["attn_q"](q_in, lora).reshape(1, M, self.n_head, D).transpose(0, 2, 1, 3)
            q = self.rope(rms_norm(q, L["attn_q_norm"], self.eps), q_pos)
            if past is not None:
                k = mx.concatenate([past[i][0], k], axis=2)
                v = mx.concatenate([past[i][1], v], axis=2)
            kvs.append((k, v))
            a = sdpa(q, k, v, D ** -0.5, rows_mask)
            a = a.transpose(0, 2, 1, 3).reshape(M, self.n_head * D)
            h = h + L["attn_output"](a, lora)
            x = rms_norm(h, L["ffn_norm"], self.eps)
            g = L["ffn_gate"](x, lora)
            u = L["ffn_up"](x, lora)
            h = h + L["ffn_down"](mx.sigmoid(g) * g * u, lora)
            if CPU_FP32:
                mx.eval(h, kvs)  # bound the memory of dequantized weights
        if out_rows is not None:
            return rms_norm(h, self.out_norm, self.eps)
        return h, kvs

    def score(self, prefix_ids, docs):
        """Relevance logits of documents (token lists) after the prefix.

        Like src/rerank.rs: the prefix is computed once and shared; documents
        are packed into one sequence per group with a mask that lets each see
        the prefix and itself; the suffix is fed as input embeddings; only the
        last suffix token of each document is output.
        """
        P = len(prefix_ids)
        n_suffix = self.suffix.shape[0]
        pos = mx.arange(P)
        mask = mx.tril(mx.ones((P, P), dtype=mx.bool_))
        _, past = self.forward(self.embed_tokens(prefix_ids), pos, mask, lora=True)
        # Only the K/V of the prefix is kept
        mx.eval(past)

        logits = []
        start = 0
        while start < len(docs):
            n, end = P, start
            while end < len(docs) and n + len(docs[end]) + n_suffix <= CONTEXT_TOKENS:
                n += len(docs[end]) + n_suffix
                end += 1
            end = max(end, start + 1)
            group = docs[start:end]
            embs, positions, seg, outs = [], [], [], []
            off = 0
            for d, ids in enumerate(group):
                embs += [self.embed_tokens(ids), self.suffix.astype(ACT)]
                L = len(ids) + n_suffix
                positions.append(mx.arange(P, P + L))
                seg.append(mx.full((L,), d))
                off += L
                outs.append(off - 1)
            h = mx.concatenate(embs, axis=0)
            pos = mx.concatenate(positions)
            seg = mx.concatenate(seg)
            N = h.shape[0]
            # Same document and not in the future, or any prefix position
            own = (seg[:, None] == seg[None, :]) & (pos[:, None] >= pos[None, :])
            mask = mx.concatenate([mx.ones((N, P), dtype=mx.bool_), own], axis=1)
            out = self.forward(h, pos, mask, past, mx.array(outs), lora=True)
            logits.append(out.astype(mx.float32) @ self.head)
            mx.eval(logits[-1])
            start = end
        return mx.concatenate(logits).tolist()


def main():
    model_path, adapter_path, tok_path = sys.argv[1:4]
    tok = Tokenizer.from_file(tok_path)
    data = sys.stdin.buffer.read().decode()
    fields = [f for f in data.split("\0")][: -1 if data.endswith("\0") else None]
    pairs = list(zip(fields[0::2], fields[1::2]))

    t0 = time.time()
    m = Qwen3(model_path, adapter_path)
    print(f"loaded in {time.time() - t0:.2f} s on {mx.default_device()}", file=sys.stderr)

    groups = []
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        query = pairs[i][0]
        prefix = f"{SYSTEM}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>:"
        prefix_ids = tok.encode(prefix, add_special_tokens=False).ids
        docs = []
        for _, d in pairs[i:j]:
            ids = tok.encode(" " + d, add_special_tokens=False).ids
            docs.append(ids[: BATCH_TOKENS - m.suffix.shape[0]])
        groups.append((prefix_ids, docs))
        i = j

    # Compiles the Metal kernels before timing
    m.score(*groups[0])
    t0 = time.time()
    out = []
    for prefix_ids, docs in groups:
        out += m.score(prefix_ids, docs)
    n_tok = sum(len(p) + sum(len(d) + m.suffix.shape[0] for d in ds) for p, ds in groups)
    for x in out:
        print(f"{x:.6f}")
    dt = time.time() - t0
    print(f"scored {len(pairs)} pairs ({n_tok} tokens) in {dt:.2f} s, "
          f"{n_tok / dt:.0f} tokens/s", file=sys.stderr)


if __name__ == "__main__":
    main()
