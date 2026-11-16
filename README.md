# nngrep

Semantic grep built on [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

```sh
nnindex ~/src/project              # build or update ~/src/project/.nngrep
nngrep "where do we retry failed uploads" ~/src/project
```

## Installation

```sh
cargo install --path .
```

nngrep needs an embedding server. It uses one of these:

1. The server at `NNGREP_SERVER`, if set.
2. A server running at `http://127.0.0.1:8000`, vLLM's default address.
3. Otherwise, its own [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`,
   started for the duration of the command. `llama-server` must be on `PATH`.
   On first use it downloads `Qwen/Qwen3-Embedding-4B-GGUF:Q8_0` (about 4 GB).

### vLLM (recommended)

vLLM indexes about 2.5 times faster than llama.cpp, but takes minutes to start,
so keep it running:

```sh
vllm serve Qwen/Qwen3-Embedding-4B --runner pooling --max-model-len 4096
```

On AMD GPUs, AMD's ROCm images work. For RDNA3 cards (RX 7900 series):

```sh
podman run -d --name vllm --device /dev/kfd --device /dev/dri --group-add keep-groups \
    --security-opt seccomp=unconfined --shm-size 8g \
    -v ~/.cache/huggingface:/root/.cache/huggingface -p 127.0.0.1:8000:8000 \
    docker.io/rocm/vllm:rocm7.13.0_gfx110X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
    vllm serve Qwen/Qwen3-Embedding-4B --runner pooling --max-model-len 4096
```

vLLM reserves 90% of GPU memory by default; lower it with `--gpu-memory-utilization`.

## nnindex

```
nnindex [DIR] [--force] [--max-file-size BYTES] [--batch-size N]
```

Splits every text file in `DIR` into chunks of up to 20 lines, ending at blank
lines where possible, and embeds them. Hidden files, binary files, non-UTF-8
files and anything ignored by git are skipped. The index is stored in `DIR/.nngrep`.

Re-running `nnindex` only embeds files whose size or modification time changed.
Progress is saved every 30 seconds, so an interrupted run resumes where it stopped.

## nngrep

```
nngrep QUERY [PATH] [-k N] [-l] [-L LINES | --full] [-s MIN_SCORE] [--color WHEN]
```

Searches the index covering `PATH` (found by looking in `PATH` and its parents),
restricted to files under `PATH`. Prints the `-k` best chunks with their cosine
similarity, or with `-l` only the best matching files.

## Configuration

| Variable              | Meaning                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `NNGREP_SERVER`       | URL of the embedding server (default `http://127.0.0.1:8000`)      |
| `NNGREP_MODEL`        | GGUF path or Hugging Face repo for a spawned `llama-server`        |
| `NNGREP_SERVER_ARGS`  | Extra arguments for a spawned `llama-server`, e.g. `-ngl 0` for CPU |
| `NNGREP_LLAMA_SERVER` | Path to the `llama-server` binary                                  |

The index records the model it was built with, ignoring quantization and file
format, so an index built through llama.cpp can be searched through vLLM.
Searching with a different model is an error, and `nnindex` re-embeds everything
when the model changes.
