# nngrep

Semantic grep built on [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

```sh
nnindex ~/src/project              # build or update ~/src/project/.nngrep
nngrep "where do we retry failed uploads" ~/src/project
```

## Requirements

- Rust (to build)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` on `PATH`.
  On first use it downloads `Qwen/Qwen3-Embedding-4B-GGUF:Q8_0` (about 4 GB) into the Hugging Face cache.

```sh
cargo install --path .
```

## nnindex

```
nnindex [DIR] [--force] [--max-file-size BYTES] [--batch-size N]
```

Splits every text file in `DIR` into chunks of up to 40 lines and embeds them.
Hidden files, binary files, non-UTF-8 files and anything ignored by git are skipped.
The index is stored in `DIR/.nngrep`.

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

| Variable              | Meaning                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `NNGREP_SERVER`       | URL of a running embedding server to use instead of starting one     |
| `NNGREP_MODEL`        | GGUF path or Hugging Face repo for the spawned server                |
| `NNGREP_SERVER_ARGS`  | Extra arguments for the spawned server, e.g. `-ngl 0` for CPU only   |
| `NNGREP_LLAMA_SERVER` | Path to the `llama-server` binary                                    |

By default every command starts its own `llama-server`, which takes a couple of
seconds. For faster repeated searches, keep one running:

```sh
llama-server -hf Qwen/Qwen3-Embedding-4B-GGUF:Q8_0 --embedding --pooling last \
    -ngl 999 -c 2048 -ub 2048 --port 8090
export NNGREP_SERVER=http://127.0.0.1:8090
```

The index records the model it was built with, and changing `NNGREP_MODEL`
re-embeds everything. nngrep cannot tell which model an `NNGREP_SERVER` runs, so
make sure it serves the model the index was built with.
