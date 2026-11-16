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

With `--cpu`, `nnindex` and `nngrep` always start their own `llama-server` and
run the model on the CPU, leaving the GPU alone. A query then takes about as
long as on the GPU, since most of the time goes to starting `llama-server`, but
indexing is much slower. `--cpu` cannot be combined with `NNGREP_SERVER`.

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
nnindex [DIR] [--force] [--max-file-size BYTES] [--batch-size N] [--cpu]
```

Splits every text file in `DIR` into chunks of up to 20 lines, ending at blank
lines where possible, and embeds them. Hidden files, binary files, non-UTF-8
files and anything ignored by git are skipped. The index is stored in `DIR/.nngrep`.

Re-running `nnindex` only embeds files whose size or modification time changed.
Progress is saved every 30 seconds, so an interrupted run resumes where it stopped.

## nngrep

```
nngrep [OPTIONS] QUERY [PATH ...]
nngrep [OPTIONS] -e QUERY ... [PATH ...]
```

Searches the files under each `PATH` (default: the current directory), using the
index covering it (found by looking in `PATH` and its parents). Paths may belong
to different indexes. Prints the `-k` best chunks, ranked by cosine similarity.

The options follow ripgrep where they apply:

| Option                     | Meaning                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `-e QUERY`                 | Search for several queries; a chunk scores its best match        |
| `-k NUM`                   | Number of results (default 10)                                   |
| `-m NUM`                   | At most `NUM` chunks per file                                    |
| `--min-score SCORE`        | Only show chunks with at least this similarity                   |
| `-l`, `-0`                 | Only print the paths of the best matching files (NUL-separated)  |
| `-q`                       | Print nothing; only set the exit status                          |
| `-g GLOB`                  | Include or exclude (`!GLOB`) files, like ripgrep                 |
| `-t TYPE`, `-T TYPE`       | Only search, or skip, files of a type (see `--type-list`)        |
| `-n`, `-N`                 | Show or hide line numbers                                        |
| `--heading`, `--no-heading`| Summary line above each chunk, or the path on every line         |
| `--lines NUM`, `--full`    | Lines of each chunk to print (default 5), or all of them         |
| `--color WHEN`             | `auto`, `always` or `never`                                      |
| `--cpu`                    | Run the model on the CPU (see [Installation](#installation))     |

As in ripgrep, output to a terminal has headings and line numbers, while piped
output prints `PATH:TEXT` lines with `--` between chunks. The exit status is 0 if
anything was found, 1 if nothing was and 2 if an error occurred.

## Configuration

| Variable              | Meaning                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `NNGREP_SERVER`       | URL of the embedding server (default `http://127.0.0.1:8000`)      |
| `NNGREP_MODEL`        | GGUF path or Hugging Face repo for a spawned `llama-server`        |
| `NNGREP_SERVER_ARGS`  | Extra arguments for a spawned `llama-server`                       |
| `NNGREP_LLAMA_SERVER` | Path to the `llama-server` binary                                  |

The index records the model it was built with, ignoring quantization and file
format, so an index built through llama.cpp can be searched through vLLM.
Searching with a different model is an error, and `nnindex` re-embeds everything
when the model changes.
