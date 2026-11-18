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

Building compiles [llama.cpp](https://github.com/ggml-org/llama.cpp), which needs
CMake and a C++ compiler. On macOS it uses Metal; elsewhere it uses Vulkan, which
also needs the Vulkan headers and `glslc`.

nngrep runs its embedding model in one of these ways:

1. On the server at `NNGREP_SERVER`, if set.
2. On a server running at `http://127.0.0.1:8000`, vLLM's default address.
3. Otherwise, in process with llama.cpp on the GPU. On first use it downloads
   `Qwen/Qwen3-Embedding-4B-GGUF:Q8_0` (about 4 GB) into the Hugging Face cache.

With `--cpu`, `nnindex` and `nngrep` always load the model in process and run it
on the CPU, leaving the GPU alone. A query then takes about as long as on the
GPU, since most of the time goes to loading the model, but indexing is much
slower. `--cpu` cannot be combined with `NNGREP_SERVER`.

### vLLM (recommended)

vLLM indexes about 1.5 times faster than llama.cpp, but takes minutes to start,
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
| `--min-score SCORE`        | Only show chunks with at least this score                        |
| `--rerank`                 | Rank the best chunks again with a reranker (see below)           |
| `--rerank-depth NUM`       | Chunks to rerank from each search, at least `-k` (default 15)    |
| `-l`, `-0`                 | Only print the paths of the best matching files (NUL-separated)  |
| `-q`                       | Print nothing; only set the exit status                          |
| `-g GLOB`                  | Include or exclude (`!GLOB`) files, like ripgrep                 |
| `-t TYPE`, `-T TYPE`       | Only search, or skip, files of a type (see `--type-list`)        |
| `-n`, `-N`                 | Show or hide line numbers                                        |
| `--heading`, `--no-heading`| Summary line above each chunk, or the path on every line         |
| `--lines NUM`, `--full`    | Lines of each chunk to print (default 5), or all of them         |
| `--color WHEN`             | `auto`, `always` or `never`                                      |
| `--cpu`                    | Run the models on the CPU (see [Installation](#installation))    |

As in ripgrep, output to a terminal has headings and line numbers, while piped
output prints `PATH:TEXT` lines with `--` between chunks. The exit status is 0 if
anything was found, 1 if nothing was and 2 if an error occurred.

### Reranking

With `--rerank`, nngrep takes the best chunks by cosine similarity, adds the
best chunks by keyword search, and ranks them all again with
[Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B), which reads
each chunk together with the query. Scores are then the reranker's estimate that
a chunk is what the query asks for, from 0 to 1.

The keyword search ranks chunks by [BM25](https://en.wikipedia.org/wiki/Okapi_BM25),
matching stemmed English words, the parts of `camelCase` and `snake_case`
identifiers, and whole identifiers. It finds chunks that the embedding misses,
such as those with an exact error message or function name. It reads every file
searched, which takes about 0.15 seconds per 100,000 chunks on 16 threads.

The reranker runs in process, after the embedding model has been unloaded. On
first use it downloads `Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp:Q4_0` (about
2.4 GB), a llama.cpp conversion of Qwen's model, since Qwen publishes none; the
`Q8_0` file is twice the size but ranks no better. The reranker uses about 3 GB
of GPU memory and, with the default `--rerank-depth`, adds about 3 seconds to a
search on an RX 7900 XT. Its time grows with the number of chunks reranked,
and on slower GPUs that dominates. On those,
`NNGREP_RERANK_MODEL=ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` needs about an
eighth of the computation, but ranks less well.

## Configuration

| Variable              | Meaning                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `NNGREP_SERVER`       | URL of the embedding server (default `http://127.0.0.1:8000`)      |
| `NNGREP_MODEL`        | GGUF path or Hugging Face `REPO[:QUANT]` for the in-process model  |
| `NNGREP_RERANK_MODEL` | Same, for the Qwen3 reranker                                       |
| `HF_HOME`, `HF_TOKEN` | Hugging Face cache directory (default `~/.cache/huggingface`) and access token |

The index records the model it was built with, ignoring quantization and file
format, so an index built through llama.cpp can be searched through vLLM.
Searching with a different model is an error, and `nnindex` re-embeds everything
when the model changes.
