# nngrep

Semantic grep built on [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

```sh
nnindex ~/src/project              # build ~/src/project/.nngrep and keep it up to date
nngrep "where do we retry failed uploads" ~/src/project
```

Both talk to `nngrepd`, a daemon that holds the model and updates indexes as
files change. They start it when it is not running.

## Installation

```sh
cargo install --path .
```

Building compiles [llama.cpp](https://github.com/ggml-org/llama.cpp), which needs
CMake and a C++ compiler. On macOS it uses Metal; elsewhere it uses Vulkan, which
also needs the Vulkan headers and `glslc`.

When it starts, nngrepd runs its embedding model in one of these ways:

1. On the server at `NNGREP_SERVER`, if set.
2. On a server running at `http://127.0.0.1:8000`, vLLM's default address.
3. Otherwise, with llama.cpp on the GPU. On first use it downloads
   `Qwen/Qwen3-Embedding-4B-GGUF:Q8_0` (about 4 GB) into the Hugging Face cache.

With `nngrepd --cpu`, it always runs the model with llama.cpp on the CPU,
leaving the GPU alone. Queries are then about as fast as on the GPU, but
indexing is much slower. `--cpu` cannot be combined with `NNGREP_SERVER`.

### vLLM (recommended)

vLLM indexes about 1.5 times faster than llama.cpp, but takes minutes to start,
so keep it running, and start it before nngrepd:

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
nnindex [DIR] [--force] [--max-file-size BYTES]
nnindex --unwatch DIR
```

Has nngrepd index `DIR` and keep the index up to date, and shows its progress.
nngrepd splits every text file in `DIR` into chunks of up to 20 lines and
embeds them. Chunks follow the layout of the text rather than its language: a
paragraph that starts no deeper than the block before it starts a new block,
such as a function, a struct or a section of prose, and each block gets its own
chunks. Hidden files, binary files, non-UTF-8 files, files larger than
`--max-file-size` (default 1 MiB, remembered by the index) and anything ignored
by git are skipped. The index is stored in `DIR/.nngrep`.

Only files whose size or modification time changed are embedded again, unless
`--force` is given. Progress is saved every 30 seconds, and interrupting
`nnindex` leaves nngrepd indexing. `--unwatch` stops keeping the index up to
date, and leaves it as it is.

## nngrepd

```
nngrepd [--cpu] [--batch-size N]
nngrepd status
nngrepd stop
```

Runs the model for `nngrep` and `nnindex`, and keeps every directory indexed
with `nnindex` up to date, also across restarts. When a file changes, nngrepd
waits for a second without changes and then embeds the changed files, between
searches: a search waits for at most one batch of embedding, about 0.3 seconds
on an RX 7900 XT. Reranking enables the reranker's adapter on the model and
indexing disables it again, so the model is loaded once. Changes are seen
through inotify on Linux, which watches every directory that is indexed; if
there are more than `fs.inotify.max_user_watches`, nngrepd logs a warning.

`nngrep` and `nnindex` start nngrepd when it is not running, with their
environment, logging to `~/.local/state/nngrep/nngrepd.log`. Run it yourself,
for instance as a systemd user service, to choose its options or to see its
log. `nngrepd status` shows the model and the state of each index, and
`nngrepd stop` stops it once the index being updated is saved, as do SIGINT and
SIGTERM. Restart nngrepd after changing its configuration or upgrading.

nngrepd uses about 4.8 GB of GPU memory, and 5.5 GB once the reranker is loaded.

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

As in ripgrep, output to a terminal has headings and line numbers, while piped
output prints `PATH:TEXT` lines with `--` between chunks. The exit status is 0 if
anything was found, 1 if nothing was and 2 if an error occurred.

### Reranking

With `--rerank`, nngrep takes the best chunks by cosine similarity, adds the
best chunks by keyword search, and ranks them all again with a reranker, which
reads each chunk together with the query. The reranker is the embedding model
with an adapter that approximates
[Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B). Scores are
then the reranker's estimate that a chunk is what the query asks for, from 0
to 1.

The keyword search ranks chunks by [BM25](https://en.wikipedia.org/wiki/Okapi_BM25),
matching stemmed English words, the parts of `camelCase` and `snake_case`
identifiers, and whole identifiers. It finds chunks that the embedding misses,
such as those with an exact error message or function name. It reads every file
searched, which takes about 0.15 seconds per 100,000 chunks on 16 threads.

nngrepd loads the reranker on first use. It runs on nngrepd's embedding model
if that runs with llama.cpp, and otherwise on a copy of it loaded from
`NNGREP_MODEL` (by default `Qwen/Qwen3-Embedding-4B-GGUF:Q8_0`) on the GPU. The
adapter is not downloaded: set `NNGREP_RERANK_ADAPTER` to its GGUF file before
nngrepd starts. With the default `--rerank-depth`, a search with `--rerank`
takes about 1.5 seconds on an RX 7900 XT.

## Emacs

`emacs/nngrep.el` runs searches from Emacs (29.1 or later):

```elisp
(use-package nngrep
  :load-path "~/src/nngrep/emacs"
  :bind ("C-c n" . nngrep-menu))
```

`M-x nngrep` searches the index covering the current directory, or offers to
build one for the current project. In the results, `RET` visits the line at
point, `n` and `p` step through the chunks, `TAB` folds one, `s` searches for
something else and `?` opens a menu with nngrep's options. `M-g n` and `M-g p`
step through the results from any buffer. `C-h m` lists every key.

## Configuration

nngrepd reads these when it starts:

| Variable              | Meaning                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `NNGREP_SERVER`       | URL of the embedding server (default `http://127.0.0.1:8000`)      |
| `NNGREP_MODEL`        | GGUF path or Hugging Face `REPO[:QUANT]` for the llama.cpp model   |
| `NNGREP_RERANK_ADAPTER` | GGUF path of the reranker's adapter (see [Reranking](#reranking)) |
| `HF_HOME`, `HF_TOKEN` | Hugging Face cache directory (default `~/.cache/huggingface`) and access token |
| `XDG_STATE_HOME`      | Directory of nngrepd's log and list of indexes (default `~/.local/state`) |

nngrepd and its clients find each other through the socket at
`NNGREP_SOCKET`, by default `$XDG_RUNTIME_DIR/nngrepd.sock`.

The index records the model it was built with, ignoring quantization and file
format, so an index built through llama.cpp can be searched through vLLM.
Searching with a different model is an error, and nngrepd re-embeds everything
when the model changes.
