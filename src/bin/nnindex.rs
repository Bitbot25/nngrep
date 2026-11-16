use anyhow::{Context, Result, bail};
use clap::Parser;
use half::f16;
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nngrep::chunk::{chunk_text, truncate};
use nngrep::embed::Embedder;
use nngrep::index::{ChunkSpan, FileEntry, INDEX_DIR, Index};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

/// Concurrent embedding requests, so the server always has work queued
const WORKERS: usize = 2;

/// Maximum length of the file path prepended to each chunk
const MAX_PATH_BYTES: usize = 256;

/// Interval between saves of a partial index
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(30);

/// Index a directory for semantic search with nngrep.
///
/// Respects .gitignore and skips hidden, binary and non-UTF-8 files. Unchanged
/// files keep their embeddings from the previous index.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Directory to index
    #[arg(default_value = ".")]
    dir: PathBuf,

    /// Re-embed all files, even unchanged ones
    #[arg(long)]
    force: bool,

    /// Skip files larger than this many bytes
    #[arg(long, default_value_t = 1 << 20)]
    max_file_size: u64,

    /// Chunks per embedding request
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Run the embedding model on the CPU in a private llama-server, even if
    /// an embedding server is running
    #[arg(long)]
    cpu: bool,
}

/// A file whose chunks still need embedding.
struct Pending {
    entry: FileEntry,
    /// Text to embed for each chunk
    texts: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let root = fs::canonicalize(&args.dir)
        .with_context(|| format!("cannot access {}", args.dir.display()))?;
    let loading = spinner("Connecting to embedding server");
    let embedder = if args.cpu {
        Embedder::start_on_cpu()?
    } else {
        Embedder::connect()?
    };
    loading.finish_and_clear();
    let model = embedder.model();

    let mut old: HashMap<String, (FileEntry, Vec<f16>)> = HashMap::new();
    let mut dim = 0;
    if !args.force && root.join(INDEX_DIR).exists() {
        match Index::load(&root) {
            Ok(index) if index.model == model => {
                dim = index.dim;
                for (file, vectors) in index.iter_files() {
                    old.insert(file.path.clone(), (file.clone(), vectors.to_vec()));
                }
            }
            Ok(index) => eprintln!(
                "Model changed from {} to {model}; re-embedding everything",
                index.model
            ),
            Err(e) => eprintln!("Ignoring existing index: {e:#}"),
        }
    }

    let mut index = Index::new(model, dim);
    let scan = spinner("Scanning files {pos}");
    let (pending, skipped) = scan_files(&root, &args, &mut old, &mut index, &scan);
    scan.finish_and_clear();

    let reused = index.files.len();
    let total_chunks: usize = pending.iter().map(|p| p.texts.len()).sum();
    eprintln!(
        "{} files: {reused} unchanged, {} to embed ({total_chunks} chunks), {skipped} skipped",
        reused + pending.len(),
        pending.len(),
    );

    embed_files(&embedder, &root, pending, &mut index, args.batch_size)?;
    index.save(&root)?;

    let chunks: usize = index.files.iter().map(|f| f.chunks.len()).sum();
    eprintln!(
        "Indexed {} files ({chunks} chunks) in {:.1?}",
        index.files.len(),
        start.elapsed()
    );
    Ok(())
}

/// Walks root, moving unchanged files from old into index and chunking the rest.
/// Returns the files to embed and the number of files skipped.
fn scan_files(
    root: &Path,
    args: &Args,
    old: &mut HashMap<String, (FileEntry, Vec<f16>)>,
    index: &mut Index,
    progress: &ProgressBar,
) -> (Vec<Pending>, usize) {
    let walker = WalkBuilder::new(root)
        .filter_entry(|e| e.file_name() != INDEX_DIR)
        .build();

    let mut pending = Vec::new();
    let mut skipped = 0;
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                progress.suspend(|| eprintln!("warning: {e}"));
                continue;
            }
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        progress.inc(1);

        let Some(path) = entry.path().strip_prefix(root).ok().and_then(Path::to_str) else {
            skipped += 1;
            continue;
        };
        let Ok(meta) = entry.metadata() else {
            skipped += 1;
            continue;
        };
        let size = meta.len();
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        if size > args.max_file_size {
            skipped += 1;
            continue;
        }

        if let Some((file, vectors)) = old.remove(path)
            && file.mtime == mtime
            && file.size == size
        {
            index.push(file, &vectors);
            continue;
        }

        let Some(text) = read_text(entry.path()) else {
            skipped += 1;
            continue;
        };
        let chunks = chunk_text(&text);
        pending.push(Pending {
            entry: FileEntry {
                path: path.to_string(),
                mtime,
                size,
                chunks: chunks
                    .iter()
                    .map(|c| ChunkSpan {
                        start: c.start_line,
                        end: c.end_line,
                    })
                    .collect(),
            },
            texts: chunks
                .iter()
                .map(|c| format!("{}\n{}", truncate(path, MAX_PATH_BYTES), c.text))
                .collect(),
        });
    }
    (pending, skipped)
}

/// Returns the contents of a UTF-8 text file, or None for binary files.
fn read_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// Embeds the chunks of pending files and adds the files to index, saving
/// partial progress periodically.
fn embed_files(
    embedder: &Embedder,
    root: &Path,
    pending: Vec<Pending>,
    index: &mut Index,
    batch_size: usize,
) -> Result<()> {
    let jobs: Vec<(usize, usize)> = pending
        .iter()
        .enumerate()
        .flat_map(|(f, p)| (0..p.texts.len()).map(move |c| (f, c)))
        .collect();
    let batches: Vec<&[(usize, usize)]> = jobs.chunks(batch_size.max(1)).collect();

    let progress = ProgressBar::new(jobs.len() as u64).with_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({percent}%) · {speed} · ETA {eta} {wide_msg:.dim}",
        )
        .unwrap()
        .with_key("speed", |s: &ProgressState, w: &mut dyn std::fmt::Write| {
            write!(w, "{:.1} chunks/s", s.per_sec()).unwrap()
        })
        .progress_chars("=> "),
    );
    progress.enable_steady_tick(Duration::from_millis(100));

    // Per-file vector buffers, filled as batches complete
    let mut buffers: Vec<Vec<f16>> = vec![Vec::new(); pending.len()];
    let mut remaining: Vec<usize> = pending.iter().map(|p| p.texts.len()).collect();
    for (p, _) in pending.iter().zip(&remaining).filter(|(_, n)| **n == 0) {
        index.push(p.entry.clone(), &[]);
    }

    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let mut last_save = Instant::now();

    thread::scope(|s| -> Result<()> {
        let (tx, rx) = mpsc::channel();
        for _ in 0..WORKERS {
            let tx = tx.clone();
            let (next, stop, batches, pending) = (&next, &stop, &batches, &pending);
            s.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(batch) = batches.get(i) else { break };
                    let texts: Vec<String> = batch
                        .iter()
                        .map(|&(f, c)| pending[f].texts[c].clone())
                        .collect();
                    if tx.send((i, embedder.embed_documents(&texts))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let result = (|| -> Result<()> {
            for (i, vectors) in rx {
                let vectors = vectors?;
                if index.dim == 0 {
                    index.dim = vectors[0].len();
                }
                let dim = index.dim;
                for (&(f, c), v) in batches[i].iter().zip(&vectors) {
                    if v.len() != dim {
                        bail!(
                            "embedding has dimension {}, index has {dim}; re-run with --force",
                            v.len()
                        );
                    }
                    let buf = &mut buffers[f];
                    if buf.is_empty() {
                        buf.resize(pending[f].texts.len() * dim, f16::ZERO);
                    }
                    for (dst, &src) in buf[c * dim..(c + 1) * dim].iter_mut().zip(v) {
                        *dst = f16::from_f32(src);
                    }
                    remaining[f] -= 1;
                    if remaining[f] == 0 {
                        index.push(pending[f].entry.clone(), &std::mem::take(buf));
                    }
                }

                progress.inc(batches[i].len() as u64);
                if let Some(&(f, _)) = batches[i].last() {
                    progress.set_message(pending[f].entry.path.clone());
                }
                if last_save.elapsed() > CHECKPOINT_INTERVAL {
                    index.save(root)?;
                    last_save = Instant::now();
                }
            }
            Ok(())
        })();
        if result.is_err() {
            stop.store(true, Ordering::Relaxed);
        }
        result
    })?;

    progress.finish_and_clear();
    Ok(())
}

fn spinner(template: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template(&format!("{{spinner:.green}} {template}")).unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}
