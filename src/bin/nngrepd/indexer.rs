use crate::log;
use crate::state::{Daemon, Event, Job};
use anyhow::{Result, bail};
use half::f16;
use ignore::WalkBuilder;
use nngrep::chunk::{chunk_text, truncate};
use nngrep::index::{ChunkSpan, DEFAULT_MAX_FILE_SIZE, FileEntry, INDEX_DIR, Index, is_under};
use nngrep::ipc::{Progress, Summary};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

/// Concurrent embedding requests, so the model always has work queued
const WORKERS: usize = 2;

/// Maximum length of the file path prepended to each chunk
const MAX_PATH_BYTES: usize = 256;

/// Interval between saves of a partial index
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(30);

/// Minimum interval between reports of scanning progress
const REPORT_INTERVAL: Duration = Duration::from_millis(100);

/// Updates indexes until nngrepd stops.
pub fn run(daemon: &Daemon) {
    while let Some(job) = daemon.next_job() {
        let (root, pass) = (job.root.clone(), job.pass);
        let event = match Pass::new(daemon, job).run() {
            Ok(summary) => {
                if summary.embedded > 0 || summary.removed > 0 {
                    log!(
                        "{}: {} files embedded ({} chunks), {} removed in {:.1} s",
                        root.display(),
                        summary.embedded,
                        summary.embedded_chunks,
                        summary.removed,
                        summary.seconds
                    );
                }
                Event::Done(pass, summary)
            }
            Err(e) => {
                log!("{}: {e:#}", root.display());
                Event::Failed(pass, format!("{e:#}"))
            }
        };
        daemon.report(&root, event);
        if !daemon.has_root(&root) {
            daemon.watches.lock().unwrap().remove_root(&root);
        }
    }
    daemon.indexer_stopped();
}

/// An update of the index of one directory.
struct Pass<'a> {
    daemon: &'a Daemon,
    job: Job,
    start: Instant,
    last_report: Instant,
    summary: Summary,
}

/// A file whose chunks still need embedding.
struct Pending {
    entry: FileEntry,
    /// Text to embed for each chunk
    texts: Vec<String>,
}

impl<'a> Pass<'a> {
    fn new(daemon: &'a Daemon, job: Job) -> Self {
        let now = Instant::now();
        Pass {
            daemon,
            job,
            start: now,
            last_report: now,
            summary: Summary::default(),
        }
    }

    /// Scans the directories of the job, embeds the files that changed and
    /// saves the index.
    fn run(mut self) -> Result<Summary> {
        let root = self.job.root.clone();
        if !root.is_dir() {
            self.daemon.unwatch(&root);
            bail!("no longer exists");
        }
        let model = self.daemon.embedder.model();
        let existing = match Index::load(&root) {
            Ok(index) if index.model == model => Some(index),
            Ok(index) => {
                log!(
                    "{}: model changed from {} to {model}; re-embedding everything",
                    root.display(),
                    index.model
                );
                None
            }
            Err(e) if root.join(INDEX_DIR).exists() => {
                log!("{}: ignoring existing index: {e:#}", root.display());
                None
            }
            Err(_) => None,
        };

        let full = self.job.full;
        let force = full.is_some_and(|f| f.force);
        let mut index = Index::new(model, existing.as_ref().map_or(0, |i| i.dim));
        index.max_file_size = full
            .and_then(|f| f.max_file_size)
            .or(existing.as_ref().map(|i| i.max_file_size))
            .unwrap_or(DEFAULT_MAX_FILE_SIZE);
        let subtrees = if full.is_some() || existing.is_none() {
            vec![root.clone()]
        } else {
            outermost(&root, &self.job.dirs)
        };
        let prefixes: Vec<&str> = subtrees
            .iter()
            .filter_map(|d| d.strip_prefix(&root).ok()?.to_str())
            .collect();

        let mut old: HashMap<String, (FileEntry, Vec<f16>)> = HashMap::new();
        if let Some(existing) = &existing {
            for (file, vectors) in existing.iter_files() {
                if !prefixes.iter().any(|p| is_under(&file.path, p)) {
                    index.push(file.clone(), vectors);
                } else if !force {
                    old.insert(file.path.clone(), (file.clone(), vectors.to_vec()));
                }
            }
        }
        let mut pending = Vec::new();
        for subtree in &subtrees {
            self.scan(subtree, &mut old, &mut index, &mut pending);
        }

        self.summary.unchanged = index.files.len();
        self.summary.removed = old.len();
        let changed = !pending.is_empty()
            || !old.is_empty()
            || existing
                .as_ref()
                .is_none_or(|e| e.max_file_size != index.max_file_size);
        if changed {
            let complete = self.embed(pending, &mut index)?;
            index.save(&root)?;
            if !complete {
                bail!("stopped before embedding every file; resuming when nngrepd starts again");
            }
        }
        self.summary.files = index.files.len();
        self.summary.chunks = index.files.iter().map(|f| f.chunks.len()).sum();
        self.summary.seconds = self.start.elapsed().as_secs_f64();
        Ok(self.summary)
    }

    fn report(&mut self, progress: Progress) {
        self.daemon
            .report(&self.job.root, Event::Progress(self.job.pass, progress));
    }

    /// Walks subtree, watching its directories, moving unchanged files from
    /// old into index and chunking the rest into pending.
    fn scan(
        &mut self,
        subtree: &Path,
        old: &mut HashMap<String, (FileEntry, Vec<f16>)>,
        index: &mut Index,
        pending: &mut Vec<Pending>,
    ) {
        let root = self.job.root.clone();
        let walker = WalkBuilder::new(subtree)
            .filter_entry(|e| e.file_name() != INDEX_DIR)
            .build();
        let mut dirs = HashSet::new();
        let mut files = 0;
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    log!("{}: {e}", root.display());
                    continue;
                }
            };
            let Some(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                self.daemon.watches.lock().unwrap().add(&root, entry.path());
                dirs.insert(entry.path().to_path_buf());
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            files += 1;
            if self.last_report.elapsed() > REPORT_INTERVAL {
                self.last_report = Instant::now();
                self.report(Progress::Scanning { files });
            }

            let Some(path) = entry.path().strip_prefix(&root).ok().and_then(Path::to_str) else {
                self.summary.skipped += 1;
                continue;
            };
            let Ok(meta) = entry.metadata() else {
                self.summary.skipped += 1;
                continue;
            };
            let size = meta.len();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            if size > index.max_file_size {
                self.summary.skipped += 1;
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
                self.summary.skipped += 1;
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
        self.daemon
            .watches
            .lock()
            .unwrap()
            .retain(&root, subtree, &dirs);
    }

    /// Embeds the chunks of pending files and adds the files to index, saving
    /// partial progress periodically. Stops early if nngrepd is stopping.
    /// Returns whether every file was embedded.
    fn embed(&mut self, pending: Vec<Pending>, index: &mut Index) -> Result<bool> {
        let daemon = self.daemon;
        let jobs: Vec<(usize, usize)> = pending
            .iter()
            .enumerate()
            .flat_map(|(f, p)| (0..p.texts.len()).map(move |c| (f, c)))
            .collect();
        let batches: Vec<&[(usize, usize)]> = jobs.chunks(daemon.batch_size.max(1)).collect();

        // Per-file vector buffers, filled as batches complete
        let mut buffers: Vec<Vec<f16>> = vec![Vec::new(); pending.len()];
        let mut remaining: Vec<usize> = pending.iter().map(|p| p.texts.len()).collect();
        for (p, _) in pending.iter().zip(&remaining).filter(|(_, n)| **n == 0) {
            index.push(p.entry.clone(), &[]);
            self.summary.embedded += 1;
        }

        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let mut last_save = Instant::now();
        let mut done = 0;

        thread::scope(|s| -> Result<()> {
            let (tx, rx) = mpsc::channel();
            for _ in 0..WORKERS {
                let tx = tx.clone();
                let (next, stop, batches, pending) = (&next, &stop, &batches, &pending);
                s.spawn(move || {
                    while !stop.load(Ordering::Relaxed) && !daemon.stopping() {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(batch) = batches.get(i) else { break };
                        let texts: Vec<String> = batch
                            .iter()
                            .map(|&(f, c)| pending[f].texts[c].clone())
                            .collect();
                        if tx
                            .send((i, daemon.embedder.embed_documents(&texts)))
                            .is_err()
                        {
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
                                "embedding has dimension {}, index has {dim}; re-run nnindex with --force",
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
                            self.summary.embedded += 1;
                            self.summary.embedded_chunks += pending[f].texts.len();
                        }
                    }

                    done += batches[i].len();
                    let path = batches[i]
                        .last()
                        .map(|&(f, _)| pending[f].entry.path.clone());
                    self.report(Progress::Embedding {
                        done,
                        total: jobs.len(),
                        path: path.unwrap_or_default(),
                    });
                    if last_save.elapsed() > CHECKPOINT_INTERVAL {
                        index.save(&self.job.root)?;
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
        Ok(done == jobs.len())
    }
}

/// Returns the directories among dirs that are not inside another, each
/// replaced by its closest existing ancestor inside root if it is gone.
fn outermost(root: &Path, dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut existing: Vec<PathBuf> = dirs
        .iter()
        .map(|d| {
            d.ancestors()
                .take_while(|a| a.starts_with(root))
                .find(|a| a.is_dir())
                .unwrap_or(root)
                .to_path_buf()
        })
        .collect();
    existing.sort();
    existing.dedup();
    let mut outermost: Vec<PathBuf> = Vec::new();
    for dir in existing {
        if !outermost.last().is_some_and(|o| dir.starts_with(o)) {
            outermost.push(dir);
        }
    }
    outermost
}

/// Returns the contents of a UTF-8 text file, or None for binary files.
fn read_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::outermost;

    #[test]
    fn keeps_outermost_directories() {
        let root = std::env::temp_dir();
        let a = root.join("nngrepd-test-outermost");
        std::fs::create_dir_all(a.join("b")).unwrap();
        let dirs = vec![
            a.join("b"),
            a.clone(),
            a.join("gone/deeper"),
            root.join("other-gone"),
        ];
        assert_eq!(outermost(&root, &dirs), vec![root.clone()]);
        let dirs = vec![a.join("b"), a.join("gone")];
        assert_eq!(outermost(&root, &dirs), vec![a.clone()]);
        std::fs::remove_dir_all(&a).unwrap();
    }
}
