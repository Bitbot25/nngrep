use crate::log;
use crate::watch::Watches;
use anyhow::{Context, Result};
use nngrep::embed::Embedder;
use nngrep::ipc::{Progress, RootStatus, Status, Summary, state_dir};
use nngrep::rerank::Reranker;
use notify::event::{AccessKind, AccessMode, EventKind};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Time without changes to a directory before its index is updated
const QUIET_PERIOD: Duration = Duration::from_secs(1);

/// Longest wait for a quiet period before updating an index anyway
const MAX_DELAY: Duration = Duration::from_secs(10);

/// The state shared by nngrepd's threads.
pub struct Daemon {
    pub embedder: Embedder,
    /// Loaded on first use
    reranker: Mutex<Option<Arc<Reranker>>>,
    /// Chunks per embedding request
    pub batch_size: usize,
    pub watches: Mutex<Watches>,
    /// Is nngrepd stopping?
    stopping: AtomicBool,
    state: Mutex<State>,
    /// Signaled when an index needs updating, or when the indexer stops
    changed: Condvar,
}

struct State {
    /// The indexed directories kept up to date
    roots: BTreeMap<PathBuf, Root>,
    /// Has the indexer stopped?
    stopped: bool,
}

/// An indexed directory kept up to date.
#[derive(Default)]
struct Root {
    /// Requested scan of the whole directory, if any
    full: Option<FullScan>,
    /// Directories with changes since the last update
    dirty: BTreeSet<PathBuf>,
    /// When the first and the last of those changes were seen
    changed: Option<(Instant, Instant)>,
    /// Number of the next update
    next_pass: u64,
    /// Progress of the update under way
    progress: Option<Progress>,
    subscribers: Vec<Sender<Event>>,
}

/// A requested scan of a whole indexed directory
#[derive(Clone, Copy, Default)]
pub struct FullScan {
    /// Re-embed all files, even unchanged ones
    pub force: bool,
    /// New limit on the size of indexed files
    pub max_file_size: Option<u64>,
}

/// An update of an index, called a pass.
pub struct Job {
    pub root: PathBuf,
    /// Number of the pass
    pub pass: u64,
    /// Scan of the whole directory, if requested
    pub full: Option<FullScan>,
    /// Directories to scan, with their subdirectories, if not the whole
    pub dirs: Vec<PathBuf>,
}

/// News of a pass over an index
#[derive(Clone)]
pub enum Event {
    Progress(u64, Progress),
    Done(u64, Summary),
    Failed(u64, String),
}

impl Daemon {
    /// Creates the daemon's state, with the indexed directories kept up to
    /// date when nngrepd last ran, each needing a scan.
    pub fn new(
        embedder: Embedder,
        batch_size: usize,
        events: Sender<notify::Result<notify::Event>>,
    ) -> Result<Self> {
        let mut roots = BTreeMap::new();
        for root in load_roots() {
            if root.is_dir() {
                let full = Some(FullScan::default());
                roots.insert(
                    root,
                    Root {
                        full,
                        ..Root::default()
                    },
                );
            } else {
                log!(
                    "{} no longer exists; not keeping its index up to date",
                    root.display()
                );
            }
        }
        let daemon = Daemon {
            embedder,
            reranker: Mutex::new(None),
            batch_size,
            watches: Mutex::new(Watches::new(events).context("watching for changes")?),
            stopping: AtomicBool::new(false),
            state: Mutex::new(State {
                roots,
                stopped: false,
            }),
            changed: Condvar::new(),
        };
        daemon.save_roots();
        Ok(daemon)
    }

    /// Returns the reranker, loading it on first use.
    pub fn reranker(&self) -> Result<Arc<Reranker>> {
        let mut reranker = self.reranker.lock().unwrap();
        if let Some(r) = &*reranker {
            return Ok(r.clone());
        }
        log!("loading the reranker");
        let r = Arc::new(Reranker::new(&self.embedder)?);
        *reranker = Some(r.clone());
        Ok(r)
    }

    /// Requests a scan of the whole indexed directory at root, adding it to
    /// those kept up to date. Returns the news of passes over it, and the
    /// number of the pass that will include the scan.
    pub fn request_scan(&self, root: &Path, scan: FullScan) -> (Receiver<Event>, u64) {
        let (tx, rx) = mpsc::channel();
        let mut state = self.state.lock().unwrap();
        let added = !state.roots.contains_key(root);
        let r = state.roots.entry(root.into()).or_default();
        let full = r.full.unwrap_or_default();
        r.full = Some(FullScan {
            force: full.force || scan.force,
            max_file_size: scan.max_file_size.or(full.max_file_size),
        });
        if r.progress.is_some() {
            let _ = tx.send(Event::Progress(r.next_pass, Progress::Waiting));
        }
        r.subscribers.push(tx);
        let pass = r.next_pass;
        self.changed.notify_all();
        drop(state);
        if added {
            log!("keeping {} up to date", root.display());
            self.save_roots();
        }
        (rx, pass)
    }

    /// Stops keeping the index at root up to date. Returns whether it was.
    pub fn unwatch(&self, root: &Path) -> bool {
        let removed = self.state.lock().unwrap().roots.remove(root).is_some();
        if removed {
            log!("no longer keeping {} up to date", root.display());
            self.watches.lock().unwrap().remove_root(root);
            self.save_roots();
        }
        removed
    }

    /// Is the index at root kept up to date?
    pub fn has_root(&self, root: &Path) -> bool {
        self.state.lock().unwrap().roots.contains_key(root)
    }

    /// Notes the changes reported by a watched directory.
    pub fn note_event(&self, event: notify::Event) {
        if event.need_rescan() {
            let mut state = self.state.lock().unwrap();
            let roots: Vec<PathBuf> = state.roots.keys().cloned().collect();
            for root in roots {
                mark_dirty(&mut state, &root, &root);
            }
            self.changed.notify_all();
            return;
        }
        let written = matches!(
            event.kind,
            EventKind::Access(AccessKind::Close(AccessMode::Write))
        );
        if matches!(event.kind, EventKind::Access(_)) && !written {
            return;
        }
        let mut state = self.state.lock().unwrap();
        for path in event.paths.iter().filter(|p| !is_noise(p)) {
            let roots: Vec<PathBuf> = state
                .roots
                .keys()
                .filter(|root| path.starts_with(root))
                .cloned()
                .collect();
            for root in roots {
                let dir = if *path == root {
                    &root
                } else {
                    path.parent().unwrap_or(&root)
                };
                mark_dirty(&mut state, &root, dir);
            }
        }
        self.changed.notify_all();
    }

    /// Waits for the next pass to make and starts it. Returns None once
    /// nngrepd is stopping.
    pub fn next_job(&self) -> Option<Job> {
        let mut state = self.state.lock().unwrap();
        loop {
            if self.stopping() {
                return None;
            }
            let now = Instant::now();
            let mut wait: Option<Duration> = None;
            let mut ready = None;
            for (path, root) in &state.roots {
                if root.full.is_some() {
                    ready = Some(path.clone());
                    break;
                }
                if let Some((first, last)) = root.changed {
                    let at = (last + QUIET_PERIOD).min(first + MAX_DELAY);
                    if at <= now {
                        ready = Some(path.clone());
                        break;
                    }
                    wait = Some(wait.map_or(at - now, |w| w.min(at - now)));
                }
            }
            if let Some(path) = ready {
                let root = state.roots.get_mut(&path).unwrap();
                let pass = root.next_pass;
                root.next_pass += 1;
                root.progress = Some(Progress::Scanning { files: 0 });
                root.changed = None;
                return Some(Job {
                    full: root.full.take(),
                    dirs: std::mem::take(&mut root.dirty).into_iter().collect(),
                    root: path,
                    pass,
                });
            }
            state = match wait {
                Some(wait) => self.changed.wait_timeout(state, wait).unwrap().0,
                None => self.changed.wait(state).unwrap(),
            };
        }
    }

    /// Sends news of a pass over the index at root to those waiting for it.
    pub fn report(&self, root: &Path, event: Event) {
        let mut state = self.state.lock().unwrap();
        let Some(r) = state.roots.get_mut(root) else {
            return;
        };
        r.progress = match &event {
            Event::Progress(_, progress) => Some(progress.clone()),
            Event::Done(..) | Event::Failed(..) => None,
        };
        r.subscribers.retain(|s| s.send(event.clone()).is_ok());
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// Asks the indexer to stop and waits until it has.
    pub fn stop(&self) {
        let mut state = self.state.lock().unwrap();
        self.stopping.store(true, Ordering::SeqCst);
        self.changed.notify_all();
        while !state.stopped {
            state = self.changed.wait(state).unwrap();
        }
    }

    /// Notes that the indexer has stopped.
    pub fn indexer_stopped(&self) {
        self.state.lock().unwrap().stopped = true;
        self.changed.notify_all();
    }

    pub fn status(&self) -> Status {
        let reranker = self.reranker.lock().unwrap().is_some();
        let state = self.state.lock().unwrap();
        Status {
            pid: std::process::id(),
            model: self.embedder.model().into(),
            location: self.embedder.location(),
            reranker,
            roots: state
                .roots
                .iter()
                .map(|(path, root)| RootStatus {
                    path: path.clone(),
                    progress: root.progress.clone(),
                    pending: root.full.is_some() || !root.dirty.is_empty(),
                })
                .collect(),
        }
    }

    /// Records the indexed directories kept up to date, for the next start.
    fn save_roots(&self) {
        let roots: String = self
            .state
            .lock()
            .unwrap()
            .roots
            .keys()
            .filter_map(|p| p.to_str())
            .map(|p| format!("{p}\n"))
            .collect();
        let path = roots_path();
        let result = fs::create_dir_all(state_dir()).and_then(|()| fs::write(&path, roots));
        if let Err(e) = result {
            log!("cannot write {}: {e}", path.display());
        }
    }
}

/// Notes a change in dir, inside the indexed directory at root.
fn mark_dirty(state: &mut State, root: &Path, dir: &Path) {
    let Some(r) = state.roots.get_mut(root) else {
        return;
    };
    let now = Instant::now();
    r.dirty.insert(dir.into());
    r.changed = Some(r.changed.map_or((now, now), |(first, _)| (first, now)));
}

/// Is a change to path irrelevant to indexes? Hidden files are, except for
/// ignore files.
fn is_noise(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.starts_with('.') && name != ".gitignore" && name != ".ignore"
}

/// Path of the list of indexed directories kept up to date
fn roots_path() -> PathBuf {
    state_dir().join("roots")
}

fn load_roots() -> Vec<PathBuf> {
    fs::read_to_string(roots_path())
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}
