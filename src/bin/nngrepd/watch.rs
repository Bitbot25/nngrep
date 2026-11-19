use crate::log;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

/// The directories watched for changes, on behalf of the indexed directories
/// (roots) they are in. Each directory is watched by itself, without its
/// subdirectories.
pub struct Watches {
    watcher: RecommendedWatcher,
    /// Watched directories of each root
    by_root: HashMap<PathBuf, HashSet<PathBuf>>,
    /// Number of roots watching each directory
    counts: HashMap<PathBuf, usize>,
    /// Has a failure to watch been logged?
    warned: bool,
}

impl Watches {
    /// Creates an empty set of watches that sends events to events.
    pub fn new(events: Sender<notify::Result<Event>>) -> notify::Result<Self> {
        Ok(Watches {
            watcher: notify::recommended_watcher(events)?,
            by_root: HashMap::new(),
            counts: HashMap::new(),
            warned: false,
        })
    }

    /// Watches dir for root.
    pub fn add(&mut self, root: &Path, dir: &Path) {
        if !self
            .by_root
            .entry(root.into())
            .or_default()
            .insert(dir.into())
        {
            return;
        }
        let count = self.counts.entry(dir.into()).or_default();
        *count += 1;
        if *count > 1 {
            return;
        }
        if let Err(e) = self.watcher.watch(dir, RecursiveMode::NonRecursive)
            && !std::mem::replace(&mut self.warned, true)
        {
            log!(
                "cannot watch {}: {e}; changes there will be missed \
                 (on Linux, raising fs.inotify.max_user_watches may help)",
                dir.display()
            );
        }
    }

    /// Stops watching the directories inside subtree that root watches,
    /// except those in keep.
    pub fn retain(&mut self, root: &Path, subtree: &Path, keep: &HashSet<PathBuf>) {
        let Some(dirs) = self.by_root.get_mut(root) else {
            return;
        };
        let gone: Vec<PathBuf> = dirs
            .iter()
            .filter(|d| d.starts_with(subtree) && !keep.contains(*d))
            .cloned()
            .collect();
        for dir in &gone {
            dirs.remove(dir);
        }
        for dir in gone {
            self.release(&dir);
        }
    }

    /// Stops watching the directories that root watches.
    pub fn remove_root(&mut self, root: &Path) {
        for dir in self.by_root.remove(root).unwrap_or_default() {
            self.release(&dir);
        }
    }

    fn release(&mut self, dir: &Path) {
        let Some(count) = self.counts.get_mut(dir) else {
            return;
        };
        *count -= 1;
        if *count == 0 {
            self.counts.remove(dir);
            // Fails if the directory is gone, which removed its watch
            let _ = self.watcher.unwatch(dir);
        }
    }
}
