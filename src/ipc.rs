//! Requests to nngrepd, the daemon that runs the models and keeps indexes up
//! to date, over a Unix socket. Each request and response is a line of JSON.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Longest wait for a newly started nngrepd to listen
const START_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
pub enum Request {
    /// Embed search queries
    Embed {
        queries: Vec<String>,
    },
    /// Score each document's relevance to each query with the reranker
    Rerank {
        queries: Vec<String>,
        documents: Vec<String>,
    },
    /// Index a directory and keep its index up to date. Answered with
    /// progress and then the summary.
    Index {
        root: PathBuf,
        /// Re-embed all files, even unchanged ones
        force: bool,
        /// Skip files larger than this many bytes (default: as before)
        max_file_size: Option<u64>,
    },
    /// Stop keeping a directory's index up to date
    Unwatch {
        root: PathBuf,
    },
    Status,
    /// Stop nngrepd once its indexes are saved
    Stop,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "response", rename_all = "snake_case")]
pub enum Response {
    Embeddings {
        /// Name of the embedding model (see embed::canonical_model_name)
        model: String,
        /// One L2-normalized embedding per query
        vectors: Vec<Vec<f32>>,
    },
    /// Relevance from 0 to 1, by query and then by document
    Scores {
        scores: Vec<Vec<f32>>,
    },
    Progress {
        progress: Progress,
    },
    Indexed {
        summary: Summary,
    },
    Status {
        status: Status,
    },
    Done,
    Error {
        message: String,
    },
}

/// Progress of indexing a directory
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Progress {
    /// Waiting for an earlier update of the index to finish
    Waiting,
    /// Looking for changed files
    Scanning { files: usize },
    /// Embedding the chunks of changed files
    Embedding {
        done: usize,
        total: usize,
        /// Last file embedded
        path: String,
    },
}

/// Outcome of indexing a directory
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Summary {
    /// Number of files in the index
    pub files: usize,
    /// Number of chunks in the index
    pub chunks: usize,
    /// Number of files kept from before
    pub unchanged: usize,
    /// Number of files embedded
    pub embedded: usize,
    /// Number of chunks embedded
    pub embedded_chunks: usize,
    /// Number of files left out
    pub skipped: usize,
    /// Number of files removed from the index
    pub removed: usize,
    pub seconds: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Status {
    pub pid: u32,
    /// Name of the embedding model
    pub model: String,
    /// Where the embedding model runs
    pub location: String,
    /// Is the reranker loaded?
    pub reranker: bool,
    pub roots: Vec<RootStatus>,
}

/// State of an index kept up to date
#[derive(Serialize, Deserialize)]
pub struct RootStatus {
    pub path: PathBuf,
    /// Progress of the update under way, if any
    pub progress: Option<Progress>,
    /// Are changes waiting for an update?
    pub pending: bool,
}

/// A connection to nngrepd.
pub struct Client {
    stream: BufReader<UnixStream>,
}

impl Client {
    /// Connects to nngrepd, starting it first if it is not running. A started
    /// nngrepd logs to log_path().
    pub fn connect() -> Result<Self> {
        if let Some(client) = Self::connect_running()? {
            return Ok(client);
        }
        let mut child = spawn_daemon()?;
        let deadline = Instant::now() + START_TIMEOUT;
        loop {
            thread::sleep(Duration::from_millis(50));
            if let Some(client) = Self::connect_running()? {
                return Ok(client);
            }
            // nngrepd exits successfully if another one is starting
            let failed = child.try_wait()?.filter(|s| !s.success());
            if failed.is_some() || Instant::now() > deadline {
                let what = match failed {
                    Some(status) => format!("nngrepd failed to start ({status})"),
                    None => "nngrepd did not start".into(),
                };
                bail!("{what}{}", log_tail());
            }
        }
    }

    /// Connects to nngrepd if it is running.
    pub fn connect_running() -> Result<Option<Self>> {
        let path = socket_path();
        match UnixStream::connect(&path) {
            Ok(stream) => Ok(Some(Client {
                stream: BufReader::new(stream),
            })),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
                ) =>
            {
                Ok(None)
            }
            Err(e) => Err(e).with_context(|| format!("connecting to {}", path.display())),
        }
    }

    pub fn send(&mut self, request: &Request) -> Result<()> {
        let mut line = serde_json::to_vec(request)?;
        line.push(b'\n');
        self.stream
            .get_mut()
            .write_all(&line)
            .context("sending a request to nngrepd")
    }

    /// Receives the next response. Error responses are returned as errors.
    pub fn receive(&mut self) -> Result<Response> {
        let mut line = String::new();
        if self.stream.read_line(&mut line)? == 0 {
            bail!("nngrepd closed the connection{}", log_tail());
        }
        match serde_json::from_str(&line).context("parsing a response from nngrepd")? {
            Response::Error { message } => bail!("{message}"),
            response => Ok(response),
        }
    }

    /// Returns the name of the embedding model and the L2-normalized
    /// embedding of each query.
    pub fn embed(&mut self, queries: &[String]) -> Result<(String, Vec<Vec<f32>>)> {
        self.send(&Request::Embed {
            queries: queries.to_vec(),
        })?;
        match self.receive()? {
            Response::Embeddings { model, vectors } if vectors.len() == queries.len() => {
                Ok((model, vectors))
            }
            _ => bail!("unexpected response from nngrepd"),
        }
    }

    /// Returns each document's relevance to each query, from 0 to 1, by
    /// query and then by document.
    pub fn rerank(&mut self, queries: &[String], documents: &[String]) -> Result<Vec<Vec<f32>>> {
        self.send(&Request::Rerank {
            queries: queries.to_vec(),
            documents: documents.to_vec(),
        })?;
        match self.receive()? {
            Response::Scores { scores } if scores.len() == queries.len() => Ok(scores),
            _ => bail!("unexpected response from nngrepd"),
        }
    }
}

/// Starts nngrepd in its own session, logging to log_path(). The executable
/// next to the current one is preferred over the one on PATH.
fn spawn_daemon() -> Result<std::process::Child> {
    let program = env::current_exe()
        .ok()
        .map(|exe| exe.with_file_name("nngrepd"))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| "nngrepd".into());
    let log = log_path();
    if let Some(dir) = log.parent() {
        fs::create_dir_all(dir)?;
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .with_context(|| format!("opening {}", log.display()))?;
    let mut command = Command::new(&program);
    command
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // SAFETY: setsid is async-signal-safe.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    command
        .spawn()
        .with_context(|| format!("starting {}", program.display()))
}

/// Returns the end of the log of a started nngrepd, as a sentence to append to
/// an error message, if the log was written to in the last minute.
fn log_tail() -> String {
    let path = log_path();
    let Ok(mut file) = File::open(&path) else {
        return String::new();
    };
    let Ok(meta) = file.metadata() else {
        return String::new();
    };
    let recent = meta
        .modified()
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < Duration::from_secs(60));
    if !recent {
        return String::new();
    }
    let len = meta.len();
    let mut tail = String::new();
    let _ = file.seek(SeekFrom::Start(len.saturating_sub(2000)));
    let _ = file.read_to_string(&mut tail);
    let lines: Vec<&str> = tail.lines().rev().take(5).collect();
    let lines: Vec<&str> = lines.into_iter().rev().collect();
    format!(
        "; the end of {} reads:\n{}",
        path.display(),
        lines.join("\n")
    )
}

/// Path of nngrepd's socket: NNGREP_SOCKET, or nngrepd.sock in the user's
/// runtime directory
pub fn socket_path() -> PathBuf {
    if let Some(path) = env::var_os("NNGREP_SOCKET") {
        return path.into();
    }
    match env::var_os("XDG_RUNTIME_DIR") {
        Some(dir) => PathBuf::from(dir).join("nngrepd.sock"),
        // SAFETY: getuid always succeeds.
        None => env::temp_dir().join(format!("nngrepd-{}.sock", unsafe { libc::getuid() })),
    }
}

/// Path of the lock file held by the running nngrepd
pub fn lock_path() -> PathBuf {
    let mut path = socket_path().into_os_string();
    path.push(".lock");
    path.into()
}

/// Directory for nngrepd's state: $XDG_STATE_HOME/nngrep or
/// ~/.local/state/nngrep
pub fn state_dir() -> PathBuf {
    match env::var_os("XDG_STATE_HOME") {
        Some(dir) => PathBuf::from(dir).join("nngrep"),
        None => Path::new(&env::var_os("HOME").unwrap_or_default()).join(".local/state/nngrep"),
    }
}

/// Path of the log of an nngrepd started by a client
pub fn log_path() -> PathBuf {
    state_dir().join("nngrepd.log")
}
