mod indexer;
mod state;
mod watch;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use nngrep::embed::Embedder;
use nngrep::ipc::{Client, Progress, Request, Response, lock_path, socket_path};
use state::{Daemon, Event, FullScan};
use std::fs::{self, File, Permissions, TryLockError};
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::process::ExitCode;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

/// Longest wait for nngrepd to exit once asked to stop
const STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// Run the models for nngrep and nnindex, and keep indexes up to date as
/// files change.
///
/// nngrep and nnindex start nngrepd when it is not running, logging to
/// ~/.local/state/nngrep/nngrepd.log. nngrepd keeps each directory indexed
/// with nnindex up to date, also across restarts, until `nnindex --unwatch`.
#[derive(Parser)]
#[command(version)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Run the embedding model in process on the CPU, even if an embedding
    /// server is running
    #[arg(long)]
    cpu: bool,

    /// Chunks per embedding request
    #[arg(long, value_name = "N", default_value_t = 64)]
    batch_size: usize,
}

#[derive(Subcommand)]
enum Command {
    /// Print the model and the indexes kept up to date
    Status,
    /// Stop nngrepd once its indexes are saved
    Stop,
}

/// Prints a line to the log, prefixed with the local time.
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!("{} {}", $crate::timestamp(), format_args!($($arg)*))
    };
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match args.command {
        Some(Command::Status) => status(),
        Some(Command::Stop) => stop(),
        None => serve(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nngrepd: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn serve(args: &Args) -> Result<()> {
    let lock_path = lock_path();
    let lock = File::create(&lock_path).with_context(|| lock_path.display().to_string())?;
    // The lock is held by a running nngrepd, which listens, or by one that
    // is about to listen or exit
    loop {
        match lock.try_lock() {
            Ok(()) => break,
            Err(TryLockError::WouldBlock) => {
                if Client::connect_running()?.is_some() {
                    eprintln!("nngrepd is already running");
                    return Ok(());
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("locking {}", lock_path.display()));
            }
        }
    }

    let signals = block_signals();

    // Clients may connect while the model loads, and wait
    let socket = socket_path();
    let _ = fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("listening on {}", socket.display()))?;
    fs::set_permissions(&socket, Permissions::from_mode(0o600))?;
    log!("listening on {}", socket.display());

    let embedder = if args.cpu {
        Embedder::load_on_cpu()
    } else {
        Embedder::connect()
    };
    let embedder = match embedder {
        Ok(embedder) => embedder,
        Err(e) => {
            let e = e.context("loading the embedding model");
            refuse_waiting(&listener, &format!("nngrepd: {e:#}"));
            let _ = fs::remove_file(&socket);
            return Err(e);
        }
    };
    log!(
        "embedding with {} on {}",
        embedder.model(),
        embedder.location()
    );

    let (events_tx, events) = mpsc::channel();
    let daemon = Arc::new(Daemon::new(embedder, args.batch_size, events_tx)?);
    {
        let daemon = daemon.clone();
        thread::spawn(move || {
            for event in events {
                match event {
                    Ok(event) => daemon.note_event(event),
                    Err(e) => log!("watching for changes: {e}"),
                }
            }
        });
    }
    {
        let daemon = daemon.clone();
        thread::spawn(move || indexer::run(&daemon));
    }
    {
        let daemon = daemon.clone();
        thread::spawn(move || {
            wait_for_signal(&signals);
            shut_down(&daemon);
            std::process::exit(0);
        });
    }

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                log!("accepting a connection: {e}");
                continue;
            }
        };
        let daemon = daemon.clone();
        thread::spawn(move || {
            if let Err(e) = Connection::new(&daemon, stream).and_then(|c| c.serve())
                && !is_disconnect(&e)
            {
                log!("{e:#}");
            }
        });
    }
    Ok(())
}

/// Stops indexing, once the index being updated is saved, and listening.
fn shut_down(daemon: &Daemon) {
    log!("stopping");
    daemon.stop();
    let _ = fs::remove_file(socket_path());
}

/// Blocks SIGINT and SIGTERM in this thread and the threads it starts, so
/// that wait_for_signal receives them. Returns the blocked signals.
fn block_signals() -> libc::sigset_t {
    // SAFETY: The calls only write to set, a plain C struct.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        set
    }
}

/// Waits for one of the signals blocked by block_signals.
fn wait_for_signal(signals: &libc::sigset_t) {
    let mut signal = 0;
    // SAFETY: sigwait only writes to signal.
    unsafe { libc::sigwait(signals, &mut signal) };
}

/// Answers the connections waiting to be accepted with an error.
fn refuse_waiting(listener: &UnixListener, message: &str) {
    if listener.set_nonblocking(true).is_err() {
        return;
    }
    let mut line = serde_json::to_vec(&Response::Error {
        message: message.into(),
    })
    .unwrap();
    line.push(b'\n');
    while let Ok((mut stream, _)) = listener.accept() {
        let _ = stream.write_all(&line);
    }
}

/// A client's connection, which may carry several requests in turn.
struct Connection<'a> {
    daemon: &'a Daemon,
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl<'a> Connection<'a> {
    fn new(daemon: &'a Daemon, stream: UnixStream) -> Result<Self> {
        Ok(Connection {
            daemon,
            reader: BufReader::new(stream.try_clone()?),
            writer: stream,
        })
    }

    fn serve(mut self) -> Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                return Ok(());
            }
            let request = match serde_json::from_str::<Request>(&line) {
                Ok(r) => r,
                Err(e) => {
                    self.send(&Response::Error {
                        message: format!("nngrepd: bad request: {e}"),
                    })?;
                    continue;
                }
            };
            if let Err(e) = self.handle(request) {
                if is_disconnect(&e) {
                    return Err(e);
                }
                self.send(&Response::Error {
                    message: format!("nngrepd: {e:#}"),
                })?;
            }
        }
    }

    fn handle(&mut self, request: Request) -> Result<()> {
        let daemon = self.daemon;
        match request {
            Request::Embed { queries } => {
                let vectors = daemon.embedder.embed_queries(&queries)?;
                let model = daemon.embedder.model().to_string();
                self.send(&Response::Embeddings { model, vectors })
            }
            Request::Rerank { queries, documents } => {
                let reranker = daemon.reranker()?;
                let documents: Vec<&str> = documents.iter().map(String::as_str).collect();
                let scores = queries
                    .iter()
                    .map(|q| reranker.score(q, &documents))
                    .collect::<Result<_>>()?;
                self.send(&Response::Scores { scores })
            }
            Request::Index {
                root,
                force,
                max_file_size,
            } => self.index(
                &root,
                FullScan {
                    force,
                    max_file_size,
                },
            ),
            Request::Unwatch { root } => {
                let root = fs::canonicalize(&root).unwrap_or(root);
                if !daemon.unwatch(&root) {
                    bail!("{} is not kept up to date", root.display());
                }
                self.send(&Response::Done)
            }
            Request::Status => self.send(&Response::Status {
                status: daemon.status(),
            }),
            Request::Stop => {
                shut_down(daemon);
                let _ = self.send(&Response::Done);
                std::process::exit(0);
            }
        }
    }

    /// Indexes root, sending progress and then the summary.
    fn index(&mut self, root: &Path, scan: FullScan) -> Result<()> {
        let root = fs::canonicalize(root).with_context(|| root.display().to_string())?;
        if !root.is_dir() {
            bail!("{} is not a directory", root.display());
        }
        let (events, pass) = self.daemon.request_scan(&root, scan);
        for event in events {
            match event {
                Event::Progress(p, progress) if p >= pass => {
                    self.send(&Response::Progress { progress })?;
                }
                Event::Done(p, summary) if p >= pass => {
                    return self.send(&Response::Indexed { summary });
                }
                Event::Failed(p, message) if p >= pass => bail!("{message}"),
                _ => {}
            }
        }
        bail!("{} is no longer kept up to date", root.display())
    }

    fn send(&mut self, response: &Response) -> Result<()> {
        let mut line = serde_json::to_vec(response)?;
        line.push(b'\n');
        self.writer.write_all(&line)?;
        Ok(())
    }
}

/// Did the client go away?
fn is_disconnect(e: &anyhow::Error) -> bool {
    e.downcast_ref::<io::Error>().is_some_and(|e| {
        matches!(
            e.kind(),
            io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
        )
    })
}

fn status() -> Result<()> {
    let Some(mut client) = Client::connect_running()? else {
        println!("nngrepd is not running");
        return Ok(());
    };
    client.send(&Request::Status)?;
    let Response::Status { status } = client.receive()? else {
        bail!("unexpected response from nngrepd");
    };
    let reranker = if status.reranker {
        "loaded"
    } else {
        "not loaded"
    };
    println!(
        "nngrepd {}: {} on {}, reranker {reranker}",
        status.pid, status.model, status.location
    );
    if status.roots.is_empty() {
        println!("No indexes kept up to date; add one with nnindex");
    }
    for root in status.roots {
        let state = match root.progress {
            None if root.pending => "changed".to_string(),
            None => "up to date".to_string(),
            Some(Progress::Waiting) | Some(Progress::Scanning { .. }) => "scanning".to_string(),
            Some(Progress::Embedding { done, total, .. }) => {
                format!("embedding {done}/{total} chunks")
            }
        };
        println!("{}: {state}", root.path.display());
    }
    Ok(())
}

/// Stops the running nngrepd and waits for it to exit.
fn stop() -> Result<()> {
    let Some(mut client) = Client::connect_running()? else {
        println!("nngrepd is not running");
        return Ok(());
    };
    client.send(&Request::Stop)?;
    client.receive()?;
    let lock = File::open(lock_path())?;
    let deadline = Instant::now() + STOP_TIMEOUT;
    while let Err(TryLockError::WouldBlock) = lock.try_lock() {
        if Instant::now() > deadline {
            bail!("nngrepd did not exit");
        }
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

/// Returns the local time as "YYYY-MM-DD HH:MM:SS".
fn timestamp() -> String {
    // SAFETY: localtime_r only writes to tm, which is a plain C struct.
    let tm = unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut tm);
        tm
    };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
}
