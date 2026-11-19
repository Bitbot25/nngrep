use anyhow::{Context, Result, bail};
use clap::Parser;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nngrep::ipc::{Client, Progress, Request, Response};
use nngrep::progress::spinner;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

/// Index a directory for semantic search with nngrep, and keep the index up
/// to date.
///
/// Respects .gitignore and skips hidden, binary and non-UTF-8 files. Unchanged
/// files keep their embeddings from the previous index. The indexing is done
/// by nngrepd, which nnindex starts if it is not running, and which then
/// updates the index whenever files change. Interrupting nnindex leaves
/// nngrepd indexing.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Directory to index
    #[arg(default_value = ".")]
    dir: PathBuf,

    /// Re-embed all files, even unchanged ones
    #[arg(long)]
    force: bool,

    /// Skip files larger than this many bytes [default: as before, or 1048576]
    #[arg(long, value_name = "BYTES")]
    max_file_size: Option<u64>,

    /// Stop keeping the index of DIR up to date, leaving it as it is
    #[arg(long, conflicts_with_all = ["force", "max_file_size"])]
    unwatch: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = fs::canonicalize(&args.dir)
        .with_context(|| format!("cannot access {}", args.dir.display()))?;

    if args.unwatch {
        let Some(mut client) = Client::connect_running()? else {
            bail!("nngrepd is not running, so it keeps nothing up to date");
        };
        client.send(&Request::Unwatch { root: root.clone() })?;
        client.receive()?;
        eprintln!("No longer keeping {} up to date", root.display());
        return Ok(());
    }

    let waiting = spinner("{msg}");
    waiting.set_message("Waiting for nngrepd");
    let mut client = match Client::connect_running()? {
        Some(client) => client,
        None => {
            waiting.set_message("Starting nngrepd");
            Client::connect()?
        }
    };
    client.send(&Request::Index {
        root: root.clone(),
        force: args.force,
        max_file_size: args.max_file_size,
    })?;

    let mut bar: Option<ProgressBar> = None;
    let summary = loop {
        match client.receive()? {
            Response::Progress { progress } => match progress {
                Progress::Waiting => waiting.set_message("Waiting for an earlier update"),
                Progress::Scanning { files } => {
                    waiting.set_message(format!("Scanning files {files}"));
                }
                Progress::Embedding { done, total, path } => {
                    waiting.finish_and_clear();
                    let bar = bar.get_or_insert_with(|| embedding_bar(total));
                    bar.set_position(done as u64);
                    bar.set_message(path);
                }
            },
            Response::Indexed { summary } => break summary,
            _ => bail!("unexpected response from nngrepd"),
        }
    };
    waiting.finish_and_clear();
    if let Some(bar) = bar {
        bar.finish_and_clear();
    }

    eprintln!(
        "{} files: {} unchanged, {} embedded ({} chunks), {} skipped, {} removed",
        summary.files,
        summary.unchanged,
        summary.embedded,
        summary.embedded_chunks,
        summary.skipped,
        summary.removed
    );
    eprintln!(
        "Indexed {} files ({} chunks) in {:.1} s; nngrepd keeps the index up to date",
        summary.files, summary.chunks, summary.seconds
    );
    Ok(())
}

fn embedding_bar(total: usize) -> ProgressBar {
    let bar = ProgressBar::new(total as u64).with_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({percent}%) · {speed} · ETA {eta} {wide_msg:.dim}",
        )
        .unwrap()
        .with_key("speed", |s: &ProgressState, w: &mut dyn std::fmt::Write| {
            write!(w, "{:.1} chunks/s", s.per_sec()).unwrap()
        })
        .progress_chars("=> "),
    );
    bar.enable_steady_tick(Duration::from_millis(100));
    bar
}
