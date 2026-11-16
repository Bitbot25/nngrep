use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use half::f16;
use half::slice::HalfFloatSliceExt;
use indicatif::{ProgressBar, ProgressStyle};
use nngrep::embed::Embedder;
use nngrep::index::{ChunkSpan, FileEntry, Index};
use rayon::prelude::*;
use std::env;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

/// Search files semantically, using an index built by nnindex.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// What to search for, in natural language
    query: String,

    /// File or directory to search (must be inside an indexed directory)
    #[arg(default_value = ".")]
    path: PathBuf,

    /// Number of results
    #[arg(short = 'k', long, default_value_t = 10)]
    top: usize,

    /// Only show results with at least this similarity (0 to 1)
    #[arg(short = 's', long, default_value_t = 0.0)]
    min_score: f32,

    /// Only print the names of the best matching files
    #[arg(short = 'l', long)]
    files_with_matches: bool,

    /// Lines of each match to print
    #[arg(short = 'L', long, default_value_t = 5)]
    lines: usize,

    /// Print each match in full
    #[arg(long, conflicts_with = "lines")]
    full: bool,

    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    color: ColorChoice,
}

#[derive(Clone, Copy, ValueEnum)]
enum ColorChoice {
    Auto,
    Always,
    Never,
}

/// A scored chunk
struct Hit<'a> {
    score: f32,
    file: &'a FileEntry,
    span: ChunkSpan,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let target = fs::canonicalize(&args.path)
        .with_context(|| format!("cannot access {}", args.path.display()))?;
    let (root, index) = Index::find(&target)?;
    let prefix = target
        .strip_prefix(&root)?
        .to_str()
        .context("non-UTF-8 path")?
        .to_string();
    let files: Vec<(&FileEntry, &[f16])> = index
        .iter_files()
        .filter(|(f, _)| is_under(&f.path, &prefix))
        .collect();

    let loading = ProgressBar::new_spinner()
        .with_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap())
        .with_message("Connecting to embedding server");
    loading.enable_steady_tick(Duration::from_millis(100));
    let embedder = Embedder::connect()?;
    if embedder.model() != index.model {
        bail!(
            "index was built with {} but the embedding server runs {}",
            index.model,
            embedder.model()
        );
    }
    let query = embedder.embed_query(&args.query)?;
    loading.finish_and_clear();
    drop(embedder);

    let mut hits = score(&files, &query, index.dim);
    hits.retain(|h| h.score >= args.min_score);
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));

    let color = match args.color {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => std::io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none(),
    };
    let style = Style::new(color);
    let cwd = env::current_dir()?;
    let mut out = std::io::stdout().lock();

    if args.files_with_matches {
        let mut seen = Vec::new();
        for hit in hits {
            if seen.len() == args.top {
                break;
            }
            if !seen.contains(&hit.file.path) {
                let path = display_path(&root.join(&hit.file.path), &cwd);
                writeln!(out, "{}  {}", style.score(hit.score), style.path(&path))?;
                seen.push(hit.file.path.clone());
            }
        }
        return Ok(());
    }

    for (i, hit) in hits.iter().take(args.top).enumerate() {
        let abs = root.join(&hit.file.path);
        let path = display_path(&abs, &cwd);
        if i > 0 && args.lines > 0 {
            writeln!(out)?;
        }
        write!(
            out,
            "{}  {}:{}",
            style.score(hit.score),
            style.path(&path),
            style.line(&format!("{}-{}", hit.span.start, hit.span.end)),
        )?;
        let text = read_if_current(&abs, hit.file);
        if text.is_none() {
            write!(out, " {}", style.dim("(changed since indexing)"))?;
        }
        writeln!(out)?;

        let max_lines = if args.full { usize::MAX } else { args.lines };
        if let Some(text) = text {
            let lines = text
                .lines()
                .enumerate()
                .skip(hit.span.start as usize - 1)
                .take((hit.span.end - hit.span.start + 1) as usize)
                .skip_while(|(_, l)| l.trim().is_empty())
                .take(max_lines);
            for (n, line) in lines {
                writeln!(out, "{}:{line}", style.line(&(n + 1).to_string()))?;
            }
        }
    }
    Ok(())
}

/// Is the index path inside the directory or file at prefix?
fn is_under(path: &str, prefix: &str) -> bool {
    prefix.is_empty()
        || path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Returns the cosine similarity of every chunk to the query.
fn score<'a>(files: &[(&'a FileEntry, &[f16])], query: &[f32], dim: usize) -> Vec<Hit<'a>> {
    files
        .par_iter()
        .flat_map_iter(|&(file, vectors)| {
            let mut buf = vec![0f32; dim];
            file.chunks
                .iter()
                .zip(vectors.chunks_exact(dim))
                .map(move |(&span, v)| {
                    v.convert_to_f32_slice(&mut buf);
                    let score = buf.iter().zip(query).map(|(a, b)| a * b).sum();
                    Hit { score, file, span }
                })
        })
        .collect()
}

/// Returns the file's contents if it has not been modified since indexing.
fn read_if_current(path: &Path, file: &FileEntry) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    if mtime != file.mtime || meta.len() != file.size {
        return None;
    }
    fs::read_to_string(path).ok()
}

fn display_path(path: &Path, cwd: &Path) -> String {
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

/// ANSI styling, or none
struct Style {
    color: bool,
}

impl Style {
    fn new(color: bool) -> Self {
        Style { color }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    fn path(&self, text: &str) -> String {
        self.paint("35", text)
    }

    fn line(&self, text: &str) -> String {
        self.paint("32", text)
    }

    fn score(&self, score: f32) -> String {
        self.paint("1;33", &format!("{score:.3}"))
    }

    fn dim(&self, text: &str) -> String {
        self.paint("2", text)
    }
}
