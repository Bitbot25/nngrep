use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use half::f16;
use half::slice::HalfFloatSliceExt;
use ignore::Match;
use ignore::overrides::{Override, OverrideBuilder};
use ignore::types::{Types, TypesBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use nngrep::embed::{self, Embedder};
use nngrep::index::{ChunkSpan, FileEntry, Index};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, UNIX_EPOCH};

/// Search files semantically, using indexes built by nnindex.
///
/// Prints the chunks most similar to QUERY, best first. The exit status is 0
/// if anything was found, 1 if nothing was found and 2 if an error occurred.
#[derive(Parser)]
#[command(
    version,
    override_usage = "nngrep [OPTIONS] QUERY [PATH ...]\n       \
                      nngrep [OPTIONS] -e QUERY ... [PATH ...]\n       \
                      nngrep --type-list"
)]
struct Args {
    /// What to search for, in natural language
    #[arg(required_unless_present_any = ["queries", "type_list"])]
    query: Option<OsString>,

    /// Files or directories to search, each inside an indexed directory
    /// [default: .]
    #[arg(value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Search for QUERY (may be repeated; a chunk scores its best match).
    /// All positional arguments are then paths.
    #[arg(short = 'e', long = "query", value_name = "QUERY")]
    queries: Vec<String>,

    /// Number of results
    #[arg(short = 'k', long, value_name = "NUM", default_value_t = 10)]
    top: usize,

    /// Show at most NUM chunks per file
    #[arg(short = 'm', long, value_name = "NUM")]
    max_count: Option<usize>,

    /// Only show chunks with at least this similarity (0 to 1)
    #[arg(long, value_name = "SCORE", default_value_t = 0.0)]
    min_score: f32,

    /// Only print the paths of the best matching files
    #[arg(short = 'l', long)]
    files_with_matches: bool,

    /// Follow each path printed by -l with a NUL byte instead of a newline
    #[arg(short = '0', long)]
    null: bool,

    /// Print nothing; only set the exit status
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Lines of each chunk to print
    #[arg(long, value_name = "NUM", default_value_t = 5)]
    lines: usize,

    /// Print each chunk in full
    #[arg(long, conflicts_with = "lines")]
    full: bool,

    /// Show line numbers (default when printing to a terminal)
    #[arg(short = 'n', long, overrides_with = "no_line_number")]
    line_number: bool,

    /// Hide line numbers
    #[arg(short = 'N', long, overrides_with = "line_number")]
    no_line_number: bool,

    /// Print a summary line above each chunk instead of the path on every
    /// line (default when printing to a terminal)
    #[arg(long, overrides_with = "no_heading")]
    heading: bool,

    /// Prefix every printed line with its path
    #[arg(long, overrides_with = "heading")]
    no_heading: bool,

    /// Include or exclude (with a leading '!') files matching GLOB, matched
    /// like a .gitignore pattern relative to the current directory
    #[arg(short = 'g', long = "glob", value_name = "GLOB")]
    globs: Vec<String>,

    /// Only search files of type TYPE (see --type-list)
    #[arg(short = 't', long = "type", value_name = "TYPE")]
    types: Vec<String>,

    /// Do not search files of type TYPE
    #[arg(short = 'T', long = "type-not", value_name = "TYPE")]
    types_not: Vec<String>,

    /// List the file types for -t and -T
    #[arg(long)]
    type_list: bool,

    #[arg(long, value_name = "WHEN", value_enum, default_value_t = ColorChoice::Auto)]
    color: ColorChoice,
}

#[derive(Clone, Copy, ValueEnum)]
enum ColorChoice {
    Auto,
    Always,
    Never,
}

/// An index and the paths searched in it
struct Source {
    root: PathBuf,
    index: Index,
    targets: Vec<Target>,
}

/// A file or directory to search
struct Target {
    /// Path as given on the command line (None for the default, the current
    /// directory)
    given: Option<PathBuf>,
    abs: PathBuf,
    /// Path relative to the index root, with '/' separators
    prefix: String,
}

/// An indexed file selected for searching
struct Candidate<'a> {
    source: usize,
    target: &'a Target,
    file: &'a FileEntry,
    vectors: &'a [f16],
}

/// A scored chunk
struct Hit {
    score: f32,
    /// Index into the candidates
    candidate: usize,
    span: ChunkSpan,
}

/// Result of a search
struct Outcome {
    found: bool,
    /// Was an error reported along the way?
    errors: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let quiet = args.quiet;
    match run(args) {
        Ok(Outcome { found: true, .. }) if quiet => ExitCode::SUCCESS,
        Ok(Outcome { errors: true, .. }) => ExitCode::from(2),
        Ok(Outcome { found, .. }) => ExitCode::from(if found { 0 } else { 1 }),
        Err(e) if is_broken_pipe(&e) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nngrep: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Args) -> Result<Outcome> {
    let mut types = TypesBuilder::new();
    types.add_defaults();
    if args.type_list {
        let mut out = io::stdout().lock();
        for def in types.definitions() {
            writeln!(out, "{}: {}", def.name(), def.globs().join(", "))?;
        }
        return Ok(Outcome {
            found: true,
            errors: false,
        });
    }
    for t in &args.types {
        types.select(t);
    }
    for t in &args.types_not {
        types.negate(t);
    }
    let cwd = env::current_dir()?;
    let filter = FileFilter::new(&args.globs, types.build()?, &cwd)?;

    let mut queries = args.queries.clone();
    let mut paths: Vec<Option<PathBuf>> = args.paths.iter().cloned().map(Some).collect();
    if let Some(query) = args.query.clone() {
        if queries.is_empty() {
            let query = query
                .into_string()
                .map_err(|_| anyhow!("query is not valid UTF-8"))?;
            queries.push(query);
        } else {
            paths.insert(0, Some(query.into()));
        }
    }
    if paths.is_empty() {
        paths.push(None);
    }

    let mut errors = false;
    let mut sources: Vec<Source> = Vec::new();
    for given in paths {
        if let Err(e) = add_target(&mut sources, given) {
            eprintln!("nngrep: {e:#}");
            errors = true;
        }
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for (s, source) in sources.iter().enumerate() {
        for (file, vectors) in source.index.iter_files() {
            let target = source
                .targets
                .iter()
                .find(|t| is_under(&file.path, &t.prefix) && filter.admits(t, &file.path));
            if let Some(target) = target {
                candidates.push(Candidate {
                    source: s,
                    target,
                    file,
                    vectors,
                });
            }
        }
    }
    if candidates.is_empty() {
        return Ok(Outcome {
            found: false,
            errors,
        });
    }

    let loading = ProgressBar::new_spinner()
        .with_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap())
        .with_message("Starting embedding server");
    loading.enable_steady_tick(Duration::from_millis(100));
    let embedder = Embedder::connect()?;
    let queries: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| embedder.embed_query(q))
        .collect::<Result<_>>()?;
    loading.finish_and_clear();
    drop(embedder);

    let dim = queries[0].len();
    for source in &sources {
        if source.index.dim != dim {
            eprintln!(
                "nngrep: index in {} has dimension {}, the embedding server returns {dim}",
                source.root.display(),
                source.index.dim
            );
            errors = true;
        }
    }
    candidates.retain(|c| sources[c.source].index.dim == dim);

    let mut hits = score(&candidates, &queries);
    hits.retain(|h| h.score >= args.min_score);
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));

    let selected: Vec<Hit> = if args.files_with_matches {
        let mut seen = HashSet::new();
        hits.into_iter()
            .filter(|h| seen.insert(h.candidate))
            .take(args.top)
            .collect()
    } else {
        let max_count = args.max_count.unwrap_or(usize::MAX);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        hits.into_iter()
            .filter(|h| {
                let count = counts.entry(h.candidate).or_default();
                *count += 1;
                *count <= max_count
            })
            .take(args.top)
            .collect()
    };
    let found = !selected.is_empty();
    if !args.quiet {
        Printer::new(&args).print(&selected, &candidates, &sources)?;
    }
    Ok(Outcome { found, errors })
}

/// Adds the file or directory at given (or the current directory) to the
/// source whose index covers it, loading that index if needed.
fn add_target(sources: &mut Vec<Source>, given: Option<PathBuf>) -> Result<()> {
    let path = given.as_deref().unwrap_or(Path::new("."));
    let abs = fs::canonicalize(path).with_context(|| path.display().to_string())?;
    let root = Index::find_root(&abs)?;
    let prefix = abs
        .strip_prefix(&root)?
        .to_str()
        .with_context(|| format!("{}: path is not valid UTF-8", path.display()))?
        .to_string();
    let target = Target { given, abs, prefix };

    if let Some(source) = sources.iter_mut().find(|s| s.root == root) {
        source.targets.push(target);
        return Ok(());
    }
    let index = Index::load(&root)?;
    let model = embed::model_name();
    if !embed::compatible(&index.model, &model) {
        eprintln!(
            "nngrep: warning: index in {} was built with {}, searching with {model}",
            root.display(),
            index.model
        );
    }
    sources.push(Source {
        root,
        index,
        targets: vec![target],
    });
    Ok(())
}

impl Target {
    /// Returns the part of the index path below this target ("" for the
    /// target itself).
    fn rest<'a>(&self, path: &'a str) -> &'a str {
        path[self.prefix.len()..].trim_start_matches('/')
    }

    /// Returns how to print the index path, relative to the path as given.
    fn display(&self, path: &str) -> PathBuf {
        let rest = self.rest(path);
        match &self.given {
            None => PathBuf::from(rest),
            Some(given) if rest.is_empty() => given.clone(),
            Some(given) => given.join(rest),
        }
    }
}

/// Is the index path inside the directory or file at prefix?
fn is_under(path: &str, prefix: &str) -> bool {
    prefix.is_empty()
        || path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// File selection by --glob, --type and --type-not
struct FileFilter {
    overrides: Override,
    types: Types,
}

impl FileFilter {
    fn new(globs: &[String], types: Types, cwd: &Path) -> Result<Self> {
        let mut overrides = OverrideBuilder::new(cwd);
        for glob in globs {
            overrides.add(glob)?;
        }
        Ok(FileFilter {
            overrides: overrides.build()?,
            types,
        })
    }

    /// Should the index path under target be searched? Paths given on the
    /// command line always are, and so are files under them unless they or a
    /// directory between them and the target is filtered out.
    fn admits(&self, target: &Target, path: &str) -> bool {
        if self.overrides.is_empty() && self.types.is_empty() {
            return true;
        }
        let rest = target.rest(path);
        if rest.is_empty() {
            return true;
        }
        let excluded_dir = rest
            .match_indices('/')
            .any(|(i, _)| self.excludes(&target.abs.join(&rest[..i]), true));
        !excluded_dir && !self.excludes(&target.abs.join(rest), false)
    }

    fn excludes(&self, path: &Path, is_dir: bool) -> bool {
        match self.overrides.matched(path, is_dir) {
            Match::Ignore(_) => true,
            Match::Whitelist(_) => false,
            Match::None => self.types.matched(path, is_dir).is_ignore(),
        }
    }
}

/// Returns every chunk of the candidates scored by its best cosine similarity
/// to the queries.
fn score(candidates: &[Candidate], queries: &[Vec<f32>]) -> Vec<Hit> {
    let dim = queries[0].len();
    candidates
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, c)| {
            let mut buf = vec![0f32; dim];
            c.file
                .chunks
                .iter()
                .zip(c.vectors.chunks_exact(dim))
                .map(move |(&span, v)| {
                    v.convert_to_f32_slice(&mut buf);
                    let score = queries
                        .iter()
                        .map(|q| buf.iter().zip(q).map(|(a, b)| a * b).sum::<f32>())
                        .fold(f32::NEG_INFINITY, f32::max);
                    Hit {
                        score,
                        candidate: i,
                        span,
                    }
                })
        })
        .collect()
}

/// Output settings
struct Printer {
    style: Style,
    files_only: bool,
    null: bool,
    heading: bool,
    line_number: bool,
    max_lines: usize,
}

impl Printer {
    fn new(args: &Args) -> Self {
        let tty = io::stdout().is_terminal();
        let color = match args.color {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => tty && env::var_os("NO_COLOR").is_none(),
        };
        Printer {
            style: Style { color },
            files_only: args.files_with_matches,
            null: args.null,
            heading: (tty || args.heading) && !args.no_heading,
            line_number: (tty || args.line_number) && !args.no_line_number,
            max_lines: if args.full { usize::MAX } else { args.lines },
        }
    }

    fn print(&self, hits: &[Hit], candidates: &[Candidate], sources: &[Source]) -> Result<()> {
        let style = &self.style;
        let mut out = io::stdout().lock();
        let mut first = true;
        for hit in hits {
            let c = &candidates[hit.candidate];
            let path = c.target.display(&c.file.path).display().to_string();
            if self.files_only {
                let end = if self.null { "\0" } else { "\n" };
                write!(out, "{}{end}", style.path(&path))?;
                continue;
            }

            let abs = sources[c.source].root.join(&c.file.path);
            let text = if self.max_lines > 0 {
                read_if_current(&abs, c.file)
            } else {
                None
            };
            if !self.heading && self.max_lines > 0 && text.is_none() {
                eprintln!("nngrep: {path}: changed since indexing");
                continue;
            }
            let separate = !std::mem::take(&mut first) && self.max_lines > 0;
            if self.heading || self.max_lines == 0 {
                if separate {
                    writeln!(out)?;
                }
                write!(
                    out,
                    "{}  {}:{}",
                    style.score(hit.score),
                    style.path(&path),
                    style.line(&format!("{}-{}", hit.span.start, hit.span.end)),
                )?;
                if text.is_none() && self.max_lines > 0 {
                    write!(out, " {}", style.dim("(changed since indexing)"))?;
                }
                writeln!(out)?;
            } else if separate {
                writeln!(out, "--")?;
            }

            let Some(text) = text else { continue };
            let lines = text
                .lines()
                .enumerate()
                .skip(hit.span.start as usize - 1)
                .take((hit.span.end - hit.span.start + 1) as usize)
                .skip_while(|(_, l)| l.trim().is_empty())
                .take(self.max_lines);
            for (n, line) in lines {
                if !self.heading {
                    write!(out, "{}:", style.path(&path))?;
                }
                if self.line_number {
                    write!(out, "{}:", style.line(&(n + 1).to_string()))?;
                }
                writeln!(out, "{line}")?;
            }
        }
        out.flush()?;
        Ok(())
    }
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

fn is_broken_pipe(e: &anyhow::Error) -> bool {
    e.downcast_ref::<io::Error>()
        .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
}

/// ANSI styling, or none
struct Style {
    color: bool,
}

impl Style {
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
