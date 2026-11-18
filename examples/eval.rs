//! Measures search quality on a set of queries with known answers.
//!
//! Usage: cargo run --release --example eval -- ROOT QUERIES [--no-rerank]
//!
//! QUERIES has one query per line, followed by a tab and the answers as
//! space-separated PATH:LINE pairs, PATH relative to ROOT. A chunk answers a
//! query if it contains one of its lines. Lines starting with '#' are ignored.
//!
//! For each query, prints the rank of the best answering chunk by cosine
//! similarity, by BM25 and after reranking the best 15 of each (as
//! `nngrep --rerank` does), followed by the three best chunks after reranking.
//! The environment variable DEPTH changes the number reranked.

use anyhow::{Context, Result, bail};
use half::slice::HalfFloatSliceExt;
use nngrep::bm25::Bm25;
use nngrep::chunk::{MAX_BYTES, truncate};
use nngrep::embed::Embedder;
use nngrep::index::{ChunkSpan, Index};
use nngrep::llama::Device;
use nngrep::rerank::Reranker;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Number of chunks reranked from each of cosine similarity and BM25
fn depth() -> usize {
    env::var("DEPTH")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(15)
}

struct Query {
    text: String,
    answers: Vec<(String, u32)>,
}

struct Chunk<'a> {
    path: &'a str,
    span: ChunkSpan,
    text: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let rerank = !args.iter().any(|a| a == "--no-rerank");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [root, queries] = positional[..] else {
        bail!("usage: eval ROOT QUERIES [--no-rerank]");
    };
    let root = PathBuf::from(root);
    let queries = parse_queries(&fs::read_to_string(queries)?)?;

    let index = Index::load(&root)?;
    let mut chunks = Vec::new();
    for file in &index.files {
        let text = fs::read_to_string(root.join(&file.path)).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        for &span in &file.chunks {
            let start = (span.start as usize - 1).min(lines.len());
            let end = (span.end as usize).min(lines.len());
            let text = truncate(&lines[start..end].join("\n"), MAX_BYTES).to_string();
            chunks.push(Chunk {
                path: &file.path,
                span,
                text,
            });
        }
    }

    let embedder = Embedder::connect()?;
    let query_vectors: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| embedder.embed_query(&q.text))
        .collect::<Result<_>>()?;
    drop(embedder);

    let mut chunk_vectors = vec![0f32; index.vectors.len()];
    index.vectors.convert_to_f32_slice(&mut chunk_vectors);
    let texts: Vec<String> = queries.iter().map(|q| q.text.clone()).collect();
    let bm25 = Bm25::new(&texts);
    let mut counter = bm25.counter();
    let counts: Vec<_> = chunks.iter().map(|c| counter.count(&c.text)).collect();
    let bm25_scores = bm25.scores(&counts);

    let mut reranker = if rerank {
        Some(Reranker::load(Device::Gpu)?)
    } else {
        None
    };

    let mut ranks: [Vec<Option<usize>>; 3] = Default::default();
    println!("{:>5} {:>5} {:>6}  query", "cos", "bm25", "rerank");
    for (q, query) in queries.iter().enumerate() {
        let is_answer = |c: &Chunk| {
            query
                .answers
                .iter()
                .any(|(p, l)| c.path == p && (c.span.start..=c.span.end).contains(l))
        };
        if !chunks.iter().any(is_answer) {
            bail!("no chunk answers {:?}", query.text);
        }

        let cosine: Vec<f32> = chunk_vectors
            .chunks_exact(index.dim)
            .map(|v| v.iter().zip(&query_vectors[q]).map(|(a, b)| a * b).sum())
            .collect();
        let by_cosine = order(&cosine);
        let by_bm25: Vec<usize> = order(&bm25_scores[q])
            .into_iter()
            .filter(|&i| bm25_scores[q][i] > 0.0)
            .collect();
        let rank_in = |order: &[usize]| order.iter().position(|&i| is_answer(&chunks[i]));

        let mut top = Vec::new();
        let reranked = match &mut reranker {
            Some(reranker) => {
                let pool: Vec<usize> = by_cosine
                    .iter()
                    .take(depth())
                    .chain(by_bm25.iter().take(depth()))
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let docs: Vec<&str> = pool.iter().map(|&i| chunks[i].text.as_str()).collect();
                let scores = reranker.score(&query.text, &docs)?;
                let by_rerank: Vec<usize> = order(&scores).into_iter().map(|i| pool[i]).collect();
                top = by_rerank.iter().take(3).copied().collect();
                rank_in(&by_rerank)
            }
            None => None,
        };
        if top.is_empty() {
            top = by_cosine.iter().take(3).copied().collect();
        }

        let row = [rank_in(&by_cosine), rank_in(&by_bm25), reranked];
        println!(
            "{:>5} {:>5} {:>6}  {}",
            show(row[0]),
            show(row[1]),
            if rerank { show(row[2]) } else { "".into() },
            query.text
        );
        for &i in &top {
            let mark = if is_answer(&chunks[i]) { "*" } else { " " };
            let c = &chunks[i];
            println!(
                "{:>20}{mark} {}:{}-{}",
                "", c.path, c.span.start, c.span.end
            );
        }
        for (all, rank) in ranks.iter_mut().zip(row) {
            all.push(rank);
        }
    }

    println!();
    println!(
        "{:>8} {:>6} {:>6} {:>6} {:>6}",
        "", "top1", "top3", "top10", "MRR"
    );
    let names = ["cosine", "bm25", "rerank"];
    for (name, all) in names.iter().zip(&ranks).take(if rerank { 3 } else { 2 }) {
        let within = |k: usize| all.iter().filter(|r| r.is_some_and(|r| r < k)).count();
        let mrr: f64 = all
            .iter()
            .map(|r| r.map_or(0.0, |r| 1.0 / (r + 1) as f64))
            .sum::<f64>()
            / all.len() as f64;
        println!(
            "{name:>8} {:>6} {:>6} {:>6} {mrr:>6.3}",
            within(1),
            within(3),
            within(10)
        );
    }
    Ok(())
}

fn parse_queries(text: &str) -> Result<Vec<Query>> {
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|line| {
            let (text, answers) = line.split_once('\t').context("missing tab")?;
            let answers = answers
                .split_whitespace()
                .map(|a| {
                    let (path, line) = a.rsplit_once(':').context("answer is not PATH:LINE")?;
                    Ok((path.to_string(), line.parse()?))
                })
                .collect::<Result<_>>()?;
            Ok(Query {
                text: text.to_string(),
                answers,
            })
        })
        .collect()
}

/// Returns the indexes of scores, highest score first.
fn order(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}

/// Formats a 0-based rank as 1-based, or "-" if missing.
fn show(rank: Option<usize>) -> String {
    rank.map_or("-".into(), |r| (r + 1).to_string())
}
