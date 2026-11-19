//! Scores query-document pairs with the reranker and prints the logit of each
//! relevance, one per line.
//!
//! Usage: cargo run --release --example rerank < PAIRS
//!
//! PAIRS holds a query and a document per pair, each followed by a NUL byte.
//! Consecutive pairs with the same query are reranked together, as nngrep
//! reranks the chunks found for a query.

use anyhow::{Result, bail};
use nngrep::embed::Embedder;
use nngrep::rerank::Reranker;
use std::io::{self, Read};
use std::time::Instant;

fn main() -> Result<()> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let input = String::from_utf8(input)?;
    let fields: Vec<&str> = input.split_terminator('\0').collect();
    let (pairs, rest) = fields.as_chunks::<2>();
    if !rest.is_empty() {
        bail!("PAIRS has a query without a document");
    }

    let embedder = Embedder::connect()?;
    let start = Instant::now();
    let reranker = Reranker::new(&embedder)?;
    let created = start.elapsed();

    let start = Instant::now();
    let mut relevance = Vec::with_capacity(pairs.len());
    for group in pairs.chunk_by(|a, b| a[0] == b[0]) {
        let documents: Vec<&str> = group.iter().map(|p| p[1]).collect();
        relevance.extend(reranker.score(group[0][0], &documents)?);
    }
    let scored = start.elapsed();

    for r in relevance {
        let r = f64::from(r);
        println!("{:.6}", (r / (1.0 - r)).ln());
    }
    eprintln!(
        "created in {:.2} s, scored {} pairs in {:.2} s",
        created.as_secs_f64(),
        pairs.len(),
        scored.as_secs_f64()
    );
    Ok(())
}
