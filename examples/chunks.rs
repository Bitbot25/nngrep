//! Prints the line ranges that a file is chunked into, with each chunk's
//! first line.
//!
//! Usage: cargo run --release --example chunks -- PATH

use nngrep::chunk::chunk_text;
use std::env;
use std::fs;

fn main() -> std::io::Result<()> {
    let path = env::args().nth(1).expect("usage: chunks PATH");
    for c in chunk_text(&fs::read_to_string(path)?) {
        let first = c.text.lines().next().unwrap_or("");
        println!(
            "{:>4}-{:<4} {}",
            c.start_line,
            c.end_line,
            first.chars().take(60).collect::<String>()
        );
    }
    Ok(())
}
