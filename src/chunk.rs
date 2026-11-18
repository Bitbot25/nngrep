use std::ops::Range;

/// Maximum number of lines in a chunk
pub const MAX_LINES: usize = 20;

/// Maximum size of a chunk's text in bytes. Keeps every chunk well below the
/// embedding server's per-sequence context, since a token is at least one byte.
pub const MAX_BYTES: usize = 900;

/// Chunks shorter than this many lines are joined to the next block if it fits
const MIN_LINES: usize = 4;

/// A piece of a text file to be embedded.
pub struct Chunk {
    /// First line (1-based)
    pub start_line: u32,
    /// Last line (1-based, inclusive)
    pub end_line: u32,
    pub text: String,
}

/// Splits text into chunks of whole lines, following its layout.
///
/// The text is read as blocks of paragraphs (runs of non-blank lines). A
/// paragraph starts a new block unless it is indented deeper than the first
/// line of the current block, or begins with a closing bracket. Each block
/// becomes its own chunk, except that chunks shorter than MIN_LINES are joined
/// to the next. A block too big for one chunk is split into its first
/// paragraph and the blocks among its other paragraphs, which are packed
/// together as tightly as the limits allow. Over-long lines are truncated.
pub fn chunk_text(text: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let paragraphs = paragraphs(&lines);

    let mut pieces = Vec::new();
    for block in blocks(&lines, &paragraphs) {
        split_block(&lines, &paragraphs[block], &mut pieces);
    }

    let mut spans: Vec<Range<usize>> = Vec::new();
    for piece in pieces {
        match spans.last_mut() {
            Some(last) if last.len() < MIN_LINES && fits(&lines, last.start..piece.end) => {
                last.end = piece.end;
            }
            _ => spans.push(piece),
        }
    }

    spans
        .into_iter()
        .map(|span| Chunk {
            start_line: span.start as u32 + 1,
            end_line: span.end as u32,
            text: truncate(&lines[span].join("\n"), MAX_BYTES).to_string(),
        })
        .collect()
}

/// Returns the line ranges of the runs of non-blank lines.
fn paragraphs(lines: &[&str]) -> Vec<Range<usize>> {
    let mut paragraphs = Vec::new();
    let mut start = None;
    for (i, line) in lines.iter().enumerate() {
        match (line.trim().is_empty(), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                paragraphs.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        paragraphs.push(s..lines.len());
    }
    paragraphs
}

/// Groups paragraphs into blocks, returned as ranges of indexes into
/// paragraphs.
fn blocks(lines: &[&str], paragraphs: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut blocks: Vec<Range<usize>> = Vec::new();
    for (i, p) in paragraphs.iter().enumerate() {
        let first = lines[p.start];
        let continues = blocks.last().is_some_and(|b| {
            let head = lines[paragraphs[b.start].start];
            indent(first) > indent(head) || is_closing(first)
        });
        match blocks.last_mut() {
            Some(b) if continues => b.end = i + 1,
            _ => blocks.push(i..i + 1),
        }
    }
    blocks
}

/// Appends the line ranges of the chunks of a block made of paragraphs.
fn split_block(lines: &[&str], paragraphs: &[Range<usize>], out: &mut Vec<Range<usize>>) {
    let whole = paragraphs[0].start..paragraphs[paragraphs.len() - 1].end;
    if fits(lines, whole.clone()) {
        out.push(whole);
        return;
    }
    if paragraphs.len() == 1 {
        split_lines(lines, whole, out);
        return;
    }

    let mut pieces = Vec::new();
    split_lines(lines, paragraphs[0].clone(), &mut pieces);
    let rest = &paragraphs[1..];
    for block in blocks(lines, rest) {
        split_block(lines, &rest[block], &mut pieces);
    }
    let mut packed: Vec<Range<usize>> = Vec::new();
    for piece in pieces {
        match packed.last_mut() {
            Some(last) if fits(lines, last.start..piece.end) => last.end = piece.end,
            _ => packed.push(piece),
        }
    }
    out.extend(packed);
}

/// Appends a range of lines cut into the longest runs that fit in a chunk.
fn split_lines(lines: &[&str], range: Range<usize>, out: &mut Vec<Range<usize>>) {
    let mut start = range.start;
    while start < range.end {
        let mut end = start + 1;
        while end < range.end && fits(lines, start..end + 1) {
            end += 1;
        }
        out.push(start..end);
        start = end;
    }
}

/// Do the lines in range fit in one chunk?
fn fits(lines: &[&str], range: Range<usize>) -> bool {
    range.len() <= MAX_LINES && lines[range].iter().map(|l| l.len() + 1).sum::<usize>() <= MAX_BYTES
}

/// Returns the width of the line's leading whitespace, with tab stops every
/// 8 columns.
fn indent(line: &str) -> usize {
    let mut width = 0;
    for c in line.chars() {
        match c {
            ' ' => width += 1,
            '\t' => width = (width / 8 + 1) * 8,
            _ => break,
        }
    }
    width
}

/// Does the line begin by closing a bracket or tag?
fn is_closing(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with(['}', ')', ']']) || line.starts_with("</")
}

/// Returns the longest prefix of s that is at most max bytes.
pub fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut i = max;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    &s[..i]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(text: &str) -> Vec<(u32, u32)> {
        chunk_text(text)
            .iter()
            .map(|c| (c.start_line, c.end_line))
            .collect()
    }

    #[test]
    fn splits_at_line_limit() {
        let text: String = (1..=50).map(|i| format!("line {i}\n")).collect();
        assert_eq!(spans(&text), [(1, 20), (21, 40), (41, 50)]);
    }

    #[test]
    fn keeps_declarations_apart() {
        let text = "\
/* A schema. */
struct schema {
\tint ntables;
};

/* Operators. */
enum op {
\tEQ,
\tLT,
};
";
        assert_eq!(spans(text), [(1, 4), (6, 10)]);
    }

    #[test]
    fn keeps_indented_paragraphs_with_their_block() {
        let text = "\
int f(void)
{
\tint x = 1;

\tx += 2;
\treturn x;

}

int g(void)
{
\treturn 0;
}
";
        assert_eq!(spans(text), [(1, 8), (10, 13)]);
    }

    #[test]
    fn joins_short_blocks_to_the_next() {
        let text = "\
#include <a.h>

#include <b.h>

int f(void)
{
\treturn 0;
}
";
        assert_eq!(spans(text), [(1, 8)]);
    }

    #[test]
    fn splits_big_blocks_between_their_paragraphs() {
        let body: String = (1..=12).map(|i| format!("    x{i} = 0\n")).collect();
        let text = format!("class A:\n    def f(self):\n{body}\n    def g(self):\n{body}");
        assert_eq!(spans(&text), [(1, 14), (16, 28)]);
    }

    #[test]
    fn truncates_long_lines() {
        let text = "é".repeat(MAX_BYTES);
        let chunks = chunk_text(&text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.len() <= MAX_BYTES);
    }

    #[test]
    fn skips_blank_text() {
        assert!(chunk_text("\n  \n\t\n").is_empty());
    }
}
