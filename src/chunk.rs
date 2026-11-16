/// Maximum number of lines in a chunk
pub const MAX_LINES: usize = 20;

/// Maximum size of a chunk's text in bytes. Keeps every chunk well below the
/// embedding server's per-sequence context, since a token is at least one byte.
pub const MAX_BYTES: usize = 900;

/// A piece of a text file to be embedded.
pub struct Chunk {
    /// First line (1-based)
    pub start_line: u32,
    /// Last line (1-based, inclusive)
    pub end_line: u32,
    pub text: String,
}

/// Splits text into chunks of whole lines. A chunk that hits a size limit ends
/// at its last blank line, if one falls in its second half. Whitespace-only
/// chunks are dropped and over-long lines are truncated.
pub fn chunk_text(text: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let mut end = start;
        let mut bytes = 0;
        while end < lines.len() && end - start < MAX_LINES {
            let len = lines[end].len() + 1;
            if bytes + len > MAX_BYTES && end > start {
                break;
            }
            bytes += len;
            end += 1;
        }
        if end < lines.len()
            && let Some(blank) = (start + (end - start) / 2..end)
                .rev()
                .find(|&i| lines[i].trim().is_empty())
        {
            end = blank + 1;
        }

        let body = lines[start..end].join("\n");
        let body = truncate(&body, MAX_BYTES);
        if !body.trim().is_empty() {
            chunks.push(Chunk {
                start_line: start as u32 + 1,
                end_line: end as u32,
                text: body.to_string(),
            });
        }

        start = end;
    }
    chunks
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

    fn spans(chunks: &[Chunk]) -> Vec<(u32, u32)> {
        chunks.iter().map(|c| (c.start_line, c.end_line)).collect()
    }

    #[test]
    fn splits_at_line_limit() {
        let text: String = (1..=50).map(|i| format!("line {i}\n")).collect();
        assert_eq!(spans(&chunk_text(&text)), [(1, 20), (21, 40), (41, 50)]);
    }

    #[test]
    fn ends_chunks_at_blank_lines() {
        let text: String = (1..=60)
            .map(|i| {
                if i % 15 == 0 {
                    "\n".into()
                } else {
                    format!("line {i}\n")
                }
            })
            .collect();
        assert_eq!(
            spans(&chunk_text(&text)),
            [(1, 15), (16, 30), (31, 45), (46, 60)]
        );
    }

    #[test]
    fn ignores_blank_lines_early_in_a_chunk() {
        let text: String = (1..=30)
            .map(|i| {
                if i == 5 {
                    "\n".into()
                } else {
                    format!("line {i}\n")
                }
            })
            .collect();
        assert_eq!(spans(&chunk_text(&text)), [(1, 20), (21, 30)]);
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
