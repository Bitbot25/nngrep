/// Maximum number of lines in a chunk
pub const MAX_LINES: usize = 40;

/// Maximum size of a chunk's text in bytes. Keeps every chunk well below the
/// embedding server's per-sequence context, since a token is at least one byte.
pub const MAX_BYTES: usize = 1500;

/// Number of lines shared by consecutive chunks
const OVERLAP: usize = 3;

/// A piece of a text file to be embedded.
pub struct Chunk {
    /// First line (1-based)
    pub start_line: u32,
    /// Last line (1-based, inclusive)
    pub end_line: u32,
    pub text: String,
}

/// Splits text into overlapping chunks of whole lines. Whitespace-only chunks
/// are dropped and over-long lines are truncated.
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

        let body = lines[start..end].join("\n");
        let body = truncate(&body, MAX_BYTES);
        if !body.trim().is_empty() {
            chunks.push(Chunk {
                start_line: start as u32 + 1,
                end_line: end as u32,
                text: body.to_string(),
            });
        }

        if end >= lines.len() {
            break;
        }
        start = if end - start > 2 * OVERLAP {
            end - OVERLAP
        } else {
            end
        };
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

    #[test]
    fn splits_long_text_with_overlap() {
        let text: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        let chunks = chunk_text(&text);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, MAX_LINES as u32);
        assert_eq!(chunks[1].start_line, (MAX_LINES - OVERLAP) as u32 + 1);
        assert_eq!(chunks.last().unwrap().end_line, 100);
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
