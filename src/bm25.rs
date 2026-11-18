use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Words left out of queries
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "get",
    "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "with",
];

/// Keyword scoring of chunks of text against queries by Okapi BM25.
///
/// Words are runs of letters and digits, also split at camelCase boundaries,
/// and whole identifiers. They are compared lowercased and stemmed.
pub struct Bm25 {
    stemmer: Stemmer,
    /// Distinct terms of all queries, sorted
    terms: Vec<String>,
    /// Indexes into terms of each query's terms
    queries: Vec<Vec<usize>>,
}

/// Length and query term frequencies of a chunk
pub struct Counts {
    /// Number of words
    len: u32,
    /// Occurrences of each query term
    tf: Vec<u32>,
}

impl Bm25 {
    pub fn new(queries: &[String]) -> Self {
        let stemmer = Stemmer::create(Algorithm::English);
        let mut query_terms: Vec<Vec<String>> = Vec::new();
        for query in queries {
            let mut terms = Vec::new();
            split_words(query, |word| {
                let word = word.to_lowercase();
                if !STOP_WORDS.contains(&word.as_str()) {
                    terms.push(stemmer.stem(&word).into_owned());
                }
            });
            query_terms.push(terms);
        }
        let mut terms: Vec<String> = query_terms.iter().flatten().cloned().collect();
        terms.sort();
        terms.dedup();
        let queries = query_terms
            .iter()
            .map(|q| {
                let mut ids: Vec<usize> =
                    q.iter().map(|t| terms.binary_search(t).unwrap()).collect();
                ids.sort();
                ids.dedup();
                ids
            })
            .collect();
        Bm25 {
            stemmer,
            terms,
            queries,
        }
    }

    /// Counts the words of a chunk.
    pub fn count(&self, text: &str) -> Counts {
        self.counter().count(text)
    }

    /// Returns a counter of words in chunks, for counting many chunks.
    pub fn counter(&self) -> Counter<'_> {
        Counter {
            bm25: self,
            terms: HashMap::new(),
            lower: String::new(),
        }
    }

    /// Returns the score of each chunk for each query, taking the chunks as
    /// the whole collection searched.
    pub fn scores(&self, chunks: &[Counts]) -> Vec<Vec<f32>> {
        let n = chunks.len() as f32;
        let total_len: f32 = chunks.iter().map(|c| c.len as f32).sum();
        let avg_len = (total_len / n).max(1.0);
        let idf: Vec<f32> = (0..self.terms.len())
            .map(|t| {
                let df = chunks.iter().filter(|c| c.tf[t] > 0).count() as f32;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            })
            .collect();
        self.queries
            .iter()
            .map(|query| {
                chunks
                    .iter()
                    .map(|c| {
                        let norm = K1 * (1.0 - B + B * c.len as f32 / avg_len);
                        query
                            .iter()
                            .map(|&t| {
                                let tf = c.tf[t] as f32;
                                idf[t] * tf * (K1 + 1.0) / (tf + norm)
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

/// Counts the words of chunks for Bm25, remembering which query term each
/// distinct word is.
pub struct Counter<'a> {
    bm25: &'a Bm25,
    /// Index into the query terms of each word seen, if it is one
    terms: HashMap<String, Option<usize>>,
    /// Buffer for a lowercased word
    lower: String,
}

impl Counter<'_> {
    /// Counts the words of a chunk.
    pub fn count(&mut self, text: &str) -> Counts {
        let bm25 = self.bm25;
        let mut counts = Counts {
            len: 0,
            tf: vec![0; bm25.terms.len()],
        };
        split_words(text, |word| {
            counts.len += 1;
            let term = match self.terms.get(word) {
                Some(&term) => term,
                None => {
                    self.lower.clear();
                    self.lower.extend(word.chars().flat_map(char::to_lowercase));
                    let stem = bm25.stemmer.stem(&self.lower);
                    let term = bm25.terms.binary_search_by(|t| t.as_str().cmp(&stem)).ok();
                    self.terms.insert(word.to_string(), term);
                    term
                }
            };
            if let Some(i) = term {
                counts.tf[i] += 1;
            }
        });
        counts
    }
}

/// Calls f with each word of text. Words are runs of letters and digits,
/// split before an uppercase letter that follows a lowercase letter or digit,
/// or that follows an uppercase letter and precedes a lowercase one. An
/// identifier made of several words is also a word itself, before them
/// ("parse_HTTPHeader" gives "parse_HTTPHeader", "parse", "HTTP" and "Header").
fn split_words(text: &str, mut f: impl FnMut(&str)) {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut parts = Vec::new();
    for ident in text.split(|c: char| !is_ident(c)) {
        let ident = ident.trim_matches('_');
        if ident.is_empty() {
            continue;
        }
        if !ident.contains('_') && !ident.chars().skip(1).any(char::is_uppercase) {
            f(ident);
            continue;
        }
        for run in ident.split('_').filter(|r| !r.is_empty()) {
            let mut start = 0;
            let mut prev = ' ';
            let mut chars = run.char_indices().peekable();
            while let Some((i, c)) = chars.next() {
                if i > 0
                    && c.is_uppercase()
                    && (!prev.is_uppercase()
                        || chars.peek().is_some_and(|&(_, next)| next.is_lowercase()))
                {
                    parts.push(&run[start..i]);
                    start = i;
                }
                prev = c;
            }
            parts.push(&run[start..]);
        }
        if parts.len() > 1 {
            f(ident);
        }
        parts.drain(..).for_each(&mut f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        let mut words = Vec::new();
        split_words(text, |w| words.push(w.to_string()));
        words
    }

    #[test]
    fn splits_identifiers() {
        assert_eq!(
            words("parseHTTPHeader(__max_len, utf8Str) Ünïcode"),
            [
                "parseHTTPHeader",
                "parse",
                "HTTP",
                "Header",
                "max_len",
                "max",
                "len",
                "utf8Str",
                "utf8",
                "Str",
                "Ünïcode"
            ]
        );
    }

    #[test]
    fn matches_stemmed_terms() {
        let bm25 = Bm25::new(&["where do we retry failed uploads".into()]);
        assert_eq!(bm25.terms, ["fail", "retri", "upload"]);
        let counts = bm25.count("fn retryUpload() { // Retries a failing upload");
        assert_eq!(counts.len, 8);
        assert_eq!(counts.tf, [1, 2, 2]);
    }

    #[test]
    fn reused_counter_counts_like_a_new_one() {
        let bm25 = Bm25::new(&["retry failed uploads".into()]);
        let mut counter = bm25.counter();
        for text in [
            "retry the upload",
            "Retries failed; upload_retry",
            "nothing here",
        ] {
            let (reused, new) = (counter.count(text), bm25.count(text));
            assert_eq!((reused.len, reused.tf), (new.len, new.tf), "{text}");
        }
    }

    #[test]
    fn ranks_rare_terms_higher() {
        let bm25 = Bm25::new(&["parse header".into(), "header".into()]);
        let chunks: Vec<Counts> = ["parse the header", "read the header", "write the body"]
            .iter()
            .map(|c| bm25.count(c))
            .collect();
        let scores = bm25.scores(&chunks);
        assert!(scores[0][0] > scores[0][1] && scores[0][1] > scores[0][2]);
        assert_eq!(scores[0][2], 0.0);
        assert_eq!(scores[1][0], scores[1][1]);
    }
}
