use crate::embed::Embedder;
use crate::gguf;
use crate::llama::{BATCH_TOKENS, CONTEXT_TOKENS, LocalModel, MAX_SEQS, Priority};
use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

const SYSTEM: &str = "<|im_start|>system\nJudge whether the Document meets the requirements \
                      based on the Query and the Instruct provided. Note that the answer can \
                      only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const INSTRUCT: &str = "Given a search query about a codebase, judge whether the code or text \
                        passage is what the query is looking for";

/// Maximum number of tokens in the prompt before a document
const MAX_PREFIX_TOKENS: usize = CONTEXT_TOKENS - BATCH_TOKENS;

/// Maximum number of documents decoded at once
const BATCH_SEQS: usize = 16;
const _: () = assert!(BATCH_SEQS < MAX_SEQS);

/// A reranker that runs Qwen3-Embedding-4B with an adapter approximating
/// Qwen3-Reranker-4B.
///
/// The adapter is the LoRA GGUF file at NNGREP_RERANK_ADAPTER, whose metadata
/// holds the head (`rerank.head`) and the suffix (`rerank.suffix`), as f32
/// arrays. The adapter is only enabled while reranking, so the model keeps
/// embedding as before.
pub struct Reranker {
    model: Arc<LocalModel>,
    adapter: PathBuf,
    /// Weights of the relevance logit over the model's output
    head: Vec<f32>,
    /// Input embeddings of the end of the prompt, one per token
    suffix: Vec<f32>,
}

impl Reranker {
    /// Creates a reranker from the embedding model of embedder, which must be
    /// Qwen3-Embedding-4B. If embedder runs the model on a server, the model
    /// is loaded in process on the GPU.
    pub fn new(embedder: &Embedder) -> Result<Self> {
        let adapter: PathBuf = env::var_os("NNGREP_RERANK_ADAPTER")
            .context("NNGREP_RERANK_ADAPTER is not set")?
            .into();
        let mut arrays = gguf::f32_arrays(&adapter)?;
        let mut take = |key: &str| {
            arrays
                .remove(key)
                .with_context(|| format!("{} has no {key}", adapter.display()))
        };
        let head = take("rerank.head")?;
        let suffix = take("rerank.suffix")?;
        let model = embedder.local_model()?;
        let n_embd = model.n_embd();
        if head.len() != n_embd
            || suffix.is_empty()
            || !suffix.len().is_multiple_of(n_embd)
            || suffix.len() / n_embd >= BATCH_TOKENS
        {
            bail!("{} does not match the embedding model", adapter.display());
        }
        // Loads the adapter
        model.with_context(Priority::Urgent, Some(&adapter), |_, _| Ok(()))?;
        Ok(Reranker {
            model,
            adapter,
            head,
            suffix,
        })
    }

    /// Returns the relevance of each document to the query, from 0 to 1.
    ///
    /// The prompt up to the document is decoded once, and each document
    /// continues a copy of it.
    pub fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        self.model
            .with_context(Priority::Urgent, Some(&self.adapter), |model, ctx| {
                self.score_with(model, ctx, query, documents)
            })
    }

    fn score_with(
        &self,
        model: &LlamaModel,
        ctx: &mut LlamaContext,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f32>> {
        let n_embd = self.head.len();
        let n_suffix = self.suffix.len() / n_embd;
        let vocab = model.vocab();
        let prefix = format!("{SYSTEM}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>:");
        let prefix = vocab.tokenize(prefix.as_bytes(), false, true);
        if prefix.len() > MAX_PREFIX_TOKENS {
            bail!("query is too long to rerank");
        }
        let documents: Vec<Vec<LlamaToken>> = documents
            .iter()
            .map(|d| {
                // Text in a document is never a special token
                let mut tokens = vocab.tokenize(format!(" {d}").as_bytes(), false, false);
                tokens.truncate(BATCH_TOKENS - n_suffix);
                tokens
            })
            .collect();

        let mut tokens = LlamaBatch::new(BATCH_TOKENS, 1);
        let mut suffixes = LlamaBatch::new_embd(BATCH_SEQS * n_suffix, n_embd, 1);
        ctx.clear_kv_cache_seq(None, None, None)?;
        tokens.add_sequence(&prefix, 0, false)?;
        ctx.decode(&mut tokens)?;

        let mut scores = Vec::with_capacity(documents.len());
        let mut start = 0;
        while start < documents.len() {
            let mut end = start;
            let mut n_tokens = 0;
            while end < documents.len()
                && end - start < BATCH_SEQS
                && n_tokens + documents[end].len() + n_suffix <= BATCH_TOKENS
            {
                n_tokens += documents[end].len() + n_suffix;
                end += 1;
            }

            tokens.clear();
            suffixes.clear();
            ctx.clear_kv_cache_seq(None, Some(prefix.len() as u32), None)?;
            for (i, doc) in documents[start..end].iter().enumerate() {
                let seq = i as i32 + 1;
                ctx.copy_kv_cache_seq(0, seq, None, None)?;
                for (j, &token) in doc.iter().enumerate() {
                    tokens.add(token, (prefix.len() + j) as i32, &[seq], false)?;
                }
                for (j, embd) in self.suffix.chunks_exact(n_embd).enumerate() {
                    let pos = (prefix.len() + doc.len() + j) as i32;
                    suffixes.add_embd(embd, pos, &[seq], j == n_suffix - 1)?;
                }
            }
            ctx.decode(&mut tokens)?;
            ctx.decode(&mut suffixes)?;
            for i in 0..end - start {
                let output = ctx
                    .embeddings_seq_ith(i as i32 + 1)
                    .map_err(|e| anyhow!("{e}"))?;
                let logit: f32 = output.iter().zip(&self.head).map(|(a, b)| a * b).sum();
                scores.push(1.0 / (1.0 + (-logit).exp()));
            }
            start = end;
        }
        Ok(scores)
    }
}
