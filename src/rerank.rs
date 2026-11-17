use crate::hub;
use crate::llama::{Device, Session};
use anyhow::{Result, anyhow, bail};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::LlamaToken;
use std::env;
use std::num::NonZeroU32;

/// Model loaded unless NNGREP_RERANK_MODEL is set
pub const DEFAULT_RERANK_MODEL: &str = "Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp:Q8_0";

const SYSTEM: &str = "<|im_start|>system\nJudge whether the Document meets the requirements \
                      based on the Query and the Instruct provided. Note that the answer can \
                      only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const INSTRUCT: &str = "Given a search query about a codebase, judge whether the code or text \
                        passage is what the query is looking for";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Maximum number of tokens in the prompt before a document
const MAX_PREFIX_TOKENS: usize = 1024;

/// Maximum number of document tokens decoded at once. Longer documents are
/// truncated.
const BATCH_TOKENS: usize = 2048;

/// Maximum number of documents decoded at once
const BATCH_SEQS: usize = 16;

/// A Qwen3 reranker loaded in process from NNGREP_RERANK_MODEL (a GGUF path or
/// a Hugging Face repository, default DEFAULT_RERANK_MODEL).
pub struct Reranker {
    session: Session,
}

impl Reranker {
    /// Loads the model on the device.
    pub fn load(device: Device) -> Result<Self> {
        let spec = env::var("NNGREP_RERANK_MODEL").unwrap_or_else(|_| DEFAULT_RERANK_MODEL.into());
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new((MAX_PREFIX_TOKENS + BATCH_TOKENS) as u32))
            .with_n_batch(BATCH_TOKENS as u32)
            .with_n_ubatch(BATCH_TOKENS as u32)
            .with_n_seq_max(BATCH_SEQS as u32 + 1)
            .with_kv_unified(true)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Rank);
        let mut session = Session::load(&hub::model_path(&spec)?, device, params)?;
        let (model, _) = session.parts();
        let arch = model
            .meta_val_str("general.architecture")
            .unwrap_or_default();
        if arch != "qwen3" || model.n_cls_out() != 2 {
            bail!("{spec} is not a Qwen3 reranker");
        }
        Ok(Reranker { session })
    }

    /// Returns the relevance of each document to the query, from 0 to 1.
    ///
    /// The prompt up to the document is decoded once, and each document
    /// continues a copy of it.
    pub fn score(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let (model, ctx) = self.session.parts();
        let vocab = model.vocab();
        let prefix = format!("{SYSTEM}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>:");
        let prefix = vocab.tokenize(prefix.as_bytes(), false, true);
        if prefix.len() > MAX_PREFIX_TOKENS {
            bail!("query is too long to rerank");
        }
        let suffix = vocab.tokenize(SUFFIX.as_bytes(), false, true);
        let documents: Vec<Vec<LlamaToken>> = documents
            .iter()
            .map(|d| {
                // Text in a document is never a special token
                let mut tokens = vocab.tokenize(format!(" {d}").as_bytes(), false, false);
                tokens.truncate(BATCH_TOKENS - suffix.len());
                tokens.extend_from_slice(&suffix);
                tokens
            })
            .collect();

        let mut batch = LlamaBatch::new(BATCH_TOKENS, 1);
        ctx.clear_kv_cache_seq(None, None, None)?;
        batch.add_sequence(&prefix, 0, false)?;
        ctx.decode(&mut batch)?;

        let mut scores = Vec::with_capacity(documents.len());
        let mut start = 0;
        while start < documents.len() {
            let mut end = start;
            let mut n_tokens = 0;
            while end < documents.len()
                && end - start < BATCH_SEQS
                && n_tokens + documents[end].len() <= BATCH_TOKENS
            {
                n_tokens += documents[end].len();
                end += 1;
            }

            batch.clear();
            ctx.clear_kv_cache_seq(None, Some(prefix.len() as u32), None)?;
            for (i, doc) in documents[start..end].iter().enumerate() {
                let seq = i as i32 + 1;
                ctx.copy_kv_cache_seq(0, seq, None, None)?;
                for (j, &token) in doc.iter().enumerate() {
                    let pos = (prefix.len() + j) as i32;
                    batch.add(token, pos, &[seq], j == doc.len() - 1)?;
                }
            }
            ctx.decode(&mut batch)?;
            for i in 0..end - start {
                let relevance = ctx
                    .embeddings_seq_ith(i as i32 + 1)
                    .map_err(|e| anyhow!("{e}"))?;
                scores.push(relevance[0]);
            }
            start = end;
        }
        Ok(scores)
    }
}
