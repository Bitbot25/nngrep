use crate::hub;
use crate::llama::{Device, Session, load_model};
use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::num::NonZeroU32;
use std::sync::Mutex;
use std::time::Duration;

/// Server used when NNGREP_SERVER is not set, if one is running there
pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8000";

/// Model loaded in process unless NNGREP_MODEL is set
pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0";

const QUERY_TASK: &str =
    "Given a search query, retrieve relevant code or text passages that match the query";

/// Maximum number of tokens decoded at once by an in-process model. Longer
/// texts are truncated.
const BATCH_TOKENS: usize = 2048;

/// Maximum number of texts decoded at once by an in-process model
const BATCH_SEQS: usize = 64;

/// An embedding model, either served by an OpenAI-compatible server such as
/// vLLM, or loaded in process from NNGREP_MODEL (a GGUF path or a Hugging Face
/// repository, default DEFAULT_MODEL).
pub struct Embedder {
    runner: Runner,
    model: String,
}

enum Runner {
    Server { agent: ureq::Agent, url: String },
    Local(Mutex<Session>),
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
}

impl Embedder {
    /// Connects to NNGREP_SERVER, or to DEFAULT_SERVER if that is unset. If no
    /// server is set and none is running at DEFAULT_SERVER, loads the model in
    /// process on the GPU.
    pub fn connect() -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout_read(Duration::from_secs(600))
            .build();
        let explicit = env::var("NNGREP_SERVER").ok();
        let url = explicit.as_deref().unwrap_or(DEFAULT_SERVER);
        let url = url.trim_end_matches('/').to_string();
        match served_model(&agent, &url) {
            Ok(model) => Ok(Embedder {
                runner: Runner::Server { agent, url },
                model,
            }),
            Err(e) if explicit.is_none() && is_connection_failure(&e) => Self::load(Device::Gpu),
            Err(e) => Err(e).with_context(|| format!("embedding server at {url}")),
        }
    }

    /// Loads the model in process on the CPU. NNGREP_SERVER must not be set.
    pub fn load_on_cpu() -> Result<Self> {
        if env::var_os("NNGREP_SERVER").is_some() {
            bail!("NNGREP_SERVER is set; unset it to run the model on the CPU");
        }
        Self::load(Device::Cpu)
    }

    fn load(device: Device) -> Result<Self> {
        let spec = local_model_spec();
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(BATCH_TOKENS as u32))
            .with_n_batch(BATCH_TOKENS as u32)
            .with_n_ubatch(BATCH_TOKENS as u32)
            .with_n_seq_max(BATCH_SEQS as u32)
            .with_kv_unified(true)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Last);
        let session = Session::load(&hub::model_path(&spec)?, device, params)?;
        Ok(Embedder {
            runner: Runner::Local(Mutex::new(session)),
            model: canonical_model_name(&spec),
        })
    }

    /// Name of the embedding model, independent of quantization and file format
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the model loaded in process, or loads it on the GPU if it runs
    /// on a server.
    pub(crate) fn into_local_model(self) -> Result<Box<LlamaModel>> {
        match self.runner {
            Runner::Local(session) => Ok(session.into_inner().unwrap().into_model()),
            Runner::Server { .. } => {
                load_model(&hub::model_path(&local_model_spec())?, Device::Gpu)
            }
        }
    }

    /// Returns L2-normalized embeddings of documents, in order.
    pub fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embeddings = match &self.runner {
            Runner::Server { agent, url } => embed_on_server(agent, url, texts)?,
            Runner::Local(session) => embed_locally(&mut session.lock().unwrap(), texts)?,
        };
        Ok(embeddings.into_iter().map(normalized).collect())
    }

    /// Returns the L2-normalized embedding of a search query.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = format!("Instruct: {QUERY_TASK}\nQuery:{query}");
        Ok(self.embed_documents(&[text])?.remove(0))
    }
}

fn embed_on_server(agent: &ureq::Agent, url: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let response = agent
        .post(&format!("{url}/v1/embeddings"))
        .send_json(json!({ "input": texts }));
    let response: EmbeddingResponse = match response {
        Ok(r) => r.into_json()?,
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            bail!("embedding server returned {code}: {body}");
        }
        Err(e) => return Err(e.into()),
    };

    let mut data = response.data;
    if data.len() != texts.len() {
        bail!("expected {} embeddings, got {}", texts.len(), data.len());
    }
    data.sort_by_key(|d| d.index);
    Ok(data.into_iter().map(|d| d.embedding).collect())
}

fn embed_locally(session: &mut Session, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let (model, ctx) = session.parts();
    let vocab = model.vocab();
    let tokens: Vec<Vec<LlamaToken>> = texts
        .iter()
        .map(|text| {
            let mut tokens = vocab.tokenize(text.as_bytes(), true, false);
            tokens.truncate(BATCH_TOKENS);
            tokens
        })
        .collect();

    let mut embeddings = Vec::with_capacity(texts.len());
    let mut batch = LlamaBatch::new(BATCH_TOKENS, 1);
    let mut start = 0;
    while start < tokens.len() {
        let mut end = start;
        let mut n_tokens = 0;
        while end < tokens.len()
            && end - start < BATCH_SEQS
            && n_tokens + tokens[end].len() <= BATCH_TOKENS
        {
            n_tokens += tokens[end].len();
            end += 1;
        }

        batch.clear();
        for (seq, t) in tokens[start..end].iter().enumerate() {
            batch.add_sequence(t, seq as i32, false)?;
        }
        ctx.clear_kv_cache_seq(None, None, None)?;
        ctx.decode(&mut batch)?;
        for seq in 0..end - start {
            let embedding = ctx
                .embeddings_seq_ith(seq as i32)
                .map_err(|e| anyhow!("{e}"))?;
            embeddings.push(embedding.to_vec());
        }
        start = end;
    }
    Ok(embeddings)
}

/// Returns the GGUF path or Hugging Face repository of the in-process model.
fn local_model_spec() -> String {
    env::var("NNGREP_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into())
}

/// Returns the canonical name of the first model served at url.
fn served_model(agent: &ureq::Agent, url: &str) -> Result<String> {
    let list: ModelList = agent.get(&format!("{url}/v1/models")).call()?.into_json()?;
    let id = list.data.first().map_or("unknown", |m| m.id.as_str());
    Ok(canonical_model_name(id))
}

fn is_connection_failure(e: &anyhow::Error) -> bool {
    matches!(e.downcast_ref::<ureq::Error>(),
        Some(ureq::Error::Transport(t)) if t.kind() == ureq::ErrorKind::ConnectionFailed)
}

/// Strips the organization, file extension, quantization and format from a
/// model name, so that e.g. "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0",
/// "/models/Qwen3-Embedding-4B-Q8_0.gguf" and "Qwen/Qwen3-Embedding-4B" are all
/// "Qwen3-Embedding-4B".
pub fn canonical_model_name(name: &str) -> String {
    let mut name = name.rsplit('/').next().unwrap_or(name);
    name = name.split(':').next().unwrap_or(name);
    name = name.strip_suffix(".gguf").unwrap_or(name);
    name = name.strip_suffix("-GGUF").unwrap_or(name);
    if let Some((base, suffix)) = name.rsplit_once('-') {
        let s = suffix.to_ascii_lowercase();
        let quant = s.starts_with('q') && s[1..].starts_with(|c: char| c.is_ascii_digit())
            || s.starts_with("iq")
            || ["f16", "bf16", "f32"].contains(&s.as_str());
        if quant {
            name = base;
        }
    }
    name.to_string()
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::canonical_model_name;

    #[test]
    fn canonicalizes_model_names() {
        for name in [
            "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0",
            "/models/Qwen3-Embedding-4B-Q8_0.gguf",
            "Qwen3-Embedding-4B-f16.gguf",
            "Qwen/Qwen3-Embedding-4B",
        ] {
            assert_eq!(canonical_model_name(name), "Qwen3-Embedding-4B", "{name}");
        }
    }
}
