use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;
use std::collections::VecDeque;
use std::env;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Model loaded by a spawned llama-server unless NNGREP_MODEL is set
pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0";

const QUERY_TASK: &str =
    "Given a search query, retrieve relevant code or text passages that match the query";

/// Lines of server output kept for error reports
const LOG_LINES: usize = 40;

/// A client for a llama.cpp embedding server.
///
/// Connects to the server at NNGREP_SERVER if set. Otherwise it starts a private
/// llama-server that lives as long as the embedder, loading NNGREP_MODEL (a GGUF
/// path or a Hugging Face repo, default DEFAULT_MODEL) with extra arguments from
/// NNGREP_SERVER_ARGS.
pub struct Embedder {
    agent: ureq::Agent,
    url: String,
    _server: Option<Server>,
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

impl Embedder {
    /// Connects to or starts an embedding server and waits until it is ready.
    pub fn connect() -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout_read(Duration::from_secs(600))
            .build();

        let model = model_name();
        let (url, server) = match env::var("NNGREP_SERVER") {
            Ok(url) => (url.trim_end_matches('/').to_string(), None),
            Err(_) => {
                let port = free_port()?;
                let server = Server::spawn(&model, port)?;
                (format!("http://127.0.0.1:{port}"), Some(server))
            }
        };

        let embedder = Embedder {
            agent,
            url,
            _server: server,
        };
        embedder.wait_ready()?;
        Ok(embedder)
    }

    fn wait_ready(&self) -> Result<()> {
        let health = format!("{}/health", self.url);
        loop {
            if let Some(server) = &self._server {
                server.check_alive()?;
            }
            match self.agent.get(&health).call() {
                Ok(_) => return Ok(()),
                // 503 while loading, connection refused before listening
                Err(ureq::Error::Status(503, _)) | Err(ureq::Error::Transport(_))
                    if self._server.is_some() =>
                {
                    thread::sleep(Duration::from_millis(200))
                }
                Err(e) => {
                    return Err(e).with_context(|| format!("embedding server at {}", self.url));
                }
            }
        }
    }

    /// Returns L2-normalized embeddings of documents, in order.
    pub fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self
            .agent
            .post(&format!("{}/v1/embeddings", self.url))
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
        Ok(data.into_iter().map(|d| normalized(d.embedding)).collect())
    }

    /// Returns the L2-normalized embedding of a search query.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = format!("Instruct: {QUERY_TASK}\nQuery:{query}");
        Ok(self.embed_documents(&[text])?.remove(0))
    }
}

/// Name of the model the embedder uses, as recorded in the index.
/// "external" when connecting to a server given by NNGREP_SERVER.
pub fn model_name() -> String {
    if env::var_os("NNGREP_SERVER").is_some() {
        return "external".into();
    }
    env::var("NNGREP_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into())
}

/// Could indexes built with these models be used together?
/// The model behind an external server is unknown, so it matches any model.
pub fn compatible(a: &str, b: &str) -> bool {
    a == b || a == "external" || b == "external"
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// A llama-server child process, killed on drop.
struct Server {
    child: Mutex<Child>,
    /// Most recent lines of server output
    log: Arc<Mutex<VecDeque<String>>>,
}

impl Server {
    fn spawn(model: &str, port: u16) -> Result<Self> {
        let bin = env::var("NNGREP_LLAMA_SERVER").unwrap_or_else(|_| "llama-server".into());
        let mut cmd = Command::new(&bin);
        if Path::new(model).is_file() {
            cmd.arg("-m").arg(model);
        } else {
            cmd.arg("-hf").arg(model);
        }
        // A single slot is faster than batching sequences across slots
        cmd.args(["--embedding", "--pooling", "last", "-ngl", "999"])
            .args(["-c", "2048", "-np", "1", "-b", "2048", "-ub", "2048"])
            .args(["--host", "127.0.0.1", "--port", &port.to_string()]);
        if let Ok(extra) = env::var("NNGREP_SERVER_ARGS") {
            cmd.args(extra.split_whitespace());
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to start {bin}"))?;
        let log = Arc::new(Mutex::new(VecDeque::new()));
        collect_lines(child.stdout.take().unwrap(), log.clone());
        collect_lines(child.stderr.take().unwrap(), log.clone());
        Ok(Server {
            child: Mutex::new(child),
            log,
        })
    }

    fn check_alive(&self) -> Result<()> {
        if let Some(status) = self.child.lock().unwrap().try_wait()? {
            thread::sleep(Duration::from_millis(100)); // let the log readers catch up
            let log = self.log.lock().unwrap();
            let tail: Vec<&str> = log.iter().map(String::as_str).collect();
            bail!("llama-server exited with {status}:\n{}", tail.join("\n"));
        }
        Ok(())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let child = self.child.get_mut().unwrap();
        unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
        let _ = child.wait();
    }
}

fn collect_lines(stream: impl Read + Send + 'static, log: Arc<Mutex<VecDeque<String>>>) {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            let mut log = log.lock().unwrap();
            if log.len() == LOG_LINES {
                log.pop_front();
            }
            log.push_back(line);
        }
    });
}
