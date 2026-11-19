use anyhow::{Context, Result, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;

/// Maximum number of tokens decoded at once
pub(crate) const BATCH_TOKENS: usize = 2048;

/// Maximum number of tokens in the context
pub(crate) const CONTEXT_TOKENS: usize = 3072;

/// Maximum number of sequences decoded at once
pub(crate) const MAX_SEQS: usize = 64;

/// Maximum number of tokens computed at once
pub(crate) const UBATCH_TOKENS: usize = 1024;

/// Where an in-process model runs
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

/// How urgently a caller needs the model
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// Someone is waiting for the result, such as a search
    Urgent,
    /// Background work, which waits for urgent callers
    Background,
}

/// A llama.cpp model loaded in process, with one context that callers take
/// turns using. The context runs the model with or without a LoRA adapter.
pub(crate) struct LocalModel {
    state: Mutex<State>,
    /// Signaled when a caller is done with the context
    released: Condvar,
    /// Number of urgent callers waiting for the context
    urgent: AtomicUsize,
    device: Device,
    /// Embedding dimension
    n_embd: usize,
}

struct State {
    /// None only if recreating the context failed
    session: Option<Session>,
    /// The adapter loaded on the model, and its path
    adapter: Option<(PathBuf, LlamaLoraAdapter)>,
    /// Is the adapter set on the context?
    adapter_on: bool,
}

impl LocalModel {
    /// Loads the GGUF model at path on the device.
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let model = load_model(path, device)?;
        let n_embd = model.n_embd() as usize;
        Ok(LocalModel {
            state: Mutex::new(State {
                session: Some(Session::new(model, context_params())?),
                adapter: None,
                adapter_on: false,
            }),
            released: Condvar::new(),
            urgent: AtomicUsize::new(0),
            device,
            n_embd,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Embedding dimension
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Runs f with the model and its context, waiting for the context first.
    /// The model runs with the LoRA adapter at the given path, which is loaded
    /// on first use, or with no adapter if None. Only one adapter can be
    /// loaded.
    pub fn with_context<T>(
        &self,
        priority: Priority,
        adapter: Option<&Path>,
        f: impl FnOnce(&LlamaModel, &mut LlamaContext<'static>) -> Result<T>,
    ) -> Result<T> {
        let result = {
            let mut state = self.lock(priority);
            state.use_adapter(adapter).and_then(|()| {
                let session = state.session.as_mut().context("the model has no context")?;
                let (model, ctx) = session.parts();
                f(model, ctx)
            })
        };
        self.released.notify_all();
        result
    }

    fn lock(&self, priority: Priority) -> MutexGuard<'_, State> {
        match priority {
            Priority::Urgent => {
                self.urgent.fetch_add(1, Ordering::SeqCst);
                let state = self.state.lock().unwrap();
                self.urgent.fetch_sub(1, Ordering::SeqCst);
                state
            }
            Priority::Background => {
                let mut state = self.state.lock().unwrap();
                while self.urgent.load(Ordering::SeqCst) > 0 {
                    state = self.released.wait(state).unwrap();
                }
                state
            }
        }
    }
}

impl State {
    /// Sets the adapter at path on the context, or removes it if None.
    fn use_adapter(&mut self, path: Option<&Path>) -> Result<()> {
        let Some(path) = path else {
            if let (true, Some((_, lora)), Some(session)) =
                (self.adapter_on, &mut self.adapter, &mut self.session)
            {
                session.parts().1.lora_adapter_remove(lora)?;
                self.adapter_on = false;
            }
            return Ok(());
        };
        match &self.adapter {
            Some((loaded, _)) if loaded != path => {
                bail!(
                    "cannot load {} next to {}",
                    path.display(),
                    loaded.display()
                )
            }
            Some(_) => {}
            None => self.load_adapter(path)?,
        }
        if !self.adapter_on {
            let (_, lora) = self.adapter.as_mut().unwrap();
            let session = self.session.as_mut().context("the model has no context")?;
            session.parts().1.lora_adapter_set(lora, 1.0)?;
            self.adapter_on = true;
        }
        Ok(())
    }

    /// Loads the adapter at path on the model, recreating the context, which
    /// borrows the model.
    fn load_adapter(&mut self, path: &Path) -> Result<()> {
        let mut model = self
            .session
            .take()
            .context("the model has no context")?
            .into_model();
        let lora = model
            .lora_adapter_init(path)
            .with_context(|| format!("loading {}", path.display()));
        self.session = Some(Session::new(model, context_params())?);
        self.adapter = Some((path.into(), lora?));
        Ok(())
    }
}

/// Returns the parameters of a context for embedding and reranking.
fn context_params() -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS as u32))
        .with_n_batch(BATCH_TOKENS as u32)
        .with_n_ubatch(UBATCH_TOKENS as u32)
        .with_n_seq_max(MAX_SEQS as u32)
        .with_kv_unified(true)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Last)
}

/// A llama.cpp model loaded in process, with a context for running it.
struct Session {
    // Declared before the model it borrows, so that it is dropped first
    ctx: LlamaContext<'static>,
    model: Box<LlamaModel>,
}

impl Session {
    /// Creates a context with params for the model.
    fn new(model: Box<LlamaModel>, params: LlamaContextParams) -> Result<Self> {
        let threads = thread::available_parallelism().map_or(4, |n| n.get() as i32);
        let params = params.with_n_threads(threads).with_n_threads_batch(threads);
        let ctx = model
            .new_context(backend(), params)
            .context("creating a llama.cpp context")?;
        // SAFETY: The model is boxed, so it stays in place when the session
        // moves, and it outlives the context, which is dropped first.
        let ctx = unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(ctx) };
        Ok(Session { ctx, model })
    }

    /// Returns the model and its context.
    fn parts(&mut self) -> (&LlamaModel, &mut LlamaContext<'static>) {
        (&self.model, &mut self.ctx)
    }

    /// Frees the context and returns the model.
    fn into_model(self) -> Box<LlamaModel> {
        let Session { ctx, model } = self;
        drop(ctx);
        model
    }
}

/// Loads the GGUF model at path on the device.
fn load_model(path: &Path, device: Device) -> Result<Box<LlamaModel>> {
    let model_params = match device {
        Device::Gpu => LlamaModelParams::default().with_n_gpu_layers(999),
        Device::Cpu => LlamaModelParams::default().with_devices(&[])?,
    };
    let model = LlamaModel::load_from_file(backend(), path, &model_params)
        .with_context(|| format!("loading {}", path.display()))?;
    Ok(Box::new(model))
}

/// Returns the llama.cpp backend, initializing it on first use.
fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        LlamaBackend::init().expect("the backend is only initialized here")
    })
}
