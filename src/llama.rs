use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use std::path::Path;
use std::sync::OnceLock;
use std::thread;

/// Where an in-process model runs
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

/// A llama.cpp model loaded in process, with a context for running it.
pub(crate) struct Session {
    // Declared before the model it borrows, so that it is dropped first
    ctx: LlamaContext<'static>,
    model: Box<LlamaModel>,
}

impl Session {
    /// Loads the GGUF model at path and creates a context with params.
    pub fn load(path: &Path, device: Device, params: LlamaContextParams) -> Result<Self> {
        Self::new(load_model(path, device)?, params)
    }

    /// Creates a context with params for the model.
    pub fn new(model: Box<LlamaModel>, params: LlamaContextParams) -> Result<Self> {
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
    pub fn parts(&mut self) -> (&LlamaModel, &mut LlamaContext<'static>) {
        (&self.model, &mut self.ctx)
    }

    /// Frees the context and returns the model.
    pub fn into_model(self) -> Box<LlamaModel> {
        let Session { ctx, model } = self;
        drop(ctx);
        model
    }
}

/// Loads the GGUF model at path on the device.
pub(crate) fn load_model(path: &Path, device: Device) -> Result<Box<LlamaModel>> {
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
