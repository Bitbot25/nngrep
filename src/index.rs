use anyhow::{Context, Result, bail};
use half::f16;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Directory holding the index, at the root of the indexed tree
pub const INDEX_DIR: &str = ".nngrep";

const META_FILE: &str = "index.json";
const VECTORS_FILE: &str = "vectors.f16";
const VERSION: u32 = 3;

/// Embeddings of the text files in a directory tree.
#[derive(Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub model: String,
    /// Embedding dimension
    pub dim: usize,
    pub files: Vec<FileEntry>,
    /// One vector per chunk, in file order. Stored as little-endian f16.
    #[serde(skip)]
    pub vectors: Vec<f16>,
}

/// An indexed file.
#[derive(Serialize, Deserialize, Clone)]
pub struct FileEntry {
    /// Path relative to the index root, with '/' separators
    pub path: String,
    /// Modification time in nanoseconds since the Unix epoch
    pub mtime: u128,
    pub size: u64,
    pub chunks: Vec<ChunkSpan>,
}

/// Line range of an indexed chunk (1-based, inclusive)
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct ChunkSpan {
    pub start: u32,
    pub end: u32,
}

impl Index {
    pub fn new(model: &str, dim: usize) -> Self {
        Index {
            version: VERSION,
            model: model.into(),
            dim,
            files: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Loads the index stored under root.
    pub fn load(root: &Path) -> Result<Self> {
        let dir = root.join(INDEX_DIR);
        let meta = fs::read(dir.join(META_FILE))
            .with_context(|| format!("reading index in {}", dir.display()))?;
        let mut index: Index = serde_json::from_slice(&meta).context("parsing index")?;
        if index.version != VERSION {
            bail!(
                "index in {} has version {}, expected {VERSION}; re-run nnindex",
                dir.display(),
                index.version
            );
        }

        let bytes = fs::read(dir.join(VECTORS_FILE))?;
        let (pairs, rest) = bytes.as_chunks::<2>();
        if !rest.is_empty() {
            bail!("index in {} is corrupt; re-run nnindex", dir.display());
        }
        index.vectors = pairs.iter().map(|&b| f16::from_le_bytes(b)).collect();
        let chunks: usize = index.files.iter().map(|f| f.chunks.len()).sum();
        if index.vectors.len() != chunks * index.dim {
            bail!("index in {} is corrupt; re-run nnindex", dir.display());
        }
        Ok(index)
    }

    /// Returns the root of the index covering path, the closest of path and
    /// its ancestors that holds an index.
    pub fn find_root(path: &Path) -> Result<PathBuf> {
        path.ancestors()
            .find(|dir| dir.join(INDEX_DIR).join(META_FILE).is_file())
            .map(Path::to_path_buf)
            .with_context(|| format!("no index found for {}; run nnindex first", path.display()))
    }

    /// Writes the index under root, replacing any existing one.
    pub fn save(&self, root: &Path) -> Result<()> {
        let dir = root.join(INDEX_DIR);
        fs::create_dir_all(&dir)?;
        fs::write(dir.join(".gitignore"), "*\n")?;

        let bytes: Vec<u8> = self.vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_atomic(&dir.join(VECTORS_FILE), &bytes)?;
        write_atomic(&dir.join(META_FILE), &serde_json::to_vec(self)?)
    }

    /// Appends a file and the vectors of its chunks.
    pub fn push(&mut self, file: FileEntry, vectors: &[f16]) {
        debug_assert_eq!(vectors.len(), file.chunks.len() * self.dim);
        self.files.push(file);
        self.vectors.extend_from_slice(vectors);
    }

    /// Returns each file with the vectors of its chunks.
    pub fn iter_files(&self) -> impl Iterator<Item = (&FileEntry, &[f16])> {
        let mut offset = 0;
        self.files.iter().map(move |f| {
            let len = f.chunks.len() * self.dim;
            let vectors = &self.vectors[offset..offset + len];
            offset += len;
            (f, vectors)
        })
    }
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}
