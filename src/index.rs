use anyhow::{Context, Result, bail};
use half::f16;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Directory holding the index, at the root of the indexed tree
pub const INDEX_DIR: &str = ".nngrep";

/// Files larger than this many bytes are not indexed, unless the index says
/// otherwise
pub const DEFAULT_MAX_FILE_SIZE: u64 = 1 << 20;

const META_FILE: &str = "index.json";
const VERSION: u32 = 3;

/// Embeddings of the text files in a directory tree.
#[derive(Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub model: String,
    /// Embedding dimension
    pub dim: usize,
    /// Files larger than this many bytes are not indexed
    #[serde(default = "default_max_file_size")]
    pub max_file_size: u64,
    /// Name of the file in the index directory that holds the vectors
    #[serde(default = "default_vectors_file")]
    vectors_file: String,
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
            max_file_size: DEFAULT_MAX_FILE_SIZE,
            vectors_file: default_vectors_file(),
            files: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Loads the index stored under root.
    pub fn load(root: &Path) -> Result<Self> {
        let dir = root.join(INDEX_DIR);
        let mut attempts = 0;
        let (mut index, bytes) = loop {
            let meta = fs::read(dir.join(META_FILE))
                .with_context(|| format!("reading index in {}", dir.display()))?;
            let index: Index = serde_json::from_slice(&meta).context("parsing index")?;
            if index.version != VERSION {
                bail!(
                    "index in {} has version {}, expected {VERSION}; re-run nnindex",
                    dir.display(),
                    index.version
                );
            }
            match fs::read(dir.join(&index.vectors_file)) {
                Ok(bytes) => break (index, bytes),
                // Replaced by a save since reading META_FILE
                Err(e) if e.kind() == io::ErrorKind::NotFound && attempts < 10 => attempts += 1,
                Err(e) => {
                    return Err(e).with_context(|| format!("reading index in {}", dir.display()));
                }
            }
        };

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

    /// Writes the index under root, replacing any existing one. Readers of
    /// the existing index never see a mix of the two.
    pub fn save(&mut self, root: &Path) -> Result<()> {
        let dir = root.join(INDEX_DIR);
        fs::create_dir_all(&dir)?;
        fs::write(dir.join(".gitignore"), "*\n")?;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        self.vectors_file = format!("vectors-{nanos:x}.f16");
        let bytes: Vec<u8> = self.vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dir.join(&self.vectors_file), &bytes)?;
        write_atomic(&dir.join(META_FILE), &serde_json::to_vec(self)?)?;

        for entry in fs::read_dir(&dir)? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("vectors") && name.ends_with(".f16") && name != self.vectors_file {
                // Removed at the next save if this fails
                let _ = fs::remove_file(dir.join(&*name));
            }
        }
        Ok(())
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

/// Is the index path inside the directory or file at prefix, also an index
/// path ("" for the root)?
pub fn is_under(path: &str, prefix: &str) -> bool {
    prefix.is_empty()
        || path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

fn default_max_file_size() -> u64 {
    DEFAULT_MAX_FILE_SIZE
}

/// Vectors file of indexes saved before vectors_file was recorded
fn default_vectors_file() -> String {
    "vectors.f16".into()
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}
