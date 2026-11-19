use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const TYPE_F32: u32 = 6;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;

/// Returns the metadata values of the GGUF file at path that are arrays of
/// f32, by key.
pub(crate) fn f32_arrays(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_f32_arrays(&mut BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))
}

fn read_f32_arrays(r: &mut impl Read) -> Result<HashMap<String, Vec<f32>>> {
    let mut magic = [0; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let _n_tensors = read_u64(r)?;
    let n_kv = read_u64(r)?;
    let mut arrays = HashMap::new();
    for _ in 0..n_kv {
        let key_len = read_u64(r)?;
        let key = String::from_utf8(read_bytes(r, key_len)?)?;
        let value_type = read_u32(r)?;
        if value_type != TYPE_ARRAY {
            skip_values(r, value_type, 1)?;
            continue;
        }
        let item_type = read_u32(r)?;
        let n = read_u64(r)?;
        if item_type != TYPE_F32 {
            skip_values(r, item_type, n)?;
            continue;
        }
        let bytes = read_bytes(r, n.checked_mul(4).context("array too long")?)?;
        let (values, _) = bytes.as_chunks::<4>();
        arrays.insert(key, values.iter().map(|&b| f32::from_le_bytes(b)).collect());
    }
    Ok(arrays)
}

/// Skips n values of a GGUF type.
fn skip_values(r: &mut impl Read, value_type: u32, n: u64) -> Result<()> {
    let size = match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        TYPE_STRING => {
            for _ in 0..n {
                let len = read_u64(r)?;
                skip(r, len)?;
            }
            return Ok(());
        }
        TYPE_ARRAY => {
            for _ in 0..n {
                let item_type = read_u32(r)?;
                let len = read_u64(r)?;
                skip_values(r, item_type, len)?;
            }
            return Ok(());
        }
        _ => bail!("unknown GGUF type {value_type}"),
    };
    skip(r, n.checked_mul(size).context("array too long")?)
}

fn skip(r: &mut impl Read, n: u64) -> Result<()> {
    if io::copy(&mut r.by_ref().take(n), &mut io::sink())? != n {
        bail!("unexpected end of file");
    }
    Ok(())
}

fn read_bytes(r: &mut impl Read, n: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    if r.by_ref().take(n).read_to_end(&mut bytes)? as u64 != n {
        bail!("unexpected end of file");
    }
    Ok(bytes)
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(s: &str) -> Vec<u8> {
        [&(s.len() as u64).to_le_bytes(), s.as_bytes()].concat()
    }

    #[test]
    fn reads_f32_arrays_and_skips_other_values() {
        let mut file = b"GGUF".to_vec();
        file.extend(3u32.to_le_bytes());
        file.extend(0u64.to_le_bytes());
        file.extend(5u64.to_le_bytes());
        file.extend(string("name"));
        file.extend(TYPE_STRING.to_le_bytes());
        file.extend(string("adapter"));
        file.extend(string("alpha"));
        file.extend(TYPE_F32.to_le_bytes());
        file.extend(0.5f32.to_le_bytes());
        file.extend(string("tokens"));
        file.extend(TYPE_ARRAY.to_le_bytes());
        file.extend(TYPE_STRING.to_le_bytes());
        file.extend(2u64.to_le_bytes());
        file.extend(string("a"));
        file.extend(string("bc"));
        file.extend(string("head"));
        file.extend(TYPE_ARRAY.to_le_bytes());
        file.extend(TYPE_F32.to_le_bytes());
        file.extend(2u64.to_le_bytes());
        file.extend(1.5f32.to_le_bytes());
        file.extend((-2f32).to_le_bytes());
        file.extend(string("ids"));
        file.extend(TYPE_ARRAY.to_le_bytes());
        file.extend(5u32.to_le_bytes());
        file.extend(1u64.to_le_bytes());
        file.extend(7i32.to_le_bytes());

        let arrays = read_f32_arrays(&mut file.as_slice()).unwrap();
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays["head"], [1.5, -2.0]);
    }

    #[test]
    fn rejects_truncated_files() {
        let mut file = b"GGUF".to_vec();
        file.extend(3u32.to_le_bytes());
        file.extend(0u64.to_le_bytes());
        file.extend(1u64.to_le_bytes());
        file.extend(string("head"));
        file.extend(TYPE_ARRAY.to_le_bytes());
        file.extend(TYPE_F32.to_le_bytes());
        file.extend(2u64.to_le_bytes());
        file.extend(1.5f32.to_le_bytes());
        assert!(read_f32_arrays(&mut file.as_slice()).is_err());
    }
}
