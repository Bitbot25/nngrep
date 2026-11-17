use crate::progress;
use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

const HUB_URL: &str = "https://huggingface.co";

#[derive(Deserialize)]
struct Revision {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

/// Returns the path of a GGUF model, given as a file path or as a Hugging Face
/// repository with an optional quantization, such as
/// "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0". A repository's model is downloaded into
/// the Hugging Face cache on first use.
pub fn model_path(spec: &str) -> Result<PathBuf> {
    if Path::new(spec).is_file() {
        return Ok(spec.into());
    }
    let (repo, quant) = match spec.split_once(':') {
        Some((repo, quant)) => (repo, Some(quant)),
        None => (spec, None),
    };
    let is_repo = repo.split_once('/').is_some_and(|(org, name)| {
        !org.is_empty() && !org.starts_with('.') && !name.is_empty() && !name.contains('/')
    });
    if !is_repo {
        bail!("{spec}: no such file, and not a Hugging Face repository");
    }
    let dir = cache_dir()?.join(format!("models--{}", repo.replace('/', "--")));
    if let Some(path) = cached(&dir, quant) {
        return Ok(path);
    }
    download(repo, quant, &dir).with_context(|| format!("downloading {spec}"))
}

/// Returns the Hugging Face cache directory.
fn cache_dir() -> Result<PathBuf> {
    if let Some(dir) = env::var_os("HF_HUB_CACHE") {
        return Ok(dir.into());
    }
    if let Some(dir) = env::var_os("HF_HOME") {
        return Ok(PathBuf::from(dir).join("hub"));
    }
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Returns the cached file matching quant in the latest downloaded revision of
/// the repository cached in dir.
fn cached(dir: &Path, quant: Option<&str>) -> Option<PathBuf> {
    let sha = fs::read_to_string(dir.join("refs/main")).ok()?;
    let snapshot = dir.join("snapshots").join(sha.trim());
    let names: Vec<String> = fs::read_dir(&snapshot)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    let name = pick(&names, quant).ok()?;
    Some(snapshot.join(name))
}

/// Downloads the file matching quant from the latest revision of the
/// repository into the cache directory dir.
fn download(repo: &str, quant: Option<&str>, dir: &Path) -> Result<PathBuf> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60))
        .build();
    let get = |url: &str| {
        let mut request = agent.get(url);
        if let Ok(token) = env::var("HF_TOKEN") {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        request.call().map_err(anyhow::Error::from)
    };

    let revision: Revision =
        get(&format!("{HUB_URL}/api/models/{repo}/revision/main"))?.into_json()?;
    let names: Vec<String> = revision.siblings.into_iter().map(|s| s.rfilename).collect();
    let name = pick(&names, quant)?;
    let path = dir.join("snapshots").join(&revision.sha).join(name);
    if !path.is_file() {
        fs::create_dir_all(path.parent().unwrap())?;
        let response = get(&format!("{HUB_URL}/{repo}/resolve/{}/{name}", revision.sha))?;
        let len = response
            .header("Content-Length")
            .and_then(|n| n.parse().ok());
        let bar = progress::bars().add(
            ProgressBar::new(len.unwrap_or(0)).with_style(
                ProgressStyle::with_template(
                    "Downloading {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} · ETA {eta}",
                )
                .unwrap()
                .progress_chars("=> "),
            ),
        );
        bar.set_message(format!("{repo}/{name}"));

        let mut partial = OsString::from(&path);
        partial.push(".part");
        let mut file = File::create(&partial)?;
        io::copy(&mut bar.wrap_read(response.into_reader()), &mut file)?;
        file.sync_all()?;
        fs::rename(&partial, &path)?;
        bar.finish_and_clear();
    }
    fs::create_dir_all(dir.join("refs"))?;
    fs::write(dir.join("refs/main"), &revision.sha)?;
    Ok(path)
}

/// Returns the single GGUF file among names whose name contains quant,
/// ignoring case.
fn pick<'a>(names: &'a [String], quant: Option<&str>) -> Result<&'a str> {
    let quant = quant.map(str::to_ascii_lowercase);
    let matches: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| n.ends_with(".gguf") && !n.contains("mmproj"))
        .filter(|n| {
            quant
                .as_ref()
                .is_none_or(|q| n.to_ascii_lowercase().contains(q))
        })
        .collect();
    match matches[..] {
        [name] => Ok(name),
        [] => bail!("no matching GGUF file"),
        _ => bail!(
            "several GGUF files match ({}); choose one with REPO:QUANT",
            matches.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::pick;

    #[test]
    fn picks_files_by_quantization() {
        let names: Vec<String> = [
            "README.md",
            "Model-Q4_K_M.gguf",
            "Model.Q8_0.gguf",
            "mmproj-Model-Q8_0.gguf",
        ]
        .map(String::from)
        .into();
        assert_eq!(pick(&names, Some("q8_0")).unwrap(), "Model.Q8_0.gguf");
        assert_eq!(pick(&names, Some("Q4_K_M")).unwrap(), "Model-Q4_K_M.gguf");
        assert!(pick(&names, Some("Q6_K")).is_err());
        assert!(pick(&names, None).is_err());
        assert_eq!(pick(&names[..2], None).unwrap(), "Model-Q4_K_M.gguf");
    }
}
