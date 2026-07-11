//! `ocr makebox` — generate Tesseract-style training box files from images.

use anyhow::{Context, Result};
use ocr::api::Ocr;
use ocr::core::output::format_makebox;
use std::path::{Path, PathBuf};

use super::helpers::build_config;

pub async fn handle_makebox(
    image_path: PathBuf,
    output_base: Option<PathBuf>,
    lang: &str,
    engine: &str,
    preprocess: bool,
) -> Result<()> {
    let path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;

    // PSM 11 (SparseText) tends to retain character/word boxes useful for training.
    let config = build_config(lang, preprocess, 11, 0.0, engine, false, "cpu", false);

    let ocr = Ocr::with_config(config)?;
    ocr.initialize()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let result = ocr
        .recognize_text_from_file(path_str)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let dyn_img = image::open(&image_path)
        .with_context(|| format!("failed to open {}", image_path.display()))?;
    let height = dyn_img.height();

    let box_text = format_makebox(&result, height)?;

    let base = output_base.unwrap_or_else(|| image_path.with_extension(""));
    let box_path = with_box_extension(&base);

    std::fs::write(&box_path, &box_text)
        .with_context(|| format!("failed to write {}", box_path.display()))?;

    println!(
        "Wrote {} ({} entries)",
        box_path.display(),
        box_text.lines().count()
    );
    if !result.text.trim().is_empty() {
        let preview: String = result.text.chars().take(80).collect();
        println!("Recognized: {preview}");
    }
    Ok(())
}

fn with_box_extension(base: &Path) -> PathBuf {
    let mut p = base.to_path_buf();
    if p.extension().and_then(|e| e.to_str()) == Some("box") {
        return p;
    }
    p.set_extension("box");
    p
}
