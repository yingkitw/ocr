use std::path::PathBuf;

use anyhow::Result;
use tracing::{error, info};

use ocr::api::{Ocr, TextProcessor};
use ocr::core::config::RecognitionEngine;

use super::helpers::{build_config, write_output};

pub async fn handle_batch(
    input_dir: PathBuf,
    output_dir: PathBuf,
    lang: &str,
    confidence: f32,
    max_concurrent: usize,
    engine: &str,
    dict_correct: bool,
    device: &str,
) -> Result<()> {
    info!(
        "Batch processing images from: {:?} -> {:?}",
        input_dir, output_dir
    );

    tokio::fs::create_dir_all(&output_dir).await?;

    let config = build_config(lang, true, 3, confidence, engine, dict_correct, device, false);
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut entries = tokio::fs::read_dir(&input_dir).await?;
    let mut image_files = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "webp") {
                image_files.push(path);
            }
        }
    }

    info!("Found {} image files", image_files.len());
    let mut processed = 0;

    for image_path in &image_files {
        let stem = image_path.file_stem().unwrap_or_default().to_string_lossy();
        let output_path = output_dir.join(format!("{}.txt", stem));

        match ocr.recognize_text_from_file(image_path).await {
            Ok(mut result) => {
                if dict_correct {
                    super::helpers::apply_dictionary_correction(&mut result, lang);
                }
                let text = if confidence > 0.0 {
                    TextProcessor::filter_by_confidence(&result, confidence)
                } else {
                    result
                };
                tokio::fs::write(&output_path, &text.text).await?;
                processed += 1;
                info!("Processed: {:?} -> {:?}", image_path, output_path);
            }
            Err(e) => {
                error!("Failed to process {:?}: {}", image_path, e);
            }
        }
    }

    info!(
        "Batch processing completed: {}/{} images processed",
        processed,
        image_files.len()
    );
    Ok(())
}
