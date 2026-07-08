use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Semaphore;
use tracing::{error, info};

use ocr::api::{Ocr, TextProcessor};

use super::helpers::build_config;

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
    let max_concurrent = max_concurrent.max(1);
    info!(
        "Batch processing images from: {:?} -> {:?} (concurrency: {})",
        input_dir, output_dir, max_concurrent
    );

    tokio::fs::create_dir_all(&output_dir).await?;

    let config = build_config(
        lang,
        true,
        3,
        confidence,
        engine,
        dict_correct,
        device,
        false,
    );
    let ocr = Ocr::with_config(config)?;
    ocr.initialize()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let ocr = Arc::new(ocr);

    let mut entries = tokio::fs::read_dir(&input_dir).await?;
    let mut image_files = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(
                ext.to_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "webp"
            ) {
                image_files.push(path);
            }
        }
    }

    info!("Found {} image files", image_files.len());

    // Cap concurrent recognition with a semaphore. Recognition acquires only a
    // read-lock on the engine, so tasks run in parallel across runtime worker
    // threads up to `max_concurrent`.
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let mut handles = Vec::with_capacity(image_files.len());

    for image_path in image_files {
        let ocr = Arc::clone(&ocr);
        let semaphore = Arc::clone(&semaphore);
        let output_dir = output_dir.clone();
        let lang = lang.to_string();
        handles.push(tokio::spawn(async move {
            // Wait for a concurrency slot before doing any work.
            let _permit = semaphore.acquire().await.expect("batch semaphore closed");
            let stem = image_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let output_path = output_dir.join(format!("{}.txt", stem));
            let res = process_one(&ocr, &image_path, &output_path, &lang, confidence, dict_correct)
                .await;
            (image_path, res)
        }));
    }

    let mut processed = 0usize;
    let total = handles.len();
    for handle in handles {
        match handle.await {
            Ok((path, Ok(()))) => {
                processed += 1;
                info!("Processed: {:?}", path);
            }
            Ok((path, Err(e))) => error!("Failed to process {:?}: {}", path, e),
            Err(e) => error!("Batch task panicked: {}", e),
        }
    }

    info!(
        "Batch processing completed: {}/{} images processed",
        processed, total
    );
    Ok(())
}

/// Recognize one image and write its text to `output_path`.
async fn process_one(
    ocr: &Ocr,
    image_path: &Path,
    output_path: &Path,
    lang: &str,
    confidence: f32,
    dict_correct: bool,
) -> Result<()> {
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
            tokio::fs::write(output_path, &text.text).await?;
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!("{}", e)),
    }
}
