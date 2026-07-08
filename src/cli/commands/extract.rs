use std::path::PathBuf;

use anyhow::{anyhow, Result};
use tracing::info;

use ocr::api::Ocr;

use super::helpers::{
    apply_dictionary_correction, build_config, format_result, generate_searchable_pdf,
};

pub async fn handle_extract(
    image_path: PathBuf,
    output: Option<PathBuf>,
    lang: &str,
    preprocess: bool,
    format: &str,
    psm: u8,
    confidence: f32,
    engine: &str,
    dict_correct: bool,
    device: &str,
    osd: bool,
) -> Result<()> {
    info!(
        "Starting OCR extraction for: {:?} (psm={}, format={}, engine={})",
        image_path, psm, format, engine
    );

    let config = build_config(
        lang,
        preprocess,
        psm,
        confidence,
        engine,
        dict_correct,
        device,
        osd,
    );
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let ext = image_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "pdf" {
        return super::pdf::handle_pdf_extraction(
            &ocr,
            &image_path,
            output,
            format,
            lang,
            dict_correct,
        )
        .await;
    }

    let mut result = ocr
        .recognize_text_from_file(&image_path)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    if dict_correct {
        apply_dictionary_correction(&mut result, lang);
    }

    info!(
        "OCR completed with confidence: {:.2}%",
        result.confidence * 100.0
    );

    if format.to_lowercase() == "pdf" {
        return generate_searchable_pdf(&image_path, &result, output).await;
    }

    let output_content = format_result(&result, format)?;
    super::helpers::write_output(&output_content, output)
}
