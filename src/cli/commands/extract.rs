use std::path::PathBuf;

use anyhow::{anyhow, Result};
use tracing::info;

use ocr::api::Ocr;

use super::helpers::{
    apply_dictionary_correction, build_config, format_result,
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
    dpi: u32,
) -> Result<()> {
    info!(
        "Starting OCR extraction for: {:?} (psm={}, format={}, engine={}, dpi={})",
        image_path, psm, format, engine, dpi
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
        dpi,
    );
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    // `-` means read the image from stdin (Unix pipe convention)
    let image_data = if image_path == std::path::Path::new("-") {
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut data)
            .map_err(|e| anyhow!("Failed to read from stdin: {}", e))?;
        if data.is_empty() {
            return Err(anyhow!("No image data received on stdin"));
        }
        data
    } else {
        std::fs::read(&image_path).map_err(|e| anyhow!("Failed to read image: {}", e))?
    };

    let mut results = ocr
        .recognize_text_from_bytes_multi_page(&image_data)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    // Apply dictionary correction to all pages
    if dict_correct {
        for r in &mut results {
            apply_dictionary_correction(r, lang);
        }
    }

    // PDF output is binary and needs the original image data
    #[cfg(feature = "pdf")]
    if format == "pdf" || format == "searchable-pdf" {
        let pages: Vec<(&[u8], &ocr::core::text::TextResult)> = results
            .iter()
            .map(|r| (image_data.as_slice(), r))
            .collect();
        let pdf_bytes = ocr::core::pdf::format_searchable_pdf(&pages)
            .map_err(|e| anyhow!("PDF generation failed: {}", e))?;
        return write_binary_output(&pdf_bytes, output);
    }

    if results.len() == 1 {
        // Single page — same output as before
        let result = &results[0];
        info!(
            "OCR completed with confidence: {:.2}%",
            result.confidence * 100.0
        );
        let output_content = format_result(result, format)?;
        super::helpers::write_output(&output_content, output)
    } else {
        // Multi-page — join outputs with page separators
        info!("OCR completed: {} pages", results.len());
        let mut sections = Vec::with_capacity(results.len());
        for (i, result) in results.iter().enumerate() {
            info!(
                "  Page {} confidence: {:.2}%",
                i + 1,
                result.confidence * 100.0
            );
            let page_output = format_result(result, format)?;
            sections.push(format!("--- Page {} ---\n{}", i + 1, page_output));
        }
        let output_content = sections.join("\n\n");
        super::helpers::write_output(&output_content, output)
    }
}

#[cfg(feature = "pdf")]
fn write_binary_output(data: &[u8], output: Option<PathBuf>) -> Result<()> {
    match output {
        Some(path) => {
            std::fs::write(&path, data)?;
            println!("OCR results saved to: {:?}", path);
        }
        None => {
            std::io::Write::write_all(&mut std::io::stdout(), data)?;
        }
    }
    Ok(())
}
