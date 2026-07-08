use std::path::PathBuf;

use anyhow::Result;
#[cfg(feature = "pdf")]
use tracing::info;

use ocr::api::Ocr;

#[cfg(feature = "pdf")]
use super::helpers::{apply_dictionary_correction, write_output};

#[cfg(feature = "pdf")]
pub async fn handle_pdf_extraction(
    ocr: &Ocr,
    pdf_path: &std::path::Path,
    output: Option<PathBuf>,
    format: &str,
    lang: &str,
    dict_correct: bool,
) -> Result<()> {
    use ocr::pdf::{extract_images, PdfImageFormat};

    info!("Extracting images from PDF: {:?}", pdf_path);
    let images = extract_images(pdf_path)?;

    if images.is_empty() {
        tracing::warn!(
            "No embedded images found in PDF. For vector-based PDFs, convert to images first."
        );
        return Ok(());
    }

    info!(
        "Found {} images across {} pages",
        images.len(),
        images.iter().map(|i| i.page_number).max().unwrap_or(0)
    );

    let mut all_text = Vec::new();
    for img in &images {
        let img_format = match img.format {
            PdfImageFormat::Jpeg => "jpeg",
            PdfImageFormat::Png => "png",
            PdfImageFormat::Raw => "raw",
        };
        info!(
            "Processing page {}: {}x{} ({})",
            img.page_number, img.width, img.height, img_format
        );

        match ocr.recognize_text(&img.data, img.width, img.height).await {
            Ok(mut result) => {
                if dict_correct {
                    apply_dictionary_correction(&mut result, lang);
                }
                all_text.push(format!("--- Page {} ---\n{}", img.page_number, result.text));
            }
            Err(e) => {
                tracing::warn!("Failed to OCR page {}: {}", img.page_number, e);
            }
        }
    }

    let combined = all_text.join("\n\n");
    let output_content = if format == "json" {
        serde_json::to_string_pretty(&serde_json::json!({
            "pages": images.iter().map(|i| serde_json::json!({
                "page": i.page_number,
                "width": i.width,
                "height": i.height,
            })).collect::<Vec<_>>(),
            "text": combined,
        }))?
    } else {
        combined
    };

    write_output(&output_content, output)
}

#[cfg(not(feature = "pdf"))]
pub async fn handle_pdf_extraction(
    _ocr: &Ocr,
    _pdf_path: &std::path::Path,
    _output: Option<PathBuf>,
    _format: &str,
    _lang: &str,
    _dict_correct: bool,
) -> Result<()> {
    Err(anyhow::anyhow!(
        "PDF support is not enabled. Rebuild with: cargo build --features pdf"
    ))
}
