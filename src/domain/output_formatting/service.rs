//! Output Formatting Service
//!
//! High-level service for formatting OCR results in various formats.
//! Supports plain text, JSON, hOCR, TSV, ALTO, Box, and PDF output.

use super::{OutputFormattingError, OutputFormat};
use crate::core::text::TextResult;
use crate::core::output::{format_alto, format_box, format_hocr, format_tsv, to_json_output};
use std::path::PathBuf;

pub struct OutputFormattingService;

impl OutputFormattingService {
    pub fn new() -> Self {
        Self
    }

    pub fn format_result(
        &self,
        result: &TextResult,
        format: OutputFormat,
    ) -> Result<String, OutputFormattingError> {
        match format {
            OutputFormat::PlainText => {
                if result.text.is_empty() {
                    Ok(String::new())
                } else {
                    Ok(result.text.clone())
                }
            }
            OutputFormat::Json => {
                let json = to_json_output(result);
                serde_json::to_string_pretty(&json)
                    .map_err(|e| OutputFormattingError::FormattingFailed(e.to_string()))
            }
            OutputFormat::Hocr => {
                format_hocr(result)
                    .map_err(|e| OutputFormattingError::FormattingFailed(e.to_string()))
            }
            OutputFormat::Tsv => {
                format_tsv(result)
                    .map_err(|e| OutputFormattingError::FormattingFailed(e.to_string()))
            }
            OutputFormat::Alto => {
                format_alto(result)
                    .map_err(|e| OutputFormattingError::FormattingFailed(e.to_string()))
            }
            OutputFormat::Box => {
                format_box(result)
                    .map_err(|e| OutputFormattingError::FormattingFailed(e.to_string()))
            }
            OutputFormat::Pdf => {
                Err(OutputFormattingError::PdfGenerationFailed(
                    "PDF generation requires image data. Use generate_pdf instead.".to_string()
                ))
            }
        }
    }

    pub fn write_output(
        &self,
        content: &str,
        output: Option<PathBuf>,
    ) -> Result<(), OutputFormattingError> {
        match output {
            Some(path) => {
                std::fs::write(&path, content)
                    .map_err(|e| OutputFormattingError::WriteFailed(format!("Failed to write to {:?}: {}", path, e)))?;
                println!("OCR results saved to: {:?}", path);
            }
            None => {
                if !content.is_empty() {
                    print!("{}", content);
                }
            }
        }
        Ok(())
    }

    pub fn generate_pdf(
        &self,
        image_path: &std::path::Path,
        result: &TextResult,
        output: PathBuf,
    ) -> Result<(), OutputFormattingError> {
        let img_data = std::fs::read(image_path)
            .map_err(|e| OutputFormattingError::WriteFailed(format!("Failed to read image: {}", e)))?;
        
        let dynamic_img = image::load_from_memory(&img_data)
            .map_err(|e| OutputFormattingError::WriteFailed(format!("Failed to load image: {}", e)))?;
        
        let (img_width_px, img_height_px) = dynamic_img.dimensions();

        let px_to_pt = |px: f32| printpdf::Pt(px * 72.0 / 300.0);
        let page_width_pt = px_to_pt(img_width_px as f32);
        let page_height_pt = px_to_pt(img_height_px as f32);

        let mut doc = printpdf::PdfDocument::new("OCR Output");

        let rgb_img = dynamic_img.to_rgb8();
        let (w, h) = rgb_img.dimensions();
        let raw_image = printpdf::RawImage {
            pixels: printpdf::RawImageData::U8(rgb_img.into_raw()),
            width: w as usize,
            height: h as usize,
            data_format: printpdf::RawImageFormat::RGB8,
            tag: Vec::new(),
        };
        let image_id = doc.add_image(&raw_image);

        let mut ops = Vec::new();

        ops.push(printpdf::Op::UseXobject {
            id: image_id,
            transform: printpdf::XObjectTransform {
                translate_x: Some(printpdf::Pt(0.0)),
                translate_y: Some(printpdf::Pt(0.0)),
                scale_x: Some((page_width_pt.0 / img_width_px as f32) as f32),
                scale_y: Some((page_height_pt.0 / img_height_px as f32) as f32),
                rotate: None,
                dpi: Some(300.0),
            },
        });

        for line in &result.lines {
            for word in &line.words {
                let bbox = &word.bounding_box;
                let x = px_to_pt(bbox.left as f32);
                let y = page_height_pt - px_to_pt(bbox.bottom as f32);
                let font_size = px_to_pt((bbox.bottom - bbox.top) as f32).0.max(1.0);

                ops.push(printpdf::Op::StartTextSection);
                ops.push(printpdf::Op::SetFont {
                    font: printpdf::PdfFontHandle::Builtin(printpdf::BuiltinFont::Helvetica),
                    size: printpdf::Pt(font_size),
                });
                ops.push(printpdf::Op::SetTextCursor {
                    pos: printpdf::Point { x, y },
                });
                ops.push(printpdf::Op::ShowText {
                    items: vec![printpdf::TextItem::Text(word.text.clone())],
                });
                ops.push(printpdf::Op::EndTextSection);
            }
        }

        let page = printpdf::PdfPage::new(
            printpdf::Mm(page_width_pt.0 * 25.4 / 72.0),
            printpdf::Mm(page_height_pt.0 * 25.4 / 72.0),
            ops,
        );
        doc.pages.push(page);

        let mut warnings = Vec::new();
        let pdf_bytes = doc.save(&printpdf::PdfSaveOptions::default(), &mut warnings);
        std::fs::write(&output, pdf_bytes)
            .map_err(|e| OutputFormattingError::WriteFailed(format!("Failed to write PDF: {}", e)))?;
        
        println!("Searchable PDF saved to: {:?}", output);
        Ok(())
    }

    pub fn format_from_string(&self, format: &str) -> Result<OutputFormat, OutputFormattingError> {
        OutputFormat::from_str(format)
    }
}

impl Default for OutputFormattingService {
    fn default() -> Self {
        Self::new()
    }
}
