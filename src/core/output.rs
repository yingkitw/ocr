//! OCR output format support
//!
//! This module implements various OCR output formats compatible with Tesseract:
//! - Plain text
//! - JSON (structured output with bounding boxes)
//! - hOCR (HTML OCR format)
//! - TSV (character-level output)

use crate::core::text::{BoundingBox, CharacterResult, LineResult, TextResult, WordResult};
use crate::utils::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Output format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text output
    PlainText,
    /// JSON structured output
    Json,
    /// hOCR (HTML with OCR data)
    HOcr,
    /// TSV (tab-separated values)
    Tsv,
}

/// JSON output structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrJsonOutput {
    /// Full text content
    pub text: String,
    /// Overall confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Detected language
    pub language: Option<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: Option<u64>,
    /// Image dimensions
    pub image_size: ImageSize,
    /// Lines of text
    pub lines: Vec<JsonLine>,
    /// Engine metadata
    pub engine_info: EngineInfo,
    /// Timestamp
    pub timestamp: String,
}

/// Image size information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
    pub dpi: u32,
}

/// Line in JSON output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonLine {
    /// Line text
    pub text: String,
    /// Line confidence
    pub confidence: f32,
    /// Bounding box
    pub bounding_box: BoundingBox,
    /// Line number (0-indexed)
    pub line_number: usize,
    /// Words in this line
    pub words: Vec<JsonWord>,
}

/// Word in JSON output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonWord {
    /// Word text
    pub text: String,
    /// Word confidence
    pub confidence: f32,
    /// Bounding box
    pub bounding_box: BoundingBox,
    /// Word number in line (0-indexed)
    pub word_number: usize,
    /// Characters in this word
    pub characters: Vec<JsonCharacter>,
}

/// Character in JSON output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonCharacter {
    /// Character
    pub text: char,
    /// Character confidence
    pub confidence: f32,
    /// Bounding box
    pub bounding_box: BoundingBox,
}

/// Engine information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineInfo {
    /// Engine name
    pub name: String,
    /// Engine version
    pub version: String,
    /// Model type used
    pub model_type: Option<String>,
}

/// Convert TextResult to JSON output format
pub fn to_json_output(result: &TextResult) -> OcrJsonOutput {
    let image_size = ImageSize {
        width: result.bounding_box.width(),
        height: result.bounding_box.height(),
        dpi: 300, // Default DPI if not available
    };

    let lines: Vec<JsonLine> = result
        .lines
        .iter()
        .enumerate()
        .map(|(line_num, line)| JsonLine {
            text: line.text.clone(),
            confidence: line.confidence,
            bounding_box: line.bounding_box,
            line_number: line_num,
            words: line
                .words
                .iter()
                .enumerate()
                .map(|(word_num, word)| JsonWord {
                    text: word.text.clone(),
                    confidence: word.confidence,
                    bounding_box: word.bounding_box,
                    word_number: word_num,
                    characters: word
                        .characters
                        .iter()
                        .map(|ch| JsonCharacter {
                            text: ch.character,
                            confidence: ch.confidence,
                            bounding_box: ch.bounding_box,
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect();

    OcrJsonOutput {
        text: result.text.clone(),
        confidence: result.confidence,
        language: result.language.clone(),
        processing_time_ms: None,
        image_size,
        lines,
        engine_info: EngineInfo {
            name: "MiniOCR".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_type: None,
        },
        timestamp: Utc::now().to_rfc3339(),
    }
}

/// Generate JSON output string
pub fn format_json(result: &TextResult) -> Result<String> {
    let json_output = to_json_output(result);
    serde_json::to_string_pretty(&json_output)
        .map_err(|e| crate::MiniOcrError::Internal(format!("JSON serialization error: {}", e)))
        .into()
}

/// Generate hOCR (HTML OCR format) output
///
/// hOCR is an HTML-based format for representing OCR output with spatial information.
/// It follows the hOCR specification from http://kba.cloud/hocr-spec/
pub fn format_hocr(result: &TextResult) -> Result<String> {
    let mut html = String::new();

    // HTML header
    html.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    html.push_str("<!DOCTYPE html>\n");
    html.push_str("<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
    html.push_str("<head>\n");
    html.push_str("  <title>OCR Output</title>\n");
    html.push_str("  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\" />\n");
    html.push_str("  <meta name='ocr-system' content='MiniOCR ");
    html.push_str(env!("CARGO_PKG_VERSION"));
    html.push_str("' />\n");
    html.push_str("</head>\n");
    html.push_str("<body>\n");

    // OCR page
    html.push_str("  <div class='ocr_page' id='page_1' title='image \"\"; bbox ");

    let bbox = &result.bounding_box;
    html.push_str(&format!(
        "{} {} {} {}",
        bbox.left,
        bbox.top,
        bbox.right,
        bbox.bottom
    ));
    html.push_str("; ppageno 0'>\n");

    // Lines
    for (line_idx, line) in result.lines.iter().enumerate() {
        let line_bbox = &line.bounding_box;
        html.push_str(&format!(
            "    <div class='ocr_line' id='line_1_{}' title=\"bbox {} {} {} {}; baseline ",
            line_idx + 1,
            line_bbox.left,
            line_bbox.top,
            line_bbox.right,
            line_bbox.bottom
        ));

        // Estimate baseline (use bottom of line for now)
        let baseline = line_bbox.bottom;
        html.push_str(&format!("{} 0; x_size 0; x_descenders 0; x_ascenders 0\">\n", baseline));

        // Words
        for (word_idx, word) in line.words.iter().enumerate() {
            let word_bbox = &word.bounding_box;
            html.push_str(&format!(
                "      <span class='ocr_carea' id='word_1_{}_{}' title=\"bbox {} {} {} {}\">",
                line_idx + 1,
                word_idx + 1,
                word_bbox.left,
                word_bbox.top,
                word_bbox.right,
                word_bbox.bottom
            ));

            // Characters with x_word
            for (char_idx, ch) in word.characters.iter().enumerate() {
                let ch_bbox = &ch.bounding_box;
                html.push_str(&format!(
                    "<span class='ocrx_word' id='xword_1_{}_{}_{}' title='bbox {} {} {} {}; x_wconf {:.0}' title=\"bbox {} {} {} {}\">{}</span>",
                    line_idx + 1,
                    word_idx + 1,
                    char_idx + 1,
                    ch_bbox.left,
                    ch_bbox.top,
                    ch_bbox.right,
                    ch_bbox.bottom,
                    ch.confidence * 100.0,
                    ch_bbox.left,
                    ch_bbox.top,
                    ch_bbox.right,
                    ch_bbox.bottom,
                    ch.character
                ));
            }

            html.push_str(&format!("{}</span>\n", word.text));
        }

        // Emit the full line text so it appears in the output
        html.push_str(&format!("      <span class='ocr_line_text'>{}</span>\n", line.text));
        html.push_str("    </div>\n");
    }

    html.push_str("  </div>\n");
    html.push_str("</body>\n");
    html.push_str("</html>\n");

    Ok(html)
}

/// Generate TSV (tab-separated values) output
///
/// TSV format: character-level output with tabs and confidence scores
/// Format: char\tleft\ttop\tright\tbottom\tpage_id\n
pub fn format_tsv(result: &TextResult) -> Result<String> {
    let mut output = String::new();

    // Header line (optional, commented out)
    // output.push_str("# char\tleft\ttop\tright\tbottom\tpage_id\tconf\tfont_id\n");

    let mut page_num = 1;
    let mut line_num = 1;
    let mut word_num = 1;

    for line in &result.lines {
        for word in &line.words {
            for ch in &word.characters {
                let bbox = &ch.bounding_box;
                output.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\n",
                    ch.character,
                    bbox.left,
                    bbox.top,
                    bbox.right,
                    bbox.bottom,
                    page_num,
                    ch.confidence,
                    0 // font_id (not used)
                ));
            }
            word_num += 1;
        }
        line_num += 1;
    }

    Ok(output)
}

/// Format output in the specified format
pub fn format_output(result: &TextResult, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::PlainText => Ok(result.text.clone()),
        OutputFormat::Json => format_json(result),
        OutputFormat::HOcr => format_hocr(result),
        OutputFormat::Tsv => format_tsv(result),
    }
}

/// Parse output format from string
pub fn parse_output_format(s: &str) -> Result<OutputFormat> {
    match s.to_lowercase().as_str() {
        "text" | "txt" | "plain" => Ok(OutputFormat::PlainText),
        "json" => Ok(OutputFormat::Json),
        "hocr" | "html" => Ok(OutputFormat::HOcr),
        "tsv" => Ok(OutputFormat::Tsv),
        _ => Err(crate::MiniOcrError::InvalidInput(format!(
            "Unknown output format: {}. Supported: text, json, hocr, tsv",
            s
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::text::{TextResult, BoundingBox};

    fn create_test_result() -> TextResult {
        let mut result = TextResult::new(
            "Hello World".to_string(),
            0.9,
            BoundingBox::new(0, 0, 200, 50),
        );

        // Add a line
        let mut line = LineResult::new("Hello World".to_string(), 0.9, BoundingBox::new(0, 0, 200, 20));

        // Add words
        let mut word1 = WordResult::new("Hello".to_string(), 0.95, BoundingBox::new(0, 0, 60, 20));
        let mut word2 = WordResult::new("World".to_string(), 0.85, BoundingBox::new(70, 0, 130, 20));

        // Add characters
        for (i, ch) in "Hello".chars().enumerate() {
            let bbox = BoundingBox::new(i as u32 * 10, 0, (i + 1) as u32 * 10, 20);
            word1.characters.push(CharacterResult::new(ch, 0.95, bbox));
        }

        for (i, ch) in "World".chars().enumerate() {
            let bbox = BoundingBox::new(70 + i as u32 * 10, 0, 70 + (i + 1) as u32 * 10, 20);
            word2.characters.push(CharacterResult::new(ch, 0.85, bbox));
        }

        line.words.push(word1);
        line.words.push(word2);
        result.lines.push(line);

        result
    }

    #[test]
    fn test_format_json() {
        let result = create_test_result();
        let json = format_json(&result).unwrap();

        assert!(json.contains("Hello World"));
        assert!(json.contains("\"confidence\": 0.9"));
        assert!(json.contains("Hello"));
        assert!(json.contains("World"));
    }

    #[test]
    fn test_format_hocr() {
        let result = create_test_result();
        let hocr = format_hocr(&result).unwrap();

        assert!(hocr.contains("<html"));
        assert!(hocr.contains("ocr_page"));
        assert!(hocr.contains("ocr_line"));
        assert!(hocr.contains("Hello World"));
    }

    #[test]
    fn test_format_tsv() {
        let result = create_test_result();
        let tsv = format_tsv(&result).unwrap();

        // Should have tab-separated values
        assert!(tsv.contains('\t'));
        assert!(tsv.chars().filter(|&c| c == '\n').count() > 0);
    }

    #[test]
    fn test_parse_output_format() {
        assert!(matches!(parse_output_format("json"), Ok(OutputFormat::Json)));
        assert!(matches!(parse_output_format("TEXT"), Ok(OutputFormat::PlainText)));
        assert!(matches!(parse_output_format("hocr"), Ok(OutputFormat::HOcr)));
        assert!(parse_output_format("invalid").is_err());
    }

    #[test]
    fn test_format_output_plain() {
        let result = create_test_result();
        let output = format_output(&result, OutputFormat::PlainText).unwrap();
        assert_eq!(output, "Hello World");
    }
}
