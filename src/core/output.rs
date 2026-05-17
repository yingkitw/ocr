//! OCR output format support
//!
//! This module implements various OCR output formats compatible with Tesseract:
//! - Plain text
//! - JSON (structured output with bounding boxes)
//! - hOCR (HTML OCR format)
//! - TSV (character-level output)

use crate::core::text::{BoundingBox, TextResult};
use crate::utils::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

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
    /// ALTO XML (standard library format)
    Alto,
    /// Box file for Tesseract training
    Box,
    /// Searchable PDF with invisible text layer
    Pdf,
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
            name: "OCR".to_string(),
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
        .map_err(|e| crate::OcrError::Internal(format!("JSON serialization error: {}", e)))
        .into()
}

/// Generate hOCR (HTML OCR format) output
///
/// hOCR is an HTML-based format for representing OCR output with spatial information.
/// It follows the hOCR 4.1 specification from http://kba.cloud/hocr-spec/
pub fn format_hocr(result: &TextResult) -> Result<String> {
    let mut html = String::new();

    // HTML header
    html.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    html.push_str("<!DOCTYPE html>\n");
    html.push_str("<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
    html.push_str("<head>\n");
    html.push_str("  <title>OCR Output</title>\n");
    html.push_str("  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\" />\n");
    html.push_str("  <meta name='ocr-system' content='OCR ");
    html.push_str(env!("CARGO_PKG_VERSION"));
    html.push_str("' />\n");
    html.push_str("  <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word ocr_cinfo' />\n");
    html.push_str("</head>\n");
    html.push_str("<body>\n");

    // OCR page
    let bbox = &result.bounding_box;
    html.push_str(&format!(
        "  <div class='ocr_page' id='page_1' title=\"image \"\"; bbox {} {} {} {}; ppageno 0\">\n",
        bbox.left,
        bbox.top,
        bbox.right,
        bbox.bottom
    ));

    // Content area (single carea for now)
    html.push_str(&format!(
        "    <div class='ocr_carea' id='block_1_1' title=\"bbox {} {} {} {}\">\n",
        bbox.left,
        bbox.top,
        bbox.right,
        bbox.bottom
    ));

    // Paragraph
    html.push_str(&format!(
        "      <p class='ocr_par' id='par_1_1' title=\"bbox {} {} {} {}\">\n",
        bbox.left,
        bbox.top,
        bbox.right,
        bbox.bottom
    ));

    // Lines
    for (line_idx, line) in result.lines.iter().enumerate() {
        let line_bbox = &line.bounding_box;
        html.push_str(&format!(
            "        <span class='ocr_line' id='line_1_{}' title=\"bbox {} {} {} {}; baseline {} 0; x_size {}; x_descenders {}; x_ascenders {}\">\n",
            line_idx + 1,
            line_bbox.left,
            line_bbox.top,
            line_bbox.right,
            line_bbox.bottom,
            line_bbox.bottom,
            line_bbox.height(),
            line_bbox.height() / 4,
            line_bbox.height() / 4
        ));

        // Words
        for (word_idx, word) in line.words.iter().enumerate() {
            let word_bbox = &word.bounding_box;
            html.push_str(&format!(
                "          <span class='ocrx_word' id='word_1_{}_{}' title=\"bbox {} {} {} {}; x_wconf {:.0}\">",
                line_idx + 1,
                word_idx + 1,
                word_bbox.left,
                word_bbox.top,
                word_bbox.right,
                word_bbox.bottom,
                word.confidence * 100.0
            ));

            // Characters (ocr_cinfo per hOCR spec)
            for (char_idx, ch) in word.characters.iter().enumerate() {
                let ch_bbox = &ch.bounding_box;
                html.push_str(&format!(
                    "<span class='ocr_cinfo' id='cinfo_1_{}_{}_{}' title='bbox {} {} {} {}; x_wconf {:.0}'>{}</span>",
                    line_idx + 1,
                    word_idx + 1,
                    char_idx + 1,
                    ch_bbox.left,
                    ch_bbox.top,
                    ch_bbox.right,
                    ch_bbox.bottom,
                    ch.confidence * 100.0,
                    html_escape_char(ch.character)
                ));
            }

            html.push_str("</span>\n");
        }

        html.push_str("        </span>\n");
    }

    html.push_str("      </p>\n");
    html.push_str("    </div>\n");
    html.push_str("  </div>\n");
    html.push_str("</body>\n");
    html.push_str("</html>\n");

    Ok(html)
}

/// Generate TSV (tab-separated values) output
///
/// Matches Tesseract's TSV column format:
/// level page_num block_num par_num line_num word_num left top width height conf text
pub fn format_tsv(result: &TextResult) -> Result<String> {
    let mut output = String::new();

    // Header
    output.push_str("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n");

    let page_num = 1;
    let block_num = 1;
    let par_num = 1;

    // Page level
    let pb = &result.bounding_box;
    output.push_str(&format!(
        "1\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t\n",
        page_num,
        block_num,
        par_num,
        0,
        0,
        pb.left,
        pb.top,
        pb.width(),
        pb.height(),
        result.confidence * 100.0,
    ));

    // Block level
    output.push_str(&format!(
        "2\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t\n",
        page_num,
        block_num,
        par_num,
        0,
        0,
        pb.left,
        pb.top,
        pb.width(),
        pb.height(),
        result.confidence * 100.0,
    ));

    // Paragraph level
    output.push_str(&format!(
        "3\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t\n",
        page_num,
        block_num,
        par_num,
        0,
        0,
        pb.left,
        pb.top,
        pb.width(),
        pb.height(),
        result.confidence * 100.0,
    ));

    for (line_idx, line) in result.lines.iter().enumerate() {
        let line_num = line_idx + 1;
        let lb = &line.bounding_box;

        // Line level
        output.push_str(&format!(
            "4\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t\n",
            page_num,
            block_num,
            par_num,
            line_num,
            0,
            lb.left,
            lb.top,
            lb.width(),
            lb.height(),
            line.confidence * 100.0,
        ));

        for (word_idx, word) in line.words.iter().enumerate() {
            let word_num = word_idx + 1;
            let wb = &word.bounding_box;
            let conf = word.confidence * 100.0;

            // Word level
            output.push_str(&format!(
                "5\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\n",
                page_num,
                block_num,
                par_num,
                line_num,
                word_num,
                wb.left,
                wb.top,
                wb.width(),
                wb.height(),
                conf,
                escape_tsv_text(&word.text),
            ));
        }
    }

    Ok(output)
}

/// Escape text for TSV output (tab and newline)
fn escape_tsv_text(s: &str) -> String {
    s.replace('\t', " ").replace('\n', " ")
}

/// Generate ALTO XML output
pub fn format_alto(result: &TextResult) -> Result<String> {
    let mut xml = String::new();
    let bbox = &result.bounding_box;

    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<alto xmlns=\"http://www.loc.gov/standards/alto/ns-v4#\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-4.xsd\" VERSION=\"4.4\">\n");
    xml.push_str("  <Description>\n");
    xml.push_str("    <MeasurementUnit>pixel</MeasurementUnit>\n");
    xml.push_str("    <OCRProcessing>\n");
    xml.push_str("      <ocrProcessingStep>\n");
    xml.push_str("        <processingSoftware>\n");
    xml.push_str(&format!("          <softwareName>OCR</softwareName>\n"));
    xml.push_str(&format!("          <softwareVersion>{}</softwareVersion>\n", env!("CARGO_PKG_VERSION")));
    xml.push_str("        </processingSoftware>\n");
    xml.push_str("      </ocrProcessingStep>\n");
    xml.push_str("    </OCRProcessing>\n");
    xml.push_str("  </Description>\n");

    xml.push_str(&format!(
        "  <Layout>\n    <Page WIDTH=\"{}\" HEIGHT=\"{}\" PHYSICAL_IMG_NR=\"1\" ID=\"page_1\">\n",
        bbox.right, bbox.bottom
    ));
    xml.push_str(&format!(
        "      <PrintSpace HPOS=\"0\" VPOS=\"0\" WIDTH=\"{}\" HEIGHT=\"{}\">\n",
        bbox.right, bbox.bottom
    ));

    for (line_idx, line) in result.lines.iter().enumerate() {
        let lb = &line.bounding_box;
        xml.push_str(&format!(
            "        <TextLine ID=\"line_{}\" HPOS=\"{}\" VPOS=\"{}\" WIDTH=\"{}\" HEIGHT=\"{}\">\n",
            line_idx + 1,
            lb.left,
            lb.top,
            lb.width(),
            lb.height()
        ));

        for (word_idx, word) in line.words.iter().enumerate() {
            let wb = &word.bounding_box;
            let wc = (word.confidence * 100.0).round() as u32;
            xml.push_str(&format!(
                "          <String ID=\"word_{}_{}\" HPOS=\"{}\" VPOS=\"{}\" WIDTH=\"{}\" HEIGHT=\"{}\" CONTENT=\"{}\" WC=\"0.{}\"/>\n",
                line_idx + 1,
                word_idx + 1,
                wb.left,
                wb.top,
                wb.width(),
                wb.height(),
                xml_escape(&word.text),
                wc
            ));
        }

        xml.push_str("        </TextLine>\n");
    }

    xml.push_str("      </PrintSpace>\n");
    xml.push_str("    </Page>\n");
    xml.push_str("  </Layout>\n");
    xml.push_str("</alto>\n");

    Ok(xml)
}

/// Generate box file output for Tesseract training
///
/// Format per character: char left top right bottom page
/// Spaces are represented as a special box with page -1
pub fn format_box(result: &TextResult) -> Result<String> {
    let mut output = String::new();
    let page = 0;

    for line in &result.lines {
        for word in &line.words {
            for ch in &word.characters {
                let bbox = &ch.bounding_box;
                output.push_str(&format!(
                    "{} {} {} {} {} {}\n",
                    ch.character,
                    bbox.left,
                    bbox.top,
                    bbox.right,
                    bbox.bottom,
                    page
                ));
            }
            // Add space between words
            if !word.characters.is_empty() {
                let last_ch = word.characters.last().unwrap();
                let space_left = last_ch.bounding_box.right;
                let space_right = space_left + 5;
                output.push_str(&format!(
                    "{} {} {} {} {} {}\n",
                    ' ',
                    space_left,
                    last_ch.bounding_box.top,
                    space_right,
                    last_ch.bounding_box.bottom,
                    page
                ));
            }
        }
        // Add newline between lines
        if !line.words.is_empty() {
            let last_word = line.words.last().unwrap();
            if !last_word.characters.is_empty() {
                let last_ch = last_word.characters.last().unwrap();
                output.push_str(&format!(
                    "\t {} {} {} {}\n",
                    last_ch.bounding_box.right + 5,
                    last_ch.bounding_box.top,
                    last_ch.bounding_box.right + 10,
                    last_ch.bounding_box.bottom,
                ));
            }
        }
    }

    Ok(output)
}

/// Escape text for XML attribute
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Escape a single character for HTML content
fn html_escape_char(c: char) -> String {
    match c {
        '<' => "&lt;".to_string(),
        '>' => "&gt;".to_string(),
        '&' => "&amp;".to_string(),
        '"' => "&quot;".to_string(),
        '\'' => "&#x27;".to_string(),
        _ => c.to_string(),
    }
}

/// Format output in the specified format
pub fn format_output(result: &TextResult, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::PlainText => Ok(result.text.clone()),
        OutputFormat::Json => format_json(result),
        OutputFormat::HOcr => format_hocr(result),
        OutputFormat::Tsv => format_tsv(result),
        OutputFormat::Alto => format_alto(result),
        OutputFormat::Box => format_box(result),
        OutputFormat::Pdf => Err(crate::OcrError::Internal(
            "PDF output requires image path. Use format_pdf_with_image instead.".to_string(),
        )
        .into()),
    }
}

/// Parse output format from string
pub fn parse_output_format(s: &str) -> Result<OutputFormat> {
    match s.to_lowercase().as_str() {
        "text" | "txt" | "plain" => Ok(OutputFormat::PlainText),
        "json" => Ok(OutputFormat::Json),
        "hocr" | "html" => Ok(OutputFormat::HOcr),
        "tsv" => Ok(OutputFormat::Tsv),
        "alto" | "xml" => Ok(OutputFormat::Alto),
        "box" => Ok(OutputFormat::Box),
        "pdf" => Ok(OutputFormat::Pdf),
        _ => Err(crate::OcrError::InvalidInput(format!(
            "Unknown output format: {}. Supported: text, json, hocr, tsv, alto, box, pdf",
            s
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::text::{BoundingBox, CharacterResult, LineResult, TextResult, WordResult};

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

    #[test]
    fn test_format_alto() {
        let result = create_test_result();
        let alto = format_alto(&result).unwrap();

        assert!(alto.contains("<?xml version=\"1.0\""));
        assert!(alto.contains("<alto xmlns=\"http://www.loc.gov/standards/alto/ns-v4#\""));
        assert!(alto.contains("<String"));
        assert!(alto.contains("CONTENT=\"Hello\""));
        assert!(alto.contains("CONTENT=\"World\""));
        assert!(alto.contains("</alto>"));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("A & B"), "A &amp; B");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("say \"hi\""), "say &quot;hi&quot;");
    }

    #[test]
    fn test_parse_alto_format() {
        assert!(matches!(parse_output_format("alto"), Ok(OutputFormat::Alto)));
        assert!(matches!(parse_output_format("xml"), Ok(OutputFormat::Alto)));
    }
}
