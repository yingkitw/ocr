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
    /// TSV (tab-separated values, Tesseract-style with bbox per word)
    Tsv,
    /// CSV (comma-separated values, simpler per-word export for spreadsheets)
    Csv,
    /// ALTO XML (standard library format)
    Alto,
    /// Box file for Tesseract training
    Box,
    /// Hierarchical structured JSON with document elements
    StructuredJson,
    /// Markdown output with heading/paragraph/list structure
    Markdown,
    /// RAG-friendly chunked JSON for LLM / embedding pipelines
    RagJson,
    /// PRImA PAGE XML (document layout format)
    Page,
    /// Searchable PDF (image + invisible text layer, requires `pdf` feature)
    #[cfg(feature = "pdf")]
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
}

/// Generate hOCR (HTML OCR format) output
///
/// hOCR is an HTML-based format for representing OCR output with spatial information.
/// It follows the hOCR 4.1 specification from <http://kba.cloud/hocr-spec/>
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
        bbox.left, bbox.top, bbox.right, bbox.bottom
    ));

    // Content area (single carea for now)
    html.push_str(&format!(
        "    <div class='ocr_carea' id='block_1_1' title=\"bbox {} {} {} {}\">\n",
        bbox.left, bbox.top, bbox.right, bbox.bottom
    ));

    // Paragraph
    html.push_str(&format!(
        "      <p class='ocr_par' id='par_1_1' title=\"bbox {} {} {} {}\">\n",
        bbox.left, bbox.top, bbox.right, bbox.bottom
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

            html.push_str(&xml_escape(&word.text));

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
    s.replace(['\t', '\n'], " ")
}

/// Generate CSV (RFC 4180) output: one row per word with metadata columns.
///
/// Distinct from TSV: TSV is Tesseract-training-style (5 levels, fixed column
/// order matching `tesseract image.tsv`), while CSV is a simple flat per-word
/// export designed for direct import into spreadsheets and databases.
pub fn format_csv(result: &TextResult) -> Result<String> {
    let mut output = String::new();

    output.push_str("page,line,word,text,confidence,left,top,width,height\n");

    let page = 1;
    for (line_idx, line) in result.lines.iter().enumerate() {
        for (word_idx, word) in line.words.iter().enumerate() {
            let wb = &word.bounding_box;
            output.push_str(&format!(
                "{},{},{},{},{:.2},{},{},{},{}\n",
                page,
                line_idx + 1,
                word_idx + 1,
                escape_csv_field(&word.text),
                word.confidence * 100.0,
                wb.left,
                wb.top,
                wb.width(),
                wb.height(),
            ));
        }
    }

    Ok(output)
}

/// Escape a CSV field per RFC 4180: wrap in quotes if it contains comma,
/// quote, CR, or LF; double up internal quotes.
fn escape_csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
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
    xml.push_str("          <softwareName>OCR</softwareName>\n");
    xml.push_str(&format!(
        "          <softwareVersion>{}</softwareVersion>\n",
        env!("CARGO_PKG_VERSION")
    ));
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

/// Generate PRImA PAGE XML output
///
/// PAGE XML is a standard for representing document layout and OCR results,
/// developed by the PRImA Research Lab.  Spec: http://www.primaresearch.org/PAGE
///
/// Structure: `PcGts` root → `Metadata` + `Page` → `TextRegion` → `TextLine` → `Word`,
/// each with `Coords` (polygon points) and `TextEquiv` (Unicode text + confidence).
pub fn format_page(result: &TextResult) -> Result<String> {
    let mut xml = String::new();
    let bbox = &result.bounding_box;
    let now = Utc::now().to_rfc3339();

    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<PcGts xmlns=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd\">\n");
    xml.push_str("  <Metadata>\n");
    xml.push_str("    <Creator>OCR</Creator>\n");
    xml.push_str(&format!("    <Created>{}</Created>\n", now));
    xml.push_str(&format!("    <LastChange>{}</LastChange>\n", now));
    xml.push_str("  </Metadata>\n");

    xml.push_str(&format!(
        "  <Page imageFilename=\"\" imageWidth=\"{}\" imageHeight=\"{}\">\n",
        bbox.right, bbox.bottom
    ));

    // Single text region covering the full page
    xml.push_str("    <TextRegion id=\"region_1\">\n");
    xml.push_str(&format!(
        "      <Coords points=\"{}\"/>\n",
        coords_str(bbox.left, bbox.top, bbox.right, bbox.bottom)
    ));

    for (line_idx, line) in result.lines.iter().enumerate() {
        let lb = &line.bounding_box;
        xml.push_str(&format!(
            "      <TextLine id=\"line_{}\">\n",
            line_idx + 1
        ));
        xml.push_str(&format!(
        "        <Coords points=\"{}\"/>\n",
        coords_str(lb.left, lb.top, lb.right, lb.bottom)
        ));

        // Line-level text equivalent
        xml.push_str("        <TextEquiv>\n");
        xml.push_str(&format!(
            "          <Unicode>{}</Unicode>\n",
            xml_escape(&line.text)
        ));
        xml.push_str(&format!(
            "          <Confidence>{:.4}</Confidence>\n",
            line.confidence
        ));
        xml.push_str("        </TextEquiv>\n");

        // Words
        for (word_idx, word) in line.words.iter().enumerate() {
            let wb = &word.bounding_box;
            xml.push_str(&format!(
                "        <Word id=\"word_{}_{}\">\n",
                line_idx + 1,
                word_idx + 1
            ));
            xml.push_str(&format!(
                "          <Coords points=\"{}\"/>\n",
                coords_str(wb.left, wb.top, wb.right, wb.bottom)
            ));
            xml.push_str("          <TextEquiv>\n");
            xml.push_str(&format!(
                "            <Unicode>{}</Unicode>\n",
                xml_escape(&word.text)
            ));
            xml.push_str(&format!(
                "            <Confidence>{:.4}</Confidence>\n",
                word.confidence
            ));
            xml.push_str("          </TextEquiv>\n");
            xml.push_str("        </Word>\n");
        }

        xml.push_str("      </TextLine>\n");
    }

    xml.push_str("    </TextRegion>\n");
    xml.push_str("  </Page>\n");
    xml.push_str("</PcGts>\n");

    Ok(xml)
}

/// Format a bounding box as PAGE XML `points` attribute (clockwise polygon)
fn coords_str(left: u32, top: u32, right: u32, bottom: u32) -> String {
    format!("{},{} {},{} {},{} {},{}", left, top, right, top, right, bottom, left, bottom)
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
                    ch.character, bbox.left, bbox.top, bbox.right, bbox.bottom, page
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

/// Generate a Tesseract-compatible `.box` file (makebox format).
///
/// Each line: `char left bottom right top page`
/// Coordinates use a **bottom-left** origin (Tesseract convention), so
/// `image_height` is required to convert from the OCR top-left boxes.
///
/// When per-character boxes are missing, character widths are estimated by
/// equally subdividing the parent word box — enough to bootstrap training.
pub fn format_makebox(result: &TextResult, image_height: u32) -> Result<String> {
    let mut output = String::new();
    let page = 0u32;
    let h = image_height.max(1);

    for line in &result.lines {
        for word in &line.words {
            let chars: Vec<(char, BoundingBox)> = if !word.characters.is_empty() {
                word.characters
                    .iter()
                    .map(|c| (c.character, c.bounding_box))
                    .collect()
            } else if !word.text.is_empty() {
                subdivide_word_box(&word.text, &word.bounding_box)
            } else {
                Vec::new()
            };

            for (ch, bbox) in chars {
                // Skip whitespace in classic makebox (Tesseract omits spaces);
                // include them with a thin estimated box when present.
                let left = bbox.left;
                let right = bbox.right.max(left + 1);
                let top = bbox.top;
                let bottom = bbox.bottom.max(top + 1);
                // Convert top-left → bottom-left origin
                let box_bottom = h.saturating_sub(bottom);
                let box_top = h.saturating_sub(top);
                output.push_str(&format!(
                    "{} {} {} {} {} {}\n",
                    escape_box_char(ch),
                    left,
                    box_bottom,
                    right,
                    box_top,
                    page
                ));
            }
        }
    }

    // Fallback: no structured lines — emit from top-level characters / text
    if output.is_empty() && !result.characters.is_empty() {
        for ch in &result.characters {
            let bbox = &ch.bounding_box;
            let box_bottom = h.saturating_sub(bbox.bottom);
            let box_top = h.saturating_sub(bbox.top);
            output.push_str(&format!(
                "{} {} {} {} {} {}\n",
                escape_box_char(ch.character),
                bbox.left,
                box_bottom,
                bbox.right.max(bbox.left + 1),
                box_top,
                page
            ));
        }
    }

    Ok(output)
}

fn escape_box_char(ch: char) -> String {
    match ch {
        ' ' => " ".to_string(),
        '\t' => "\\t".to_string(),
        '\n' => "\\n".to_string(),
        _ => ch.to_string(),
    }
}

fn subdivide_word_box(text: &str, bbox: &BoundingBox) -> Vec<(char, BoundingBox)> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let n = chars.len() as u32;
    let width = bbox.width().max(n);
    let cell = width / n;
    chars
        .into_iter()
        .enumerate()
        .map(|(i, ch)| {
            let left = bbox.left + i as u32 * cell;
            let right = if i as u32 + 1 == n {
                bbox.right
            } else {
                left + cell
            };
            (
                ch,
                BoundingBox::new(left, bbox.top, right.max(left + 1), bbox.bottom),
            )
        })
        .collect()
}

/// Escape text for XML attribute
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Document element type for structured output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocElement {
    Heading { level: u8, text: String },
    Paragraph { text: String },
    ListItem { text: String, ordered: bool, number: Option<usize> },
    Table { rows: Vec<Vec<String>> },
    Figure { caption: String },
    CodeBlock { text: String },
}

/// Structured document output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredDocument {
    pub title: Option<String>,
    pub elements: Vec<DocElement>,
    pub metadata: HashMap<String, String>,
}

impl StructuredDocument {
    pub fn from_text(text: &str) -> Self {
        let mut elements = Vec::new();
        let mut current_paragraph = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !current_paragraph.is_empty() {
                    elements.push(DocElement::Paragraph {
                        text: current_paragraph.trim().to_string(),
                    });
                    current_paragraph.clear();
                }
                continue;
            }

            // Check for list items
            if let Some(list_item) = Self::parse_list_item(trimmed) {
                if !current_paragraph.is_empty() {
                    elements.push(DocElement::Paragraph {
                        text: current_paragraph.trim().to_string(),
                    });
                    current_paragraph.clear();
                }
                elements.push(list_item);
                continue;
            }

            // Check for heading (simple heuristic: short line, all caps, or ends without period)
            if Self::is_heading(trimmed, &current_paragraph) {
                if !current_paragraph.is_empty() {
                    elements.push(DocElement::Paragraph {
                        text: current_paragraph.trim().to_string(),
                    });
                    current_paragraph.clear();
                }
                let level = Self::heading_level(trimmed);
                elements.push(DocElement::Heading {
                    level,
                    text: trimmed.to_string(),
                });
                continue;
            }

            if !current_paragraph.is_empty() {
                current_paragraph.push(' ');
            }
            current_paragraph.push_str(trimmed);
        }

        if !current_paragraph.is_empty() {
            elements.push(DocElement::Paragraph {
                text: current_paragraph.trim().to_string(),
            });
        }

        StructuredDocument {
            title: None,
            elements,
            metadata: HashMap::new(),
        }
    }

    fn parse_list_item(text: &str) -> Option<DocElement> {
        let trimmed = text.trim_start();
        for prefix in ["• ", "- ", "* "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                return Some(DocElement::ListItem {
                    text: rest.to_string(),
                    ordered: false,
                    number: None,
                });
            }
        }
        // Numbered list: "1. text" or "1) text"
        let mut chars = trimmed.chars();
        let first = chars.next()?;
        if first.is_numeric() {
            let mut num_str = first.to_string();
            for ch in chars {
                if ch.is_numeric() {
                    num_str.push(ch);
                } else if ch == '.' || ch == ')' {
                    let rest = trimmed[num_str.len() + 1..].trim_start().to_string();
                    return Some(DocElement::ListItem {
                        text: rest,
                        ordered: true,
                        number: num_str.parse().ok(),
                    });
                } else {
                    break;
                }
            }
        }
        None
    }

    fn is_heading(text: &str, prev_context: &str) -> bool {
        if text.len() < 3 || text.len() > 120 {
            return false;
        }
        // All caps heading
        if text.chars().all(|c| c.is_uppercase() || c.is_numeric() || c.is_whitespace() || c == '-' || c == '_')
            && text.chars().any(|c| c.is_alphabetic())
            && text.len() < 80
        {
            return true;
        }
        // Short line without ending punctuation (likely heading)
        let words = text.split_whitespace().count();
        if words <= 8 && !text.ends_with('.') && !text.ends_with(',') && !text.ends_with(';') {
            // If previous context exists, this is likely a new section
            if !prev_context.is_empty() && text.len() > 3 {
                return true;
            }
        }
        false
    }

    fn heading_level(text: &str) -> u8 {
        if text.chars().all(|c| c.is_uppercase() || c.is_numeric() || c.is_whitespace() || c == '-' || c == '_') {
            1
        } else if text.len() < 40 {
            2
        } else {
            3
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        for element in &self.elements {
            match element {
                DocElement::Heading { level, text } => {
                    let hashes = "#".repeat(*level as usize);
                    md.push_str(&format!("{} {}\n\n", hashes, text));
                }
                DocElement::Paragraph { text } => {
                    md.push_str(&format!("{}\n\n", text));
                }
                DocElement::ListItem { text, ordered: false, .. } => {
                    md.push_str(&format!("- {}\n", text));
                }
                DocElement::ListItem { text, ordered: true, number: Some(n) } => {
                    md.push_str(&format!("{}. {}\n", n, text));
                }
                DocElement::ListItem { text, ordered: true, number: None } => {
                    md.push_str(&format!("1. {}\n", text));
                }
                DocElement::Table { rows } => {
                    if !rows.is_empty() {
                        md.push_str(&markdown_table(rows));
                        md.push('\n');
                    }
                }
                DocElement::Figure { caption } => {
                    md.push_str(&format!("*Figure: {}*\n\n", caption));
                }
                DocElement::CodeBlock { text } => {
                    md.push_str(&format!("```\n{}\n```\n\n", text));
                }
            }
        }
        md
    }
}

/// Generate structured JSON output with document elements
pub fn format_structured_json(result: &TextResult) -> Result<String> {
    let doc = StructuredDocument::from_text(&result.text);
    serde_json::to_string_pretty(&doc)
        .map_err(|e| crate::OcrError::Internal(format!("JSON serialization error: {}", e)))
}

/// Render table rows as GitHub-flavored Markdown
fn markdown_table(rows: &[Vec<String>]) -> String {
    let mut md = String::new();
    for (i, row) in rows.iter().enumerate() {
        let cells: Vec<String> = row.iter().map(|c| format!("| {} ", c)).collect();
        md.push_str(&cells.join(""));
        md.push_str("|\n");
        if i == 0 {
            let sep = "|---".repeat(row.len());
            md.push_str(&format!("{}|\n", sep));
        }
    }
    md
}

/// A retrieval-friendly chunk for RAG / LLM pipelines.
///
/// One chunk per document element (heading, paragraph, list item, table,
/// figure, code block), tagged with the heading path that anchors it so
/// downstream embedders keep section context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChunk {
    /// Zero-based chunk index in reading order
    pub id: usize,
    /// Chunk kind: `heading`, `paragraph`, `list_item`, `table`, `figure`, `code`
    pub kind: String,
    /// Chunk text (tables are rendered as Markdown)
    pub text: String,
    /// Heading path anchoring this chunk (e.g. `["Introduction", "Scope"]`)
    pub section: Vec<String>,
}

/// Split a document into RAG-friendly chunks with heading context.
pub fn rag_chunks(result: &TextResult) -> Vec<RagChunk> {
    let doc = StructuredDocument::from_text(&result.text);
    let mut chunks: Vec<RagChunk> = Vec::new();
    // Stack of (level, heading text) currently in scope
    let mut section: Vec<(u8, String)> = Vec::new();

    for element in &doc.elements {
        match element {
            DocElement::Heading { level, text } => {
                // Pop shallower-or-equal headings: a level-2 heading closes
                // prior level-2/3 sections but stays nested under level-1.
                while section.last().is_some_and(|(l, _)| l >= level) {
                    section.pop();
                }
                chunks.push(RagChunk {
                    id: chunks.len(),
                    kind: "heading".to_string(),
                    text: text.clone(),
                    section: section.iter().map(|(_, t)| t.clone()).collect(),
                });
                section.push((*level, text.clone()));
            }
            DocElement::Paragraph { text } => {
                push_chunk(&mut chunks, "paragraph", text, &section)
            }
            DocElement::ListItem { text, .. } => {
                push_chunk(&mut chunks, "list_item", text, &section)
            }
            DocElement::Table { rows } => {
                push_chunk(&mut chunks, "table", &markdown_table(rows), &section)
            }
            DocElement::Figure { caption } => push_chunk(&mut chunks, "figure", caption, &section),
            DocElement::CodeBlock { text } => push_chunk(&mut chunks, "code", text, &section),
        }
    }
    chunks
}

fn push_chunk(chunks: &mut Vec<RagChunk>, kind: &str, text: &str, section: &[(u8, String)]) {
    chunks.push(RagChunk {
        id: chunks.len(),
        kind: kind.to_string(),
        text: text.to_string(),
        section: section.iter().map(|(_, t)| t.clone()).collect(),
    });
}

/// Generate RAG-friendly chunked JSON output for LLM / embedding pipelines
pub fn format_rag_json(result: &TextResult) -> Result<String> {
    let chunks = rag_chunks(result);
    serde_json::to_string_pretty(&serde_json::json!({ "chunks": chunks }))
        .map_err(|e| crate::OcrError::Internal(format!("JSON serialization error: {}", e)))
}

/// Generate Markdown output with document structure
pub fn format_markdown(result: &TextResult) -> Result<String> {
    let doc = StructuredDocument::from_text(&result.text);
    Ok(doc.to_markdown())
}

/// Format output in the specified format
pub fn format_output(result: &TextResult, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::PlainText => Ok(result.text.clone()),
        OutputFormat::Json => format_json(result),
        OutputFormat::HOcr => format_hocr(result),
        OutputFormat::Tsv => format_tsv(result),
        OutputFormat::Csv => format_csv(result),
        OutputFormat::Alto => format_alto(result),
        OutputFormat::Box => format_box(result),
        OutputFormat::StructuredJson => format_structured_json(result),
        OutputFormat::Markdown => format_markdown(result),
        OutputFormat::RagJson => format_rag_json(result),
        OutputFormat::Page => format_page(result),
        #[cfg(feature = "pdf")]
        OutputFormat::Pdf => Err(crate::OcrError::InvalidInput(
            "PDF output requires binary data — use format_searchable_pdf() with image bytes".to_string()
        )),
    }
}

/// Parse output format from string
pub fn parse_output_format(s: &str) -> Result<OutputFormat> {
    match s.to_lowercase().as_str() {
        "text" | "txt" | "plain" => Ok(OutputFormat::PlainText),
        "json" => Ok(OutputFormat::Json),
        "hocr" | "html" => Ok(OutputFormat::HOcr),
        "tsv" => Ok(OutputFormat::Tsv),
        "csv" => Ok(OutputFormat::Csv),
        "alto" | "xml" => Ok(OutputFormat::Alto),
        "box" => Ok(OutputFormat::Box),
        "md" | "markdown" => Ok(OutputFormat::Markdown),
        "structured" | "structured-json" | "doc" => Ok(OutputFormat::StructuredJson),
        "rag" | "rag-json" | "chunks" => Ok(OutputFormat::RagJson),
        "page" | "page-xml" | "primapage" => Ok(OutputFormat::Page),
        #[cfg(feature = "pdf")]
        "pdf" | "searchable-pdf" => Ok(OutputFormat::Pdf),
        _ => Err(crate::OcrError::InvalidInput(format!(
            "Unknown output format: {}. Supported: text, json, hocr, tsv, csv, alto, box, markdown, structured-json, rag, page{}",
            s,
            if cfg!(feature = "pdf") { ", pdf" } else { "" }
        ))),
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
        let mut line = LineResult::new(
            "Hello World".to_string(),
            0.9,
            BoundingBox::new(0, 0, 200, 20),
        );

        // Add words
        let mut word1 = WordResult::new("Hello".to_string(), 0.95, BoundingBox::new(0, 0, 60, 20));
        let mut word2 =
            WordResult::new("World".to_string(), 0.85, BoundingBox::new(70, 0, 130, 20));

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
        assert!(hocr.contains("Hello"));
        assert!(hocr.contains("World"));
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
    fn test_format_csv() {
        let result = create_test_result();
        let csv = format_csv(&result).unwrap();

        // Header row
        assert!(csv.starts_with("page,line,word,text,confidence,left,top,width,height\n"));
        // One row per word (Hello + World = 2 words)
        assert_eq!(csv.lines().count(), 3);
        // Comma-separated, not tab-separated
        assert!(csv.contains("Hello,95.00"));
        assert!(csv.contains("World,85.00"));
        assert!(!csv.contains('\t'));
    }

    #[test]
    fn test_format_csv_rfc4180_escaping() {
        // Fields containing commas, quotes, or newlines must be quoted
        assert_eq!(escape_csv_field("hello"), "hello");
        assert_eq!(escape_csv_field("hello,world"), "\"hello,world\"");
        assert_eq!(escape_csv_field("she said \"hi\""), "\"she said \"\"hi\"\"\"");
        assert_eq!(escape_csv_field("line1\nline2"), "\"line1\nline2\"");
    }

    #[test]
    fn test_format_csv_alias() {
        assert_eq!(parse_output_format("csv").unwrap(), OutputFormat::Csv);
    }

    #[test]
    fn test_format_makebox_bottom_left_origin() {
        let result = create_test_result();
        let image_height = 50u32;
        let box_out = format_makebox(&result, image_height).unwrap();
        assert!(!box_out.is_empty());
        // First char of "Hello" at top-left y=0..20 → bottom-left bottom=30, top=50
        let first = box_out.lines().next().unwrap();
        let parts: Vec<&str> = first.split_whitespace().collect();
        assert!(parts.len() >= 6);
        assert_eq!(parts[0], "H");
        // page index
        assert_eq!(parts[5], "0");
        let bottom: u32 = parts[2].parse().unwrap();
        let top: u32 = parts[4].parse().unwrap();
        assert!(top > bottom, "top should be above bottom in BL origin");
        assert_eq!(top, image_height); // char top was 0
    }

    #[test]
    fn test_subdivide_word_box() {
        let bbox = BoundingBox::new(0, 0, 50, 10);
        let parts = subdivide_word_box("Hi", &bbox);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].0, 'H');
        assert_eq!(parts[1].0, 'i');
        assert!(parts[0].1.right <= parts[1].1.left || parts[0].1.right == 25);
    }

    #[test]
    fn test_parse_output_format() {
        assert!(matches!(
            parse_output_format("json"),
            Ok(OutputFormat::Json)
        ));
        assert!(matches!(
            parse_output_format("TEXT"),
            Ok(OutputFormat::PlainText)
        ));
        assert!(matches!(
            parse_output_format("hocr"),
            Ok(OutputFormat::HOcr)
        ));
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
        assert!(matches!(
            parse_output_format("alto"),
            Ok(OutputFormat::Alto)
        ));
        assert!(matches!(parse_output_format("xml"), Ok(OutputFormat::Alto)));
    }

    #[test]
    fn test_parse_markdown_format() {
        assert!(matches!(
            parse_output_format("markdown"),
            Ok(OutputFormat::Markdown)
        ));
        assert!(matches!(parse_output_format("md"), Ok(OutputFormat::Markdown)));
    }

    #[test]
    fn test_parse_structured_json_format() {
        assert!(matches!(
            parse_output_format("structured-json"),
            Ok(OutputFormat::StructuredJson)
        ));
        assert!(matches!(parse_output_format("doc"), Ok(OutputFormat::StructuredJson)));
    }

    #[test]
    fn test_structured_document_from_text() {
        let text = "INTRODUCTION\n\nThis is a paragraph with multiple sentences. It continues on.\n\n• First bullet\n• Second bullet\n\n1. Ordered item\n2. Another ordered\n\nConclusion text here.";
        let doc = StructuredDocument::from_text(text);
        assert!(!doc.elements.is_empty());
        assert!(matches!(doc.elements[0], DocElement::Heading { level: 1, .. }));
    }

    #[test]
    fn test_markdown_output() {
        let text = "INTRODUCTION\n\nThis is a paragraph.\n\n• Bullet item\n\n1. Numbered item\n\nSUBSECTION\n\nMore text.";
        let doc = StructuredDocument::from_text(text);
        let md = doc.to_markdown();
        assert!(md.contains("# INTRODUCTION"));
        assert!(md.contains("- Bullet item"));
        assert!(md.contains("1. Numbered item"));
        assert!(md.contains("# SUBSECTION"));
    }

    #[test]
    fn test_format_markdown_output() {
        let mut result = create_test_result();
        result.text = "HEADING\n\nBody text here.".to_string();
        let md = format_markdown(&result).unwrap();
        assert!(md.contains("# HEADING"));
        assert!(md.contains("Body text here."));
    }

    #[test]
    fn test_format_structured_json_output() {
        let mut result = create_test_result();
        result.text = "INTRO\n\nParagraph.".to_string();
        let json = format_structured_json(&result).unwrap();
        assert!(json.contains("\"elements\""));
        assert!(json.contains("\"Heading\""));
    }

    #[test]
    fn test_rag_chunks_heading_context() {
        let mut result = create_test_result();
        // Note: the heading heuristic only fires for a non-all-caps heading
        // when it directly follows paragraph text (no blank line in between).
        result.text = "INTRO\n\nFirst paragraph.\nMore Details\n\nSecond paragraph.\n\n- item one\n- item two"
            .to_string();
        let chunks = rag_chunks(&result);

        // heading, para, heading, para, list x2
        assert_eq!(chunks.len(), 6);
        assert_eq!(chunks[0].kind, "heading");
        assert_eq!(chunks[0].text, "INTRO");
        assert!(chunks[0].section.is_empty());
        assert_eq!(chunks[1].kind, "paragraph");
        assert_eq!(chunks[1].section, vec!["INTRO".to_string()]);
        // "More Details" is not all-caps → level 2, nests under level-1 "INTRO"
        assert_eq!(chunks[2].kind, "heading");
        assert_eq!(chunks[2].section, vec!["INTRO".to_string()]);
        assert_eq!(chunks[3].section, vec!["INTRO".to_string(), "More Details".to_string()]);
        assert_eq!(chunks[4].kind, "list_item");
        assert_eq!(chunks[4].text, "item one");
        assert_eq!(chunks[5].text, "item two");
        // ids are sequential
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.id, i);
        }
    }

    #[test]
    fn test_markdown_table() {
        let md = markdown_table(&[vec!["A".into(), "B".into()], vec!["1".into(), "2".into()]]);
        assert!(md.contains("| A | B |\n"));
        assert!(md.contains("|---|---|\n"));
        assert!(md.contains("| 1 | 2 |\n"));
    }

    #[test]
    fn test_rag_chunks_sibling_heading_closes_section() {
        let mut result = create_test_result();
        result.text = "ONE\n\npara a\n\nTWO\n\npara b".to_string();
        let chunks = rag_chunks(&result);
        // "TWO" is a sibling of "ONE": para b's section must not contain "ONE"
        assert_eq!(chunks[3].section, vec!["TWO".to_string()]);
    }

    #[test]
    fn test_format_rag_json_output() {
        let mut result = create_test_result();
        result.text = "TITLE\n\nBody text.".to_string();
        let json = format_rag_json(&result).unwrap();
        assert!(json.contains("\"chunks\""));
        assert!(json.contains("\"kind\": \"heading\""));
        assert!(json.contains("\"section\": []"));
        // Empty input yields an empty chunk list, not an error
        result.text = String::new();
        let json = format_rag_json(&result).unwrap();
        assert_eq!(json.replace(char::is_whitespace, ""), r#"{"chunks":[]}"#);
    }

    #[test]
    fn test_format_page() {
        let result = create_test_result();
        let page = format_page(&result).unwrap();

        assert!(page.contains("<?xml version=\"1.0\""));
        assert!(page.contains("<PcGts xmlns=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15\""));
        assert!(page.contains("<Metadata>"));
        assert!(page.contains("<Creator>OCR</Creator>"));
        assert!(page.contains("<Page imageFilename=\"\""));
        assert!(page.contains("<TextRegion id=\"region_1\""));
        assert!(page.contains("<TextLine id=\"line_1\""));
        assert!(page.contains("<Word id=\"word_1_1\""));
        assert!(page.contains("<Word id=\"word_1_2\""));
        assert!(page.contains("<Unicode>Hello</Unicode>"));
        assert!(page.contains("<Unicode>World</Unicode>"));
        assert!(page.contains("<Confidence>0.9500</Confidence>"));
        assert!(page.contains("<Confidence>0.8500</Confidence>"));
        assert!(page.contains("</PcGts>"));
    }

    #[test]
    fn test_format_page_coords_polygon() {
        let result = create_test_result();
        let page = format_page(&result).unwrap();
        // Coords are clockwise polygon: left,top right,top right,bottom left,bottom
        let lb = &result.lines[0].bounding_box;
        let expected = format!(
            "points=\"{},{} {},{} {},{} {},{}\"",
            lb.left, lb.top, lb.right, lb.top, lb.right, lb.bottom, lb.left, lb.bottom
        );
        assert!(page.contains(&expected), "expected coords polygon not found");
    }

    #[test]
    fn test_format_page_xml_escaping() {
        let mut result = create_test_result();
        let raw = "A & B < C > D";
        result.lines[0].words[0].text = raw.to_string();
        let page = format_page(&result).unwrap();
        // The Unicode element must contain the XML-escaped text
        let escaped = xml_escape(raw);
        assert!(page.contains(&format!("<Unicode>{}</Unicode>", escaped)));
        // Raw unescaped text must NOT appear
        assert!(!page.contains(&format!("<Unicode>{}</Unicode>", raw)));
    }

    #[test]
    fn test_parse_page_format() {
        assert!(matches!(parse_output_format("page"), Ok(OutputFormat::Page)));
        assert!(matches!(parse_output_format("page-xml"), Ok(OutputFormat::Page)));
        assert!(matches!(parse_output_format("PAGE"), Ok(OutputFormat::Page)));
    }

    #[test]
    fn test_format_page_empty_result() {
        let result = TextResult::new(String::new(), 0.0, BoundingBox::new(0, 0, 100, 100));
        let page = format_page(&result).unwrap();
        assert!(page.contains("<PcGts"));
        assert!(page.contains("<Page imageFilename=\"\" imageWidth=\"100\" imageHeight=\"100\">"));
        assert!(page.contains("<TextRegion id=\"region_1\""));
        // No lines → no TextLine elements
        assert!(!page.contains("<TextLine"));
        assert!(page.contains("</PcGts>"));
    }
}
