//! Text processing utilities for OCR API

use crate::api::error::{ApiError, ApiResult};
use crate::core::text::{BoundingBox, TextResult};
use serde::{Deserialize, Serialize};

/// Text processing operations
pub struct TextProcessor;

impl TextProcessor {
    /// Clean and normalize text
    pub fn clean_text(text: &str) -> String {
        text.trim()
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract text from result
    pub fn extract_text(result: &TextResult) -> String {
        result.text.clone()
    }

    /// Extract text with confidence filtering
    pub fn extract_text_with_confidence(result: &TextResult, min_confidence: f32) -> String {
        result
            .characters
            .iter()
            .filter(|c| c.confidence >= min_confidence)
            .map(|c| c.character)
            .collect()
    }

    /// Get text statistics
    pub fn get_text_statistics(result: &TextResult) -> TextStatistics {
        let character_count = result.characters.len();
        let word_count = result.words.len();
        let line_count = result.lines.len();

        let avg_character_confidence = if character_count > 0 {
            result.characters.iter().map(|c| c.confidence).sum::<f32>() / character_count as f32
        } else {
            0.0
        };

        let avg_word_confidence = if word_count > 0 {
            result.words.iter().map(|w| w.confidence).sum::<f32>() / word_count as f32
        } else {
            0.0
        };

        let avg_line_confidence = if line_count > 0 {
            result.lines.iter().map(|l| l.confidence).sum::<f32>() / line_count as f32
        } else {
            0.0
        };

        TextStatistics {
            character_count,
            word_count,
            line_count,
            avg_character_confidence,
            avg_word_confidence,
            avg_line_confidence,
            overall_confidence: result.confidence,
        }
    }

    /// Filter results by confidence
    pub fn filter_by_confidence(result: &TextResult, min_confidence: f32) -> TextResult {
        let mut filtered = result.clone();

        // Filter characters
        filtered
            .characters
            .retain(|c| c.confidence >= min_confidence);

        // Filter words
        filtered.words.retain(|w| w.confidence >= min_confidence);

        // Filter lines
        filtered.lines.retain(|l| l.confidence >= min_confidence);

        // Rebuild text from filtered characters if we have character-level results
        // Otherwise, preserve the original text if overall confidence meets threshold
        if !filtered.characters.is_empty() {
            filtered.text = filtered.characters.iter().map(|c| c.character).collect();
        } else if result.confidence < min_confidence {
            // If overall confidence is below threshold and no characters, clear text
            filtered.text = String::new();
        }
        // Otherwise, keep the original text (no character-level filtering needed)

        filtered
    }

    /// Merge multiple text results
    pub fn merge_results(results: &[TextResult]) -> TextResult {
        if results.is_empty() {
            return TextResult::new(String::new(), 0.0, BoundingBox::new(0, 0, 0, 0));
        }

        let mut merged = results[0].clone();

        for result in &results[1..] {
            merged.text.push_str(&result.text);
            merged.characters.extend(result.characters.clone());
            merged.words.extend(result.words.clone());
            merged.lines.extend(result.lines.clone());
        }

        // Recalculate confidence
        let total_confidence: f32 = results.iter().map(|r| r.confidence).sum();
        merged.confidence = total_confidence / results.len() as f32;

        merged
    }

    /// Split text into lines
    pub fn split_into_lines(text: &str) -> Vec<String> {
        text.lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    }

    /// Split text into words
    pub fn split_into_words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.trim().to_string())
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Calculate text similarity
    pub fn calculate_similarity(text1: &str, text2: &str) -> f32 {
        // Simple Levenshtein distance-based similarity
        let distance = Self::levenshtein_distance(text1, text2);
        let max_len = text1.len().max(text2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f32 / max_len as f32)
        }
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s1_len = s1_chars.len();
        let s2_len = s2_chars.len();

        if s1_len == 0 {
            return s2_len;
        }
        if s2_len == 0 {
            return s1_len;
        }

        let mut matrix = vec![vec![0; s2_len + 1]; s1_len + 1];

        for i in 0..=s1_len {
            matrix[i][0] = i;
        }

        for j in 0..=s2_len {
            matrix[0][j] = j;
        }

        for i in 1..=s1_len {
            for j in 1..=s2_len {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[s1_len][s2_len]
    }
}

/// Text statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    /// Number of characters
    pub character_count: usize,
    /// Number of words
    pub word_count: usize,
    /// Number of lines
    pub line_count: usize,
    /// Average character confidence
    pub avg_character_confidence: f32,
    /// Average word confidence
    pub avg_word_confidence: f32,
    /// Average line confidence
    pub avg_line_confidence: f32,
    /// Overall confidence
    pub overall_confidence: f32,
}

/// Text export formats
pub enum TextExportFormat {
    /// Plain text
    PlainText,
    /// JSON
    Json,
    /// XML
    Xml,
    /// CSV
    Csv,
    /// Markdown
    Markdown,
}

/// Text exporter
pub struct TextExporter;

impl TextExporter {
    /// Export text result to string
    pub fn export_to_string(result: &TextResult, format: TextExportFormat) -> ApiResult<String> {
        match format {
            TextExportFormat::PlainText => Ok(result.text.clone()),
            TextExportFormat::Json => {
                serde_json::to_string_pretty(result).map_err(|e| ApiError::Serialization(e))
            }
            TextExportFormat::Xml => Self::export_to_xml(result),
            TextExportFormat::Csv => Self::export_to_csv(result),
            TextExportFormat::Markdown => Self::export_to_markdown(result),
        }
    }

    /// Export to XML
    fn export_to_xml(result: &TextResult) -> ApiResult<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<ocr_result>\n");
        xml.push_str(&format!(
            "  <text>{}</text>\n",
            Self::escape_xml(&result.text)
        ));
        xml.push_str(&format!(
            "  <confidence>{}</confidence>\n",
            result.confidence
        ));
        xml.push_str("</ocr_result>\n");
        Ok(xml)
    }

    /// Export to CSV
    fn export_to_csv(result: &TextResult) -> ApiResult<String> {
        let mut csv = String::new();
        csv.push_str("text,confidence,character_count,word_count,line_count\n");
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            Self::escape_csv(&result.text),
            result.confidence,
            result.characters.len(),
            result.words.len(),
            result.lines.len()
        ));
        Ok(csv)
    }

    /// Export to Markdown
    fn export_to_markdown(result: &TextResult) -> ApiResult<String> {
        let mut md = String::new();
        md.push_str("# OCR Result\n\n");
        md.push_str(&format!(
            "**Confidence:** {:.2}%\n\n",
            result.confidence * 100.0
        ));
        md.push_str(&format!("**Text:**\n\n{}\n\n", result.text));
        md.push_str(&format!("**Statistics:**\n"));
        md.push_str(&format!("- Characters: {}\n", result.characters.len()));
        md.push_str(&format!("- Words: {}\n", result.words.len()));
        md.push_str(&format!("- Lines: {}\n", result.lines.len()));
        Ok(md)
    }

    /// Escape XML special characters
    fn escape_xml(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Escape CSV special characters
    fn escape_csv(text: &str) -> String {
        if text.contains(',') || text.contains('"') || text.contains('\n') {
            format!("\"{}\"", text.replace('"', "\"\""))
        } else {
            text.to_string()
        }
    }
}
