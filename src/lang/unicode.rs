//! Unicode handling operations
//!
//! Script detection and classification for multi-language OCR routing.

use crate::utils::Result;

/// Unicode handler
pub struct UnicodeHandler;

impl UnicodeHandler {
    /// Normalize text
    pub fn normalize(text: &str) -> Result<String> {
        Ok(text.to_string())
    }
}

/// Supported Unicode scripts for OCR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    CJK, // Chinese, Japanese, Korean
    Arabic,
    Cyrillic,
    Greek,
    Hebrew,
    Thai,
    Devanagari,
    Other,
}

impl Script {
    /// Detect the dominant script in a text sample
    pub fn detect(text: &str) -> Script {
        let mut counts = std::collections::HashMap::new();
        for ch in text.chars() {
            let script = Self::classify_char(ch);
            *counts.entry(script).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(script, _)| script)
            .unwrap_or(Script::Latin)
    }

    /// Detect scripts with percentages
    pub fn detect_distribution(text: &str) -> Vec<(Script, f32)> {
        if text.is_empty() {
            return vec![(Script::Latin, 1.0)];
        }

        let mut counts = std::collections::HashMap::new();
        let total = text.chars().count() as f32;

        for ch in text.chars() {
            let script = Self::classify_char(ch);
            *counts.entry(script).or_insert(0usize) += 1;
        }

        let mut result: Vec<(Script, f32)> = counts
            .into_iter()
            .map(|(s, c)| (s, c as f32 / total))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }

    /// Classify a single character to its script
    pub fn classify_char(ch: char) -> Script {
        let cp = ch as u32;
        match cp {
            // Latin
            0x0041..=0x007A | 0x00C0..=0x017F | 0x0180..=0x024F => Script::Latin,
            // CJK
            0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x3000..=0x303F
            | 0x3040..=0x309F
            | 0x30A0..=0x30FF
            | 0xAC00..=0xD7AF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF => Script::CJK,
            // Arabic
            0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => Script::Arabic,
            // Cyrillic
            0x0400..=0x04FF | 0x0500..=0x052F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => {
                Script::Cyrillic
            }
            // Greek
            0x0370..=0x03FF | 0x1F00..=0x1FFF => Script::Greek,
            // Hebrew
            0x0590..=0x05FF | 0xFB1D..=0xFB4F => Script::Hebrew,
            // Thai
            0x0E00..=0x0E7F => Script::Thai,
            // Devanagari
            0x0900..=0x097F | 0xA8E0..=0xA8FF => Script::Devanagari,
            // Common (digits, punctuation, symbols)
            0x0020..=0x0040 | 0x007B..=0x007E => {
                Script::Latin // Default to Latin for common chars
            }
            _ => Script::Other,
        }
    }

    /// Recommended OCR engine config for this script
    pub fn recommended_config(&self) -> &'static str {
        match self {
            Script::Latin => "crnn_latin",
            Script::CJK => "crnn_cjk",
            Script::Arabic => "crnn_arabic",
            Script::Cyrillic => "crnn_cyrillic",
            Script::Greek => "crnn_greek",
            Script::Hebrew => "crnn_hebrew",
            Script::Thai => "crnn_thai",
            Script::Devanagari => "crnn_devanagari",
            Script::Other => "crnn_latin",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_latin() {
        assert_eq!(Script::detect("Hello World"), Script::Latin);
    }

    #[test]
    fn test_detect_cjk() {
        assert_eq!(Script::detect("你好世界"), Script::CJK);
    }

    #[test]
    fn test_detect_cyrillic() {
        assert_eq!(Script::detect("Привет"), Script::Cyrillic);
    }

    #[test]
    fn test_classify_char() {
        assert_eq!(Script::classify_char('A'), Script::Latin);
        assert_eq!(Script::classify_char('漢'), Script::CJK);
        assert_eq!(Script::classify_char('1'), Script::Latin);
    }

    #[test]
    fn test_detect_distribution() {
        let dist = Script::detect_distribution("Hello 你好");
        assert!(dist.iter().any(|(s, _)| *s == Script::Latin));
        assert!(dist.iter().any(|(s, _)| *s == Script::CJK));
    }
}
