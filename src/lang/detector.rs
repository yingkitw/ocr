//! Language detection from recognized text.
//!
//! Uses Unicode script classification plus lightweight dictionary hit-rates
//! for Latin languages. This enables `--lang auto` without an external LM.

use crate::lang::dictionary::DictionaryHandler;
use crate::lang::unicode::Script;
use crate::utils::Result;

/// Language detector
pub struct LanguageDetector;

/// Detection result with confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageGuess {
    pub language: String,
    pub confidence: f32,
    pub script: Script,
}

impl LanguageDetector {
    /// Detect language from text. Returns ISO-ish codes used by this crate
    /// (`en`, `fr`, `de`, `es`, `zh`, `ja`, `ko`, `ru`, …).
    pub fn detect_language(text: &str) -> Result<String> {
        Ok(Self::detect(text).language)
    }

    /// Full detection with confidence and dominant script.
    pub fn detect(text: &str) -> LanguageGuess {
        let trimmed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        if trimmed.is_empty() {
            return LanguageGuess {
                language: "en".into(),
                confidence: 0.1,
                script: Script::Latin,
            };
        }

        let script = Script::detect(text);
        match script {
            Script::CJK => detect_cjk(text),
            Script::Cyrillic => LanguageGuess {
                language: "ru".into(),
                confidence: 0.85,
                script,
            },
            Script::Arabic => LanguageGuess {
                language: "ar".into(),
                confidence: 0.85,
                script,
            },
            Script::Greek => LanguageGuess {
                language: "el".into(),
                confidence: 0.85,
                script,
            },
            Script::Hebrew => LanguageGuess {
                language: "he".into(),
                confidence: 0.85,
                script,
            },
            Script::Thai => LanguageGuess {
                language: "th".into(),
                confidence: 0.85,
                script,
            },
            Script::Devanagari => LanguageGuess {
                language: "hi".into(),
                confidence: 0.85,
                script,
            },
            Script::Latin | Script::Other => detect_latin(text),
        }
    }
}

fn detect_cjk(text: &str) -> LanguageGuess {
    let mut hiragana = 0u32;
    let mut katakana = 0u32;
    let mut hangul = 0u32;
    let mut han = 0u32;

    for c in text.chars() {
        let u = c as u32;
        match u {
            0x3040..=0x309F => hiragana += 1,
            0x30A0..=0x30FF => katakana += 1,
            0xAC00..=0xD7AF => hangul += 1,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF => han += 1,
            _ => {}
        }
    }

    if hangul > 0 && hangul >= hiragana + katakana {
        return LanguageGuess {
            language: "ko".into(),
            confidence: 0.9,
            script: Script::CJK,
        };
    }
    if hiragana + katakana > 0 {
        return LanguageGuess {
            language: "ja".into(),
            confidence: 0.9,
            script: Script::CJK,
        };
    }
    if han > 0 {
        return LanguageGuess {
            language: "zh".into(),
            confidence: 0.8,
            script: Script::CJK,
        };
    }
    LanguageGuess {
        language: "zh".into(),
        confidence: 0.5,
        script: Script::CJK,
    }
}

fn detect_latin(text: &str) -> LanguageGuess {
    let candidates = ["en", "fr", "de", "es", "it", "pt"];
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| w.len() >= 2)
        .collect();

    if words.is_empty() {
        return LanguageGuess {
            language: "en".into(),
            confidence: 0.3,
            script: Script::Latin,
        };
    }

    let mut best = ("en", 0.0f32);
    for &lang in &candidates {
        let handler = DictionaryHandler::new_for_language(lang);
        let hits = words
            .iter()
            .filter(|w| handler.is_word_valid(w))
            .count() as f32;
        let score = hits / words.len() as f32;
        if score > best.1 {
            best = (lang, score);
        }
    }

    // Accent heuristics as a light tie-breaker / boost
    let lower = text.to_lowercase();
    if lower.contains('ñ') || lower.contains('¿') || lower.contains('¡') {
        if best.0 == "es" || best.1 < 0.4 {
            best = ("es", best.1.max(0.55));
        }
    }
    if lower.contains('ß') || lower.contains("sch") {
        if best.0 == "de" || best.1 < 0.4 {
            best = ("de", best.1.max(0.55));
        }
    }
    if lower.contains('ç') || lower.contains("ção") {
        if best.0 == "pt" || best.1 < 0.4 {
            best = ("pt", best.1.max(0.5));
        }
    }

    LanguageGuess {
        language: best.0.into(),
        confidence: best.1.clamp(0.15, 0.99),
        script: Script::Latin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        let g = LanguageDetector::detect("the quick brown fox jumps over the lazy dog");
        assert_eq!(g.language, "en");
        assert!(g.confidence > 0.3);
    }

    #[test]
    fn test_detect_cyrillic() {
        let g = LanguageDetector::detect("Привет мир");
        assert_eq!(g.language, "ru");
    }

    #[test]
    fn test_detect_chinese() {
        let g = LanguageDetector::detect("你好世界");
        assert_eq!(g.language, "zh");
    }

    #[test]
    fn test_detect_japanese() {
        let g = LanguageDetector::detect("こんにちは世界");
        assert_eq!(g.language, "ja");
    }

    #[test]
    fn test_detect_korean() {
        let g = LanguageDetector::detect("안녕하세요");
        assert_eq!(g.language, "ko");
    }

    #[test]
    fn test_detect_empty_defaults_en() {
        let g = LanguageDetector::detect("   ");
        assert_eq!(g.language, "en");
    }

    #[test]
    fn test_detect_language_api() {
        let lang = LanguageDetector::detect_language("Bonjour le monde").unwrap();
        assert!(!lang.is_empty());
    }
}
