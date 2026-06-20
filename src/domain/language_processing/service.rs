//! Language Processing Service
//!
//! High-level service for language-specific text processing.
//! Handles dictionary correction, language detection, and CJK text processing.

use super::{LanguageProcessingError, LanguageProcessingResult};
use crate::lang::dictionary::Dictionary;
use crate::core::text::TextResult;

pub struct LanguageProcessingService {
    dictionaries: std::collections::HashMap<String, Dictionary>,
}

impl LanguageProcessingService {
    pub fn new() -> Self {
        let mut service = Self {
            dictionaries: std::collections::HashMap::new(),
        };
        service.load_default_dictionaries();
        service
    }

    fn load_default_dictionaries(&mut self) {
        let mut en_dict = Dictionary::new();
        en_dict.load_words(&[
            "the", "this", "that", "and", "for", "are", "was", "but", "not", "you",
            "all", "can", "had", "her", "was", "one", "our", "out", "has", "have",
            "from", "they", "been", "with", "their", "would", "about", "which",
            "there", "could", "should", "people", "water", "where", "after",
            "still", "world", "hello", "ocr", "text", "image", "file",
        ]);
        self.dictionaries.insert("en".to_string(), en_dict);

        let mut zh_dict = Dictionary::new();
        zh_dict.load_words(&[
            "的", "一", "是", "不", "了", "人", "我", "在", "有", "他",
            "这", "中", "大", "来", "上", "国", "个", "到", "说", "们",
        ]);
        self.dictionaries.insert("zh".to_string(), zh_dict);

        let mut ja_dict = Dictionary::new();
        ja_dict.load_words(&[
            "の", "に", "を", "は", "が", "た", "で", "て", "と", "し",
            "れ", "る", "か", "な", "い", "あ", "こ", "さ", "き", "ま",
        ]);
        self.dictionaries.insert("ja".to_string(), ja_dict);
    }

    pub fn apply_dictionary_correction(
        &self,
        result: &mut TextResult,
        language: &str,
    ) -> Result<(), LanguageProcessingError> {
        let dict = self.dictionaries.get(language)
            .ok_or_else(|| LanguageProcessingError::DictionaryNotFound(language.to_string()))?;

        for word in result.words.iter_mut() {
            let word_text = word.text.trim();
            if word_text.len() > 2 && !dict.contains(word_text) {
                let corrected = dict.correct_word(word_text);
                if corrected != word_text {
                    word.text = corrected;
                }
            }
        }

        result.text = result.words.iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(())
    }

    pub fn correct_text(
        &self,
        text: &str,
        language: &str,
    ) -> Result<LanguageProcessingResult, LanguageProcessingError> {
        let dict = self.dictionaries.get(language)
            .ok_or_else(|| LanguageProcessingError::DictionaryNotFound(language.to_string()))?;

        let words: Vec<&str> = text.split_whitespace().collect();
        let corrected_words: Vec<String> = words.iter()
            .map(|word| {
                if word.len() > 2 && !dict.contains(word) {
                    dict.correct_word(word)
                } else {
                    word.to_string()
                }
            })
            .collect();

        let corrected_text = corrected_words.join(" ");
        
        Ok(LanguageProcessingResult {
            original_text: text.to_string(),
            corrected_text,
            detected_language: Some(language.to_string()),
            correction_confidence: 0.8,
        })
    }

    pub fn add_dictionary(&mut self, language: String, dictionary: Dictionary) {
        self.dictionaries.insert(language, dictionary);
    }

    pub fn get_supported_languages(&self) -> Vec<String> {
        self.dictionaries.keys().cloned().collect()
    }

    pub fn has_language(&self, language: &str) -> bool {
        self.dictionaries.contains_key(language)
    }
}

impl Default for LanguageProcessingService {
    fn default() -> Self {
        Self::new()
    }
}
