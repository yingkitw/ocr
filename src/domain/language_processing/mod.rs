//! Language Processing Domain Module
//!
//! Provides language-specific processing capabilities including
//! dictionary correction, language detection, and CJK support.

pub use service::LanguageProcessingService;

mod service;

/// Language processing domain error types
#[derive(Debug, thiserror::Error)]
pub enum LanguageProcessingError {
    #[error("Language detection failed: {0}")]
    DetectionFailed(String),
    
    #[error("Dictionary not found for language: {0}")]
    DictionaryNotFound(String),
    
    #[error("Text correction failed: {0}")]
    CorrectionFailed(String),
    
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
}

/// Language processing result
#[derive(Debug, Clone)]
pub struct LanguageProcessingResult {
    /// Original text
    pub original_text: String,
    /// Corrected text
    pub corrected_text: String,
    /// Detected language
    pub detected_language: Option<String>,
    /// Confidence in correction
    pub correction_confidence: f32,
}
