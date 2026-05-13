//! Modern recognition engine interface for MiniOCR
//!
//! This module provides a flexible abstraction for different OCR model types,
//! including LSTM, Transformer, Vision Transformer, and other modern architectures.

use crate::core::geometry::TBox;
use crate::core::image::OcrImage;
use crate::core::recognition::TrainableModel;
use crate::utils::{MiniOcrError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recognition result with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionResult {
    pub text: String,
    pub confidence: f32,
    pub bounding_boxes: Vec<TBox>,
    pub character_results: Vec<CharacterRecognitionResult>,
    pub word_results: Vec<WordRecognitionResult>,
    pub line_results: Vec<LineRecognitionResult>,
    pub language: Option<String>,
    pub model_type: ModelType,
    pub processing_time_ms: u64,
}

impl RecognitionResult {
    /// Create a new recognition result
    pub fn new(text: String, confidence: f32) -> Self {
        Self {
            text,
            confidence,
            bounding_boxes: Vec::new(),
            character_results: Vec::new(),
            word_results: Vec::new(),
            line_results: Vec::new(),
            language: None,
            model_type: ModelType::LSTM,
            processing_time_ms: 0,
        }
    }
}

/// Character-level recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterRecognitionResult {
    pub character: char,
    pub confidence: f32,
    pub bounding_box: TBox,
    pub unicode_category: UnicodeCategory,
    pub script: ScriptType,
}

/// Word-level recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordRecognitionResult {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: TBox,
    pub characters: Vec<CharacterRecognitionResult>,
    pub language: Option<String>,
}

/// Line-level recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineRecognitionResult {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: TBox,
    pub words: Vec<WordRecognitionResult>,
    pub reading_order: ReadingOrder,
}

// ModelType is defined in ocr crate
use crate::core::ModelType;

/// Unicode character categories for CJK and other scripts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnicodeCategory {
    // Latin scripts
    Latin,
    LatinExtended,

    // CJK scripts
    CJKUnifiedIdeographs,
    CJKUnifiedIdeographsExtensionA,
    CJKUnifiedIdeographsExtensionB,
    CJKUnifiedIdeographsExtensionC,
    CJKUnifiedIdeographsExtensionD,
    CJKUnifiedIdeographsExtensionE,
    CJKUnifiedIdeographsExtensionF,
    CJKUnifiedIdeographsExtensionG,
    CJKUnifiedIdeographsExtensionH,
    CJKUnifiedIdeographsExtensionI,

    // Japanese specific
    Hiragana,
    Katakana,
    KatakanaPhoneticExtensions,

    // Korean specific
    HangulSyllables,
    HangulJamo,
    HangulCompatibilityJamo,

    // Chinese specific
    CJKRadicals,
    CJKStrokes,
    CJKSymbols,
    CJKCompatibility,

    // Other scripts
    Arabic,
    Devanagari,
    Cyrillic,
    Greek,
    Hebrew,
    Thai,
    Other,
}

/// Script types for language detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScriptType {
    Latin,
    Chinese,
    Japanese,
    Korean,
    Arabic,
    Devanagari,
    Cyrillic,
    Greek,
    Hebrew,
    Thai,
    Other,
}

/// Text reading order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReadingOrder {
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
    Mixed,
}

/// Language variants for CJK
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageVariant {
    English,
    ChineseSimplified,
    ChineseTraditional,
    Japanese,
    Korean,
    Arabic,
    Hindi,
    Russian,
    Other(String),
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub model_path: String,
    pub supported_languages: Vec<LanguageVariant>,
    pub input_shape: (usize, usize, usize), // (height, width, channels)
    pub max_text_length: Option<usize>,
    pub confidence_threshold: f32,
    pub device: DeviceType,
    pub quantization: Option<QuantizationType>,
}

/// Device types for model inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    GPU,
    NPU, // Neural Processing Unit
    Auto,
}

/// Quantization types for model optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    FP32,
    FP16,
    INT8,
    Dynamic,
}

/// Core OCR model trait
pub trait OcrModel: Send + Sync {
    /// Perform inference on the given input
    fn predict(&self, input: &[u8]) -> Result<RecognitionResult>;

    /// Get the model type
    fn model_type(&self) -> ModelType;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageVariant>;

    /// Get input shape requirements
    fn input_shape(&self) -> (usize, usize, usize);

    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Check if the model supports a specific language
    fn supports_language(&self, language: &LanguageVariant) -> bool;

    /// Get trainable interface if supported
    fn as_trainable(&mut self) -> Option<&mut dyn TrainableModel> {
        None
    }

    /// Get trainable interface (immutable) if supported
    fn as_trainable_ref(&self) -> Option<&dyn TrainableModel> {
        None
    }
}

/// Recognition engine trait
#[allow(async_fn_in_trait)]
pub trait RecognitionEngine: Send + Sync {
    /// Recognize text from an image
    async fn recognize(&self, image: &OcrImage) -> Result<RecognitionResult>;

    /// Recognize text from a specific region
    async fn recognize_region(&self, image: &OcrImage, region: &TBox) -> Result<RecognitionResult>;

    /// Recognize text with specific language hints
    async fn recognize_with_language(
        &self,
        image: &OcrImage,
        language_hint: Option<LanguageVariant>,
    ) -> Result<RecognitionResult>;

    /// Get the current model type
    fn model_type(&self) -> ModelType;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageVariant>;

    /// Switch to a different model
    async fn switch_model(&mut self, model_type: ModelType) -> Result<()>;
}

/// Model manager for handling multiple models
pub struct ModelManager {
    models: HashMap<ModelType, Box<dyn OcrModel>>,
    active_model: Option<ModelType>,
    device: DeviceType,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(device: DeviceType) -> Self {
        Self {
            models: HashMap::new(),
            active_model: None,
            device,
        }
    }

    /// Load a model
    pub async fn load_model<M: OcrModel + 'static>(&mut self, model: M) -> Result<()> {
        let model_type = model.model_type();
        self.models.insert(model_type.clone(), Box::new(model));

        if self.active_model.is_none() {
            self.active_model = Some(model_type);
        }

        Ok(())
    }

    /// Switch to a different model
    pub fn switch_model(&mut self, model_type: ModelType) -> Result<()> {
        if self.models.contains_key(&model_type) {
            self.active_model = Some(model_type);
            Ok(())
        } else {
            Err(MiniOcrError::ModelNotFound(format!("Model {:?} not found", model_type)).into())
        }
    }

    /// Get the active model
    pub fn active_model(&self) -> Option<&dyn OcrModel> {
        self.active_model
            .as_ref()
            .and_then(|model_type| self.models.get(model_type).map(|m| m.as_ref()))
    }

    /// Get a specific model
    pub fn get_model(&self, model_type: ModelType) -> Option<&dyn OcrModel> {
        self.models.get(&model_type).map(|m| m.as_ref())
    }

    /// List available models
    pub fn available_models(&self) -> Vec<ModelType> {
        self.models.keys().cloned().collect()
    }
}

/// Utility functions for CJK text processing
pub struct CJKProcessor;

impl CJKProcessor {
    /// Detect if a character is CJK
    pub fn is_cjk_character(c: char) -> bool {
        let code = c as u32;
        matches!(
            code,
            0x4E00..=0x9FFF | // CJK Unified Ideographs
            0x3400..=0x4DBF | // CJK Unified Ideographs Extension A
            0x20000..=0x2A6DF | // CJK Unified Ideographs Extension B
            0x2A700..=0x2B73F | // CJK Unified Ideographs Extension C
            0x2B740..=0x2B81F | // CJK Unified Ideographs Extension D
            0x2B820..=0x2CEAF | // CJK Unified Ideographs Extension E
            0x2CEB0..=0x2EBEF | // CJK Unified Ideographs Extension F
            0x30000..=0x3134F | // CJK Unified Ideographs Extension G
            0x31350..=0x323AF | // CJK Unified Ideographs Extension H
            0x323B0..=0x32B2F | // CJK Unified Ideographs Extension I
            0x3040..=0x309F | // Hiragana
            0x30A0..=0x30FF | // Katakana
            0xAC00..=0xD7AF   // Hangul Syllables
        )
    }

    /// Detect if a character is Chinese
    pub fn is_chinese_character(c: char) -> bool {
        let code = c as u32;
        matches!(
            code,
            0x4E00..=0x9FFF | // CJK Unified Ideographs
            0x3400..=0x4DBF | // CJK Unified Ideographs Extension A
            0x20000..=0x2A6DF | // CJK Unified Ideographs Extension B
            0x2A700..=0x2B73F | // CJK Unified Ideographs Extension C
            0x2B740..=0x2B81F | // CJK Unified Ideographs Extension D
            0x2B820..=0x2CEAF | // CJK Unified Ideographs Extension E
            0x2CEB0..=0x2EBEF | // CJK Unified Ideographs Extension F
            0x30000..=0x3134F | // CJK Unified Ideographs Extension G
            0x31350..=0x323AF | // CJK Unified Ideographs Extension H
            0x323B0..=0x32B2F   // CJK Unified Ideographs Extension I
        )
    }

    /// Detect if a character is Japanese
    pub fn is_japanese_character(c: char) -> bool {
        let code = c as u32;
        matches!(
            code,
            0x3040..=0x309F | // Hiragana
            0x30A0..=0x30FF | // Katakana
            0x4E00..=0x9FFF   // CJK Unified Ideographs (shared with Chinese)
        )
    }

    /// Detect if a character is Korean
    pub fn is_korean_character(c: char) -> bool {
        let code = c as u32;
        matches!(
            code,
            0xAC00..=0xD7AF | // Hangul Syllables
            0x1100..=0x11FF | // Hangul Jamo
            0x3130..=0x318F   // Hangul Compatibility Jamo
        )
    }

    /// Get Unicode category for a character
    pub fn get_unicode_category(c: char) -> UnicodeCategory {
        let code = c as u32;
        match code {
            0x4E00..=0x9FFF => UnicodeCategory::CJKUnifiedIdeographs,
            0x3400..=0x4DBF => UnicodeCategory::CJKUnifiedIdeographsExtensionA,
            0x20000..=0x2A6DF => UnicodeCategory::CJKUnifiedIdeographsExtensionB,
            0x3040..=0x309F => UnicodeCategory::Hiragana,
            0x30A0..=0x30FF => UnicodeCategory::Katakana,
            0xAC00..=0xD7AF => UnicodeCategory::HangulSyllables,
            0x1100..=0x11FF => UnicodeCategory::HangulJamo,
            0x3130..=0x318F => UnicodeCategory::HangulCompatibilityJamo,
            0x0000..=0x007F => UnicodeCategory::Latin,
            0x0080..=0x00FF => UnicodeCategory::LatinExtended,
            _ => UnicodeCategory::Other,
        }
    }

    /// Get script type for a character
    pub fn get_script_type(c: char) -> ScriptType {
        if Self::is_chinese_character(c) {
            ScriptType::Chinese
        } else if Self::is_japanese_character(c) {
            ScriptType::Japanese
        } else if Self::is_korean_character(c) {
            ScriptType::Korean
        } else {
            ScriptType::Latin
        }
    }
}

/// Basic recognition engine implementation (legacy compatibility)
pub struct BasicRecognitionEngine {
    config: ModelConfig,
}

impl BasicRecognitionEngine {
    /// Create a new basic recognition engine
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl RecognitionEngine for BasicRecognitionEngine {
    async fn recognize(&self, image: &OcrImage) -> Result<RecognitionResult> {
        let core_result = super::basic_ocr::BasicOcrEngine::new().recognize_sync(image)?;
        let bbox = TBox::new(0, 0, image.width as i32, image.height as i32);
        Ok(Self::convert_core_result(
            core_result,
            bbox,
            self.config.model_type.clone(),
        ))
    }

    async fn recognize_region(&self, image: &OcrImage, region: &TBox) -> Result<RecognitionResult> {
        let left = region.left().min(region.right()).max(0) as u32;
        let right = region.left().max(region.right()).min(image.width as i32) as u32;
        let top = region.bottom().min(region.top()).max(0) as u32;
        let bottom = region.bottom().max(region.top()).min(image.height as i32) as u32;

        if right <= left || bottom <= top {
            return Ok(RecognitionResult::new(String::new(), 0.0));
        }

        let cropped = image.crop(left, top, right - left, bottom - top)?;
        let core_result = super::basic_ocr::BasicOcrEngine::new().recognize_sync(&cropped)?;
        let bbox = TBox::new(left as i32, top as i32, right as i32, bottom as i32);
        Ok(Self::convert_core_result(
            core_result,
            bbox,
            self.config.model_type.clone(),
        ))
    }

    async fn recognize_with_language(
        &self,
        image: &OcrImage,
        language_hint: Option<LanguageVariant>,
    ) -> Result<RecognitionResult> {
        let mut core_result = super::basic_ocr::BasicOcrEngine::new().recognize_sync(image)?;
        if let Some(lang) = language_hint {
            core_result.language = Some(format!("{:?}", lang));
        }
        let bbox = TBox::new(0, 0, image.width as i32, image.height as i32);
        Ok(Self::convert_core_result(
            core_result,
            bbox,
            self.config.model_type.clone(),
        ))
    }

    fn model_type(&self) -> ModelType {
        self.config.model_type.clone()
    }

    fn supported_languages(&self) -> Vec<LanguageVariant> {
        self.config.supported_languages.clone()
    }

    async fn switch_model(&mut self, model_type: ModelType) -> Result<()> {
        self.config.model_type = model_type;
        Ok(())
    }
}

impl BasicRecognitionEngine {
    fn convert_core_result(
        core_result: crate::core::recognition::RecognitionResult,
        bbox: TBox,
        model_type: ModelType,
    ) -> RecognitionResult {
        let mut character_results = Vec::new();
        for ch in &core_result.characters {
            character_results.push(CharacterRecognitionResult {
                character: ch.character,
                confidence: ch.confidence,
                bounding_box: bbox,
                unicode_category: CJKProcessor::get_unicode_category(ch.character),
                script: CJKProcessor::get_script_type(ch.character),
            });
        }

        let mut word_results = Vec::new();
        for w in &core_result.words {
            let mut word_chars = Vec::new();
            for ch in &w.characters {
                word_chars.push(CharacterRecognitionResult {
                    character: ch.character,
                    confidence: ch.confidence,
                    bounding_box: bbox,
                    unicode_category: CJKProcessor::get_unicode_category(ch.character),
                    script: CJKProcessor::get_script_type(ch.character),
                });
            }

            word_results.push(WordRecognitionResult {
                text: w.word.clone(),
                confidence: w.confidence,
                bounding_box: bbox,
                characters: word_chars,
                language: core_result.language.clone(),
            });
        }

        let mut line_results = Vec::new();
        for l in &core_result.lines {
            line_results.push(LineRecognitionResult {
                text: l.line.clone(),
                confidence: l.confidence,
                bounding_box: bbox,
                words: Vec::new(),
                reading_order: ReadingOrder::LeftToRight,
            });
        }

        let has_text = !core_result.text.trim().is_empty();
        let text = core_result.text;

        RecognitionResult {
            text,
            confidence: core_result.confidence,
            bounding_boxes: if has_text { vec![bbox] } else { Vec::new() },
            character_results,
            word_results,
            line_results,
            language: core_result.language,
            model_type,
            processing_time_ms: core_result.processing_time_ms.unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cjk_character_detection() {
        assert!(CJKProcessor::is_cjk_character('中'));
        assert!(CJKProcessor::is_cjk_character('日'));
        assert!(CJKProcessor::is_cjk_character('한'));
        assert!(CJKProcessor::is_cjk_character('ひ'));
        assert!(CJKProcessor::is_cjk_character('カ'));
        assert!(!CJKProcessor::is_cjk_character('A'));
        assert!(!CJKProcessor::is_cjk_character('1'));
    }

    #[test]
    fn test_language_detection() {
        assert!(CJKProcessor::is_chinese_character('中'));
        assert!(CJKProcessor::is_japanese_character('ひ'));
        assert!(CJKProcessor::is_korean_character('한'));
    }

    #[test]
    fn test_unicode_categories() {
        assert_eq!(
            CJKProcessor::get_unicode_category('中'),
            UnicodeCategory::CJKUnifiedIdeographs
        );
        assert_eq!(
            CJKProcessor::get_unicode_category('ひ'),
            UnicodeCategory::Hiragana
        );
        assert_eq!(
            CJKProcessor::get_unicode_category('カ'),
            UnicodeCategory::Katakana
        );
        assert_eq!(
            CJKProcessor::get_unicode_category('한'),
            UnicodeCategory::HangulSyllables
        );
        assert_eq!(
            CJKProcessor::get_unicode_category('A'),
            UnicodeCategory::Latin
        );
    }
}
