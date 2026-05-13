//! Text recognition data structures for MiniOCR

// Recognition engine traits and structures
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::image::OcrImage;
use crate::core::text::BoundingBox;
use crate::utils::Result;
use ndarray::Array2;

/// Trait for text recognition backends
#[allow(async_fn_in_trait)]
pub trait TextRecognizer: Send + Sync {
    /// Recognize text from an image
    async fn recognize(&self, image: &OcrImage) -> Result<RecognitionResult>;
}

/// Trait for models that can be trained
pub trait TrainableModel: Send + Sync {
    /// Forward pass for training
    /// Returns the logits/predictions as an Array2
    fn forward_train(&self, input: &Array2<f32>) -> Result<Array2<f32>>;

    /// Backward pass to compute gradients
    fn backward_train(&mut self, input: &Array2<f32>, output_grad: &Array2<f32>) -> Result<()>;

    /// Get pairs of (parameter, gradient) for optimization
    fn get_params_and_grads(&mut self) -> Vec<(&mut Array2<f32>, &Array2<f32>)>;
}

/// Model type used for recognition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    LSTM,
    Transformer,
    VisionTransformer,
    CNN,
    Hybrid,
    EndToEnd,
    Custom(String),
}

/// Recognition engine type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecognitionEngine {
    /// LSTM-based recognition
    LSTM,
    /// Traditional pattern matching
    PatternMatching,
    /// Hybrid approach
    Hybrid,
}

/// Recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// Recognition engine to use
    pub engine: RecognitionEngine,
    /// Language code
    pub language: String,
    /// Confidence threshold
    pub confidence_threshold: f32,
    /// Character whitelist
    pub character_whitelist: Option<Vec<char>>,
    /// Character blacklist
    pub character_blacklist: Option<Vec<char>>,
    /// Enable dictionary correction
    pub enable_dictionary_correction: bool,
    /// Enable language model
    pub enable_language_model: bool,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            engine: RecognitionEngine::LSTM,
            language: "en".to_string(),
            confidence_threshold: 0.5,
            character_whitelist: None,
            character_blacklist: None,
            enable_dictionary_correction: true,
            enable_language_model: true,
            parameters: HashMap::new(),
        }
    }
}

/// Recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionResult {
    /// Recognized text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Character-level results
    pub characters: Vec<CharacterRecognition>,
    /// Word-level results
    pub words: Vec<WordRecognition>,
    /// Line-level results
    pub lines: Vec<LineRecognition>,
    /// Recognition metadata
    pub metadata: RecognitionMetadata,
    /// Model type used for recognition
    pub model_type: Option<ModelType>,
    /// Processing time in milliseconds
    pub processing_time_ms: Option<u64>,
    /// Detected language
    pub language: Option<String>,
    /// Character-level results (alias for characters)
    pub character_results: Vec<CharacterRecognition>,
    /// Word-level results (alias for words)
    pub word_results: Vec<WordRecognition>,
    /// Line-level results (alias for lines)
    pub line_results: Vec<LineRecognition>,
}

impl RecognitionResult {
    /// Create a new recognition result
    pub fn new(text: String, confidence: f32) -> Self {
        Self {
            text,
            confidence,
            characters: Vec::new(),
            words: Vec::new(),
            lines: Vec::new(),
            metadata: RecognitionMetadata::default(),
            model_type: None,
            processing_time_ms: None,
            language: None,
            character_results: Vec::new(),
            word_results: Vec::new(),
            line_results: Vec::new(),
        }
    }
}

/// Character recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterRecognition {
    /// Recognized character
    pub character: char,
    /// Confidence score
    pub confidence: f32,
    /// Character code point
    pub code_point: u32,
    /// Character properties
    pub properties: CharacterProperties,
    /// Alternative characters
    pub alternatives: Vec<CharacterAlternative>,
    /// Bounding box of the character in the source image
    pub bounding_box: Option<BoundingBox>,
    /// Extracted INT_FEATURE vectors (baseline-normalized, character-normalized)
    pub features: Option<(Vec<u8>, Vec<u8>)>,
}

impl CharacterRecognition {
    /// Create a new character recognition
    pub fn new(character: char, confidence: f32) -> Self {
        Self {
            character,
            confidence,
            code_point: character as u32,
            properties: CharacterProperties::default(),
            alternatives: Vec::new(),
            bounding_box: None,
            features: None,
        }
    }

    /// Create a new character recognition with bounding box
    pub fn with_bounding_box(character: char, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            character,
            confidence,
            code_point: character as u32,
            properties: CharacterProperties::default(),
            alternatives: Vec::new(),
            bounding_box: Some(bounding_box),
            features: None,
        }
    }
}

/// Character alternative
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterAlternative {
    /// Alternative character
    pub character: char,
    /// Confidence score
    pub confidence: f32,
}

/// Character properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterProperties {
    /// Whether character is a digit
    pub is_digit: bool,
    /// Whether character is a letter
    pub is_letter: bool,
    /// Whether character is whitespace
    pub is_whitespace: bool,
    /// Whether character is punctuation
    pub is_punctuation: bool,
    /// Character width
    pub width: f32,
    /// Character height
    pub height: f32,
    /// Character baseline
    pub baseline: f32,
    /// Character x-height
    pub x_height: f32,
}

impl Default for CharacterProperties {
    fn default() -> Self {
        Self {
            is_digit: false,
            is_letter: false,
            is_whitespace: false,
            is_punctuation: false,
            width: 0.0,
            height: 0.0,
            baseline: 0.0,
            x_height: 0.0,
        }
    }
}

/// Word recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordRecognition {
    /// Recognized word
    pub word: String,
    /// Confidence score
    pub confidence: f32,
    /// Character recognitions
    pub characters: Vec<CharacterRecognition>,
    /// Word properties
    pub properties: WordProperties,
    /// Alternative words
    pub alternatives: Vec<WordAlternative>,
    /// Bounding box of the word in the source image
    pub bounding_box: Option<BoundingBox>,
}

impl WordRecognition {
    /// Create a new word recognition
    pub fn new(word: String, confidence: f32) -> Self {
        Self {
            word,
            confidence,
            characters: Vec::new(),
            properties: WordProperties::default(),
            alternatives: Vec::new(),
            bounding_box: None,
        }
    }

    /// Create a new word recognition with bounding box
    pub fn with_bounding_box(word: String, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            word,
            confidence,
            characters: Vec::new(),
            properties: WordProperties::default(),
            alternatives: Vec::new(),
            bounding_box: Some(bounding_box),
        }
    }
}

/// Word alternative
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordAlternative {
    /// Alternative word
    pub word: String,
    /// Confidence score
    pub confidence: f32,
}

/// Word properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordProperties {
    /// Whether word is in dictionary
    pub is_dictionary_word: bool,
    /// Word length
    pub length: usize,
    /// Average character width
    pub average_character_width: f32,
    /// Average character height
    pub average_character_height: f32,
    /// Word language
    pub language: Option<String>,
}

impl Default for WordProperties {
    fn default() -> Self {
        Self {
            is_dictionary_word: false,
            length: 0,
            average_character_width: 0.0,
            average_character_height: 0.0,
            language: None,
        }
    }
}

/// Line recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineRecognition {
    /// Recognized line
    pub line: String,
    /// Confidence score
    pub confidence: f32,
    /// Word recognitions
    pub words: Vec<WordRecognition>,
    /// Line properties
    pub properties: LineProperties,
    /// Bounding box of the line in the source image
    pub bounding_box: Option<BoundingBox>,
}

impl LineRecognition {
    /// Create a new line recognition
    pub fn new(line: String, confidence: f32) -> Self {
        Self {
            line,
            confidence,
            words: Vec::new(),
            properties: LineProperties::default(),
            bounding_box: None,
        }
    }

    /// Create a new line recognition with bounding box
    pub fn with_bounding_box(line: String, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            line,
            confidence,
            words: Vec::new(),
            properties: LineProperties::default(),
            bounding_box: Some(bounding_box),
        }
    }
}

/// Line properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineProperties {
    /// Line height
    pub height: f32,
    /// Line baseline
    pub baseline: f32,
    /// Line spacing
    pub line_spacing: f32,
    /// Line alignment
    pub alignment: TextAlignment,
    /// Line reading order
    pub reading_order: ReadingOrder,
}

impl Default for LineProperties {
    fn default() -> Self {
        Self {
            height: 0.0,
            baseline: 0.0,
            line_spacing: 0.0,
            alignment: TextAlignment::Left,
            reading_order: ReadingOrder::LeftToRight,
        }
    }
}

/// Recognition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionMetadata {
    /// Recognition engine used
    pub engine: RecognitionEngine,
    /// Language detected
    pub language: Option<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Model version
    pub model_version: Option<String>,
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

impl Default for RecognitionMetadata {
    fn default() -> Self {
        Self {
            engine: RecognitionEngine::LSTM,
            language: None,
            processing_time_ms: 0,
            model_version: None,
            additional: HashMap::new(),
        }
    }
}

/// Text alignment enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlignment {
    /// Left aligned
    Left,
    /// Right aligned
    Right,
    /// Center aligned
    Center,
    /// Justified
    Justified,
}

/// Reading order enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadingOrder {
    /// Left to right
    LeftToRight,
    /// Right to left
    RightToLeft,
    /// Top to bottom
    TopToBottom,
    /// Bottom to top
    BottomToTop,
}

/// Recognition trait for different engines
pub trait RecognitionEngineTrait {
    /// Initialize the recognition engine
    fn initialize(&mut self, config: &RecognitionConfig) -> anyhow::Result<()>;

    /// Recognize text from image data
    fn recognize(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<RecognitionResult>;

    /// Recognize text from image region
    fn recognize_region(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
        region: &ImageRegion,
    ) -> anyhow::Result<RecognitionResult>;

    /// Get engine capabilities
    fn capabilities(&self) -> EngineCapabilities;

    /// Get engine information
    fn info(&self) -> EngineInfo;
}

/// Engine capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Maximum image size
    pub max_image_size: (u32, u32),
    /// Minimum image size
    pub min_image_size: (u32, u32),
    /// Supported image formats
    pub supported_formats: Vec<String>,
    /// Supports character-level recognition
    pub supports_character_level: bool,
    /// Supports word-level recognition
    pub supports_word_level: bool,
    /// Supports line-level recognition
    pub supports_line_level: bool,
    /// Supports confidence scores
    pub supports_confidence: bool,
    /// Supports alternative results
    pub supports_alternatives: bool,
}

/// Engine information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineInfo {
    /// Engine name
    pub name: String,
    /// Engine version
    pub version: String,
    /// Engine description
    pub description: String,
    /// Engine author
    pub author: String,
    /// Engine license
    pub license: String,
}

/// Image region for recognition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRegion {
    /// Left coordinate
    pub left: u32,
    /// Top coordinate
    pub top: u32,
    /// Right coordinate
    pub right: u32,
    /// Bottom coordinate
    pub bottom: u32,
}

impl ImageRegion {
    /// Create a new image region
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Get region width
    pub fn width(&self) -> u32 {
        self.right - self.left
    }

    /// Get region height
    pub fn height(&self) -> u32 {
        self.bottom - self.top
    }

    /// Get region area
    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }
}
