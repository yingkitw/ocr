//! Configuration structures for MiniOCR

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main OCR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrConfig {
    /// Recognition configuration
    pub recognition: RecognitionConfig,
    /// Image processing configuration
    pub image_processing: ImageProcessingConfig,
    /// Layout analysis configuration
    pub layout_analysis: LayoutAnalysisConfig,
    /// Language configuration
    pub language: LanguageConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Debug configuration
    pub debug: DebugConfig,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            recognition: RecognitionConfig::default(),
            image_processing: ImageProcessingConfig::default(),
            layout_analysis: LayoutAnalysisConfig::default(),
            language: LanguageConfig::default(),
            performance: PerformanceConfig::default(),
            debug: DebugConfig::default(),
        }
    }
}

/// Recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// Recognition engine
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

/// Recognition engine enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecognitionEngine {
    /// LSTM-based recognition
    LSTM,
    /// Traditional pattern matching
    PatternMatching,
    /// Hybrid approach
    Hybrid,
}

/// Image processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingConfig {
    /// Enable image preprocessing
    pub enable_preprocessing: bool,
    /// Enable image enhancement
    pub enable_enhancement: bool,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable contrast enhancement
    pub enable_contrast_enhancement: bool,
    /// Enable sharpening
    pub enable_sharpening: bool,
    /// Enable deskewing
    pub enable_deskewing: bool,
    /// Enable binarization
    pub enable_binarization: bool,
    /// Binarization threshold
    pub binarization_threshold: f32,
    /// Binarization method
    pub binarization_method: BinarizationMethod,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for ImageProcessingConfig {
    fn default() -> Self {
        Self {
            enable_preprocessing: true,
            enable_enhancement: true,
            enable_noise_reduction: true,
            enable_contrast_enhancement: true,
            enable_sharpening: false,
            enable_deskewing: true,
            enable_binarization: true,
            binarization_threshold: 0.5,
            binarization_method: BinarizationMethod::Otsu,
            parameters: HashMap::new(),
        }
    }
}

/// Binarization method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinarizationMethod {
    /// Otsu's method
    Otsu,
    /// Adaptive thresholding
    Adaptive,
    /// Fixed threshold
    Fixed,
    /// Sauvola's method
    Sauvola,
}

/// Layout analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutAnalysisConfig {
    /// Enable layout analysis
    pub enable_layout_analysis: bool,
    /// Enable text region detection
    pub enable_text_region_detection: bool,
    /// Enable image region detection
    pub enable_image_region_detection: bool,
    /// Enable table detection
    pub enable_table_detection: bool,
    /// Enable reading order detection
    pub enable_reading_order_detection: bool,
    /// Enable orientation detection
    pub enable_orientation_detection: bool,
    /// Page segmentation mode (Tesseract PSM)
    pub page_seg_mode: PageSegMode,
    /// Minimum text region size
    pub min_text_region_size: (u32, u32),
    /// Maximum text region size
    pub max_text_region_size: (u32, u32),
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for LayoutAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_layout_analysis: true,
            enable_text_region_detection: true,
            enable_image_region_detection: true,
            enable_table_detection: true,
            enable_reading_order_detection: true,
            enable_orientation_detection: true,
            page_seg_mode: PageSegMode::Auto,
            min_text_region_size: (10, 10),
            max_text_region_size: (10000, 10000),
            parameters: HashMap::new(),
        }
    }
}

/// Page segmentation mode (Tesseract --psm equivalent)
///
/// Controls how the page is segmented into text regions before recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageSegMode {
    /// PSM 3: Fully automatic page segmentation (default)
    Auto,
    /// PSM 6: Assume a single uniform block of text
    SingleBlock,
    /// PSM 4: Assume a single column of text of variable sizes
    SingleColumn,
    /// PSM 11: Sparse text — find as much text as possible with no ordering
    SparseText,
    /// PSM 12: Sparse text with orientation and script detection
    SparseTextWithOsd,
    /// PSM 13: Raw line mode — treat input as a single text line
    SingleLine,
    /// PSM 7: Treat the image as a single text line (skip segmentation)
    SingleLineRaw,
    /// PSM 8: Treat the image as a single word
    SingleWord,
    /// PSM 10: Treat the image as a single character
    SingleChar,
}

/// Language configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    /// Primary language
    pub primary_language: String,
    /// Secondary languages
    pub secondary_languages: Vec<String>,
    /// Enable language detection
    pub enable_language_detection: bool,
    /// Language detection confidence threshold
    pub language_detection_threshold: f32,
    /// Character set
    pub character_set: CharacterSet,
    /// Text direction
    pub text_direction: TextDirection,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            primary_language: "en".to_string(),
            secondary_languages: Vec::new(),
            enable_language_detection: true,
            language_detection_threshold: 0.7,
            character_set: CharacterSet::Unicode,
            text_direction: TextDirection::LeftToRight,
            parameters: HashMap::new(),
        }
    }
}

/// Character set enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterSet {
    /// ASCII character set
    Ascii,
    /// Latin-1 character set
    Latin1,
    /// Unicode character set
    Unicode,
    /// Custom character set
    Custom,
}

/// Text direction enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDirection {
    /// Left to right
    LeftToRight,
    /// Right to left
    RightToLeft,
    /// Top to bottom
    TopToBottom,
    /// Bottom to top
    BottomToTop,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum number of threads
    pub max_threads: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Cache size in MB
    pub cache_size_mb: usize,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            enable_simd: true,
            enable_gpu: false,
            memory_limit_mb: 1024,
            cache_size_mb: 256,
            enable_parallel_processing: true,
            parameters: HashMap::new(),
        }
    }
}

/// Debug configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable debug logging
    pub enable_debug_logging: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable intermediate image saving
    pub enable_intermediate_saving: bool,
    /// Debug output directory
    pub debug_output_dir: Option<String>,
    /// Log level
    pub log_level: LogLevel,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_debug_logging: false,
            enable_profiling: false,
            enable_intermediate_saving: false,
            debug_output_dir: None,
            log_level: LogLevel::Info,
            parameters: HashMap::new(),
        }
    }
}

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    /// Error level
    Error,
    /// Warning level
    Warning,
    /// Info level
    Info,
    /// Debug level
    Debug,
    /// Trace level
    Trace,
}

impl OcrConfig {
    /// Create a new OCR configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate recognition config
        if self.recognition.confidence_threshold < 0.0
            || self.recognition.confidence_threshold > 1.0
        {
            return Err(anyhow::anyhow!(
                "Confidence threshold must be between 0.0 and 1.0"
            ));
        }

        // Validate image processing config
        if self.image_processing.binarization_threshold < 0.0
            || self.image_processing.binarization_threshold > 1.0
        {
            return Err(anyhow::anyhow!(
                "Binarization threshold must be between 0.0 and 1.0"
            ));
        }

        // Validate layout analysis config
        if self.layout_analysis.min_text_region_size.0
            >= self.layout_analysis.max_text_region_size.0
            || self.layout_analysis.min_text_region_size.1
                >= self.layout_analysis.max_text_region_size.1
        {
            return Err(anyhow::anyhow!(
                "Minimum text region size must be smaller than maximum"
            ));
        }

        // Validate language config
        if self.language.language_detection_threshold < 0.0
            || self.language.language_detection_threshold > 1.0
        {
            return Err(anyhow::anyhow!(
                "Language detection threshold must be between 0.0 and 1.0"
            ));
        }

        // Validate performance config
        if self.performance.max_threads == 0 {
            return Err(anyhow::anyhow!("Maximum threads must be greater than 0"));
        }

        if self.performance.memory_limit_mb == 0 {
            return Err(anyhow::anyhow!("Memory limit must be greater than 0"));
        }

        Ok(())
    }

    /// Get a configuration parameter
    pub fn get_parameter(&self, key: &str) -> Option<&String> {
        // Check recognition parameters
        if let Some(value) = self.recognition.parameters.get(key) {
            return Some(value);
        }

        // Check image processing parameters
        if let Some(value) = self.image_processing.parameters.get(key) {
            return Some(value);
        }

        // Check layout analysis parameters
        if let Some(value) = self.layout_analysis.parameters.get(key) {
            return Some(value);
        }

        // Check language parameters
        if let Some(value) = self.language.parameters.get(key) {
            return Some(value);
        }

        // Check performance parameters
        if let Some(value) = self.performance.parameters.get(key) {
            return Some(value);
        }

        // Check debug parameters
        if let Some(value) = self.debug.parameters.get(key) {
            return Some(value);
        }

        None
    }

    /// Set a configuration parameter
    pub fn set_parameter(&mut self, key: &str, value: String) {
        // Set in recognition parameters by default
        self.recognition.parameters.insert(key.to_string(), value);
    }
}
