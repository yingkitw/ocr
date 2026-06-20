//! Core OCR engine implementation

use crate::utils::{Profiler, Timer};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::core::{
    config::OcrConfig,
    image::OcrImage,
    layout::{LayoutResult, PageSize, TextRegion},
    recognition::RecognitionResult,
    text::TextResult,
};

/// Main OCR engine
pub struct OcrEngine {
    /// Engine configuration
    config: OcrConfig,
    /// Engine state
    state: RwLock<EngineState>,
    /// Performance profiler
    profiler: Profiler,
    /// Engine metadata
    metadata: EngineMetadata,
    /// Compute backend for neural network operations
    compute_backend: Option<Box<dyn crate::compute::ComputeBackend>>,
}

/// Engine state
#[derive(Debug, Clone)]
pub struct EngineState {
    /// Whether engine is initialized
    pub initialized: bool,
    /// Current language
    pub current_language: String,
    /// Engine statistics
    pub statistics: EngineStatistics,
    /// Cache for processed images
    pub image_cache: HashMap<String, OcrImage>,
    /// Cache for recognition results
    pub result_cache: HashMap<String, TextResult>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            initialized: false,
            current_language: "en".to_string(),
            statistics: EngineStatistics::default(),
            image_cache: HashMap::new(),
            result_cache: HashMap::new(),
        }
    }
}

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatistics {
    /// Total images processed
    pub total_images_processed: u64,
    /// Total text recognized
    pub total_text_recognized: u64,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: u64,
    /// Average processing time per image
    pub average_processing_time_ms: f64,
    /// Average confidence score
    pub average_confidence: f64,
    /// Error count
    pub error_count: u64,
    /// Last processed timestamp
    pub last_processed: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for EngineStatistics {
    fn default() -> Self {
        Self {
            total_images_processed: 0,
            total_text_recognized: 0,
            total_processing_time_ms: 0,
            average_processing_time_ms: 0.0,
            average_confidence: 0.0,
            error_count: 0,
            last_processed: None,
        }
    }
}

/// Engine metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetadata {
    /// Engine name
    pub name: String,
    /// Engine version
    pub version: String,
    /// Engine description
    pub description: String,
    /// Engine capabilities
    pub capabilities: EngineCapabilities,
    /// Supported languages
    pub supported_languages: Vec<String>,
}

/// Engine capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    /// Supports image preprocessing
    pub supports_preprocessing: bool,
    /// Supports layout analysis
    pub supports_layout_analysis: bool,
    /// Supports text recognition
    pub supports_text_recognition: bool,
    /// Supports language detection
    pub supports_language_detection: bool,
    /// Supports confidence scoring
    pub supports_confidence_scoring: bool,
    /// Supports parallel processing
    pub supports_parallel_processing: bool,
    /// Maximum image size
    pub max_image_size: (u32, u32),
    /// Minimum image size
    pub min_image_size: (u32, u32),
}

impl OcrEngine {
    /// Create a new OCR engine with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(OcrConfig::default())
    }

    /// Create a new OCR engine with custom configuration
    pub fn with_config(config: OcrConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Initialize compute backend based on device setting
        let compute_backend = Self::create_compute_backend(&config);

        let engine = Self {
            config,
            state: RwLock::new(EngineState::default()),
            profiler: Profiler::new(),
            metadata: EngineMetadata {
                name: "OCR Engine".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: "A modern OCR engine written in Rust".to_string(),
                capabilities: EngineCapabilities {
                    supports_preprocessing: true,
                    supports_layout_analysis: true,
                    supports_text_recognition: true,
                    supports_language_detection: true,
                    supports_confidence_scoring: true,
                    supports_parallel_processing: true,
                    max_image_size: (10000, 10000),
                    min_image_size: (10, 10),
                },
                supported_languages: vec![
                    "en".to_string(),
                    "es".to_string(),
                    "fr".to_string(),
                    "de".to_string(),
                    "it".to_string(),
                    "pt".to_string(),
                    "ru".to_string(),
                    "zh".to_string(),
                    "ja".to_string(),
                    "ko".to_string(),
                    "nl".to_string(),
                    "pl".to_string(),
                    "sv".to_string(),
                    "da".to_string(),
                    "fi".to_string(),
                    "no".to_string(),
                    "tr".to_string(),
                    "el".to_string(),
                    "hi".to_string(),
                    "th".to_string(),
                    "vi".to_string(),
                    "ar".to_string(),
                    "he".to_string(),
                    "id".to_string(),
                    "ms".to_string(),
                    "uk".to_string(),
                    "cs".to_string(),
                    "hu".to_string(),
                    "ro".to_string(),
                    "bg".to_string(),
                ],
            },
            compute_backend,
        };

        Ok(engine)
    }

    /// Initialize the OCR engine
    pub async fn initialize(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.initialized {
            return Ok(());
        }

        // Initialize recognition engine
        self.initialize_recognition_engine().await?;

        // Initialize image processing pipeline
        self.initialize_image_processing().await?;

        // Initialize layout analysis
        self.initialize_layout_analysis().await?;

        // Initialize language detection
        self.initialize_language_detection().await?;

        state.initialized = true;
        state.current_language = self.config.recognition.language.clone();

        Ok(())
    }

    /// Process an image and return text result
    pub async fn process_image(&self, image: OcrImage) -> Result<TextResult> {
        let timer = Timer::new();

        // Check if engine is initialized
        {
            let state = self.state.read().await;
            if !state.initialized {
                return Err(anyhow::anyhow!("Engine not initialized"));
            }
        }

        // Validate image
        self.validate_image(&image)?;

        // Process image
        let result = self.process_image_internal(image).await?;

        // Update statistics
        self.update_statistics(timer.elapsed_ms()).await?;

        Ok(result)
    }

    /// Process image data and return text result
    pub async fn process_image_data(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<TextResult> {
        // Convert image data to OcrImage
        let image = OcrImage::from_raw_pixels(
            width,
            height,
            image_data.to_vec(),
            crate::core::image::ImageFormat::Grayscale,
            self.config
                .recognition
                .parameters
                .get("dpi")
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
        )?;

        self.process_image(image).await
    }

    /// Get engine statistics
    pub async fn get_statistics(&self) -> Result<EngineStatistics> {
        let state = self.state.read().await;
        Ok(state.statistics.clone())
    }

    /// Get engine metadata
    pub fn get_metadata(&self) -> &EngineMetadata {
        &self.metadata
    }

    /// Get engine configuration
    pub fn get_config(&self) -> &OcrConfig {
        &self.config
    }

    /// Update engine configuration
    pub async fn update_config(&mut self, config: OcrConfig) -> Result<()> {
        config.validate()?;
        self.config = config;

        // Reinitialize if already initialized
        let state = self.state.read().await;
        if state.initialized {
            drop(state);
            self.initialize().await?;
        }

        Ok(())
    }

    /// Clear engine cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.image_cache.clear();
        state.result_cache.clear();
        Ok(())
    }

    /// Reset engine statistics
    pub async fn reset_statistics(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.statistics = EngineStatistics::default();
        Ok(())
    }

    /// Get performance profiler
    pub fn get_profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// Internal method to process image
    async fn process_image_internal(&self, image: OcrImage) -> Result<TextResult> {
        let mut profiler = self.profiler.clone();
        profiler.start_operation("total");

        // Step 1: Image preprocessing
        profiler.start_operation("preprocessing");
        let preprocessed_image = if self.config.image_processing.enable_preprocessing {
            self.preprocess_image(&image).await?
        } else {
            image
        };
        profiler.stop_operation();

        // Step 2: Layout analysis
        profiler.start_operation("layout_analysis");
        let layout_result = if self.config.layout_analysis.enable_layout_analysis {
            self.analyze_layout(&preprocessed_image).await?
        } else {
            LayoutResult::new(crate::core::layout::PageSize::new(
                preprocessed_image.width,
                preprocessed_image.height,
                preprocessed_image.dpi,
            ))
        };
        profiler.stop_operation();

        // Step 3: Text recognition
        profiler.start_operation("recognition");
        let recognition_result = self
            .recognize_text(&preprocessed_image, &layout_result)
            .await?;
        profiler.stop_operation();

        // Step 4: Post-processing
        profiler.start_operation("post_processing");
        let text_result = self
            .post_process_result(recognition_result, &preprocessed_image)
            .await?;
        profiler.stop_operation();

        profiler.stop_operation(); // total

        // Log profiling info
        if self.config.debug.enable_profiling {
            for name in profiler.operation_names() {
                if let Some(stats) = profiler.get_stats(&name) {
                    tracing::info!(
                        "Stage '{}': avg={:.2}ms min={:.2}ms max={:.2}ms count={}",
                        name,
                        stats.average_ms(),
                        stats.min_ms(),
                        stats.max_ms(),
                        stats.count
                    );
                }
            }
        }

        Ok(text_result)
    }

    /// Validate image
    fn validate_image(&self, image: &OcrImage) -> Result<()> {
        // Check image size
        if image.width < self.metadata.capabilities.min_image_size.0
            || image.height < self.metadata.capabilities.min_image_size.1
        {
            return Err(anyhow::anyhow!(
                "Image too small: {}x{}, minimum: {}x{}",
                image.width,
                image.height,
                self.metadata.capabilities.min_image_size.0,
                self.metadata.capabilities.min_image_size.1
            ));
        }

        if image.width > self.metadata.capabilities.max_image_size.0
            || image.height > self.metadata.capabilities.max_image_size.1
        {
            return Err(anyhow::anyhow!(
                "Image too large: {}x{}, maximum: {}x{}",
                image.width,
                image.height,
                self.metadata.capabilities.max_image_size.0,
                self.metadata.capabilities.max_image_size.1
            ));
        }

        Ok(())
    }

    /// Initialize recognition engine
    async fn initialize_recognition_engine(&self) -> Result<()> {
        // Log compute backend information
        if let Some(ref backend) = self.compute_backend {
            tracing::info!(
                "Compute backend: {} ({})",
                backend.backend_type().name(),
                backend.device_name()
            );
        } else {
            tracing::info!("Compute backend: CPU (fallback)");
        }
        Ok(())
    }

    /// Create compute backend based on configuration
    fn create_compute_backend(
        config: &OcrConfig,
    ) -> Option<Box<dyn crate::compute::ComputeBackend>> {
        use crate::compute::{create_backend, BackendType};

        let device_str = config.performance.device.to_lowercase();
        let backend_type = match device_str.as_str() {
            "cpu" => BackendType::Cpu,
            "gpu" => BackendType::detect(), // Auto-detect best GPU backend
            "auto" => BackendType::detect(), // Auto-detect best available backend
            #[cfg(feature = "cuda")]
            "cuda" => BackendType::Cuda,
            #[cfg(feature = "opencl")]
            "opencl" => BackendType::OpenCl,
            _ => {
                tracing::warn!(
                    "Unknown device '{}', falling back to auto-detection",
                    device_str
                );
                BackendType::detect()
            }
        };

        match create_backend(backend_type) {
            Ok(backend) => Some(backend),
            Err(e) => {
                tracing::warn!(
                    "Failed to create compute backend: {}, using CPU fallback",
                    e
                );
                None
            }
        }
    }

    /// Initialize image processing
    async fn initialize_image_processing(&self) -> Result<()> {
        // TODO: Initialize image processing pipeline
        Ok(())
    }

    /// Initialize layout analysis
    async fn initialize_layout_analysis(&self) -> Result<()> {
        // TODO: Initialize layout analysis
        Ok(())
    }

    /// Initialize language detection
    async fn initialize_language_detection(&self) -> Result<()> {
        // TODO: Initialize language detection
        Ok(())
    }

    /// Preprocess image
    async fn preprocess_image(&self, image: &OcrImage) -> Result<OcrImage> {
        let mut processed = image.clone();

        // Convert to grayscale first for all further operations
        if processed.format != crate::core::image::ImageFormat::Grayscale {
            processed = processed.to_grayscale();
        }

        // Detect if image has dark background (inverted colors)
        let avg_brightness = self.detect_background_brightness(&processed)?;
        if avg_brightness < 128.0 {
            processed = processed.invert()?;
        }

        // Orientation detection and correction (auto-rotate 0/90/180/270)
        if self.config.layout_analysis.enable_orientation_detection {
            tracing::debug!("Running orientation detection");
            processed = crate::image::ImageEnhancer::correct_orientation(&processed)?;
        }

        Ok(processed)
    }

    /// Detect background brightness by sampling corner pixels
    fn detect_background_brightness(&self, image: &OcrImage) -> Result<f32> {
        use image::GenericImageView;

        let sample_size = 10u32;
        let mut total_brightness = 0u32;
        let mut sample_count = 0u32;

        // Convert to grayscale for brightness detection
        let gray_image = image.data.to_luma8();

        // Sample corners
        let corners = vec![
            (0, 0),
            (image.width.saturating_sub(sample_size), 0),
            (0, image.height.saturating_sub(sample_size)),
            (
                image.width.saturating_sub(sample_size),
                image.height.saturating_sub(sample_size),
            ),
        ];

        for (x, y) in corners {
            for dy in 0..sample_size.min(image.height.saturating_sub(y)) {
                for dx in 0..sample_size.min(image.width.saturating_sub(x)) {
                    let px = gray_image.get_pixel(x + dx, y + dy);
                    let gray = px[0] as u32;
                    total_brightness += gray;
                    sample_count += 1;
                }
            }
        }

        if sample_count > 0 {
            Ok((total_brightness as f32) / (sample_count as f32))
        } else {
            Ok(128.0) // Default to medium brightness
        }
    }

    /// Analyze layout
    async fn analyze_layout(&self, image: &OcrImage) -> Result<LayoutResult> {
        use crate::core::config::PageSegMode;

        let page_size = PageSize::new(image.width, image.height, image.dpi);

        match self.config.layout_analysis.page_seg_mode {
            PageSegMode::SingleLine | PageSegMode::SingleLineRaw => {
                let mut result = LayoutResult::new(page_size);
                result.text_regions.push(TextRegion::new(
                    "single_line".to_string(),
                    crate::core::text::BoundingBox::new(0, 0, image.width, image.height),
                    String::new(),
                ));
                return Ok(result);
            }
            PageSegMode::SingleWord => {
                let mut result = LayoutResult::new(page_size);
                result.text_regions.push(TextRegion::new(
                    "single_word".to_string(),
                    crate::core::text::BoundingBox::new(0, 0, image.width, image.height),
                    String::new(),
                ));
                return Ok(result);
            }
            PageSegMode::SingleChar => {
                let mut result = LayoutResult::new(page_size);
                result.text_regions.push(TextRegion::new(
                    "single_char".to_string(),
                    crate::core::text::BoundingBox::new(0, 0, image.width, image.height),
                    String::new(),
                ));
                return Ok(result);
            }
            PageSegMode::SingleBlock | PageSegMode::SingleColumn => {
                let mut result = LayoutResult::new(page_size);
                result.text_regions.push(TextRegion::new(
                    "full_page".to_string(),
                    crate::core::text::BoundingBox::new(0, 0, image.width, image.height),
                    String::new(),
                ));
                return Ok(result);
            }
            PageSegMode::SparseText | PageSegMode::SparseTextWithOsd | PageSegMode::Auto => {
                Ok(crate::layout::LayoutAnalyzer::analyze_layout(image)?)
            }
        }
    }

    /// Recognize text
    async fn recognize_text(
        &self,
        image: &OcrImage,
        layout: &LayoutResult,
    ) -> Result<RecognitionResult> {
        match self.config.recognition.engine {
            crate::core::config::RecognitionEngine::LSTM => {
                self.recognize_with_crnn(image, layout).await
            }
            _ => self.recognize_with_basic_ocr(image, layout).await,
        }
    }

    async fn recognize_with_basic_ocr(
        &self,
        image: &OcrImage,
        layout: &LayoutResult,
    ) -> Result<RecognitionResult> {
        use crate::recognition::basic_ocr::BasicOcrEngine;

        let ocr_engine = BasicOcrEngine::new();

        if layout.text_regions.is_empty() {
            return Ok(ocr_engine.recognize_sync(image)?);
        }

        let mut full_text = String::new();
        let mut all_characters = Vec::new();
        let mut all_words = Vec::new();
        let mut all_lines = Vec::new();
        let mut conf_sum = 0.0f32;
        let mut conf_count = 0u32;

        for region in &layout.text_regions {
            let bbox = &region.bounding_box;
            let cropped = image.crop(bbox.left, bbox.top, bbox.width(), bbox.height())?;
            let region_result = ocr_engine.recognize_sync(&cropped)?;

            if !region_result.text.trim().is_empty() {
                if !full_text.is_empty() {
                    full_text.push('\n');
                }
                full_text.push_str(region_result.text.trim());
            }

            for ch in region_result.characters {
                conf_sum += ch.confidence;
                conf_count += 1;
                all_characters.push(ch);
            }
            all_words.extend(region_result.words);
            all_lines.extend(region_result.lines);
        }

        let confidence = if conf_count == 0 {
            0.0
        } else {
            conf_sum / (conf_count as f32)
        };

        Ok(RecognitionResult {
            text: full_text,
            confidence,
            characters: all_characters,
            words: all_words,
            lines: all_lines,
            metadata: Default::default(),
            model_type: Some(crate::core::recognition::ModelType::Custom(
                "BasicOCR".to_string(),
            )),
            processing_time_ms: None,
            language: Some(self.config.recognition.language.clone()),
            character_results: Vec::new(),
            word_results: Vec::new(),
            line_results: Vec::new(),
        })
    }

    async fn recognize_with_crnn(
        &self,
        image: &OcrImage,
        layout: &LayoutResult,
    ) -> Result<RecognitionResult> {
        use crate::recognition::crnn::{CrnnConfig, CrnnModel};

        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);

        let mut full_text = String::new();
        let mut all_characters = Vec::new();
        let mut all_words = Vec::new();
        let mut all_lines = Vec::new();
        let mut conf_sum = 0.0f32;
        let mut conf_count = 0u32;

        let regions = if layout.text_regions.is_empty() {
            vec![image.clone()]
        } else {
            layout
                .text_regions
                .iter()
                .map(|r| {
                    image
                        .crop(
                            r.bounding_box.left,
                            r.bounding_box.top,
                            r.bounding_box.width(),
                            r.bounding_box.height(),
                        )
                        .unwrap_or_else(|_| image.clone())
                })
                .collect()
        };

        for region_img in &regions {
            let text = model.recognize(region_img).unwrap_or_default();
            if !text.trim().is_empty() {
                if !full_text.is_empty() {
                    full_text.push('\n');
                }
                full_text.push_str(text.trim());
            }
            // Approximate confidence for CRNN output
            conf_sum += 0.7;
            conf_count += 1;
        }

        let confidence = if conf_count == 0 {
            0.0
        } else {
            conf_sum / conf_count as f32
        };

        Ok(RecognitionResult {
            text: full_text,
            confidence,
            characters: all_characters,
            words: all_words,
            lines: all_lines,
            metadata: Default::default(),
            model_type: Some(crate::core::recognition::ModelType::LSTM),
            processing_time_ms: None,
            language: Some(self.config.recognition.language.clone()),
            character_results: Vec::new(),
            word_results: Vec::new(),
            line_results: Vec::new(),
        })
    }

    /// Post-process result
    async fn post_process_result(
        &self,
        recognition: RecognitionResult,
        image: &OcrImage,
    ) -> Result<TextResult> {
        use crate::lang::dictionary::DictionaryHandler;

        let whitelist = &self.config.recognition.character_whitelist;
        let blacklist = &self.config.recognition.character_blacklist;
        let confidence_threshold = self.config.recognition.confidence_threshold;
        let enable_dict_correction = self.config.recognition.enable_dictionary_correction;
        let enable_font_analysis = self.config.recognition.enable_font_attribute_detection;

        let dict = if enable_dict_correction {
            Some(DictionaryHandler::new_for_language(
                &self.config.recognition.language,
            ))
        } else {
            None
        };

        let dynamic_img = image.data.clone();

        let mut text_result = TextResult::new(
            recognition.text.clone(),
            recognition.confidence,
            crate::core::text::BoundingBox::new(0, 0, 0, 0),
        );

        for line_rec in &recognition.lines {
            let line_bbox = line_rec
                .bounding_box
                .unwrap_or_else(|| crate::core::text::BoundingBox::new(0, 0, 0, 0));

            let mut filtered_words = Vec::new();
            let mut filtered_line_text = String::new();

            for word_rec in &line_rec.words {
                let word_bbox = word_rec
                    .bounding_box
                    .unwrap_or_else(|| crate::core::text::BoundingBox::new(0, 0, 0, 0));

                let filtered_chars: Vec<_> = word_rec
                    .characters
                    .iter()
                    .filter(|c| {
                        if c.confidence < confidence_threshold {
                            return false;
                        }
                        if let Some(allowed) = whitelist {
                            if !allowed.contains(&c.character) {
                                return false;
                            }
                        }
                        if let Some(blocked) = blacklist {
                            if blocked.contains(&c.character) {
                                return false;
                            }
                        }
                        true
                    })
                    .cloned()
                    .collect();

                if filtered_chars.is_empty() {
                    continue;
                }

                let mut corrected_text: String =
                    filtered_chars.iter().map(|c| c.character).collect();

                // Apply dictionary correction if enabled
                if let Some(ref dict) = dict {
                    corrected_text = dict.correct(&corrected_text);
                }

                let avg_conf = filtered_chars.iter().map(|c| c.confidence).sum::<f32>()
                    / filtered_chars.len() as f32;

                let mut word_result =
                    crate::core::text::WordResult::new(corrected_text.clone(), avg_conf, word_bbox);

                for char_rec in &filtered_chars {
                    let char_bbox = char_rec
                        .bounding_box
                        .unwrap_or_else(|| crate::core::text::BoundingBox::new(0, 0, 0, 0));
                    word_result
                        .characters
                        .push(crate::core::text::CharacterResult::new(
                            char_rec.character,
                            char_rec.confidence,
                            char_bbox,
                        ));
                }

                if word_result.confidence < confidence_threshold {
                    continue;
                }

                // Font attribute detection
                if enable_font_analysis {
                    if let Ok((is_bold, is_italic, is_monospace)) =
                        crate::image::font_analysis::analyze_font_attributes(
                            &dynamic_img,
                            &word_result,
                        )
                    {
                        word_result.properties.is_bold = is_bold;
                        word_result.properties.is_italic = is_italic;
                        word_result.properties.is_monospace = is_monospace;
                    }
                }

                if !filtered_line_text.is_empty() {
                    filtered_line_text.push(' ');
                }
                filtered_line_text.push_str(&word_result.text);
                filtered_words.push(word_result);
            }

            if filtered_words.is_empty() {
                continue;
            }

            let line_conf = filtered_words.iter().map(|w| w.confidence).sum::<f32>()
                / filtered_words.len() as f32;

            let mut line_result =
                crate::core::text::LineResult::new(filtered_line_text, line_conf, line_bbox);
            line_result.words = filtered_words;

            // Vertical text detection: tall narrow lines are likely vertical CJK
            if self.config.layout_analysis.enable_vertical_text_detection {
                let line_width = line_bbox.right.saturating_sub(line_bbox.left);
                let line_height = line_bbox.bottom.saturating_sub(line_bbox.top);
                if line_height > line_width.saturating_mul(2) {
                    line_result.properties.is_vertical = true;
                    line_result.properties.reading_order =
                        crate::core::text::ReadingOrder::TopToBottom;
                }
            }

            text_result.lines.push(line_result);
        }

        for char_result in &recognition.characters {
            let char_bbox = char_result
                .bounding_box
                .unwrap_or_else(|| crate::core::text::BoundingBox::new(0, 0, 0, 0));

            let passes_filter = char_result.confidence >= confidence_threshold
                && whitelist
                    .as_ref()
                    .map(|wl| wl.contains(&char_result.character))
                    .unwrap_or(true)
                && blacklist
                    .as_ref()
                    .map(|bl| !bl.contains(&char_result.character))
                    .unwrap_or(true);

            if passes_filter {
                text_result
                    .characters
                    .push(crate::core::text::CharacterResult::new(
                        char_result.character,
                        char_result.confidence,
                        char_bbox,
                    ));
            }
        }

        for word_result in &recognition.words {
            let word_bbox = word_result
                .bounding_box
                .unwrap_or_else(|| crate::core::text::BoundingBox::new(0, 0, 0, 0));

            let filtered_chars: Vec<_> = word_result
                .characters
                .iter()
                .filter(|c| {
                    c.confidence >= confidence_threshold
                        && whitelist
                            .as_ref()
                            .map(|wl| wl.contains(&c.character))
                            .unwrap_or(true)
                        && blacklist
                            .as_ref()
                            .map(|bl| !bl.contains(&c.character))
                            .unwrap_or(true)
                })
                .cloned()
                .collect();

            if filtered_chars.is_empty() {
                continue;
            }

            let mut corrected_text: String = filtered_chars.iter().map(|c| c.character).collect();

            if let Some(ref d) = dict {
                corrected_text = d.correct(&corrected_text);
            }

            let mut wr = crate::core::text::WordResult::new(
                corrected_text,
                word_result.confidence,
                word_bbox,
            );

            for char_rec in &filtered_chars {
                let char_bbox = char_rec
                    .bounding_box
                    .unwrap_or_else(|| crate::core::text::BoundingBox::new(0, 0, 0, 0));
                wr.characters.push(crate::core::text::CharacterResult::new(
                    char_rec.character,
                    char_rec.confidence,
                    char_bbox,
                ));
            }

            text_result.words.push(wr);
        }

        if let Some(lang) = recognition.language {
            text_result.language = Some(lang);
        }

        // Compute overall bounding box from all processed lines
        if !text_result.lines.is_empty() {
            let mut min_x = u32::MAX;
            let mut min_y = u32::MAX;
            let mut max_x = 0u32;
            let mut max_y = 0u32;
            for line in &text_result.lines {
                let bb = &line.bounding_box;
                if bb.right > bb.left && bb.bottom > bb.top {
                    min_x = min_x.min(bb.left);
                    min_y = min_y.min(bb.top);
                    max_x = max_x.max(bb.right);
                    max_y = max_y.max(bb.bottom);
                }
            }
            if max_x > min_x && max_y > min_y {
                text_result.bounding_box =
                    crate::core::text::BoundingBox::new(min_x, min_y, max_x, max_y);
            }
        }

        Ok(text_result)
    }

    /// Update statistics
    async fn update_statistics(&self, processing_time_ms: u64) -> Result<()> {
        let mut state = self.state.write().await;

        state.statistics.total_images_processed += 1;
        state.statistics.total_processing_time_ms += processing_time_ms;
        state.statistics.average_processing_time_ms = state.statistics.total_processing_time_ms
            as f64
            / state.statistics.total_images_processed as f64;
        state.statistics.last_processed = Some(chrono::Utc::now());

        Ok(())
    }
}

impl Default for OcrEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create OCR engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    fn glyph_rows(ch: char) -> [&'static str; 7] {
        match ch {
            // Uppercase A-Z
            'A' => [
                "01110", "10001", "10001", "11111", "10001", "10001", "10001",
            ],
            'B' => [
                "11110", "10001", "10001", "11110", "10001", "10001", "11110",
            ],
            'C' => [
                "01110", "10001", "10000", "10000", "10000", "10001", "01110",
            ],
            'D' => [
                "11110", "10001", "10001", "10001", "10001", "10001", "11110",
            ],
            'E' => [
                "11111", "10000", "10000", "11110", "10000", "10000", "11111",
            ],
            'F' => [
                "11111", "10000", "10000", "11110", "10000", "10000", "10000",
            ],
            'G' => [
                "01110", "10001", "10000", "10111", "10001", "10001", "01110",
            ],
            'H' => [
                "10001", "10001", "10001", "11111", "10001", "10001", "10001",
            ],
            'I' => [
                "01110", "00100", "00100", "00100", "00100", "00100", "01110",
            ],
            'J' => [
                "00001", "00001", "00001", "00001", "10001", "10001", "01110",
            ],
            'K' => [
                "10001", "10010", "10100", "11000", "10100", "10010", "10001",
            ],
            'L' => [
                "10000", "10000", "10000", "10000", "10000", "10000", "11111",
            ],
            'M' => [
                "10001", "11011", "10101", "10001", "10001", "10001", "10001",
            ],
            'N' => [
                "10001", "11001", "10101", "10011", "10001", "10001", "10001",
            ],
            'O' => [
                "01110", "10001", "10001", "10001", "10001", "10001", "01110",
            ],
            'P' => [
                "11110", "10001", "10001", "11110", "10000", "10000", "10000",
            ],
            'Q' => [
                "01110", "10001", "10001", "10001", "10101", "10010", "01101",
            ],
            'R' => [
                "11110", "10001", "10001", "11110", "10100", "10010", "10001",
            ],
            'S' => [
                "01111", "10000", "10000", "01110", "00001", "00001", "11110",
            ],
            'T' => [
                "11111", "00100", "00100", "00100", "00100", "00100", "00100",
            ],
            'U' => [
                "10001", "10001", "10001", "10001", "10001", "10001", "01110",
            ],
            'V' => [
                "10001", "10001", "10001", "10001", "10001", "01010", "00100",
            ],
            'W' => [
                "10001", "10001", "10001", "10101", "10101", "10101", "01010",
            ],
            'X' => [
                "10001", "10001", "01010", "00100", "01010", "10001", "10001",
            ],
            'Y' => [
                "10001", "10001", "10001", "01010", "00100", "00100", "00100",
            ],
            'Z' => [
                "11111", "00001", "00010", "00100", "01000", "10000", "11111",
            ],
            // Digits 0-9
            '0' => [
                "01110", "10011", "10101", "10101", "10101", "11001", "01110",
            ],
            '1' => [
                "00100", "01100", "00100", "00100", "00100", "00100", "01110",
            ],
            '2' => [
                "01110", "10001", "00001", "00010", "00100", "01000", "11111",
            ],
            '3' => [
                "01110", "10001", "00001", "00110", "00001", "10001", "01110",
            ],
            '4' => [
                "00010", "00110", "01010", "10010", "11111", "00010", "00010",
            ],
            '5' => [
                "11111", "10000", "11110", "00001", "00001", "10001", "01110",
            ],
            '6' => [
                "01110", "10000", "10000", "11110", "10001", "10001", "01110",
            ],
            '7' => [
                "11111", "00001", "00010", "00100", "00100", "00100", "00100",
            ],
            '8' => [
                "01110", "10001", "10001", "01110", "10001", "10001", "01110",
            ],
            '9' => [
                "01110", "10001", "10001", "01111", "00001", "00001", "00001",
            ],
            // Common symbols
            '.' => [
                "00000", "00000", "00000", "00000", "00000", "00000", "00100",
            ],
            '-' => [
                "00000", "00000", "00000", "11111", "00000", "00000", "00000",
            ],
            ',' => [
                "00000", "00000", "00000", "00000", "00000", "00100", "01000",
            ],
            '\'' => [
                "00100", "00100", "00000", "00000", "00000", "00000", "00000",
            ],
            '!' => [
                "00100", "00100", "00100", "00100", "00100", "00000", "00100",
            ],
            '?' => [
                "01110", "10001", "00001", "00010", "00100", "00000", "00100",
            ],
            '/' => [
                "00001", "00010", "00100", "01000", "10000", "00000", "00000",
            ],
            ':' => [
                "00000", "00100", "00100", "00000", "00100", "00100", "00000",
            ],
            _ => [
                "00000", "00000", "00000", "00000", "00000", "00000", "00000",
            ],
        }
    }

    fn render_text_5x7(text: &str, scale: u32, char_spacing: u32, line_spacing: u32) -> GrayImage {
        let lines: Vec<&str> = text.lines().collect();
        let glyph_w = 5 * scale;
        let glyph_h = 7 * scale;

        let max_line_len = lines
            .iter()
            .map(|l| l.chars().count() as u32)
            .max()
            .unwrap_or(0);
        let width = if max_line_len == 0 {
            1
        } else {
            max_line_len * glyph_w + max_line_len.saturating_sub(1) * char_spacing + scale * 2
        };
        let height = if lines.is_empty() {
            1
        } else {
            (lines.len() as u32) * glyph_h
                + (lines.len() as u32).saturating_sub(1) * line_spacing
                + scale * 2
        };

        let mut img = GrayImage::from_pixel(width.max(10), height.max(10), Luma([255u8]));

        let mut y = scale;
        for line in lines {
            let mut x = scale;
            for ch in line.chars() {
                if ch == ' ' {
                    x += glyph_w + char_spacing;
                    continue;
                }
                let rows = glyph_rows(ch.to_ascii_uppercase());
                for (ry, row) in rows.iter().enumerate() {
                    for (rx, b) in row.as_bytes().iter().enumerate() {
                        if *b == b'1' {
                            for dy in 0..scale {
                                for dx in 0..scale {
                                    img.put_pixel(
                                        x + (rx as u32) * scale + dx,
                                        y + (ry as u32) * scale + dy,
                                        Luma([0u8]),
                                    );
                                }
                            }
                        }
                    }
                }
                x += glyph_w + char_spacing;
            }
            y += glyph_h + line_spacing;
        }

        img
    }

    #[tokio::test]
    async fn test_ocr_engine_process_image_simple_text() {
        let mut config = crate::core::OcrConfig::default();
        config.image_processing.enable_preprocessing = false;
        config.layout_analysis.enable_layout_analysis = false;

        let engine = OcrEngine::with_config(config).unwrap();
        engine.initialize().await.unwrap();

        let img = render_text_5x7("HELLO", 6, 6, 12);
        let ocr_image = OcrImage::new(DynamicImage::ImageLuma8(img), 300);

        let result = engine.process_image(ocr_image).await.unwrap();
        let text = result.text.trim();
        assert!(!text.is_empty());
        assert!(text.starts_with("HELL"));
        assert!(result.confidence > 0.1);
    }

    #[tokio::test]
    async fn test_ocr_engine_process_image_multiline_text() {
        let mut config = crate::core::OcrConfig::default();
        config.image_processing.enable_preprocessing = false;
        config.layout_analysis.enable_layout_analysis = false;

        let engine = OcrEngine::with_config(config).unwrap();
        engine.initialize().await.unwrap();

        let img = render_text_5x7("HELLO\nWORLD", 6, 6, 12);
        let ocr_image = OcrImage::new(DynamicImage::ImageLuma8(img), 300);

        let result = engine.process_image(ocr_image).await.unwrap();
        let text = result.text.trim();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("HELL"));
        assert!(!lines[1].is_empty());
        assert!(result.confidence > 0.1);
    }
}
