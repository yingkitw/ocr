//! Main OCR API for OCR

use crate::api::error::{ApiError, ApiResult};
use crate::core::{
    LayoutResult, OcrConfig, OcrEngine, OcrImage, TextResult, image::ImageFormat,
};
use crate::utils::{OcrError, Result, Timer};
use chrono::{DateTime, Utc};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Main OCR API
pub struct Ocr {
    /// OCR engine
    engine: RwLock<OcrEngine>,
    /// Configuration
    config: OcrConfig,
    /// API metadata
    metadata: ApiMetadata,
}

/// API metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMetadata {
    /// API name
    pub name: String,
    /// API version
    pub version: String,
    /// API description
    pub description: String,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Supported image formats
    pub supported_image_formats: Vec<String>,
}

impl Ocr {
    /// Create a new OCR instance with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(OcrConfig::default())
    }

    /// Create a new OCR instance with custom configuration
    pub fn with_config(config: OcrConfig) -> Result<Self> {
        let engine = OcrEngine::with_config(config.clone())?;

        Ok(Self {
            engine: RwLock::new(engine),
            config,
            metadata: ApiMetadata {
                name: "OCR API".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: "A modern OCR API written in Rust".to_string(),
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
                ],
                supported_image_formats: vec![
                    "png".to_string(),
                    "jpg".to_string(),
                    "jpeg".to_string(),
                    "tiff".to_string(),
                    "bmp".to_string(),
                    "gif".to_string(),
                ],
            },
        })
    }

    /// Initialize the OCR engine
    pub async fn initialize(&self) -> ApiResult<()> {
        let engine = self.engine.write().await;
        engine.initialize().await?;
        Ok(())
    }

    /// Recognize text from image data
    pub async fn recognize_text(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> ApiResult<TextResult> {
        let timer = Timer::new();

        // Convert image data to OcrImage
        let image = OcrImage::from_raw_pixels(
            width,
            height,
            image_data.to_vec(),
            ImageFormat::Grayscale,
            self.config
                .recognition
                .parameters
                .get("dpi")
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
        )?;

        // Process image
        let engine = self.engine.read().await;
        let result = engine.process_image(image).await?;

        tracing::debug!("Text recognition completed in {}ms", timer.elapsed_ms());
        Ok(result)
    }

    /// Recognize text from image file
    pub async fn recognize_text_from_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> ApiResult<TextResult> {
        let image_data = tokio::fs::read(path).await?;
        let dynamic_image = image::load_from_memory(&image_data)
            .map_err(|e| ApiError::ImageProcessing(format!("Failed to load image: {}", e)))?;
        let (width, height) = dynamic_image.dimensions();

        // Convert to grayscale and extract raw pixel data
        let gray_image = dynamic_image.to_luma8();
        let raw_pixels: Vec<u8> = gray_image.pixels().map(|p| p[0]).collect();

        self.recognize_text(&raw_pixels, width, height).await
    }

    /// Recognize text from image with custom configuration
    pub async fn recognize_text_with_config(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
        config: &OcrConfig,
    ) -> ApiResult<TextResult> {
        // Create temporary engine with custom config
        let temp_engine = OcrEngine::with_config(config.clone())
            .map_err(|e| ApiError::Configuration(e.to_string()))?;
        temp_engine
            .initialize()
            .await
            .map_err(|e| ApiError::OcrProcessing(OcrError::Recognition(e.to_string())))?;

        // Convert image data to OcrImage
        let image = OcrImage::from_raw_pixels(
            width,
            height,
            image_data.to_vec(),
            ImageFormat::Grayscale,
            config
                .recognition
                .parameters
                .get("dpi")
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
        )
        .map_err(|e| ApiError::ImageProcessing(e.to_string()))?;

        // Process image
        temp_engine
            .process_image(image)
            .await
            .map_err(|e| ApiError::OcrProcessing(OcrError::Recognition(e.to_string())))
    }

    /// Analyze page layout
    pub async fn analyze_layout(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> ApiResult<LayoutResult> {
        // Convert image data to OcrImage
        let image = OcrImage::from_raw_pixels(
            width,
            height,
            image_data.to_vec(),
            ImageFormat::Grayscale,
            self.config
                .recognition
                .parameters
                .get("dpi")
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
        )?;

        let _engine = self.engine.read().await;
        Ok(crate::layout::LayoutAnalyzer::analyze_layout(&image)?)
    }

    /// Get supported languages
    pub fn get_supported_languages(&self) -> Vec<String> {
        let engine = self.engine.try_read().ok();
        if let Some(e) = engine {
            e.get_metadata().supported_languages.clone()
        } else {
            self.metadata.supported_languages.clone()
        }
    }

    /// Get supported image formats
    pub fn get_supported_image_formats(&self) -> &[String] {
        &self.metadata.supported_image_formats
    }

    /// Get API metadata
    pub fn get_metadata(&self) -> &ApiMetadata {
        &self.metadata
    }

    /// Get current configuration
    pub fn get_config(&self) -> &OcrConfig {
        &self.config
    }

    /// Update configuration
    pub async fn update_config(&self, config: OcrConfig) -> ApiResult<()> {
        let mut engine = self.engine.write().await;
        engine.update_config(config).await?;
        Ok(())
    }

    /// Get engine statistics
    pub async fn get_statistics(&self) -> ApiResult<crate::core::EngineStatistics> {
        let engine = self.engine.read().await;
        engine
            .get_statistics()
            .await
            .map_err(|e| ApiError::Internal(e.to_string()))
    }

    /// Clear engine cache
    pub async fn clear_cache(&self) -> ApiResult<()> {
        let engine = self.engine.read().await;
        engine.clear_cache().await?;
        Ok(())
    }

    /// Reset engine statistics
    pub async fn reset_statistics(&self) -> ApiResult<()> {
        let engine = self.engine.read().await;
        engine.reset_statistics().await?;
        Ok(())
    }

    /// Check if engine is initialized
    pub async fn is_initialized(&self) -> bool {
        let engine = self.engine.read().await;
        // TODO: Add is_initialized method to OcrEngine
        true
    }
}

impl Default for Ocr {
    fn default() -> Self {
        Self::new().expect("Failed to create OCR instance")
    }
}

/// Batch processing for multiple images
pub struct BatchProcessor {
    /// OCR engine
    ocr: Ocr,
    /// Batch configuration
    config: BatchConfig,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of concurrent images
    pub max_concurrent: usize,
    /// Enable progress tracking
    pub enable_progress_tracking: bool,
    /// Progress callback
    pub progress_callback: Option<String>, // Function name for callback
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            enable_progress_tracking: true,
            progress_callback: None,
        }
    }
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(ocr: Ocr, config: BatchConfig) -> Self {
        Self { ocr, config }
    }

    /// Process multiple images
    pub async fn process_images(&self, images: Vec<ImageInput>) -> ApiResult<Vec<TextResult>> {
        use futures::stream::{FuturesUnordered, StreamExt};

        let semaphore =
            std::sync::Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent));
        let mut tasks: FuturesUnordered<_> = FuturesUnordered::new();
        let ocr = &self.ocr;

        for image in images {
            let semaphore = semaphore.clone();
            tasks.push(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .map_err(|e| ApiError::Internal(format!("Semaphore acquire failed: {}", e)))?;
                match image {
                    ImageInput::Data {
                        data,
                        width,
                        height,
                    } => ocr.recognize_text(&data, width, height).await,
                    ImageInput::File { path } => ocr.recognize_text_from_file(path).await,
                }
            });
        }

        let mut results = Vec::new();
        while let Some(result) = tasks.next().await {
            results.push(result?);
        }

        Ok(results)
    }
}

/// Image input for batch processing
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// Image data with dimensions
    Data {
        data: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Image file path
    File { path: String },
}

/// OCR result with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// Text result
    pub text_result: TextResult,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Image metadata
    pub image_metadata: ImageMetadata,
    /// Engine metadata
    pub engine_metadata: EngineMetadata,
}

/// Image metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Image format
    pub format: String,
    /// Image DPI
    pub dpi: u32,
    /// Image size in bytes
    pub size_bytes: usize,
}

/// Engine metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetadata {
    /// Engine name
    pub name: String,
    /// Engine version
    pub version: String,
    /// Processing timestamp
    pub timestamp: DateTime<Utc>,
    /// Configuration used
    pub config: OcrConfig,
}
