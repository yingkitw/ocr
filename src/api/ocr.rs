//! Main OCR API for OCR

use crate::api::error::{ApiError, ApiResult};
use crate::core::{image::ImageFormat, LayoutResult, OcrConfig, OcrEngine, OcrImage, TextResult};
use crate::utils::{OcrError, Result, Timer};
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

    /// Recognize text from encoded image bytes (PNG, JPEG, etc.)
    ///
    /// For multi-page TIFF, only the first page is recognized. Use
    /// `recognize_text_from_bytes_multi_page` to get all pages.
    pub async fn recognize_text_from_bytes(&self, image_data: &[u8]) -> ApiResult<TextResult> {
        let dynamic_image = image::load_from_memory(image_data)
            .map_err(|e| ApiError::ImageProcessing(format!("Failed to load image: {}", e)))?;
        let (width, height) = dynamic_image.dimensions();

        // Convert to grayscale and extract raw pixel data
        let gray_image = dynamic_image.to_luma8();
        let raw_pixels: Vec<u8> = gray_image.pixels().map(|p| p[0]).collect();

        self.recognize_text(&raw_pixels, width, height).await
    }

    /// Recognize text from encoded image bytes, returning one `TextResult` per page.
    ///
    /// Multi-page TIFF files produce multiple results; all other formats produce
    /// a single-element vector. Each page is recognized independently.
    pub async fn recognize_text_from_bytes_multi_page(
        &self,
        image_data: &[u8],
    ) -> ApiResult<Vec<TextResult>> {
        // Try multi-page TIFF decoding first
        if let Some(pages) = decode_multi_page_tiff(image_data)? {
            let mut results = Vec::with_capacity(pages.len());
            for (pixels, width, height) in pages {
                results.push(self.recognize_text(&pixels, width, height).await?);
            }
            return Ok(results);
        }

        // Fall back to single-image decode for non-TIFF or single-page TIFF
        let result = self.recognize_text_from_bytes(image_data).await?;
        Ok(vec![result])
    }

    /// Recognize text from image file, returning one `TextResult` per page.
    ///
    /// Multi-page TIFF files produce multiple results; all other formats produce
    /// a single-element vector.
    pub async fn recognize_text_from_file_multi_page<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> ApiResult<Vec<TextResult>> {
        let image_data = tokio::fs::read(path).await?;
        self.recognize_text_from_bytes_multi_page(&image_data).await
    }

    /// Recognize text from image file
    pub async fn recognize_text_from_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> ApiResult<TextResult> {
        let image_data = tokio::fs::read(path).await?;
        self.recognize_text_from_bytes(&image_data).await
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
        self.engine.read().await.is_initialized().await
    }
}

impl Default for Ocr {
    fn default() -> Self {
        Self::new().expect("Failed to create OCR instance")
    }
}

/// One decoded TIFF page: (grayscale pixels, width, height).
type TiffPage = (Vec<u8>, u32, u32);

/// Decode a multi-page TIFF from raw bytes, returning one grayscale pixel buffer
/// per page. Returns `None` if the data is not a TIFF or has only a single page
/// (in which case the caller should fall back to `image::load_from_memory`).
///
/// Uses the `tiff` crate directly because `image::load_from_memory` only decodes
/// the first frame of a multi-page TIFF.
fn decode_multi_page_tiff(data: &[u8]) -> ApiResult<Option<Vec<TiffPage>>> {
    use std::io::Cursor;
    use tiff::decoder::{Decoder, DecodingResult};

    // Quick magic-number check: TIFF starts with "II*\x00" (little-endian)
    // or "MM\x00*" (big-endian).
    if data.len() < 4 || !(data.starts_with(b"II*\x00") || data.starts_with(b"MM\x00*")) {
        return Ok(None);
    }

    let cursor = Cursor::new(data);
    let mut decoder = match Decoder::new(cursor) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };

    let mut pages = Vec::new();

    loop {
        let (width, height) = match decoder.dimensions() {
            Ok(d) => d,
            Err(_) => break,
        };

        let result = match decoder.read_image() {
            Ok(r) => r,
            Err(_) => break,
        };

        // Convert to grayscale u8 pixels
        let pixels: Vec<u8> = match result {
            DecodingResult::U8(v) => v,
            DecodingResult::U16(v) => v.iter().map(|&x| (x >> 8) as u8).collect(),
            DecodingResult::U32(v) => v.iter().map(|&x| (x >> 24) as u8).collect(),
            DecodingResult::U64(v) => v.iter().map(|&x| (x >> 56) as u8).collect(),
            DecodingResult::F32(v) => v.iter().map(|&x| (x * 255.0) as u8).collect(),
            _ => Vec::new(),
        };

        // If the pixel count matches width*height*channels, take just the first
        // channel of each pixel (grayscale conversion for RGB/RGBA).
        let expected = (width as usize) * (height as usize);
        let gray: Vec<u8> = if pixels.len() == expected {
            pixels
        } else if pixels.len() == expected * 3 {
            // RGB → grayscale via luminance
            pixels
                .chunks_exact(3)
                .map(|c| {
                    (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) as u8
                })
                .collect()
        } else if pixels.len() == expected * 4 {
            // RGBA → grayscale via luminance (ignore alpha)
            pixels
                .chunks_exact(4)
                .map(|c| {
                    (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) as u8
                })
                .collect()
        } else {
            // Unexpected channel count — take first N pixels
            pixels.into_iter().take(expected).collect()
        };

        pages.push((gray, width, height));

        if !decoder.more_images() {
            break;
        }
        match decoder.next_image() {
            Ok(()) => {}
            Err(_) => break,
        }
    }

    if pages.len() <= 1 {
        Ok(None) // single-page or failed — let caller use the normal path
    } else {
        Ok(Some(pages))
    }
}

