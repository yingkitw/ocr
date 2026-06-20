//! Text Recognition Service
//!
//! High-level service for text recognition operations.
//! Coordinates between OCR engine, image processing, and layout analysis.

use super::{RecognitionResult, TextRecognitionError};
use crate::core::{OcrEngine, config::OcrConfig, text::TextResult, image::OcrImage};
use crate::layout::LayoutResult;
use crate::utils::Timer;
use tokio::sync::RwLock;

pub struct TextRecognitionService {
    engine: RwLock<OcrEngine>,
    config: OcrConfig,
}

impl TextRecognitionService {
    pub fn new(engine: OcrEngine, config: OcrConfig) -> Self {
        Self {
            engine: RwLock::new(engine),
            config,
        }
    }

    pub async fn initialize(&self) -> Result<(), TextRecognitionError> {
        let engine = self.engine.write().await;
        engine.initialize()
            .await
            .map_err(|e| TextRecognitionError::InitializationFailed(e.to_string()))?;
        Ok(())
    }

    pub async fn recognize_image(
        &self,
        image: OcrImage,
    ) -> Result<RecognitionResult, TextRecognitionError> {
        let timer = Timer::new();
        let engine = self.engine.read().await;
        
        let text_result = engine
            .process_image(image)
            .await
            .map_err(|e| TextRecognitionError::RecognitionFailed(e.to_string()))?;

        let processing_time = timer.elapsed_ms();
        let engine_name = format!("{:?}", self.config.recognition.engine);
        
        Ok(RecognitionResult::new(text_result, processing_time, engine_name))
    }

    pub async fn recognize_from_pixels(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<RecognitionResult, TextRecognitionError> {
        use crate::core::image::ImageFormat;
        
        let dpi = self.config
            .recognition
            .parameters
            .get("dpi")
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        let image = OcrImage::from_raw_pixels(width, height, data.to_vec(), ImageFormat::Grayscale, dpi)
            .map_err(|e| TextRecognitionError::ImageProcessingFailed(e.to_string()))?;

        self.recognize_image(image).await
    }

    pub async fn recognize_from_file(
        &self,
        path: &std::path::Path,
    ) -> Result<RecognitionResult, TextRecognitionError> {
        let image_data = tokio::fs::read(path)
            .await
            .map_err(|e| TextRecognitionError::ImageProcessingFailed(format!("Failed to read image: {}", e)))?;
        
        let dynamic_image = image::load_from_memory(&image_data)
            .map_err(|e| TextRecognitionError::ImageProcessingFailed(format!("Failed to load image: {}", e)))?;
        
        let (width, height) = dynamic_image.dimensions();
        let gray_image = dynamic_image.to_luma8();
        let raw_pixels: Vec<u8> = gray_image.pixels().map(|p| p[0]).collect();

        self.recognize_from_pixels(&raw_pixels, width, height).await
    }

    pub async fn analyze_layout(
        &self,
        image: OcrImage,
    ) -> Result<LayoutResult, TextRecognitionError> {
        crate::layout::LayoutAnalyzer::analyze_layout(&image)
            .map_err(|e| TextRecognitionError::LayoutAnalysisFailed(e.to_string()))
    }

    pub fn get_config(&self) -> &OcrConfig {
        &self.config
    }

    pub async fn update_config(&mut self, config: OcrConfig) -> Result<(), TextRecognitionError> {
        let mut engine = self.engine.write().await;
        engine.update_config(config.clone())
            .await
            .map_err(|e| TextRecognitionError::InvalidConfiguration(e.to_string()))?;
        self.config = config;
        Ok(())
    }
}
