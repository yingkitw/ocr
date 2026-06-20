//! Image Processing Service
//!
//! High-level service for image preprocessing and enhancement.
//! Coordinates between image loading, enhancement, and preprocessing components.

use super::{ImageProcessingError, ImageProcessingResult};
use crate::image::ImageProcessor;
use crate::core::image::OcrImage;
use crate::utils::Timer;

pub struct ImageProcessingService {
    max_dimensions: (u32, u32),
}

impl ImageProcessingService {
    pub fn new() -> Self {
        Self {
            max_dimensions: (10000, 10000),
        }
    }

    pub fn with_max_dimensions(width: u32, height: u32) -> Self {
        Self {
            max_dimensions: (width, height),
        }
    }

    pub async fn process_image_from_file(
        &self,
        path: &std::path::Path,
    ) -> Result<ImageProcessingResult, ImageProcessingError> {
        let timer = Timer::new();
        
        let original_data = tokio::fs::read(path)
            .await
            .map_err(|e| ImageProcessingError::LoadFailed(format!("Failed to read image: {}", e)))?;

        let image = image::load_from_memory(&original_data)
            .map_err(|e| ImageProcessingError::LoadFailed(format!("Failed to load image: {}", e)))?;

        let (width, height) = image.dimensions();
        if width > self.max_dimensions.0 || height > self.max_dimensions.1 {
            return Err(ImageProcessingError::DimensionsTooLarge { width, height });
        }

        let applied_ops = vec!["preprocessing".to_string()];
        let ocr_image = ImageProcessor::to_ocr_image(image, 300);
        let processed_image = ImageProcessor::preprocess_for_ocr(&ocr_image)
            .map_err(|e| ImageProcessingError::ProcessingFailed(e.to_string()))?;
        
        let processed_data = Self::image_to_bytes(processed_image);

        Ok(ImageProcessingResult {
            original_data,
            processed_data,
            processing_time_ms: timer.elapsed_ms(),
            applied_operations: applied_ops,
        })
    }

    pub async fn process_image_from_bytes(
        &self,
        data: &[u8],
    ) -> Result<ImageProcessingResult, ImageProcessingError> {
        let timer = Timer::new();
        
        let original_data = data.to_vec();
        let image = image::load_from_memory(&original_data)
            .map_err(|e| ImageProcessingError::LoadFailed(format!("Failed to load image: {}", e)))?;

        let applied_ops = vec!["preprocessing".to_string()];
        let ocr_image = ImageProcessor::to_ocr_image(image, 300);
        let processed_image = ImageProcessor::preprocess_for_ocr(&ocr_image)
            .map_err(|e| ImageProcessingError::ProcessingFailed(e.to_string()))?;
        
        let processed_data = Self::image_to_bytes(processed_image);

        Ok(ImageProcessingResult {
            original_data,
            processed_data,
            processing_time_ms: timer.elapsed_ms(),
            applied_operations: applied_ops,
        })
    }

    pub fn create_ocr_image(&self, data: &[u8], width: u32, height: u32) -> Result<OcrImage, ImageProcessingError> {
        OcrImage::from_raw_pixels(width, height, data.to_vec(), crate::core::image::ImageFormat::Grayscale, 300)
            .map_err(|e| ImageProcessingError::ProcessingFailed(e.to_string()))
    }

    fn image_to_bytes(ocr_image: &OcrImage) -> Vec<u8> {
        let gray = ocr_image.to_luma8();
        gray.into_raw()
    }
}

impl Default for ImageProcessingService {
    fn default() -> Self {
        Self::new()
    }
}
