//! Image processing operations

use crate::core::image::OcrImage;
use crate::utils::{MiniOcrError, Result};
use image::GenericImageView;

/// Image processor for basic operations
pub struct ImageProcessor;

impl ImageProcessor {
    /// Load image from memory
    pub fn load_from_memory(data: &[u8]) -> Result<image::DynamicImage> {
        let img = image::load_from_memory(data)
            .map_err(|e| MiniOcrError::ImageProcessing(format!("Failed to load image: {}", e)))?;
        Ok(img)
    }

    /// Convert dynamic image to OCR image
    pub fn to_ocr_image(img: image::DynamicImage, dpi: u32) -> OcrImage {
        let mut ocr_image = OcrImage::new(img, dpi);

        // Extract basic metadata
        ocr_image.metadata.insert(
            "color_type".to_string(),
            format!("{:?}", ocr_image.data.color()),
        );
        let (w, h) = ocr_image.data.dimensions();
        ocr_image
            .metadata
            .insert("dimensions".to_string(), format!("{}x{}", w, h));

        ocr_image
    }

    /// Preprocess image for OCR
    pub fn preprocess_for_ocr(img: &OcrImage) -> Result<OcrImage> {
        // Convert to grayscale if needed
        let gray_img = if img.format != crate::core::image::ImageFormat::Grayscale {
            img.to_grayscale()
        } else {
            img.clone()
        };

        // Apply basic preprocessing
        let preprocessed = Self::apply_basic_preprocessing(&gray_img)?;

        Ok(preprocessed)
    }

    /// Apply basic preprocessing
    fn apply_basic_preprocessing(img: &OcrImage) -> Result<OcrImage> {
        // TODO: Implement basic preprocessing
        // - Noise reduction
        // - Contrast enhancement
        // - Sharpening
        // - Deskewing
        Ok(img.clone())
    }
}
