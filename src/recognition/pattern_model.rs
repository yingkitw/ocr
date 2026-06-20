//! Pattern-based OCR model implementation
//!
//! This module provides a concrete implementation of the OcrModel trait
//! for pattern matching based recognition.

use super::engine::*;
use crate::core::geometry::TBox;
use crate::core::ModelType;
use crate::utils::{OcrError, Result};
use image::{GrayImage, ImageBuffer, Luma};

/// Template for pattern matching
pub struct Template {
    pub character: char,
    pub image: GrayImage,
}

/// Pattern matching model implementation
pub struct PatternModel {
    config: ModelConfig,
    templates: Vec<Template>,
}

impl PatternModel {
    /// Create a new pattern model
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            templates: Vec::new(),
        }
    }

    /// Add a template to the model
    pub fn add_template(
        &mut self,
        character: char,
        image_data: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (req_h, req_w, _req_c) = self.config.input_shape;

        // Check dimensions
        if width as usize != req_w || height as usize != req_h {
            return Err(OcrError::ImageProcessing(format!(
                "Template size mismatch. Expected {}x{}, got {}x{}",
                req_w, req_h, width, height
            ))
            .into());
        }

        let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, image_data)
            .ok_or_else(|| {
                OcrError::ImageProcessing("Invalid image data for template".to_string())
            })?;

        self.templates.push(Template {
            character,
            image: img,
        });

        Ok(())
    }

    /// Calculate similarity between two images (0.0 to 1.0)
    /// Using simple pixel-wise difference (L1 norm inverted)
    fn calculate_similarity(&self, img1: &GrayImage, img2: &GrayImage) -> f32 {
        let mut diff_sum: u64 = 0;
        let total_pixels = (img1.width() * img1.height()) as u64;

        if total_pixels == 0 {
            return 0.0;
        }

        for (p1, p2) in img1.pixels().zip(img2.pixels()) {
            let v1 = p1.0[0] as i32;
            let v2 = p2.0[0] as i32;
            diff_sum += (v1 - v2).abs() as u64;
        }

        let max_diff = total_pixels * 255;
        1.0 - (diff_sum as f32 / max_diff as f32)
    }
}

impl OcrModel for PatternModel {
    fn predict(&self, input: &[u8]) -> Result<RecognitionResult> {
        if self.templates.is_empty() {
            // If no templates, return a placeholder/dummy result to avoid crashing
            // or maybe this is expected in "skeleton" mode?
            // But let's return a "not found" or low confidence result.
            let mut result = RecognitionResult::new("".to_string(), 0.0);
            result.model_type = self.model_type();
            return Ok(result);
        }

        let (req_h, req_w, _req_c) = self.config.input_shape;

        // Parse input image
        // Assuming input is raw bytes matching input_shape
        if input.len() != req_h * req_w {
            return Err(OcrError::ImageProcessing(format!(
                "Input size mismatch. Expected {} bytes, got {}",
                req_h * req_w,
                input.len()
            ))
            .into());
        }

        let input_img =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(req_w as u32, req_h as u32, input.to_vec())
                .ok_or_else(|| {
                    OcrError::ImageProcessing("Failed to create input image".to_string())
                })?;

        // Find best match
        let mut best_char = '?';
        let mut best_score = 0.0;

        for template in &self.templates {
            let score = self.calculate_similarity(&input_img, &template.image);
            if score > best_score {
                best_score = score;
                best_char = template.character;
            }
        }

        let mut result = RecognitionResult::new(best_char.to_string(), best_score);
        result.model_type = self.model_type();
        result.processing_time_ms = 10;

        // Populate character result
        result.character_results = vec![CharacterRecognitionResult {
            character: best_char,
            confidence: best_score,
            bounding_box: TBox::new(0, 0, req_w as i32, req_h as i32), // Relative to input
            unicode_category: UnicodeCategory::Latin,                  // Simplified
            script: ScriptType::Latin,                                 // Simplified
        }];

        Ok(result)
    }

    fn model_type(&self) -> ModelType {
        ModelType::Custom("PatternMatching".to_string())
    }

    fn supported_languages(&self) -> Vec<LanguageVariant> {
        // Pattern matching typically works best for specific fonts/languages
        // For this skeleton, we assume it supports Latin scripts
        vec![LanguageVariant::English]
    }

    fn input_shape(&self) -> (usize, usize, usize) {
        self.config.input_shape
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn supports_language(&self, language: &LanguageVariant) -> bool {
        self.supported_languages().contains(language)
    }
}
