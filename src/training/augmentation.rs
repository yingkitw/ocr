//! Data augmentation for training

use crate::core::geometry::TBox;
use crate::training::data::TrainingSample;
use anyhow::Result;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Data augmentation pipeline
pub struct AugmentationPipeline {
    config: AugmentationConfig,
    rng: rand::rngs::StdRng,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    pub enabled: bool,
    pub rotation_range: (f32, f32),
    pub scale_range: (f32, f32),
    pub translation_range: (f32, f32),
    pub brightness_range: (f32, f32),
    pub contrast_range: (f32, f32),
    pub noise_level: f32,
    pub blur_probability: f32,
    pub crop_probability: f32,
    pub perspective_probability: f32,
    pub elastic_deformation_probability: f32,
    pub color_jitter_probability: f32,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rotation_range: (-15.0, 15.0),
            scale_range: (0.8, 1.2),
            translation_range: (-20.0, 20.0),
            brightness_range: (0.7, 1.3),
            contrast_range: (0.7, 1.3),
            noise_level: 0.1,
            blur_probability: 0.2,
            crop_probability: 0.3,
            perspective_probability: 0.1,
            elastic_deformation_probability: 0.1,
            color_jitter_probability: 0.3,
        }
    }
}

impl AugmentationPipeline {
    pub fn new(config: AugmentationConfig) -> Self {
        Self {
            config,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Apply augmentation to a training sample
    pub fn augment_sample(&mut self, sample: &mut TrainingSample) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Apply random rotation
        if self.rng.r#gen::<f32>() < 0.5 {
            self.apply_rotation(sample)?;
        }

        // Apply random scaling
        if self.rng.r#gen::<f32>() < 0.5 {
            self.apply_scaling(sample)?;
        }

        // Apply random translation
        if self.rng.r#gen::<f32>() < 0.5 {
            self.apply_translation(sample)?;
        }

        // Apply brightness adjustment
        if self.rng.r#gen::<f32>() < 0.5 {
            self.apply_brightness(sample)?;
        }

        // Apply contrast adjustment
        if self.rng.r#gen::<f32>() < 0.5 {
            self.apply_contrast(sample)?;
        }

        // Apply noise
        if self.rng.r#gen::<f32>() < 0.3 {
            self.apply_noise(sample)?;
        }

        // Apply blur
        if self.rng.r#gen::<f32>() < self.config.blur_probability {
            self.apply_blur(sample)?;
        }

        // Apply random crop
        if self.rng.r#gen::<f32>() < self.config.crop_probability {
            self.apply_crop(sample)?;
        }

        // Apply perspective transformation
        if self.rng.r#gen::<f32>() < self.config.perspective_probability {
            self.apply_perspective(sample)?;
        }

        // Apply elastic deformation
        if self.rng.r#gen::<f32>() < self.config.elastic_deformation_probability {
            self.apply_elastic_deformation(sample)?;
        }

        // Apply color jitter
        if self.rng.r#gen::<f32>() < self.config.color_jitter_probability {
            self.apply_color_jitter(sample)?;
        }

        Ok(())
    }

    /// Apply random rotation
    fn apply_rotation(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let angle = self
            .rng
            .gen_range(self.config.rotation_range.0..self.config.rotation_range.1);
        let angle_rad = angle * PI / 180.0;

        // Rotate image
        sample.image = sample.image.rotate(angle)?;

        // Update bounding boxes
        for bbox in &mut sample.bounding_boxes {
            *bbox = self.rotate_bbox(*bbox, angle_rad, sample.image.dimensions());
        }

        Ok(())
    }

    /// Apply random scaling
    fn apply_scaling(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let scale = self
            .rng
            .gen_range(self.config.scale_range.0..self.config.scale_range.1);

        // Scale image
        let (width, height) = sample.image.dimensions();
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        sample.image = sample.image.resize(new_width, new_height)?;

        // Update bounding boxes
        for bbox in &mut sample.bounding_boxes {
            *bbox = self.scale_bbox(*bbox, scale);
        }

        Ok(())
    }

    /// Apply random translation
    fn apply_translation(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let tx = self
            .rng
            .gen_range(self.config.translation_range.0..self.config.translation_range.1);
        let ty = self
            .rng
            .gen_range(self.config.translation_range.0..self.config.translation_range.1);

        // Translate image
        sample.image = sample.image.translate(tx, ty)?;

        // Update bounding boxes
        for bbox in &mut sample.bounding_boxes {
            *bbox = self.translate_bbox(*bbox, tx, ty);
        }

        Ok(())
    }

    /// Apply brightness adjustment
    fn apply_brightness(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let factor = self
            .rng
            .gen_range(self.config.brightness_range.0..self.config.brightness_range.1);
        sample.image = sample.image.adjust_brightness(factor)?;
        Ok(())
    }

    /// Apply contrast adjustment
    fn apply_contrast(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let factor = self
            .rng
            .gen_range(self.config.contrast_range.0..self.config.contrast_range.1);
        sample.image = sample.image.adjust_contrast(factor)?;
        Ok(())
    }

    /// Apply noise
    fn apply_noise(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let noise_level = self.config.noise_level;
        sample.image = sample.image.add_noise(noise_level)?;
        Ok(())
    }

    /// Apply blur
    fn apply_blur(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let blur_radius = self.rng.gen_range(1.0..3.0);
        sample.image = sample.image.gaussian_blur(blur_radius)?;
        Ok(())
    }

    /// Apply random crop
    fn apply_crop(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let (width, height) = sample.image.dimensions();
        let crop_ratio = self.rng.gen_range(0.7..1.0);

        let crop_width = (width as f32 * crop_ratio) as u32;
        let crop_height = (height as f32 * crop_ratio) as u32;

        let x = self.rng.gen_range(0..width.saturating_sub(crop_width));
        let y = self.rng.gen_range(0..height.saturating_sub(crop_height));

        sample.image = sample.image.crop(x, y, crop_width, crop_height)?;

        // Update bounding boxes
        for bbox in &mut sample.bounding_boxes {
            *bbox = self.crop_bbox(*bbox, x, y, crop_width, crop_height);
        }

        Ok(())
    }

    /// Apply perspective transformation
    fn apply_perspective(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let (width, height) = sample.image.dimensions();
        let perspective_strength = self.rng.gen_range(0.1..0.3);

        // Generate random perspective points
        let src_points = vec![
            (0.0, 0.0),
            (width as f32, 0.0),
            (width as f32, height as f32),
            (0.0, height as f32),
        ];

        let mut dst_points = src_points.clone();

        for (_i, (x, y)) in dst_points.iter_mut().enumerate() {
            let offset_x = (self.rng.r#gen::<f32>() - 0.5) * width as f32 * perspective_strength;
            let offset_y = (self.rng.r#gen::<f32>() - 0.5) * height as f32 * perspective_strength;
            *x += offset_x;
            *y += offset_y;
        }

        sample.image = sample
            .image
            .perspective_transform(&src_points, &dst_points)?;

        // Update bounding boxes (simplified)
        for bbox in &mut sample.bounding_boxes {
            *bbox = self.perspective_transform_bbox(*bbox, &src_points, &dst_points);
        }

        Ok(())
    }

    /// Apply elastic deformation
    fn apply_elastic_deformation(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let (width, height) = sample.image.dimensions();
        let alpha = self.rng.gen_range(50.0..200.0);
        let sigma = self.rng.gen_range(5.0..15.0);

        sample.image = sample.image.elastic_deformation(alpha, sigma)?;

        // Update bounding boxes
        for bbox in &mut sample.bounding_boxes {
            *bbox = self.elastic_deform_bbox(*bbox, alpha, sigma, width, height);
        }

        Ok(())
    }

    /// Apply color jitter
    fn apply_color_jitter(&mut self, sample: &mut TrainingSample) -> Result<()> {
        let hue_shift = self.rng.gen_range(-30.0..30.0);
        let saturation_factor = self.rng.gen_range(0.7..1.3);
        let value_factor = self.rng.gen_range(0.7..1.3);

        sample.image = sample
            .image
            .color_jitter(hue_shift, saturation_factor, value_factor)?;
        Ok(())
    }

    // Helper methods for bounding box transformations

    fn rotate_bbox(&self, bbox: TBox, angle: f32, (width, height): (u32, u32)) -> TBox {
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Rotate corners
        let corners = [
            (
                bbox.left() as f32 - center_x,
                bbox.bottom() as f32 - center_y,
            ),
            (
                bbox.right() as f32 - center_x,
                bbox.bottom() as f32 - center_y,
            ),
            (bbox.right() as f32 - center_x, bbox.top() as f32 - center_y),
            (bbox.left() as f32 - center_x, bbox.top() as f32 - center_y),
        ];

        let rotated_corners: Vec<(f32, f32)> = corners
            .iter()
            .map(|(x, y)| {
                let new_x = x * cos_a - y * sin_a + center_x;
                let new_y = x * sin_a + y * cos_a + center_y;
                (new_x, new_y)
            })
            .collect();

        // Find bounding box of rotated corners
        let min_x = rotated_corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f32::INFINITY, f32::min);
        let max_x = rotated_corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = rotated_corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f32::INFINITY, f32::min);
        let max_y = rotated_corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f32::NEG_INFINITY, f32::max);

        TBox::new(min_x as i32, min_y as i32, max_x as i32, max_y as i32)
    }

    fn scale_bbox(&self, bbox: TBox, scale: f32) -> TBox {
        TBox::new(
            (bbox.left() as f32 * scale) as i32,
            (bbox.bottom() as f32 * scale) as i32,
            (bbox.right() as f32 * scale) as i32,
            (bbox.top() as f32 * scale) as i32,
        )
    }

    fn translate_bbox(&self, bbox: TBox, tx: f32, ty: f32) -> TBox {
        TBox::new(
            bbox.left() + tx as i32,
            bbox.bottom() + ty as i32,
            bbox.right() + tx as i32,
            bbox.top() + ty as i32,
        )
    }

    fn crop_bbox(
        &self,
        bbox: TBox,
        crop_x: u32,
        crop_y: u32,
        crop_width: u32,
        crop_height: u32,
    ) -> TBox {
        let new_left = (bbox.left() - crop_x as i32).max(0);
        let new_bottom = (bbox.bottom() - crop_y as i32).max(0);
        let new_right = (bbox.right() - crop_x as i32).min(crop_width as i32);
        let new_top = (bbox.top() - crop_y as i32).min(crop_height as i32);

        TBox::new(new_left, new_bottom, new_right, new_top)
    }

    fn perspective_transform_bbox(
        &self,
        bbox: TBox,
        _src_points: &[(f32, f32)],
        _dst_points: &[(f32, f32)],
    ) -> TBox {
        // Simplified perspective transformation for bounding boxes
        // In practice, this would be more complex
        bbox
    }

    fn elastic_deform_bbox(
        &self,
        bbox: TBox,
        _alpha: f32,
        _sigma: f32,
        _width: u32,
        _height: u32,
    ) -> TBox {
        // Simplified elastic deformation for bounding boxes
        // In practice, this would apply the same deformation field
        bbox
    }
}

/// Augmentation strategies for different types of data
pub enum AugmentationStrategy {
    /// Conservative augmentation for high-quality data
    Conservative,
    /// Moderate augmentation for balanced datasets
    Moderate,
    /// Aggressive augmentation for small datasets
    Aggressive,
    /// Custom augmentation with specific parameters
    Custom(AugmentationConfig),
}

impl AugmentationStrategy {
    pub fn get_config(&self) -> AugmentationConfig {
        match self {
            AugmentationStrategy::Conservative => AugmentationConfig {
                enabled: true,
                rotation_range: (-5.0, 5.0),
                scale_range: (0.9, 1.1),
                translation_range: (-10.0, 10.0),
                brightness_range: (0.9, 1.1),
                contrast_range: (0.9, 1.1),
                noise_level: 0.05,
                blur_probability: 0.1,
                crop_probability: 0.1,
                perspective_probability: 0.0,
                elastic_deformation_probability: 0.0,
                color_jitter_probability: 0.1,
            },
            AugmentationStrategy::Moderate => AugmentationConfig::default(),
            AugmentationStrategy::Aggressive => AugmentationConfig {
                enabled: true,
                rotation_range: (-30.0, 30.0),
                scale_range: (0.6, 1.4),
                translation_range: (-50.0, 50.0),
                brightness_range: (0.5, 1.5),
                contrast_range: (0.5, 1.5),
                noise_level: 0.2,
                blur_probability: 0.4,
                crop_probability: 0.5,
                perspective_probability: 0.3,
                elastic_deformation_probability: 0.3,
                color_jitter_probability: 0.5,
            },
            AugmentationStrategy::Custom(config) => config.clone(),
        }
    }
}

/// Augmentation utilities
pub struct AugmentationUtils;

impl AugmentationUtils {
    /// Create augmentation pipeline from strategy
    pub fn create_pipeline(strategy: AugmentationStrategy) -> AugmentationPipeline {
        AugmentationPipeline::new(strategy.get_config())
    }

    /// Validate augmentation configuration
    pub fn validate_config(config: &AugmentationConfig) -> Result<()> {
        if config.rotation_range.0 >= config.rotation_range.1 {
            return Err(anyhow::anyhow!("Invalid rotation range"));
        }
        if config.scale_range.0 <= 0.0 || config.scale_range.0 >= config.scale_range.1 {
            return Err(anyhow::anyhow!("Invalid scale range"));
        }
        if config.brightness_range.0 <= 0.0
            || config.brightness_range.0 >= config.brightness_range.1
        {
            return Err(anyhow::anyhow!("Invalid brightness range"));
        }
        if config.contrast_range.0 <= 0.0 || config.contrast_range.0 >= config.contrast_range.1 {
            return Err(anyhow::anyhow!("Invalid contrast range"));
        }
        if config.noise_level < 0.0 || config.noise_level > 1.0 {
            return Err(anyhow::anyhow!("Invalid noise level"));
        }
        Ok(())
    }

    /// Estimate augmentation impact on dataset size
    pub fn estimate_augmented_size(original_size: usize, config: &AugmentationConfig) -> usize {
        if !config.enabled {
            return original_size;
        }

        let mut multiplier = 1.0;

        // Each augmentation type can potentially double the dataset
        if config.rotation_range.0 != config.rotation_range.1 {
            multiplier *= 1.5;
        }
        if config.scale_range.0 != config.scale_range.1 {
            multiplier *= 1.5;
        }
        if config.translation_range.0 != config.translation_range.1 {
            multiplier *= 1.5;
        }
        if config.brightness_range.0 != config.brightness_range.1 {
            multiplier *= 1.2;
        }
        if config.contrast_range.0 != config.contrast_range.1 {
            multiplier *= 1.2;
        }
        if config.noise_level > 0.0 {
            multiplier *= 1.2;
        }
        if config.blur_probability > 0.0 {
            multiplier *= 1.2;
        }
        if config.crop_probability > 0.0 {
            multiplier *= 1.3;
        }
        if config.perspective_probability > 0.0 {
            multiplier *= 1.2;
        }
        if config.elastic_deformation_probability > 0.0 {
            multiplier *= 1.2;
        }
        if config.color_jitter_probability > 0.0 {
            multiplier *= 1.2;
        }

        (original_size as f32 * multiplier) as usize
    }
}
