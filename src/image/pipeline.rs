//! Image processing pipeline

use crate::core::image::OcrImage;
use crate::utils::Result;

/// Image preprocessing pipeline
pub struct ImagePreprocessingPipeline {
    config: ImageEnhancementConfig,
}

/// Image enhancement configuration
#[derive(Debug, Clone)]
pub struct ImageEnhancementConfig {
    pub enable_contrast_enhancement: bool,
    pub contrast_factor: f32,
    pub enable_noise_reduction: bool,
    pub noise_reduction_strength: f32,
    pub enable_sharpening: bool,
    pub sharpening_strength: f32,
    pub enable_deskewing: bool,
    pub deskewing_threshold: f32,
    pub enable_perspective_dewarp: bool,
    pub enable_curve_rectification: bool,
    pub enable_super_resolution: bool,
    pub enable_speckle_removal: bool,
    pub max_speckle_area: u32,
    pub enable_border_removal: bool,
    pub enable_orientation_correction: bool,
}

impl Default for ImageEnhancementConfig {
    fn default() -> Self {
        Self {
            enable_contrast_enhancement: true,
            contrast_factor: 1.2,
            enable_noise_reduction: true,
            noise_reduction_strength: 0.5,
            enable_sharpening: false,
            sharpening_strength: 0.5,
            enable_deskewing: true,
            deskewing_threshold: 0.1,
            enable_perspective_dewarp: true,
            enable_curve_rectification: true,
            enable_super_resolution: true,
            enable_speckle_removal: true,
            max_speckle_area: 8,
            enable_border_removal: true,
            enable_orientation_correction: false,
        }
    }
}

impl ImagePreprocessingPipeline {
    pub fn new(config: ImageEnhancementConfig) -> Self {
        Self { config }
    }

    pub fn process(&self, img: &OcrImage) -> Result<OcrImage> {
        let mut processed = img.clone();

        if self.config.enable_orientation_correction {
            processed = crate::image::enhancement::ImageEnhancer::correct_orientation(&processed)?;
        }

        if self.config.enable_border_removal {
            processed = crate::image::enhancement::ImageEnhancer::remove_borders(&processed)?;
        }

        if self.config.enable_speckle_removal {
            processed = crate::image::enhancement::ImageEnhancer::remove_speckle(
                &processed,
                self.config.max_speckle_area,
            )?;
        }

        if self.config.enable_contrast_enhancement {
            processed = crate::image::enhancement::ImageEnhancer::enhance_contrast(
                &processed,
                self.config.contrast_factor,
            )?;
        }

        if self.config.enable_super_resolution {
            let sr = crate::image::super_resolution::TextSuperResolution::default();
            processed = sr.upscale_if_needed(&processed)?.image;
        }

        if self.config.enable_noise_reduction {
            processed = crate::image::enhancement::ImageEnhancer::reduce_noise(&processed)?;
        }

        if self.config.enable_sharpening {
            processed = crate::image::enhancement::ImageEnhancer::sharpen(&processed)?;
        }

        if self.config.enable_deskewing {
            processed = crate::image::enhancement::ImageEnhancer::deskew(&processed)?;
        }

        if self.config.enable_perspective_dewarp {
            processed = crate::image::dewarp::PerspectiveDewarp::default().dewarp(&processed)?;
        }

        Ok(processed)
    }
}
