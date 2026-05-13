//! Image quality assessment

use crate::core::image::OcrImage;
use imageproc::filter::filter3x3;

/// Image quality assessor
pub struct ImageQualityAssessor;

/// DPI estimation result
#[derive(Debug, Clone)]
pub struct DpiEstimate {
    /// Estimated DPI
    pub dpi: u32,
    /// Confidence in the estimate (0.0 to 1.0)
    pub confidence: f32,
    /// Estimation method used
    pub method: DpiEstimationMethod,
}

/// Method used for DPI estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpiEstimationMethod {
    /// From image metadata (EXIF)
    Metadata,
    /// From character stroke width analysis
    StrokeWidth,
    /// From font size analysis
    FontSize,
    /// Default fallback value
    Default,
}

impl ImageQualityAssessor {
    /// Estimate DPI from image
    ///
    /// Tries multiple methods in order:
    /// 1. Image metadata (EXIF)
    /// 2. Stroke width analysis (for printed text)
    /// 3. Font size analysis
    /// 4. Default value (300 DPI)
    pub fn estimate_dpi(img: &OcrImage) -> DpiEstimate {
        // Try metadata first
        if let Some(estimate) = Self::estimate_dpi_from_metadata(img) {
            if estimate.confidence > 0.7 {
                return estimate;
            }
        }

        // Try stroke width analysis for binary/grayscale images
        if img.format == crate::core::image::ImageFormat::Binary
            || img.format == crate::core::image::ImageFormat::Grayscale
        {
            if let Some(estimate) = Self::estimate_dpi_from_stroke_width(img) {
                if estimate.confidence > 0.5 {
                    return estimate;
                }
            }
        }

        // Try font size analysis
        if let Some(estimate) = Self::estimate_dpi_from_font_size(img) {
            if estimate.confidence > 0.4 {
                return estimate;
            }
        }

        // Default to 300 DPI
        DpiEstimate {
            dpi: 300,
            confidence: 0.3,
            method: DpiEstimationMethod::Default,
        }
    }

    /// Estimate DPI from image metadata
    fn estimate_dpi_from_metadata(img: &OcrImage) -> Option<DpiEstimate> {
        // Check if DPI is already set in the image
        if img.dpi > 0 {
            return Some(DpiEstimate {
                dpi: img.dpi,
                confidence: if img.dpi >= 72 && img.dpi <= 600 {
                    0.9
                } else {
                    0.6
                },
                method: DpiEstimationMethod::Metadata,
            });
        }

        // Check metadata for resolution information
        if let Some(dpi_str) = img.metadata.get("dpi") {
            if let Ok(dpi) = dpi_str.parse::<u32>() {
                return Some(DpiEstimate {
                    dpi,
                    confidence: 0.8,
                    method: DpiEstimationMethod::Metadata,
                });
            }
        }

        None
    }

    /// Estimate DPI from stroke width analysis
    ///
    /// This works for printed text where stroke width correlates with DPI.
    /// Typical printed text has stroke widths of 1-2 pixels at 300 DPI.
    fn estimate_dpi_from_stroke_width(img: &OcrImage) -> Option<DpiEstimate> {
        use image::GenericImageView;

        // Convert to grayscale if needed
        let gray = if img.format == crate::core::image::ImageFormat::Grayscale {
            img.data.to_luma8()
        } else {
            img.data.to_luma8()
        };

        // Sample horizontal strokes (e.g., from '-', 'E', 'T', etc.)
        let mut stroke_widths = Vec::new();
        let (width, height) = gray.dimensions();

        // Sample from center region of image (avoiding edges)
        let margin = width.min(height) / 10;
        let sample_width = (width - 2 * margin).min(500);
        let sample_height = (height - 2 * margin).min(500);
        let sample_x = margin;
        let sample_y = margin;

        if sample_width < 10 || sample_height < 10 {
            return None;
        }

        // Find black-white transitions to estimate stroke width
        for y in sample_y..(sample_y + sample_height).min(height) {
            let mut in_stroke = false;
            let mut stroke_start = 0;
            let mut transition_count = 0u32;

            for x in sample_x..(sample_x + sample_width).min(width) {
                let pixel = gray.get_pixel(x, y);
                let is_dark = pixel[0] < 128;

                match (in_stroke, is_dark) {
                    (false, true) => {
                        in_stroke = true;
                        stroke_start = x;
                    }
                    (true, false) => {
                        in_stroke = false;
                        let stroke_width = x - stroke_start;
                        if stroke_width > 0 && stroke_width < 50 {
                            stroke_widths.push(stroke_width);
                        }
                        transition_count += 1;
                    }
                    _ => {}
                }
            }

            // Only count rows with reasonable number of transitions (text lines)
            if transition_count >= 4 && transition_count <= 50 {
                // This row likely contains text
            }
        }

        if stroke_widths.is_empty() {
            return None;
        }

        // Calculate median stroke width
        stroke_widths.sort_unstable();
        let median_stroke_width = stroke_widths[stroke_widths.len() / 2] as f32;

        // Typical printed text at 300 DPI has ~1.5 pixel stroke width
        // DPI = (1.5 / stroke_width) * 300
        // But we use a more robust formula based on typical print ranges
        // At 300 DPI: 1-2 pixels
        // At 200 DPI: 1-1.3 pixels
        // At 600 DPI: 2-4 pixels
        let estimated_dpi = if median_stroke_width < 1.0 {
            600
        } else if median_stroke_width < 1.5 {
            300
        } else if median_stroke_width < 2.5 {
            200
        } else if median_stroke_width < 4.0 {
            150
        } else {
            100
        };

        // Confidence based on consistency of stroke widths
        let variance = if stroke_widths.len() > 1 {
            let mean = stroke_widths.iter().sum::<u32>() as f32 / stroke_widths.len() as f32;
            let variance = stroke_widths
                .iter()
                .map(|&sw| (sw as f32 - mean).powi(2))
                .sum::<f32>()
                / stroke_widths.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let confidence = if variance < 0.5 {
            0.8
        } else if variance < 1.0 {
            0.6
        } else {
            0.4
        };

        Some(DpiEstimate {
            dpi: estimated_dpi,
            confidence,
            method: DpiEstimationMethod::StrokeWidth,
        })
    }

    /// Estimate DPI from font size analysis
    ///
    /// Typical font sizes: 10-12pt for body text
    /// At 300 DPI, 1pt = 300/72 = 4.17 pixels
    /// So 12pt font = ~50 pixels tall
    fn estimate_dpi_from_font_size(img: &OcrImage) -> Option<DpiEstimate> {
        use image::GenericImageView;

        // Convert to grayscale if needed
        let gray = if img.format == crate::core::image::ImageFormat::Grayscale {
            img.data.to_luma8()
        } else {
            img.data.to_luma8()
        };

        let (width, height) = gray.dimensions();

        // Use ImageThresholder and Union-Find CCL to find character heights
        use crate::image::ImageThresholder;
        use crate::layout::connected_components_4connectivity;

        let mut thresholder = ImageThresholder::new();
        thresholder.set_image(img.clone()).ok()?;
        let binary = thresholder.threshold(crate::image::ThresholdMethod::Otsu).ok()?;
        let binary_gray = binary.data.to_luma8();

        let ccl_result = connected_components_4connectivity(&binary_gray);

        // Collect component heights
        let mut char_heights: Vec<u32> = ccl_result
            .bounding_boxes
            .iter()
            .skip(1) // Skip index 0 (background)
            .map(|bbox| bbox.height())
            .filter(|&h| h > 5 && h < 200)
            .collect();

        if char_heights.is_empty() {
            return None;
        }

        // Calculate median character height
        char_heights.sort_unstable();
        let median_height = char_heights[char_heights.len() / 2] as f32;

        // Typical body text is 10-12pt
        // At 300 DPI: 10pt = 42 pixels, 12pt = 50 pixels
        // So DPI = (median_height / 50) * 300 for 12pt equivalent
        // We use a range to account for different font sizes
        let estimated_dpi = ((median_height / 50.0) * 300.0).round() as u32;
        let estimated_dpi = estimated_dpi.clamp(72, 600);

        // Confidence based on consistency of character heights
        let variance = if char_heights.len() > 1 {
            let mean = char_heights.iter().sum::<u32>() as f32 / char_heights.len() as f32;
            let variance = char_heights
                .iter()
                .map(|&h| (h as f32 - mean).powi(2))
                .sum::<f32>()
                / char_heights.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let confidence = if variance < 10.0 {
            0.7
        } else if variance < 20.0 {
            0.5
        } else {
            0.3
        };

        Some(DpiEstimate {
            dpi: estimated_dpi,
            confidence,
            method: DpiEstimationMethod::FontSize,
        })
    }

    /// Assess image quality for OCR
    pub fn assess_quality(img: &OcrImage) -> ImageQualityScore {
        let stats = img.statistics();

        // Calculate quality metrics
        let contrast = Self::calculate_contrast(&stats);
        let sharpness = Self::calculate_sharpness(img);
        let noise_level = Self::calculate_noise_level(&stats);
        let resolution = Self::calculate_resolution_score(img);

        // Calculate overall quality score
        let overall_score = (contrast + sharpness + (1.0 - noise_level) + resolution) / 4.0;

        ImageQualityScore {
            overall_score,
            contrast,
            sharpness,
            noise_level,
            resolution,
            recommendations: Self::generate_recommendations(
                overall_score,
                contrast,
                sharpness,
                noise_level,
                resolution,
            ),
        }
    }

    /// Calculate contrast score
    fn calculate_contrast(stats: &crate::core::image::ImageStatistics) -> f32 {
        if stats.max == stats.min {
            0.0
        } else {
            (stats.max - stats.min) as f32 / 255.0
        }
    }

    /// Calculate sharpness score
    fn calculate_sharpness(img: &OcrImage) -> f32 {
        let gray = img.to_grayscale();
        // Laplacian kernel for edge detection
        let kernel: [i32; 9] = [0, 1, 0, 1, -4, 1, 0, 1, 0];

        if let Some(buf) = gray.data.as_luma8() {
            let filtered = filter3x3::<_, _, u8>(buf, &kernel);

            // Calculate variance of the Laplacian
            let mut sum = 0u64;
            let mut sq_sum = 0u64;
            let count = filtered.len() as u64;

            if count == 0 {
                return 0.0;
            }

            for p in filtered.pixels() {
                let val = p[0] as u64;
                sum += val;
                sq_sum += val * val;
            }

            let mean = sum as f64 / count as f64;
            let variance = (sq_sum as f64 / count as f64) - (mean * mean);

            // Normalize variance. Higher variance means sharper edges.
            // A variance of > 500 is considered sharp.
            (variance / 500.0).min(1.0) as f32
        } else {
            0.0
        }
    }

    /// Calculate noise level
    fn calculate_noise_level(stats: &crate::core::image::ImageStatistics) -> f32 {
        // Simple noise estimation based on pixel count and variance
        if stats.pixel_count == 0 {
            0.0
        } else {
            // This is a simplified noise estimation
            // In practice, you'd use more sophisticated methods
            let variance = stats.mean * (1.0 - stats.mean / 255.0);
            (variance / 255.0).min(1.0)
        }
    }

    /// Calculate resolution score
    fn calculate_resolution_score(img: &OcrImage) -> f32 {
        // Score based on image dimensions and DPI
        let pixel_count = img.width * img.height;
        let dpi_score = (img.dpi as f32 / 300.0).min(1.0);
        let size_score = (pixel_count as f32 / (1920.0 * 1080.0)).min(1.0);

        (dpi_score + size_score) / 2.0
    }

    /// Generate quality improvement recommendations
    fn generate_recommendations(
        overall_score: f32,
        contrast: f32,
        sharpness: f32,
        noise_level: f32,
        resolution: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if overall_score < 0.5 {
            recommendations.push(
                "Image quality is poor. Consider using a higher resolution image.".to_string(),
            );
        }

        if contrast < 0.3 {
            recommendations.push("Low contrast detected. Consider enhancing contrast.".to_string());
        }

        if sharpness < 0.3 {
            recommendations.push(
                "Image appears blurry. Consider sharpening or using a higher resolution image."
                    .to_string(),
            );
        }

        if noise_level > 0.7 {
            recommendations
                .push("High noise level detected. Consider applying noise reduction.".to_string());
        }

        if resolution < 0.5 {
            recommendations
                .push("Low resolution detected. Consider using a higher DPI image.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Image quality is good for OCR processing.".to_string());
        }

        recommendations
    }
}

/// Image quality score
#[derive(Debug, Clone)]
pub struct ImageQualityScore {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Contrast score (0.0 to 1.0)
    pub contrast: f32,
    /// Sharpness score (0.0 to 1.0)
    pub sharpness: f32,
    /// Noise level (0.0 to 1.0, higher is worse)
    pub noise_level: f32,
    /// Resolution score (0.0 to 1.0)
    pub resolution: f32,
    /// Quality improvement recommendations
    pub recommendations: Vec<String>,
}

impl ImageQualityScore {
    /// Check if image quality is acceptable for OCR
    pub fn is_acceptable(&self, threshold: f32) -> bool {
        self.overall_score >= threshold
    }

    /// Get quality grade
    pub fn get_grade(&self) -> QualityGrade {
        match self.overall_score {
            score if score >= 0.8 => QualityGrade::Excellent,
            score if score >= 0.6 => QualityGrade::Good,
            score if score >= 0.4 => QualityGrade::Fair,
            score if score >= 0.2 => QualityGrade::Poor,
            _ => QualityGrade::VeryPoor,
        }
    }
}

/// Quality grade enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    /// Excellent quality
    Excellent,
    /// Good quality
    Good,
    /// Fair quality
    Fair,
    /// Poor quality
    Poor,
    /// Very poor quality
    VeryPoor,
}
