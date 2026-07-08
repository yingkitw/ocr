//! Font attribute detection
//!
//! Detects visual font attributes (bold, italic, monospace) from text region images
//! using stroke analysis, slant measurement, and character spacing consistency.

use image::{DynamicImage, GrayImage};
#[cfg(test)]
use image::Luma;

/// Detected font attributes for a text region
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontAttributes {
    pub bold: bool,
    pub italic: bool,
    pub monospace: bool,
}

impl Default for FontAttributes {
    fn default() -> Self {
        Self {
            bold: false,
            italic: false,
            monospace: false,
        }
    }
}

/// Analyzes text region images to detect font attributes
pub struct FontAttributeDetector {
    /// Stroke width threshold for bold detection (> 1.5x median = bold)
    pub bold_threshold: f32,
    /// Slant angle threshold for italic detection (degrees)
    pub italic_threshold: f32,
    /// Width variance threshold for monospace detection (CV < 0.15 = monospace)
    pub monospace_cv_threshold: f32,
}

impl Default for FontAttributeDetector {
    fn default() -> Self {
        Self {
            bold_threshold: 1.5,
            italic_threshold: 8.0,
            monospace_cv_threshold: 0.15,
        }
    }
}

impl FontAttributeDetector {
    /// Detect font attributes from a text region image
    pub fn detect(&self, image: &DynamicImage) -> FontAttributes {
        let gray = image.to_luma8();
        FontAttributes {
            bold: self.detect_bold(&gray),
            italic: self.detect_italic(&gray),
            monospace: self.detect_monospace(&gray),
        }
    }

    /// Detect bold by measuring average stroke thickness
    fn detect_bold(&self, gray: &GrayImage) -> bool {
        let thickness = self.estimate_stroke_thickness(gray);
        // Normal text stroke is ~1-2px; bold is ~3-4px+
        thickness >= self.bold_threshold
    }

    /// Estimate average stroke thickness by median run-length of dark pixels
    /// Measures both horizontal and vertical runs, takes the median of the shorter runs
    fn estimate_stroke_thickness(&self, gray: &GrayImage) -> f32 {
        let (w, h) = (gray.width() as usize, gray.height() as usize);
        if w == 0 || h == 0 {
            return 1.0;
        }

        let threshold = 128u8;
        let mut run_lengths = Vec::new();

        // Horizontal runs (per row)
        for y in 0..h {
            let mut run_len = 0usize;
            for x in 0..w {
                if gray.get_pixel(x as u32, y as u32).0[0] < threshold {
                    run_len += 1;
                } else if run_len > 0 {
                    if run_len < w / 2 {
                        run_lengths.push(run_len);
                    }
                    run_len = 0;
                }
            }
            if run_len > 0 && run_len < w / 2 {
                run_lengths.push(run_len);
            }
        }

        // Vertical runs (per column)
        for x in 0..w {
            let mut run_len = 0usize;
            for y in 0..h {
                if gray.get_pixel(x as u32, y as u32).0[0] < threshold {
                    run_len += 1;
                } else if run_len > 0 {
                    if run_len < h / 2 {
                        run_lengths.push(run_len);
                    }
                    run_len = 0;
                }
            }
            if run_len > 0 && run_len < h / 2 {
                run_lengths.push(run_len);
            }
        }

        if run_lengths.is_empty() {
            return 1.0;
        }

        run_lengths.sort_unstable();
        let median = if run_lengths.len() % 2 == 1 {
            run_lengths[run_lengths.len() / 2] as f32
        } else {
            let mid = run_lengths.len() / 2;
            (run_lengths[mid - 1] + run_lengths[mid]) as f32 / 2.0
        };

        median.max(1.0)
    }

    /// Detect italic by measuring slant via projection profiles at different angles
    fn detect_italic(&self, gray: &GrayImage) -> bool {
        let slant = self.estimate_slant(gray);
        slant.abs() >= self.italic_threshold
    }

    /// Estimate text slant by finding the shear angle that maximizes horizontal projection variance
    fn estimate_slant(&self, gray: &GrayImage) -> f32 {
        let (w, h) = (gray.width() as i32, gray.height() as i32);
        if w < 3 || h < 3 {
            return 0.0;
        }

        let threshold = 128u8;
        let mut best_slant = 0.0f32;
        let mut best_variance = 0.0f32;

        for slant_deg in (-20..=20).step_by(2) {
            let slant = slant_deg as f32;
            let variance = self.slant_projection_variance(gray, slant, threshold);
            // Require meaningful variance to avoid noise picking a random angle
            if variance > best_variance && variance > 1.0 {
                best_variance = variance;
                best_slant = slant;
            }
        }

        best_slant
    }

    fn slant_projection_variance(&self, gray: &GrayImage, slant_deg: f32, threshold: u8) -> f32 {
        let (w, h) = (gray.width() as i32, gray.height() as i32);
        let tan_slant = slant_deg.to_radians().tan();
        let mut projection = vec![0i32; w as usize];

        for y in 0..h {
            for x in 0..w {
                let shifted_x = (x as f32 + (y as f32 - h as f32 / 2.0) * tan_slant) as i32;
                if shifted_x >= 0 && shifted_x < w {
                    let px = gray.get_pixel(x as u32, y as u32).0[0];
                    if px < threshold {
                        projection[shifted_x as usize] += 1;
                    }
                }
            }
        }

        if projection.is_empty() {
            return 0.0;
        }

        let mean = projection.iter().sum::<i32>() as f32 / projection.len() as f32;
        let variance = projection.iter().map(|&v| {
            let d = v as f32 - mean;
            d * d
        }).sum::<f32>() / projection.len() as f32;

        variance
    }

    /// Detect monospace by measuring character width consistency
    fn detect_monospace(&self, gray: &GrayImage) -> bool {
        let widths = self.estimate_character_widths(gray);
        if widths.len() < 2 {
            return false;
        }

        let mean = widths.iter().sum::<f32>() / widths.len() as f32;
        if mean <= 0.0 {
            return false;
        }

        let variance = widths.iter().map(|&w| {
            let d = w - mean;
            d * d
        }).sum::<f32>() / widths.len() as f32;

        let std_dev = variance.sqrt();
        let cv = std_dev / mean;

        cv < self.monospace_cv_threshold
    }

    /// Estimate character widths by analyzing vertical projection profile peaks
    fn estimate_character_widths(&self, gray: &GrayImage) -> Vec<f32> {
        let (w, h) = (gray.width() as usize, gray.height() as usize);
        if w < 2 || h < 2 {
            return vec![];
        }

        let threshold = 128u8;
        // Vertical projection: sum of dark pixels per column
        let mut projection = vec![0usize; w];
        for y in 0..h {
            for x in 0..w {
                if gray.get_pixel(x as u32, y as u32).0[0] < threshold {
                    projection[x] += 1;
                }
            }
        }

        // Find connected groups of columns with non-zero projection (character regions)
        let mut widths = Vec::new();
        let mut in_char = false;
        let mut start = 0usize;
        let min_gap = 2usize; // minimum gap between characters

        for x in 0..w {
            let has_ink = projection[x] > 0;
            if has_ink && !in_char {
                start = x;
                in_char = true;
            } else if !has_ink && in_char {
                let gap_start = x;
                // Look ahead to see if this gap is wide enough
                let mut gap_end = gap_start;
                while gap_end < w && projection[gap_end] == 0 {
                    gap_end += 1;
                }
                let gap_len = gap_end - gap_start;
                if gap_len >= min_gap || gap_end >= w {
                    widths.push((gap_start - start) as f32);
                    in_char = false;
                }
            }
        }

        if in_char {
            widths.push((w - start) as f32);
        }

        widths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_attributes_default() {
        let attrs = FontAttributes::default();
        assert!(!attrs.bold);
        assert!(!attrs.italic);
        assert!(!attrs.monospace);
    }

    #[test]
    fn test_detector_default_thresholds() {
        let det = FontAttributeDetector::default();
        assert_eq!(det.bold_threshold, 1.5);
        assert_eq!(det.italic_threshold, 8.0);
        assert_eq!(det.monospace_cv_threshold, 0.15);
    }

    #[test]
    fn test_empty_image() {
        let img = GrayImage::from_pixel(20, 10, Luma([255]));
        let detector = FontAttributeDetector::default();
        let attrs = detector.detect(&DynamicImage::ImageLuma8(img));
        assert!(!attrs.bold);
        assert!(!attrs.italic);
        assert!(!attrs.monospace);
    }

    #[test]
    fn test_stroke_thickness_computation() {
        // 3px thick vertical strokes
        let mut img = GrayImage::from_pixel(40, 10, Luma([255]));
        for x in 5..8 {
            for y in 0..10 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for x in 15..18 {
            for y in 0..10 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let detector = FontAttributeDetector::default();
        let thickness = detector.estimate_stroke_thickness(&img);
        assert!(thickness >= 2.0, "Expected thickness >= 2, got {}", thickness);
    }

    #[test]
    fn test_slant_estimation_upright() {
        let mut img = GrayImage::from_pixel(50, 10, Luma([255]));
        for x in 10..15 {
            for y in 1..9 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let detector = FontAttributeDetector::default();
        let slant = detector.estimate_slant(&img);
        assert!(slant.abs() < 5.0, "Expected near-zero slant, got {}", slant);
    }

    #[test]
    fn test_slant_estimation_slanted() {
        let mut img = GrayImage::from_pixel(60, 10, Luma([255]));
        // Draw a slanted bar: x increases by ~1 per y
        for y in 1..9 {
            let x = 10 + y;
            for dx in 0..3 {
                img.put_pixel(x + dx, y, Luma([0]));
            }
        }
        let detector = FontAttributeDetector::default();
        let slant = detector.estimate_slant(&img);
        assert!(slant.abs() >= 5.0, "Expected meaningful slant, got {}", slant);
    }

    #[test]
    fn test_monospace_widths_uniform() {
        let mut img = GrayImage::from_pixel(60, 10, Luma([255]));
        // 4 chars of width 8, spacing 4
        for c in 0..4 {
            let cx = 4 + c * 12;
            for dx in 0..8 {
                for y in 1..9 {
                    img.put_pixel(cx + dx as u32, y, Luma([0]));
                }
            }
        }
        let detector = FontAttributeDetector::default();
        let widths = detector.estimate_character_widths(&img);
        assert!(!widths.is_empty());
        // Should find mostly uniform widths
        let cv = {
            let mean = widths.iter().sum::<f32>() / widths.len() as f32;
            let var = widths.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / widths.len() as f32;
            var.sqrt() / mean
        };
        assert!(cv < 0.3, "Expected low CV for uniform widths, got {} with widths {:?}", cv, widths);
    }

    #[test]
    fn test_monospace_widths_variable() {
        let mut img = GrayImage::from_pixel(60, 10, Luma([255]));
        // Variable-width chars
        let widths = [14u32, 5, 16, 4];
        let spacing = 4u32;
        let mut x = spacing;
        for &cw in &widths {
            for dx in 0..cw {
                for y in 1..9 {
                    img.put_pixel(x + dx, y, Luma([0]));
                }
            }
            x += cw + spacing;
            if x >= img.width() {
                break;
            }
        }
        let detector = FontAttributeDetector::default();
        let found_widths = detector.estimate_character_widths(&img);
        assert!(!found_widths.is_empty());
        let cv = {
            let mean = found_widths.iter().sum::<f32>() / found_widths.len() as f32;
            let var = found_widths.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / found_widths.len() as f32;
            var.sqrt() / mean
        };
        assert!(cv > 0.2, "Expected high CV for variable widths, got {} with widths {:?}", cv, found_widths);
    }

    #[test]
    fn test_detector_does_not_panic_on_noise() {
        let mut img = GrayImage::from_pixel(30, 10, Luma([255]));
        // Random noise-like pattern
        for x in 0..30 {
            for y in 0..10 {
                if (x + y) % 3 == 0 {
                    img.put_pixel(x, y, Luma([0]));
                }
            }
        }
        let detector = FontAttributeDetector::default();
        let _ = detector.detect(&DynamicImage::ImageLuma8(img));
    }
}
