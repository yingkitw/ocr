//! Text line feature extraction
//!
//! This module implements Tesseract-style text line analysis including:
//! - Baseline estimation (fitting a polynomial to the bottom of characters)
//! - X-height estimation (height of lowercase letters like 'x', 'a', 'c', 'e')
//! - Ascender/descender detection
//! - Line height estimation

use crate::core::text::BoundingBox;
use image::GrayImage;
#[cfg(test)]
use image::Luma;

/// Features extracted from a text line
#[derive(Debug, Clone)]
pub struct TextLineFeatures {
    /// Estimated baseline polynomial coefficients
    /// For a horizontal line: [slope, offset]
    /// For a quadratic: [a, b, c] where y = ax² + bx + c
    pub baseline: BaselineEstimate,
    /// Estimated x-height (height of lowercase letters like 'x')
    pub x_height: f32,
    /// Estimated ascender height (height above x-height for letters like 'l', 'h', 'k')
    pub ascender_height: f32,
    /// Estimated descender depth (depth below baseline for letters like 'g', 'j', 'p', 'q')
    pub descender_depth: f32,
    /// Estimated cap height (height of uppercase letters)
    pub cap_height: f32,
    /// Overall line height
    pub line_height: f32,
    /// Confidence in the estimates (0.0 to 1.0)
    pub confidence: f32,
}

/// Baseline estimate for a text line
#[derive(Debug, Clone)]
pub enum BaselineEstimate {
    /// Horizontal line (most common for printed text)
    Horizontal { y: f32 },
    /// Linear (slanted) line
    Linear { slope: f32, intercept: f32 },
    /// Quadratic curve (for curved text)
    Quadratic { a: f32, b: f32, c: f32 },
}

impl BaselineEstimate {
    /// Get the baseline y-coordinate at a given x position
    pub fn y_at_x(&self, x: f32) -> f32 {
        match self {
            BaselineEstimate::Horizontal { y } => *y,
            BaselineEstimate::Linear { slope, intercept } => slope * x + intercept,
            BaselineEstimate::Quadratic { a, b, c } => a * x * x + b * x + c,
        }
    }

    /// Check if the baseline is roughly horizontal
    pub fn is_horizontal(&self, tolerance: f32) -> bool {
        match self {
            BaselineEstimate::Horizontal { .. } => true,
            BaselineEstimate::Linear { slope, .. } => slope.abs() < tolerance,
            BaselineEstimate::Quadratic { a, b, .. } => a.abs() < tolerance && b.abs() < tolerance,
        }
    }
}

/// Estimate baseline for a text line using bottom pixel analysis
///
/// This implements Tesseract's approach of finding the bottom edge of
/// characters and fitting a curve to them.
pub fn estimate_baseline(binary_image: &GrayImage, line_bbox: &BoundingBox) -> BaselineEstimate {
    let width = line_bbox.width() as usize;
    let height = line_bbox.height() as usize;
    let left = line_bbox.left as usize;
    let top = line_bbox.top as usize;

    if width < 3 || height < 3 {
        // Too small - return horizontal baseline at bottom
        return BaselineEstimate::Horizontal {
            y: (line_bbox.bottom) as f32,
        };
    }

    // Collect bottom pixels for each column
    let mut bottom_y: Vec<Option<f32>> = vec![None; width];
    let mut has_content = false;

    for x in 0..width {
        let img_x = left + x;
        // Scan from bottom to top to find the first dark pixel
        for y_rel in (0..height).rev() {
            let img_y = top + y_rel;
            if img_x >= binary_image.width() as usize || img_y >= binary_image.height() as usize {
                break;
            }

            let pixel = binary_image.get_pixel(img_x as u32, img_y as u32);
            if pixel[0] < 128 {
                // Found bottom of a character
                bottom_y[x] = Some((img_y) as f32);
                has_content = true;
                break;
            }
        }
    }

    if !has_content {
        return BaselineEstimate::Horizontal {
            y: (line_bbox.bottom) as f32,
        };
    }

    // Filter out outliers (likely noise or ascenders/descenders)
    let valid_points: Vec<(f32, f32)> = bottom_y
        .iter()
        .enumerate()
        .filter_map(|(x, &y_opt)| y_opt.map(|y| (x as f32 + left as f32, y)))
        .collect();

    if valid_points.is_empty() {
        return BaselineEstimate::Horizontal {
            y: (line_bbox.bottom) as f32,
        };
    }

    // Calculate mean and standard deviation of y values
    let mean_y = valid_points.iter().map(|&(_, y)| y).sum::<f32>() / valid_points.len() as f32;
    let variance = valid_points
        .iter()
        .map(|&(_, y)| (y - mean_y).powi(2))
        .sum::<f32>()
        / valid_points.len() as f32;
    let std_dev = variance.sqrt();

    // Filter points within 2 standard deviations
    let filtered_points: Vec<(f32, f32)> = valid_points
        .into_iter()
        .filter(|&(_, y)| (y - mean_y).abs() < 2.0 * std_dev)
        .collect();

    if filtered_points.len() < 3 {
        // Not enough points - return horizontal baseline
        return BaselineEstimate::Horizontal { y: mean_y.round() };
    }

    // Try fitting linear baseline
    let (slope, intercept) = fit_linear_baseline(&filtered_points);

    // Calculate residual error
    let residual_error = filtered_points
        .iter()
        .map(|&(x, y)| {
            let predicted = slope * x + intercept;
            (y - predicted).abs()
        })
        .sum::<f32>()
        / filtered_points.len() as f32;

    // If residual is low enough, use linear; otherwise use horizontal
    if residual_error < 2.0 && slope.abs() < 0.1 {
        BaselineEstimate::Linear { slope, intercept }
    } else {
        BaselineEstimate::Horizontal { y: mean_y.round() }
    }
}

/// Fit a linear baseline using least squares
fn fit_linear_baseline(points: &[(f32, f32)]) -> (f32, f32) {
    if points.len() < 2 {
        return (0.0, 0.0);
    }

    let n = points.len() as f32;
    let sum_x: f32 = points.iter().map(|&(x, _)| x).sum();
    let sum_y: f32 = points.iter().map(|&(_, y)| y).sum();
    let sum_xx: f32 = points.iter().map(|&(x, _)| x * x).sum();
    let sum_xy: f32 = points.iter().map(|&(x, y)| x * y).sum();

    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < 1e-6 {
        return (0.0, sum_y / n);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

/// Estimate x-height for a text line
///
/// X-height is the height of lowercase letters without ascenders or descenders,
/// such as 'x', 'a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'z'.
/// This is a key metric for Tesseract's character normalization.
pub fn estimate_x_height(
    binary_image: &GrayImage,
    line_bbox: &BoundingBox,
    _baseline: &BaselineEstimate,
) -> (f32, f32) {
    let left = line_bbox.left as usize;
    let top = line_bbox.top as usize;
    let width = line_bbox.width() as usize;
    let height = line_bbox.height() as usize;

    if width < 5 || height < 5 {
        return (height as f32, 0.3); // Default to line height
    }

    // Collect heights of character-like components
    let mut char_heights = Vec::new();

    // Scan horizontally for vertical dark runs
    for x in 0..width {
        let img_x = left + x;
        if img_x >= binary_image.width() as usize {
            continue;
        }

        let mut in_run = false;
        let mut run_top = 0;

        for y_rel in 0..height {
            let img_y = top + y_rel;
            if img_y >= binary_image.height() as usize {
                break;
            }

            let pixel = binary_image.get_pixel(img_x as u32, img_y as u32);
            let is_dark = pixel[0] < 128;

            match (in_run, is_dark) {
                (false, true) => {
                    in_run = true;
                    run_top = img_y;
                }
                (true, false) => {
                    let run_height = img_y.saturating_sub(run_top);
                    if run_height > 2 && run_height < height / 2 {
                        char_heights.push(run_height as f32);
                    }
                    in_run = false;
                }
                _ => {}
            }
        }
    }

    if char_heights.is_empty() {
        return (height as f32 * 0.5, 0.2);
    }

    // Use median to avoid outliers
    char_heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_height = char_heights[char_heights.len() / 2];

    // Estimate x-height as ~60% of median character height
    // This is because typical text has ascenders (like 'h', 'l') that are taller
    let x_height = median_height * 0.6;

    // Confidence based on consistency
    let variance = if char_heights.len() > 1 {
        let mean = char_heights.iter().sum::<f32>() / char_heights.len() as f32;
        let variance = char_heights
            .iter()
            .map(|&h| (h - mean).powi(2))
            .sum::<f32>()
            / char_heights.len() as f32;
        variance.sqrt()
    } else {
        0.0
    };

    let confidence = if variance < median_height * 0.2 {
        0.8
    } else if variance < median_height * 0.4 {
        0.5
    } else {
        0.3
    };

    (x_height.max(1.0), confidence)
}

/// Estimate ascender and descender heights
///
/// Ascenders are parts of lowercase letters that rise above x-height (h, k, l, b, d, f)
/// Descenders are parts that fall below the baseline (g, j, p, q, y)
pub fn estimate_ascender_descender(
    binary_image: &GrayImage,
    line_bbox: &BoundingBox,
    baseline: &BaselineEstimate,
    x_height: f32,
) -> (f32, f32) {
    let _left = line_bbox.left as usize;
    let _top = line_bbox.top as usize;
    let _height = line_bbox.height() as usize;

    // Get baseline y-position at the center of the line
    let center_x = (line_bbox.left + line_bbox.right) as f32 / 2.0;
    let baseline_y = baseline.y_at_x(center_x);

    // Find the top of the text (excluding outliers)
    let mut top_y = baseline_y as u32;
    let mut bottom_y = baseline_y as u32;

    for x in line_bbox.left..line_bbox.right {
        if x >= binary_image.width() {
            break;
        }

        // Find top of text at this column
        for y in line_bbox.top..line_bbox.bottom {
            if y >= binary_image.height() {
                break;
            }

            let pixel = binary_image.get_pixel(x, y);
            if pixel[0] < 128 {
                if y < top_y {
                    top_y = y;
                }
                break;
            }
        }

        // Find bottom of text at this column
        for y in (line_bbox.top..line_bbox.bottom).rev() {
            if y >= binary_image.height() {
                continue;
            }

            let pixel = binary_image.get_pixel(x, y);
            if pixel[0] < 128 {
                if y > bottom_y {
                    bottom_y = y;
                }
                break;
            }
        }
    }

    // Estimate ascender height (distance from top to baseline - x-height)
    let ascender_zone = (baseline_y - top_y as f32) - x_height;
    let ascender_height = ascender_zone.max(0.0).min(x_height * 0.8);

    // Estimate descender depth (distance from baseline to bottom)
    let descender_depth = (bottom_y as f32 - baseline_y).max(0.0).min(x_height * 0.5);

    (ascender_height, descender_depth)
}

/// Estimate cap height (height of uppercase letters)
pub fn estimate_cap_height(
    _binary_image: &GrayImage,
    _line_bbox: &BoundingBox,
    _baseline: &BaselineEstimate,
    x_height: f32,
    ascender_height: f32,
) -> f32 {
    // Cap height is typically between x-height and x-height + ascender
    // For most fonts: cap_height ≈ x_height + (ascender_height * 0.7)
    x_height + ascender_height * 0.7
}

/// Extract all text line features
pub fn extract_text_line_features(
    binary_image: &GrayImage,
    line_bbox: &BoundingBox,
) -> TextLineFeatures {
    let baseline = estimate_baseline(binary_image, line_bbox);
    let (x_height, x_height_conf) = estimate_x_height(binary_image, line_bbox, &baseline);
    let (ascender_height, descender_depth) =
        estimate_ascender_descender(binary_image, line_bbox, &baseline, x_height);
    let cap_height = estimate_cap_height(
        binary_image,
        line_bbox,
        &baseline,
        x_height,
        ascender_height,
    );

    let line_height = x_height + ascender_height + descender_depth;

    let confidence = x_height_conf;

    TextLineFeatures {
        baseline,
        x_height,
        ascender_height,
        descender_depth,
        cap_height,
        line_height,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn create_test_text_line() -> (GrayImage, BoundingBox) {
        // Create a white image (background = 255)
        let mut img = GrayImage::from_pixel(200, 50, Luma([255u8]));

        // Draw some "characters" (simple rectangles with dark pixels = 0)
        // Row of text from y=20 to y=30 (exclusive, so last dark row is y=29)
        for x in 10..30 {
            for y in 20..30 {
                img.put_pixel(x, y, Luma([0u8]));
            }
        }
        for x in 35..55 {
            for y in 20..30 {
                img.put_pixel(x, y, Luma([0u8]));
            }
        }
        for x in 60..80 {
            for y in 20..30 {
                img.put_pixel(x, y, Luma([0u8]));
            }
        }

        (img, BoundingBox::new(0, 0, 200, 50))
    }

    #[test]
    fn test_estimate_baseline() {
        let (img, bbox) = create_test_text_line();
        let baseline = estimate_baseline(&img, &bbox);

        // Baseline should be around y=29-30 (bottom of the text)
        // The baseline estimation finds the bottom edge of characters
        let y_at_center = baseline.y_at_x(100.0);
        // Allow a wider range since we're sampling from the center (x=100) where there's no text
        // The algorithm should extrapolate from the text regions
        assert!(
            y_at_center >= 20.0 && y_at_center <= 50.0,
            "Baseline at center {} is not in expected range [20.0, 50.0]",
            y_at_center
        );

        // Check that baseline is more reasonable at the actual text positions
        let y_at_text = baseline.y_at_x(20.0);
        assert!(
            y_at_text >= 25.0 && y_at_text <= 35.0,
            "Baseline at text position {} is not in expected range [25.0, 35.0]",
            y_at_text
        );
    }

    #[test]
    fn test_baseline_is_horizontal() {
        let (img, bbox) = create_test_text_line();
        let baseline = estimate_baseline(&img, &bbox);

        assert!(baseline.is_horizontal(0.05));
    }

    #[test]
    fn test_estimate_x_height() {
        let (img, bbox) = create_test_text_line();
        let baseline = estimate_baseline(&img, &bbox);
        let (x_height, _conf) = estimate_x_height(&img, &bbox, &baseline);

        // X-height should be around 10 (height of our test characters)
        assert!(x_height > 5.0 && x_height < 20.0);
    }

    #[test]
    fn test_extract_text_line_features() {
        let (img, bbox) = create_test_text_line();
        let features = extract_text_line_features(&img, &bbox);

        assert!(features.x_height > 0.0);
        assert!(features.line_height > 0.0);
        assert!(features.confidence > 0.0);
    }

    #[test]
    fn test_baseline_y_at_x() {
        let baseline = BaselineEstimate::Horizontal { y: 42.0 };
        assert_eq!(baseline.y_at_x(0.0), 42.0);
        assert_eq!(baseline.y_at_x(100.0), 42.0);

        let baseline = BaselineEstimate::Linear {
            slope: 0.5,
            intercept: 10.0,
        };
        assert_eq!(baseline.y_at_x(0.0), 10.0);
        assert_eq!(baseline.y_at_x(10.0), 15.0);
    }
}
