//! Orientation and Script Detection (OSD)
//!
//! Detects text orientation (0°, 90°, 180°, 270°) and dominant script
//! in an image before OCR recognition, similar to Tesseract's OSD.

use crate::core::image::OcrImage;
use crate::utils::Result;

/// Detected text orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextOrientation {
    /// Normal orientation (0°)
    Normal,
    /// Rotated 90° clockwise
    Rotated90,
    /// Rotated 180°
    Rotated180,
    /// Rotated 270° clockwise (90° counter-clockwise)
    Rotated270,
}

impl TextOrientation {
    /// Get the rotation angle in degrees
    pub fn angle_degrees(&self) -> u16 {
        match self {
            TextOrientation::Normal => 0,
            TextOrientation::Rotated90 => 90,
            TextOrientation::Rotated180 => 180,
            TextOrientation::Rotated270 => 270,
        }
    }

    /// Convert to a rotation description
    pub fn description(&self) -> &'static str {
        match self {
            TextOrientation::Normal => "Normal (0°)",
            TextOrientation::Rotated90 => "Rotated 90° clockwise",
            TextOrientation::Rotated180 => "Rotated 180°",
            TextOrientation::Rotated270 => "Rotated 270° clockwise (90° counter-clockwise)",
        }
    }
}

/// Detected script type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectedScript {
    Latin,
    Cjk,
    Arabic,
    Cyrillic,
    Devanagari,
    Unknown,
}

impl DetectedScript {
    pub fn name(&self) -> &'static str {
        match self {
            DetectedScript::Latin => "Latin",
            DetectedScript::Cjk => "CJK",
            DetectedScript::Arabic => "Arabic",
            DetectedScript::Cyrillic => "Cyrillic",
            DetectedScript::Devanagari => "Devanagari",
            DetectedScript::Unknown => "Unknown",
        }
    }
}

/// OSD result containing orientation and script information
#[derive(Debug, Clone)]
pub struct OsdResult {
    /// Detected text orientation
    pub orientation: TextOrientation,
    /// Confidence in orientation detection (0.0 to 1.0)
    pub orientation_confidence: f32,
    /// Detected dominant script
    pub script: DetectedScript,
    /// Confidence in script detection (0.0 to 1.0)
    pub script_confidence: f32,
    /// Secondary scripts detected (if any)
    pub secondary_scripts: Vec<(DetectedScript, f32)>,
}

impl OsdResult {
    /// Create a new OSD result with default values
    pub fn new() -> Self {
        Self {
            orientation: TextOrientation::Normal,
            orientation_confidence: 0.0,
            script: DetectedScript::Latin,
            script_confidence: 0.0,
            secondary_scripts: Vec::new(),
        }
    }

    /// Check if the image likely needs rotation
    pub fn needs_rotation(&self) -> bool {
        self.orientation != TextOrientation::Normal
    }
}

impl Default for OsdResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Orientation and Script Detector
pub struct OsdDetector;

impl OsdDetector {
    /// Create a new OSD detector
    pub fn new() -> Self {
        Self
    }

    /// Detect orientation and script from an image
    ///
    /// Uses a heuristic-based approach:
    /// 1. Analyze pixel projection profiles along X and Y axes
    /// 2. Strong horizontal text lines create peaks in the Y projection
    /// 3. If Y projection has stronger periodicity than X, text is horizontal (0° or 180°)
    /// 4. If X projection has stronger periodicity, text is vertical (90° or 270°)
    /// 5. For script detection, analyze Unicode ranges of connected components
    pub fn detect(image: &OcrImage) -> Result<OsdResult> {
        let mut result = OsdResult::new();

        // Get image dimensions and pixel data
        let width = image.width;
        let height = image.height;
        let gray = image.data.to_luma8();
        let pixels = gray.as_raw();

        if width == 0 || height == 0 || pixels.is_empty() {
            return Ok(result);
        }

        // Compute horizontal and vertical projections
        let (y_projection, x_projection) = Self::compute_projections(pixels, width, height);

        // Detect orientation from projection profiles
        let (orientation, orientation_conf) = Self::detect_orientation_from_projections(
            &y_projection,
            &x_projection,
            width,
            height,
        );
        result.orientation = orientation;
        result.orientation_confidence = orientation_conf;

        // Detect script from pixel analysis
        let (script, script_conf) = Self::detect_script_from_pixels(pixels, width, height);
        result.script = script;
        result.script_confidence = script_conf;

        Ok(result)
    }

    /// Compute horizontal (Y) and vertical (X) pixel projections
    fn compute_projections(
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        let w = width as usize;
        let h = height as usize;

        let mut y_proj = vec![0u32; h];
        let mut x_proj = vec![0u32; w];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if idx < pixels.len() {
                    // Invert: dark pixels (text) have higher values
                    let val = 255 - pixels[idx];
                    if val > 128 {
                        y_proj[y] += val as u32;
                        x_proj[x] += val as u32;
                    }
                }
            }
        }

        (y_proj, x_proj)
    }

    /// Detect orientation by comparing projection periodicities
    fn detect_orientation_from_projections(
        y_proj: &[u32],
        x_proj: &[u32],
        _width: u32,
        _height: u32,
    ) -> (TextOrientation, f32) {
        // Compute autocorrelation to find periodicity
        let y_periodicity = Self::compute_periodicity(y_proj);
        let x_periodicity = Self::compute_periodicity(x_proj);

        // Text lines create periodic patterns in Y projection
        // If Y projection is more periodic, text is horizontal (0° or 180°)
        // If X projection is more periodic, text is vertical (90° or 270°)
        let is_horizontal = y_periodicity >= x_periodicity;

        if is_horizontal {
            // Distinguish 0° vs 180° by analyzing baseline position
            // For Latin text, descenders (g, j, p, q, y) create asymmetry
            let orientation = Self::detect_horizontal_flip(y_proj);
            let confidence = (y_periodicity - x_periodicity).abs().min(1.0);
            (orientation, confidence)
        } else {
            // Distinguish 90° vs 270°
            let orientation = Self::detect_vertical_flip(x_proj);
            let confidence = (x_periodicity - y_periodicity).abs().min(1.0);
            (orientation, confidence)
        }
    }

    /// Compute periodicity score from a projection profile (0.0 to 1.0)
    fn compute_periodicity(projection: &[u32]) -> f32 {
        if projection.len() < 4 {
            return 0.0;
        }

        // Find peaks in the projection
        let mut peaks = Vec::new();
        for i in 1..projection.len() - 1 {
            if projection[i] > projection[i - 1] && projection[i] > projection[i + 1] {
                peaks.push(i);
            }
        }

        if peaks.len() < 2 {
            return 0.0;
        }

        // Measure consistency of peak spacing
        let mut spacings = Vec::new();
        for window in peaks.windows(2) {
            spacings.push((window[1] - window[0]) as f32);
        }

        if spacings.is_empty() {
            return 0.0;
        }

        let mean = spacings.iter().sum::<f32>() / spacings.len() as f32;
        if mean == 0.0 {
            return 0.0;
        }

        let variance = spacings
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>()
            / spacings.len() as f32;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;

        // High periodicity = low variation in spacing
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Detect if horizontal text is flipped (0° vs 180°)
    fn detect_horizontal_flip(y_proj: &[u32]) -> TextOrientation {
        if y_proj.is_empty() {
            return TextOrientation::Normal;
        }

        let mid = y_proj.len() / 2;
        let top_sum: u32 = y_proj[..mid].iter().sum();
        let bottom_sum: u32 = y_proj[mid..].iter().sum();

        // In normal orientation, text baselines tend to be in the lower part
        // For simple heuristic: if bottom half has more mass, likely normal
        if top_sum > bottom_sum * 2 {
            // More mass in top half suggests upside-down
            TextOrientation::Rotated180
        } else {
            TextOrientation::Normal
        }
    }

    /// Detect if vertical text is flipped (90° vs 270°)
    fn detect_vertical_flip(x_proj: &[u32]) -> TextOrientation {
        if x_proj.is_empty() {
            return TextOrientation::Rotated90;
        }

        let mid = x_proj.len() / 2;
        let left_sum: u32 = x_proj[..mid].iter().sum();
        let right_sum: u32 = x_proj[mid..].iter().sum();

        // Heuristic: if left side is heavier, might be 270° (RTL vertical)
        if left_sum > right_sum * 2 {
            TextOrientation::Rotated270
        } else {
            TextOrientation::Rotated90
        }
    }

    /// Detect dominant script from image pixels (simplified heuristic)
    fn detect_script_from_pixels(
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> (DetectedScript, f32) {
        let w = width as usize;
        let h = height as usize;

        // Count connected component characteristics
        let mut horizontal_runs = 0u32;
        let mut vertical_runs = 0u32;
        let mut dense_regions = 0u32;

        for y in 0..h {
            let mut in_run = false;
            for x in 0..w {
                let idx = y * w + x;
                if idx < pixels.len() && pixels[idx] < 128 {
                    if !in_run {
                        horizontal_runs += 1;
                        in_run = true;
                    }
                } else {
                    in_run = false;
                }
            }
        }

        for x in 0..w {
            let mut in_run = false;
            for y in 0..h {
                let idx = y * w + x;
                if idx < pixels.len() && pixels[idx] < 128 {
                    if !in_run {
                        vertical_runs += 1;
                        in_run = true;
                    }
                } else {
                    in_run = false;
                }
            }
        }

        // CJK scripts tend to have more square/dense characters with similar H/V complexity
        // Latin has more horizontal complexity (many short horizontal strokes)
        // Arabic has more connected horizontal runs
        let total_pixels = (w * h) as f32;
        let coverage = if total_pixels > 0.0 {
            pixels.iter().filter(|&&p| p < 128).count() as f32 / total_pixels
        } else {
            0.0
        };

        // Heuristic: CJK text has ~2-5% coverage and balanced H/V runs
        // Latin text has ~1-3% coverage and more horizontal runs
        let run_ratio = if vertical_runs > 0 {
            horizontal_runs as f32 / vertical_runs as f32
        } else {
            10.0
        };

        if coverage > 0.03 && run_ratio > 0.5 && run_ratio < 2.0 {
            // Dense, balanced -> likely CJK
            (DetectedScript::Cjk, 0.6)
        } else if run_ratio > 3.0 {
            // Strong horizontal dominance -> Latin-like
            (DetectedScript::Latin, 0.7)
        } else {
            (DetectedScript::Unknown, 0.3)
        }
    }
}

impl Default for OsdDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply rotation to an image based on OSD result
pub fn apply_rotation(image: &OcrImage, osd: &OsdResult) -> Result<OcrImage> {
    match osd.orientation {
        TextOrientation::Normal => Ok(image.clone()),
        TextOrientation::Rotated90 => {
            let rotated = image.data.rotate90();
            Ok(OcrImage::new(rotated, image.dpi))
        }
        TextOrientation::Rotated180 => {
            let rotated = image.data.rotate180();
            Ok(OcrImage::new(rotated, image.dpi))
        }
        TextOrientation::Rotated270 => {
            let rotated = image.data.rotate270();
            Ok(OcrImage::new(rotated, image.dpi))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_orientation_angles() {
        assert_eq!(TextOrientation::Normal.angle_degrees(), 0);
        assert_eq!(TextOrientation::Rotated90.angle_degrees(), 90);
        assert_eq!(TextOrientation::Rotated180.angle_degrees(), 180);
        assert_eq!(TextOrientation::Rotated270.angle_degrees(), 270);
    }

    #[test]
    fn test_osd_result_default() {
        let osd = OsdResult::new();
        assert_eq!(osd.orientation, TextOrientation::Normal);
        assert!(!osd.needs_rotation());
    }

    #[test]
    fn test_osd_result_needs_rotation() {
        let mut osd = OsdResult::new();
        osd.orientation = TextOrientation::Rotated180;
        assert!(osd.needs_rotation());
    }

    #[test]
    fn test_periodicity_flat() {
        let flat = vec![10u32; 20];
        let score = OsdDetector::compute_periodicity(&flat);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_periodicity_peaks() {
        let mut peaks = vec![0u32; 100];
        for i in (10..100).step_by(15) {
            peaks[i] = 100;
        }
        let score = OsdDetector::compute_periodicity(&peaks);
        assert!(score > 0.5, "Expected periodicity > 0.5, got {}", score);
    }

    #[test]
    fn test_script_names() {
        assert_eq!(DetectedScript::Latin.name(), "Latin");
        assert_eq!(DetectedScript::Cjk.name(), "CJK");
        assert_eq!(DetectedScript::Arabic.name(), "Arabic");
    }
}
