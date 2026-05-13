//! Tesseract-style feature extraction for character classification
//!
//! This module implements Tesseract's feature extraction algorithms:
//! - INT_FEATURE: Integer features from outlines
//! - Feature extraction from blob outlines
//! - Normalization and denormalization

use crate::core::geometry::{FCoord, ICoord};
use crate::recognition::tesseract_blob::{ChainOutline, TBlob};
use crate::utils::Result;

/// Integer feature structure (equivalent to Tesseract's INT_FEATURE_STRUCT)
#[derive(Debug, Clone, Copy)]
pub struct IntFeature {
    /// X position (0-255)
    pub x: u8,
    /// Y position (0-255)
    pub y: u8,
    /// Direction (0-255, binary degrees)
    pub theta: u8,
}

impl IntFeature {
    /// Create a new integer feature
    pub fn new(x: u8, y: u8, theta: u8) -> Self {
        Self { x, y, theta }
    }
}

/// Feature extraction result (equivalent to Tesseract's INT_FX_RESULT_STRUCT)
#[derive(Debug, Clone)]
pub struct FeatureExtractionResult {
    /// Number of baseline features
    pub num_bl: usize,
    /// Number of character-normalized features
    pub num_cn: usize,
    /// Y bottom
    pub y_bottom: i32,
    /// Y top
    pub y_top: i32,
    /// Width
    pub width: i32,
    /// Length (perimeter)
    pub length: i32,
    /// X mean (center of mass)
    pub x_mean: i32,
    /// Y mean (center of mass)
    pub y_mean: i32,
    /// Rx (rounded y second moment)
    pub rx: i32,
    /// Ry (rounded x second moment)
    pub ry: i32,
}

impl Default for FeatureExtractionResult {
    fn default() -> Self {
        Self {
            num_bl: 0,
            num_cn: 0,
            y_bottom: 0,
            y_top: 0,
            width: 0,
            length: 0,
            x_mean: 0,
            y_mean: 0,
            rx: 0,
            ry: 0,
        }
    }
}

/// Extract features from a blob (equivalent to Tesseract's ExtractFeatures)
pub fn extract_features(
    blob: &TBlob,
    nonlinear_norm: bool,
) -> Result<(Vec<IntFeature>, Vec<IntFeature>, FeatureExtractionResult)> {
    let mut bl_features = Vec::new();
    let mut cn_features = Vec::new();
    let mut fx_result = FeatureExtractionResult::default();

    // Compute bounding box
    let bbox = blob.bounding_box();
    fx_result.y_bottom = bbox.bottom();
    fx_result.y_top = bbox.top();
    fx_result.width = bbox.width();

    // Compute moments
    let (center, second_moments, length) = compute_moments(blob)?;
    fx_result.length = length;
    fx_result.x_mean = center.x as i32;
    fx_result.y_mean = center.y as i32;
    fx_result.rx = second_moments.y as i32;
    fx_result.ry = second_moments.x as i32;

    // Extract features from outlines
    for outline in blob.outlines() {
        extract_features_from_outline(
            outline,
            &center,
            &second_moments,
            nonlinear_norm,
            &mut bl_features,
            &mut cn_features,
        )?;
    }

    fx_result.num_bl = bl_features.len();
    fx_result.num_cn = cn_features.len();

    Ok((bl_features, cn_features, fx_result))
}

/// Compute first and second moments of a blob
fn compute_moments(blob: &TBlob) -> Result<(FCoord, FCoord, i32)> {
    let mut total_length = 0i32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_x2 = 0.0f32;
    let mut sum_y2 = 0.0f32;

    // Iterate over all outlines
    for outline in blob.outlines() {
        let mut pos = outline.start;
        total_length += outline.steps.len() as i32;

        for &step in &outline.steps {
            let (dx, dy) = step.to_offset();
            pos = ICoord::new(pos.x + dx, pos.y + dy);

            let x = pos.x as f32;
            let y = pos.y as f32;

            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
    }

    if total_length == 0 {
        return Ok((FCoord::zero(), FCoord::zero(), 0));
    }

    let n = total_length as f32;
    let center = FCoord::new(sum_x / n, sum_y / n);

    // Second moments (variance)
    let second_x = (sum_x2 / n) - (center.x * center.x);
    let second_y = (sum_y2 / n) - (center.y * center.y);
    let second_moments = FCoord::new(second_x.max(0.0), second_y.max(0.0));

    Ok((center, second_moments, total_length))
}

/// Extract features from an outline
fn extract_features_from_outline(
    outline: &ChainOutline,
    center: &FCoord,
    second_moments: &FCoord,
    nonlinear_norm: bool,
    bl_features: &mut Vec<IntFeature>,
    cn_features: &mut Vec<IntFeature>,
) -> Result<()> {
    let mut pos = FCoord::new(outline.start.x as f32, outline.start.y as f32);

    for &step in &outline.steps {
        let (dx, dy) = step.to_offset();
        pos = FCoord::new(pos.x + dx as f32, pos.y + dy as f32);

        // Compute direction (angle)
        let direction = FCoord::new(dx as f32, dy as f32);
        let angle = direction.angle();

        // Convert angle to binary degrees (0-255)
        let theta = ((angle + std::f32::consts::PI) * 128.0 / std::f32::consts::PI) as u8;

        // Baseline-normalized features
        let bl_x = ((pos.x - center.x) + 128.0) as u8;
        let bl_y = ((pos.y - center.y) + 128.0) as u8;
        bl_features.push(IntFeature::new(bl_x, bl_y, theta));

        // Character-normalized features
        let scale_x = if second_moments.x > 0.0 {
            (51.2 / second_moments.x.sqrt()).min(10.0)
        } else {
            1.0
        };
        let scale_y = if second_moments.y > 0.0 {
            (51.2 / second_moments.y.sqrt()).min(10.0)
        } else {
            1.0
        };

        let cn_x = ((pos.x - center.x) * scale_x + 128.0) as u8;
        let cn_y = ((pos.y - center.y) * scale_y + 128.0) as u8;
        cn_features.push(IntFeature::new(cn_x, cn_y, theta));
    }

    Ok(())
}
