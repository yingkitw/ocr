//! Document perspective dewarp and curved-line rectification.
//!
//! PaddleOCR / docTR rectify page perspective and curved text baselines before
//! recognition. This module provides pure-Rust equivalents using
//! `imageproc` projective warps and vertical baseline remapping.

use crate::core::image::OcrImage;
use crate::utils::{OcrError, Result};
use image::{DynamicImage, GrayImage, Luma};
use imageproc::geometric_transformations::{warp, Interpolation, Projection};

/// Perspective (quadrilateral → rectangle) document dewarp.
#[derive(Debug, Clone)]
pub struct PerspectiveDewarp {
    /// Minimum relative corner offset (vs min(w,h)) required to apply a warp.
    pub min_distortion: f32,
    /// Dark-pixel threshold for content detection (0–255).
    pub ink_threshold: u8,
}

impl Default for PerspectiveDewarp {
    fn default() -> Self {
        Self {
            min_distortion: 0.02,
            ink_threshold: 128,
        }
    }
}

impl PerspectiveDewarp {
    pub fn new() -> Self {
        Self::default()
    }

    /// Dewarp a document image if a significant perspective distortion is detected.
    /// Returns the original image when corners are nearly rectangular.
    pub fn dewarp(&self, img: &OcrImage) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (w, h) = (gray.width(), gray.height());
        if w < 16 || h < 16 {
            return Ok(img.clone());
        }

        let Some(corners) = estimate_content_corners(&gray, self.ink_threshold) else {
            return Ok(img.clone());
        };

        if !self.is_significantly_distorted(corners, w, h) {
            return Ok(img.clone());
        }

        let dst = [
            (0.0, 0.0),
            (w as f32 - 1.0, 0.0),
            (w as f32 - 1.0, h as f32 - 1.0),
            (0.0, h as f32 - 1.0),
        ];

        let projection = Projection::from_control_points(corners, dst).ok_or_else(|| {
            OcrError::ImageProcessing("Failed to build perspective projection".into())
        })?;

        let warped = warp(&gray, &projection, Interpolation::Bilinear, Luma([255]));
        Ok(OcrImage::new(DynamicImage::ImageLuma8(warped), img.dpi))
    }

    fn is_significantly_distorted(&self, corners: [(f32, f32); 4], w: u32, h: u32) -> bool {
        let min_dim = w.min(h) as f32;
        let threshold = self.min_distortion * min_dim;
        let xs: Vec<f32> = corners.iter().map(|c| c.0).collect();
        let ys: Vec<f32> = corners.iter().map(|c| c.1).collect();
        let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let aabb = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ];
        corners
            .iter()
            .zip(aabb.iter())
            .any(|(c, a)| ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt() > threshold)
    }
}

/// Estimate TL, TR, BR, BL corners of dark content.
fn estimate_content_corners(gray: &GrayImage, ink_threshold: u8) -> Option<[(f32, f32); 4]> {
    let (w, h) = (gray.width(), gray.height());
    let mut points: Vec<(f32, f32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if gray.get_pixel(x, y)[0] < ink_threshold {
                points.push((x as f32, y as f32));
            }
        }
    }

    if points.len() < 16 {
        return None;
    }

    // Extreme-point corner heuristic (common for scanned docs):
    // TL = min(x+y), TR = max(x-y), BR = max(x+y), BL = min(x-y)
    let mut tl = points[0];
    let mut tr = points[0];
    let mut br = points[0];
    let mut bl = points[0];
    let mut best_tl = f32::INFINITY;
    let mut best_tr = f32::NEG_INFINITY;
    let mut best_br = f32::NEG_INFINITY;
    let mut best_bl = f32::INFINITY;

    for &(x, y) in &points {
        let sum = x + y;
        let diff = x - y;
        if sum < best_tl {
            best_tl = sum;
            tl = (x, y);
        }
        if diff > best_tr {
            best_tr = diff;
            tr = (x, y);
        }
        if sum > best_br {
            best_br = sum;
            br = (x, y);
        }
        if diff < best_bl {
            best_bl = diff;
            bl = (x, y);
        }
    }

    Some([tl, tr, br, bl])
}

/// Rectify a curved (or wavy) text-line image by flattening its baseline.
#[derive(Debug, Clone)]
pub struct CurveRectifier {
    /// Minimum |quadratic a| * width² to trigger remapping.
    pub min_curvature: f32,
    pub ink_threshold: u8,
}

impl Default for CurveRectifier {
    fn default() -> Self {
        Self {
            min_curvature: 2.0,
            ink_threshold: 128,
        }
    }
}

impl CurveRectifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Flatten a curved text line so the baseline becomes horizontal.
    pub fn rectify(&self, img: &OcrImage) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (w, h) = (gray.width() as usize, gray.height() as usize);
        if w < 8 || h < 4 {
            return Ok(img.clone());
        }

        let baseline = column_baselines(&gray, self.ink_threshold);
        let Some((a, _b, _c)) = fit_quadratic(&baseline) else {
            return Ok(img.clone());
        };

        // Curvature magnitude over the line width
        let curvature = (a * (w as f32).powi(2)).abs();
        if curvature < self.min_curvature {
            return Ok(img.clone());
        }

        let mean_base: f32 = {
            let valid: Vec<f32> = baseline.iter().copied().filter(|y| y.is_finite()).collect();
            if valid.is_empty() {
                return Ok(img.clone());
            }
            valid.iter().sum::<f32>() / valid.len() as f32
        };

        let mut out = GrayImage::new(w as u32, h as u32);
        for x in 0..w {
            let shift = baseline[x] - mean_base;
            for y in 0..h {
                let src_y = y as f32 + shift;
                let sample = sample_bilinear(&gray, x as f32, src_y);
                out.put_pixel(x as u32, y as u32, Luma([sample]));
            }
        }

        Ok(OcrImage::new(DynamicImage::ImageLuma8(out), img.dpi))
    }
}

fn column_baselines(gray: &GrayImage, ink_threshold: u8) -> Vec<f32> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let mut baselines = vec![h as f32 * 0.75; w];

    for x in 0..w {
        let mut found = None;
        for y in (0..h).rev() {
            if gray.get_pixel(x as u32, y as u32)[0] < ink_threshold {
                found = Some(y as f32);
                break;
            }
        }
        if let Some(y) = found {
            baselines[x] = y;
        }
    }

    // 5-tap moving average to damp noise
    let mut smoothed = baselines.clone();
    for x in 0..w {
        let lo = x.saturating_sub(2);
        let hi = (x + 2).min(w - 1);
        let slice = &baselines[lo..=hi];
        smoothed[x] = slice.iter().sum::<f32>() / slice.len() as f32;
    }
    smoothed
}

/// Fit y = a x² + b x + c via normal equations on (index, baseline).
fn fit_quadratic(ys: &[f32]) -> Option<(f32, f32, f32)> {
    let n = ys.len();
    if n < 5 {
        return None;
    }

    // Accumulate Σ x^k and Σ x^k y for k=0..4
    let mut s = [0.0f64; 5]; // s[k] = sum x^k
    let mut t = [0.0f64; 3]; // t[k] = sum x^k * y
    for (i, &y) in ys.iter().enumerate() {
        if !y.is_finite() {
            continue;
        }
        let x = i as f64;
        let mut xp = 1.0;
        for k in 0..5 {
            s[k] += xp;
            if k < 3 {
                t[k] += xp * y as f64;
            }
            xp *= x;
        }
    }

    // Solve 3x3 system for [c, b, a]:
    // | s0 s1 s2 |   | c |   | t0 |
    // | s1 s2 s3 | * | b | = | t1 |
    // | s2 s3 s4 |   | a |   | t2 |
    let m = [
        [s[0], s[1], s[2]],
        [s[1], s[2], s[3]],
        [s[2], s[3], s[4]],
    ];
    let rhs = [t[0], t[1], t[2]];
    let sol = solve3(m, rhs)?;
    Some((sol[2] as f32, sol[1] as f32, sol[0] as f32)) // a, b, c
}

fn solve3(m: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    let mut a = m;
    // Gaussian elimination with partial pivoting
    for col in 0..3 {
        let mut pivot = col;
        for row in col + 1..3 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let div = a[col][col];
        for j in col..3 {
            a[col][j] /= div;
        }
        b[col] /= div;
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..3 {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

fn sample_bilinear(img: &GrayImage, x: f32, y: f32) -> u8 {
    let w = img.width() as i32;
    let h = img.height() as i32;
    if y < 0.0 || y >= (h - 1) as f32 || x < 0.0 || x >= (w - 1) as f32 {
        // Out of bounds → white background
        if y < 0.0 || y >= h as f32 || x < 0.0 || x >= w as f32 {
            return 255;
        }
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let x0 = x0.clamp(0, w - 1);
    let y0 = y0.clamp(0, h - 1);

    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);

    let p00 = img.get_pixel(x0 as u32, y0 as u32)[0] as f32;
    let p10 = img.get_pixel(x1 as u32, y0 as u32)[0] as f32;
    let p01 = img.get_pixel(x0 as u32, y1 as u32)[0] as f32;
    let p11 = img.get_pixel(x1 as u32, y1 as u32)[0] as f32;

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bot = p01 * (1.0 - fx) + p11 * fx;
    (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    fn make_rect_text_image(w: u32, h: u32) -> OcrImage {
        // White page with a dark rectangle (axis-aligned content)
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in h / 4..3 * h / 4 {
            for x in w / 4..3 * w / 4 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        OcrImage::new(DynamicImage::ImageLuma8(img), 72)
    }

    fn make_trapezoid_image(w: u32, h: u32) -> OcrImage {
        // Dark trapezoid: top narrower than bottom → perspective cue
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 10..h - 10 {
            let t = (y - 10) as f32 / (h - 20) as f32;
            let left = (w as f32 * (0.3 - 0.15 * t)) as u32;
            let right = (w as f32 * (0.7 + 0.15 * t)) as u32;
            for x in left..right.min(w) {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        OcrImage::new(DynamicImage::ImageLuma8(img), 72)
    }

    fn make_curved_line(w: u32, h: u32) -> OcrImage {
        // A 3px-thick arc (quadratic) of dark ink
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        let a = 4.0 / (w as f32).powi(2);
        let c = h as f32 * 0.3;
        for x in 0..w {
            let y = (a * (x as f32).powi(2) + c) as i32;
            for dy in 0..3 {
                let yy = y + dy;
                if yy >= 0 && yy < h as i32 {
                    img.put_pixel(x, yy as u32, Luma([0]));
                }
            }
        }
        OcrImage::new(DynamicImage::ImageLuma8(img), 72)
    }

    #[test]
    fn test_perspective_skips_axis_aligned() {
        let img = make_rect_text_image(80, 60);
        let out = PerspectiveDewarp::default().dewarp(&img).unwrap();
        // Should return same dimensions; content remains roughly centered
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
    }

    #[test]
    fn test_perspective_warps_trapezoid() {
        let img = make_trapezoid_image(100, 80);
        let out = PerspectiveDewarp {
            min_distortion: 0.01,
            ..Default::default()
        }
        .dewarp(&img)
        .unwrap();
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        // Output should still be a valid grayscale image
        let gray = out.data.to_luma8();
        assert_eq!(gray.width(), 100);
    }

    #[test]
    fn test_curve_rectify_flattens_arc() {
        let img = make_curved_line(120, 40);
        let rectifier = CurveRectifier {
            min_curvature: 1.0,
            ..Default::default()
        };
        let out = rectifier.rectify(&img).unwrap();
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);

        // After rectification, column baselines should have lower variance
        let before = column_baselines(&img.data.to_luma8(), 128);
        let after = column_baselines(&out.data.to_luma8(), 128);
        let var = |ys: &[f32]| {
            let mean = ys.iter().sum::<f32>() / ys.len() as f32;
            ys.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / ys.len() as f32
        };
        assert!(
            var(&after) < var(&before),
            "rectify should reduce baseline variance: before={} after={}",
            var(&before),
            var(&after)
        );
    }

    #[test]
    fn test_curve_skips_flat_line() {
        let mut img: GrayImage = ImageBuffer::from_pixel(60, 20, Luma([255]));
        for x in 5..55 {
            for y in 12..15 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let ocr = OcrImage::new(DynamicImage::ImageLuma8(img), 72);
        let out = CurveRectifier::default().rectify(&ocr).unwrap();
        // Flat line → clone path; dimensions unchanged
        assert_eq!(out.width, 60);
    }

    #[test]
    fn test_fit_quadratic_recovers_parabola() {
        let ys: Vec<f32> = (0..50)
            .map(|x| 0.01 * (x as f32).powi(2) + 0.5 * x as f32 + 3.0)
            .collect();
        let (a, b, c) = fit_quadratic(&ys).unwrap();
        assert!((a - 0.01).abs() < 0.002, "a={a}");
        assert!((b - 0.5).abs() < 0.05, "b={b}");
        assert!((c - 3.0).abs() < 0.5, "c={c}");
    }
}
