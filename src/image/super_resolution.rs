//! Super-resolution / upscaling for tiny or low-DPI text.
//!
//! PaddleOCR and similar engines upscale low-resolution input before detection
//! and recognition. This module uses classical (Lanczos) upscaling triggered by
//! stroke-width / DPI heuristics — no learned SR weights required.

use crate::core::image::OcrImage;
use crate::utils::Result;

/// Upscales images when text appears too small for reliable OCR.
#[derive(Debug, Clone)]
pub struct TextSuperResolution {
    /// Target effective DPI after upscaling (typically 300).
    pub target_dpi: u32,
    /// Upscale when median dark-stroke width is below this (pixels).
    pub min_stroke_px: f32,
    /// Upscale when estimated text-line height is below this (pixels).
    pub min_line_height_px: u32,
    /// Maximum scale factor to apply (caps memory/CPU).
    pub max_scale: f32,
    /// Apply a light unsharp after upscaling (helps thin strokes).
    pub sharpen_after: bool,
}

impl Default for TextSuperResolution {
    fn default() -> Self {
        Self {
            target_dpi: 300,
            min_stroke_px: 2.5,
            min_line_height_px: 16,
            max_scale: 4.0,
            sharpen_after: true,
        }
    }
}

/// Decision / result of an upscale attempt.
#[derive(Debug, Clone)]
pub struct UpscaleResult {
    pub image: OcrImage,
    /// Scale factor applied (1.0 = unchanged).
    pub scale: f32,
    pub reason: UpscaleReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleReason {
    NotNeeded,
    LowDpi,
    ThinStroke,
    ShortLineHeight,
}

impl TextSuperResolution {
    pub fn new() -> Self {
        Self::default()
    }

    /// Upscale `img` if tiny-text heuristics fire; otherwise return a clone.
    pub fn upscale_if_needed(&self, img: &OcrImage) -> Result<UpscaleResult> {
        let (scale, reason) = self.recommend_scale(img);
        if (scale - 1.0).abs() < 1e-3 {
            return Ok(UpscaleResult {
                image: img.clone(),
                scale: 1.0,
                reason: UpscaleReason::NotNeeded,
            });
        }

        let new_w = ((img.width as f32) * scale).round().max(1.0) as u32;
        let new_h = ((img.height as f32) * scale).round().max(1.0) as u32;
        // Cap absolute size to avoid pathological blow-ups
        let (new_w, new_h) = clamp_dims(new_w, new_h, 4096);

        let mut out = img.resize(new_w, new_h)?;
        // Update DPI metadata to reflect effective resolution
        out.dpi = ((img.dpi as f32) * scale).round().clamp(72.0, 1200.0) as u32;

        if self.sharpen_after {
            out = crate::image::enhancement::ImageEnhancer::sharpen(&out).unwrap_or(out);
        }

        Ok(UpscaleResult {
            image: out,
            scale,
            reason,
        })
    }

    /// Recommend a scale factor and why.
    pub fn recommend_scale(&self, img: &OcrImage) -> (f32, UpscaleReason) {
        // Skip pathological / noise-like images (SR would amplify speckles).
        if looks_like_noise(img) {
            return (1.0, UpscaleReason::NotNeeded);
        }

        let mut scale = 1.0f32;
        let mut reason = UpscaleReason::NotNeeded;

        let dpi = if img.dpi > 0 {
            img.dpi
        } else {
            crate::image::enhancement::ImageEnhancer::estimate_dpi(img)
        };
        let line_h = estimate_content_band_height(img);
        let stroke = estimate_median_stroke_px(img);
        let has_text_structure = line_h.map(|h| h >= 3 && h < img.height).unwrap_or(false);

        // 1) Declared / estimated DPI — only when text-like structure is present
        if has_text_structure && dpi > 0 && dpi < self.target_dpi {
            let dpi_scale = (self.target_dpi as f32 / dpi as f32).clamp(1.0, self.max_scale);
            if dpi_scale > scale {
                scale = dpi_scale;
                reason = UpscaleReason::LowDpi;
            }
        }

        // 2) Short line height (tiny rendered glyphs)
        if let Some(line_h) = line_h {
            if line_h > 0 && line_h < self.min_line_height_px {
                let h_scale =
                    (self.min_line_height_px as f32 / line_h as f32).clamp(1.0, self.max_scale);
                if h_scale > scale {
                    scale = h_scale;
                    reason = UpscaleReason::ShortLineHeight;
                }
            }
        }

        // 3) Thin stroke — only reinforce when DPI is low or lines are already short
        if let Some(stroke) = stroke {
            let dpi_low = dpi > 0 && dpi < self.target_dpi;
            let lines_short = line_h
                .map(|h| h < self.min_line_height_px)
                .unwrap_or(false);
            if stroke > 0.0 && stroke < self.min_stroke_px && (dpi_low || lines_short) {
                let stroke_scale =
                    (self.min_stroke_px / stroke).clamp(1.0, self.max_scale);
                if stroke_scale > scale {
                    scale = stroke_scale;
                    reason = UpscaleReason::ThinStroke;
                }
            }
        }

        let scale = quantize_scale(scale, self.max_scale);
        if (scale - 1.0).abs() < 1e-3 {
            (1.0, UpscaleReason::NotNeeded)
        } else {
            (scale, reason)
        }
    }

    /// Force a 2× Lanczos upscale (used for very short recognition crops).
    pub fn upscale_2x(img: &OcrImage) -> Result<OcrImage> {
        let new_w = img.width.saturating_mul(2).min(4096);
        let new_h = img.height.saturating_mul(2).min(4096);
        let mut out = img.resize(new_w, new_h)?;
        out.dpi = img.dpi.saturating_mul(2).min(1200);
        Ok(out)
    }
}

fn looks_like_noise(img: &OcrImage) -> bool {
    let gray = img.data.to_luma8();
    let (w, h) = gray.dimensions();
    if w * h == 0 {
        return true;
    }
    let mut dark = 0u64;
    let total = (w * h) as u64;
    // Subsample for speed
    let step = ((w * h) / 10_000).max(1);
    let mut sampled = 0u64;
    for (i, p) in gray.pixels().enumerate() {
        if (i as u32) % step != 0 {
            continue;
        }
        sampled += 1;
        if p[0] < 128 {
            dark += 1;
        }
    }
    if sampled == 0 {
        return false;
    }
    let ratio = dark as f32 / sampled as f32;
    // Uniform noise / dense speckles sit near 50% dark; clean text is typically < 25%.
    ratio > 0.35 || (total > 50_000 && ratio > 0.28)
}

fn clamp_dims(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let max_dim = w.max(h);
    if max_dim <= max_side {
        return (w.max(1), h.max(1));
    }
    let s = max_side as f32 / max_dim as f32;
    (
        ((w as f32) * s).round().max(1.0) as u32,
        ((h as f32) * s).round().max(1.0) as u32,
    )
}

fn quantize_scale(scale: f32, max_scale: f32) -> f32 {
    if scale <= 1.05 {
        return 1.0;
    }
    let candidates = [1.5f32, 2.0, 3.0, 4.0];
    candidates
        .iter()
        .copied()
        .filter(|&c| c <= max_scale + 1e-3)
        .find(|&c| c + 0.01 >= scale)
        .unwrap_or(max_scale.min(4.0))
}

/// Median run-length of dark pixels on a mid-row scan.
fn estimate_median_stroke_px(img: &OcrImage) -> Option<f32> {
    let gray = img.data.to_luma8();
    let (w, h) = gray.dimensions();
    if w < 8 || h < 4 {
        return None;
    }

    let mut runs = Vec::new();
    for sample_y in [h / 4, h / 2, 3 * h / 4] {
        let mut in_dark = gray.get_pixel(0, sample_y)[0] < 128;
        let mut run = 0u32;
        for x in 0..w {
            let dark = gray.get_pixel(x, sample_y)[0] < 128;
            if dark == in_dark {
                run += 1;
            } else {
                if in_dark && run > 0 && run < w / 3 {
                    runs.push(run);
                }
                run = 1;
                in_dark = dark;
            }
        }
        if in_dark && run > 0 && run < w / 3 {
            runs.push(run);
        }
    }

    if runs.len() < 3 {
        return None;
    }
    runs.sort_unstable();
    Some(runs[runs.len() / 2] as f32)
}

/// Estimate typical dark horizontal band height (proxy for x-height / line height).
fn estimate_content_band_height(img: &OcrImage) -> Option<u32> {
    let gray = img.data.to_luma8();
    let (w, h) = gray.dimensions();
    if w < 8 || h < 4 {
        return None;
    }

    let mut row_dark = vec![0u32; h as usize];
    for y in 0..h {
        let mut dark = 0u32;
        for x in 0..w {
            if gray.get_pixel(x, y)[0] < 128 {
                dark += 1;
            }
        }
        row_dark[y as usize] = dark;
    }

    let threshold = (w / 20).max(2);
    let mut band_heights = Vec::new();
    let mut in_band = false;
    let mut band_start = 0u32;
    for y in 0..h {
        let dark_enough = row_dark[y as usize] >= threshold;
        if dark_enough && !in_band {
            in_band = true;
            band_start = y;
        } else if !dark_enough && in_band {
            band_heights.push(y - band_start);
            in_band = false;
        }
    }
    if in_band {
        band_heights.push(h - band_start);
    }

    if band_heights.is_empty() {
        return None;
    }
    band_heights.sort_unstable();
    // Prefer mid-sized bands (skip full-page fills)
    let mid = band_heights[band_heights.len() / 2];
    if mid >= h {
        None
    } else {
        Some(mid)
    }
}

/// Convenience: upscale a recognition crop when it is shorter than `min_height`.
pub fn upscale_short_crop(img: &OcrImage, min_height: u32) -> Result<OcrImage> {
    if img.height >= min_height {
        return Ok(img.clone());
    }
    let scale = (min_height as f32 / img.height.max(1) as f32)
        .clamp(1.0, 4.0)
        .ceil();
    let new_w = ((img.width as f32) * scale).round() as u32;
    let new_h = ((img.height as f32) * scale).round() as u32;
    let (new_w, new_h) = clamp_dims(new_w.max(1), new_h.max(1), 2048);
    let mut out = img.resize(new_w, new_h)?;
    out.dpi = ((img.dpi as f32) * scale).round() as u32;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    fn tiny_text_image() -> OcrImage {
        // 40×12 canvas with a 1–2px horizontal stroke (tiny)
        let mut img = GrayImage::from_pixel(40, 12, Luma([255]));
        for x in 4..36 {
            img.put_pixel(x, 5, Luma([0]));
            img.put_pixel(x, 6, Luma([0]));
        }
        let mut ocr = OcrImage::new(DynamicImage::ImageLuma8(img), 72);
        ocr.dpi = 72;
        ocr
    }

    fn large_text_image() -> OcrImage {
        let mut img = GrayImage::from_pixel(120, 48, Luma([255]));
        for x in 10..110 {
            for y in 18..30 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let mut ocr = OcrImage::new(DynamicImage::ImageLuma8(img), 300);
        ocr.dpi = 300;
        ocr
    }

    #[test]
    fn test_recommends_upscale_for_low_dpi() {
        let img = tiny_text_image();
        let sr = TextSuperResolution::default();
        let (scale, reason) = sr.recommend_scale(&img);
        assert!(scale > 1.0, "expected upscale, got {scale}");
        assert!(
            matches!(
                reason,
                UpscaleReason::LowDpi | UpscaleReason::ThinStroke | UpscaleReason::ShortLineHeight
            ),
            "reason={reason:?}"
        );
    }

    #[test]
    fn test_skips_adequate_image() {
        let img = large_text_image();
        let sr = TextSuperResolution {
            min_stroke_px: 2.0,
            min_line_height_px: 10,
            ..Default::default()
        };
        let (scale, reason) = sr.recommend_scale(&img);
        assert_eq!(scale, 1.0);
        assert_eq!(reason, UpscaleReason::NotNeeded);
    }

    #[test]
    fn test_upscale_increases_dimensions() {
        let img = tiny_text_image();
        let result = TextSuperResolution::default()
            .upscale_if_needed(&img)
            .unwrap();
        assert!(result.scale > 1.0);
        assert!(result.image.width > img.width);
        assert!(result.image.height > img.height);
        assert!(result.image.dpi >= img.dpi);
    }

    #[test]
    fn test_upscale_short_crop() {
        let img = tiny_text_image();
        let out = upscale_short_crop(&img, 32).unwrap();
        assert!(out.height >= 32 || out.height == img.height * 4); // capped by scale
        assert!(out.height > img.height);
    }

    #[test]
    fn test_skips_noise_image() {
        let mut img = GrayImage::from_pixel(100, 80, Luma([128]));
        for y in 0..80u32 {
            for x in 0..100u32 {
                let n = ((x.wrapping_mul(y + 1).wrapping_mul(1103515245) >> 16) & 0xFF) as u8;
                img.put_pixel(x, y, Luma([n]));
            }
        }
        let ocr = OcrImage::new(DynamicImage::ImageLuma8(img), 300);
        let (scale, reason) = TextSuperResolution::default().recommend_scale(&ocr);
        assert_eq!(scale, 1.0);
        assert_eq!(reason, UpscaleReason::NotNeeded);
    }

    #[test]
    fn test_quantize_scale() {
        assert_eq!(quantize_scale(1.0, 4.0), 1.0);
        assert_eq!(quantize_scale(1.2, 4.0), 1.5);
        assert_eq!(quantize_scale(1.8, 4.0), 2.0);
        assert_eq!(quantize_scale(2.5, 4.0), 3.0);
    }
}
