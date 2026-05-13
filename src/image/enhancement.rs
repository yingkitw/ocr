//! Image enhancement operations

use crate::core::image::OcrImage;
use crate::utils::{OcrError, Result};
use image::{imageops, DynamicImage, GrayImage, Luma};

/// Image enhancement operations
pub struct ImageEnhancer;

impl ImageEnhancer {
    /// Enhance image contrast
    pub fn enhance_contrast(img: &OcrImage, factor: f32) -> Result<OcrImage> {
        let mut enhanced = img.data.clone();
        imageops::contrast(&mut enhanced, factor);
        Ok(OcrImage::new(enhanced, img.dpi))
    }

    /// Reduce image noise (using Gaussian blur)
    pub fn reduce_noise(img: &OcrImage) -> Result<OcrImage> {
        let blurred = imageops::blur(&img.data, 0.5);
        Ok(OcrImage::new(DynamicImage::ImageRgba8(blurred), img.dpi))
    }

    /// Sharpen image
    pub fn sharpen(img: &OcrImage) -> Result<OcrImage> {
        let mut sharpened = img.data.clone();
        imageops::unsharpen(&mut sharpened, 1.0, 1);
        Ok(OcrImage::new(sharpened, img.dpi))
    }

    /// Deskew image
    pub fn deskew(img: &OcrImage) -> Result<OcrImage> {
        let gray = img.to_grayscale();
        let binary = gray.threshold(200);

        let mut best_angle = 0.0;
        let mut max_variance = 0.0;

        let range = 20;

        for i in -range..=range {
            let angle_deg = i as f32 * 0.1;
            let angle_rad = angle_deg.to_radians();

            let rotated = binary.rotate(angle_rad)?;

            let variance = Self::calculate_projection_variance(&rotated);

            if variance > max_variance {
                max_variance = variance;
                best_angle = angle_rad;
            }
        }

        if best_angle.abs() > 0.001 {
            img.rotate(best_angle).map_err(OcrError::from)
        } else {
            Ok(img.clone())
        }
    }

    /// Estimate DPI from image content by analyzing stroke widths
    ///
    /// Returns an estimated DPI. Falls back to the image's stored DPI if
    /// estimation is not possible. Typical printed text has strokes ~1pt wide,
    /// which at 300 DPI is ~4px.
    pub fn estimate_dpi(img: &OcrImage) -> u32 {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width < 50 || height < 50 {
            return img.dpi.max(72);
        }

        let mid_y = height / 2;
        let mut run_lengths = Vec::new();
        let mut current_run = 0u32;
        let mut in_dark = gray.get_pixel(0, mid_y)[0] < 128;

        for x in 0..width {
            let dark = gray.get_pixel(x, mid_y)[0] < 128;
            if dark == in_dark {
                current_run += 1;
            } else {
                if in_dark && current_run > 0 && current_run < width / 4 {
                    run_lengths.push(current_run);
                }
                current_run = 1;
                in_dark = dark;
            }
        }

        if run_lengths.len() < 3 {
            return img.dpi.max(72);
        }

        run_lengths.sort_unstable();
        let median_stroke = run_lengths[run_lengths.len() / 2];

        let estimated = (median_stroke as f32 * 72.0) as u32;
        estimated.clamp(72, 1200).max(img.dpi)
    }

    /// Detect image orientation (0, 90, 180, 270 degrees)
    ///
    /// Uses horizontal projection variance: upright text has higher
    /// variance because text lines create distinct dark bands separated
    /// by white space.
    pub fn detect_orientation(img: &OcrImage) -> u32 {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width < 50 || height < 50 {
            return 0;
        }

        let binary = Self::simple_threshold(&gray);

        let score_0 = Self::orientation_score(&binary, width, height);
        let score_90 = Self::orientation_score(&Self::rotate_90_cw(&binary), height, width);
        let score_180 = Self::orientation_score(&Self::rotate_180(&binary), width, height);
        let score_270 = Self::orientation_score(&Self::rotate_90_ccw(&binary), height, width);

        let scores = [
            (0u32, score_0),
            (90, score_90),
            (180, score_180),
            (270, score_270),
        ];
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        best.0
    }

    /// Rotate image to correct orientation based on detected text direction
    pub fn correct_orientation(img: &OcrImage) -> Result<OcrImage> {
        let orientation = Self::detect_orientation(img);

        match orientation {
            0 => Ok(img.clone()),
            90 => img
                .rotate(std::f32::consts::FRAC_PI_2)
                .map_err(OcrError::from),
            180 => img.rotate(std::f32::consts::PI).map_err(OcrError::from),
            270 => img
                .rotate(-std::f32::consts::FRAC_PI_2)
                .map_err(OcrError::from),
            _ => Ok(img.clone()),
        }
    }

    /// Remove document borders/margins
    ///
    /// Scans inward from each edge to find the first row/column containing
    /// dark pixels, then crops with a small margin.
    pub fn remove_borders(img: &OcrImage) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width < 20 || height < 20 {
            return Ok(img.clone());
        }

        let threshold = 200u8;
        let margin = 5u32;

        let mut left = margin;
        let mut right = width - margin - 1;
        let mut top = margin;
        let mut bottom = height - margin - 1;

        for x in margin..(width - margin) {
            let mut has_content = false;
            for y in margin..(height - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                left = x.saturating_sub(margin);
                break;
            }
        }

        for x in (margin..(width - margin)).rev() {
            let mut has_content = false;
            for y in margin..(height - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                right = (x + margin).min(width);
                break;
            }
        }

        for y in margin..(height - margin) {
            let mut has_content = false;
            for x in margin..(width - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                top = y.saturating_sub(margin);
                break;
            }
        }

        for y in (margin..(height - margin)).rev() {
            let mut has_content = false;
            for x in margin..(width - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                bottom = (y + margin).min(height);
                break;
            }
        }

        let crop_width = right.saturating_sub(left);
        let crop_height = bottom.saturating_sub(top);

        if crop_width < 20 || crop_height < 20 || crop_width >= width || crop_height >= height {
            return Ok(img.clone());
        }

        img.crop(left, top, crop_width, crop_height)
            .map_err(OcrError::from)
    }

    /// Remove speckle noise (small connected components of dark pixels)
    ///
    /// Finds connected components of dark pixels using two-pass labeling
    /// and removes those whose area is smaller than `max_speckle_area`.
    pub fn remove_speckle(img: &OcrImage, max_speckle_area: u32) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width == 0 || height == 0 {
            return Ok(img.clone());
        }

        let total_pixels = (width * height) as usize;
        let mut labels = vec![0u32; total_pixels];
        let mut next_label = 1u32;
        let mut areas = vec![0u32; total_pixels + 1];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if gray.get_pixel(x, y)[0] >= 128 {
                    continue;
                }

                let up = if y > 0 {
                    labels[((y - 1) * width + x) as usize]
                } else {
                    0
                };
                let left_label = if x > 0 {
                    labels[(y * width + x - 1) as usize]
                } else {
                    0
                };

                if up == 0 && left_label == 0 {
                    labels[idx] = next_label;
                    next_label += 1;
                } else if up > 0 && left_label == 0 {
                    labels[idx] = up;
                } else if left_label > 0 && up == 0 {
                    labels[idx] = left_label;
                } else {
                    labels[idx] = up.min(left_label);
                }
            }
        }

        for &label in &labels {
            if label > 0 && (label as usize) < areas.len() {
                areas[label as usize] += 1;
            }
        }

        let mut result = gray.clone();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let label = labels[idx];
                if label > 0
                    && (label as usize) < areas.len()
                    && areas[label as usize] < max_speckle_area
                {
                    result.put_pixel(x, y, Luma([255u8]));
                }
            }
        }

        Ok(OcrImage::new(DynamicImage::ImageLuma8(result), img.dpi))
    }

    fn simple_threshold(gray: &GrayImage) -> GrayImage {
        let (width, height) = gray.dimensions();

        let mut histogram = [0u32; 256];
        for pixel in gray.pixels() {
            histogram[pixel[0] as usize] += 1;
        }

        let total = (width * height) as u32;
        let mut sum = 0u32;
        for i in 0..256 {
            sum += histogram[i] * (i as u32);
        }

        let mut sum_b = 0u32;
        let mut w_b = 0u32;
        let mut max_variance = 0f64;
        let mut threshold = 128u8;

        for i in 0..256 {
            w_b += histogram[i];
            if w_b == 0 {
                continue;
            }
            let w_f = total - w_b;
            if w_f == 0 {
                break;
            }
            sum_b += (i as u32) * histogram[i];
            let m_b = sum_b as f64 / w_b as f64;
            let m_f = (sum - sum_b) as f64 / w_f as f64;
            let v = (w_b as f64) * (w_f as f64) * (m_b - m_f) * (m_b - m_f);
            if v > max_variance {
                max_variance = v;
                threshold = i as u8;
            }
        }

        let mut result = GrayImage::new(width, height);
        for (x, y, pixel) in gray.enumerate_pixels() {
            let value = if pixel[0] > threshold { 255u8 } else { 0u8 };
            result.put_pixel(x, y, Luma([value]));
        }
        result
    }

    fn orientation_score(binary: &GrayImage, width: u32, height: u32) -> f64 {
        let mut row_sums = Vec::with_capacity(height as usize);
        for y in 0..height {
            let mut sum = 0u64;
            for x in 0..width {
                if binary.get_pixel(x, y)[0] < 128 {
                    sum += 1;
                }
            }
            row_sums.push(sum as f64);
        }

        if row_sums.is_empty() {
            return 0.0;
        }

        let mean = row_sums.iter().sum::<f64>() / row_sums.len() as f64;
        let variance = row_sums
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>()
            / row_sums.len() as f64;

        variance
    }

    fn rotate_90_cw(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        let mut result = GrayImage::new(height, width);
        for y in 0..height {
            for x in 0..width {
                let src_pixel = img.get_pixel(x, y);
                result.put_pixel(height - 1 - y, x, *src_pixel);
            }
        }
        result
    }

    fn rotate_90_ccw(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        let mut result = GrayImage::new(height, width);
        for y in 0..height {
            for x in 0..width {
                let src_pixel = img.get_pixel(x, y);
                result.put_pixel(y, width - 1 - x, *src_pixel);
            }
        }
        result
    }

    fn rotate_180(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        let mut result = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let src_pixel = img.get_pixel(x, y);
                result.put_pixel(width - 1 - x, height - 1 - y, *src_pixel);
            }
        }
        result
    }

    fn calculate_projection_variance(img: &OcrImage) -> f64 {
        let (width, height) = img.dimensions();
        let mut row_sums = Vec::with_capacity(height as usize);

        if let Some(buf) = img.data.as_luma8() {
            for y in 0..height {
                let mut sum = 0u64;
                for x in 0..width {
                    sum += buf.get_pixel(x, y)[0] as u64;
                }
                row_sums.push(sum as f64);
            }
        } else {
            return 0.0;
        }

        if row_sums.is_empty() {
            return 0.0;
        }

        let mean = row_sums.iter().sum::<f64>() / row_sums.len() as f64;
        let variance = row_sums
            .iter()
            .map(|s| {
                let diff = s - mean;
                diff * diff
            })
            .sum::<f64>()
            / row_sums.len() as f64;

        variance
    }
}
