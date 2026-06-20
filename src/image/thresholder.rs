//! Image thresholding for OCR preprocessing
//!
//! Ported from Tesseract's thresholder.h/cpp and otsuthr.h/cpp
//! Provides various thresholding methods to convert grayscale images to binary

use crate::core::image::{ImageFormat, OcrImage};
use crate::utils::{OcrError, Result};
use image::{DynamicImage, GrayImage, Luma};

/// Thresholding method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMethod {
    /// Tesseract's legacy Otsu thresholding
    Otsu,
    /// Standard Otsu thresholding
    LeptonicaOtsu,
    /// Sauvola adaptive thresholding
    Sauvola,
    /// Niblack adaptive thresholding
    Niblack,
}

/// Image thresholder for converting grayscale images to binary
///
/// Ported from Tesseract's ImageThresholder class
pub struct ImageThresholder {
    /// Source image
    image: Option<OcrImage>,
    /// Processing rectangle
    rect: Option<Rect>,
    /// Estimated resolution
    estimated_res: u32,
    /// Scale factor
    scale: f32,
}

/// Rectangle for processing region
#[derive(Debug, Clone, Copy)]
struct Rect {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
}

impl ImageThresholder {
    /// Create a new image thresholder
    pub fn new() -> Self {
        Self {
            image: None,
            rect: None,
            estimated_res: 300,
            scale: 1.0,
        }
    }

    /// Check if no image has been set
    pub fn is_empty(&self) -> bool {
        self.image.is_none()
    }

    /// Clear the current image
    pub fn clear(&mut self) {
        self.image = None;
        self.rect = None;
    }

    /// Set the image to process
    ///
    /// Supports grayscale (8 bpp), RGB (24 bpp), and RGBA (32 bpp) images
    pub fn set_image(&mut self, image: OcrImage) -> Result<()> {
        // Validate image format
        match image.format {
            ImageFormat::Grayscale | ImageFormat::Rgb | ImageFormat::Rgba => {
                self.image = Some(image);
                Ok(())
            }
            ImageFormat::Binary => {
                // Binary images are already thresholded
                self.image = Some(image);
                Ok(())
            }
        }
    }

    /// Set the rectangle to process
    ///
    /// Doesn't actually do any thresholding, just stores the coordinates
    pub fn set_rectangle(&mut self, left: u32, top: u32, width: u32, height: u32) {
        self.rect = Some(Rect {
            left,
            top,
            width,
            height,
        });
    }

    /// Get image sizes and rectangle information
    pub fn get_image_sizes(&self) -> Option<(u32, u32, u32, u32, u32, u32)> {
        let image = self.image.as_ref()?;
        let rect = self.rect?;

        Some((
            rect.left,
            rect.top,
            rect.width,
            rect.height,
            image.width,
            image.height,
        ))
    }

    /// Check if the source image is color
    pub fn is_color(&self) -> bool {
        self.image
            .as_ref()
            .map(|img| matches!(img.format, ImageFormat::Rgb | ImageFormat::Rgba))
            .unwrap_or(false)
    }

    /// Threshold the image using the specified method
    ///
    /// Returns a binary OcrImage
    pub fn threshold(&self, method: ThresholdMethod) -> Result<OcrImage> {
        let image = self
            .image
            .as_ref()
            .ok_or_else(|| OcrError::ImageProcessing("No image set".to_string()))?;

        // Convert to grayscale if needed
        let gray_image = match image.format {
            ImageFormat::Grayscale => image.data.to_luma8(),
            ImageFormat::Rgb | ImageFormat::Rgba => image.data.to_luma8(),
            ImageFormat::Binary => {
                // Already binary, return as-is
                return Ok(image.clone());
            }
        };

        // Apply thresholding method
        let thresholded = match method {
            ThresholdMethod::Otsu | ThresholdMethod::LeptonicaOtsu => {
                Self::otsu_threshold(&gray_image)?
            }
            ThresholdMethod::Sauvola => Self::sauvola_threshold(&gray_image)?,
            ThresholdMethod::Niblack => Self::niblack_threshold(&gray_image)?,
        };

        // Convert to OcrImage
        let dynamic_image = DynamicImage::ImageLuma8(thresholded);
        let mut ocr_image = OcrImage::new(dynamic_image, image.dpi);
        ocr_image.format = ImageFormat::Binary;
        ocr_image.metadata = image.metadata.clone();

        Ok(ocr_image)
    }

    /// Otsu thresholding algorithm
    ///
    /// Ported from Tesseract's Otsu thresholding implementation
    fn otsu_threshold(gray_image: &GrayImage) -> Result<GrayImage> {
        let (width, height) = gray_image.dimensions();
        let mut histogram = [0u32; 256];

        // Build histogram
        for pixel in gray_image.pixels() {
            histogram[pixel[0] as usize] += 1;
        }

        // Calculate Otsu threshold
        let total_pixels = (width * height) as u32;

        // Calculate total sum first
        let mut sum = 0u32;
        for i in 0..256 {
            sum += histogram[i] * (i as u32);
        }

        let mut sum_b = 0u32;
        let mut w_b = 0u32;
        let mut w_f: u32;
        let mut max_variance = 0f64;
        let mut threshold = 0u8;

        for i in 0..256 {
            w_b += histogram[i];
            if w_b == 0 {
                continue;
            }
            w_f = total_pixels - w_b;
            if w_f == 0 {
                break;
            }

            sum_b += (i as u32) * histogram[i];
            let m_b = sum_b as f64 / w_b as f64;
            let sum_f = sum - sum_b;
            let m_f = sum_f as f64 / w_f as f64;

            let variance_between = (w_b as f64) * (w_f as f64) * (m_b - m_f) * (m_b - m_f);

            if variance_between > max_variance {
                max_variance = variance_between;
                threshold = i as u8;
            }
        }

        // Apply threshold
        let mut result = GrayImage::new(width, height);
        for (x, y, pixel) in gray_image.enumerate_pixels() {
            let value = if pixel[0] > threshold { 255u8 } else { 0u8 };
            result.put_pixel(x, y, Luma([value]));
        }

        Ok(result)
    }

    /// Sauvola adaptive thresholding algorithm (fast, integral-image based)
    ///
    /// T = mean * (1 + k * (std_dev/R - 1)), k=0.2, R=128
    /// Works well for document images with uneven lighting.
    fn sauvola_threshold(gray_image: &GrayImage) -> Result<GrayImage> {
        let (width, height) = gray_image.dimensions();
        let window_size = 31u32;
        let k = 0.2f64;
        let r = 128.0f64;
        let w = width as usize;

        let (integral, integral_sq) = Self::build_integral_images(gray_image);

        let half = (window_size / 2) as i32;
        let mut result = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let x0 = (x as i32 - half).max(0) as u32;
                let y0 = (y as i32 - half).max(0) as u32;
                let x1 = (x as i32 + half).min(width as i32 - 1) as u32;
                let y1 = (y as i32 + half).min(height as i32 - 1) as u32;

                let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
                let sum = Self::integral_rect(&integral, w, x0, y0, x1, y1) as f64;
                let sum_sq = Self::integral_rect(&integral_sq, w, x0, y0, x1, y1) as f64;

                let mean = sum / area;
                let variance = (sum_sq / area) - (mean * mean);
                let std_dev = variance.max(0.0).sqrt();

                let threshold = mean * (1.0 + k * (std_dev / r - 1.0));

                let pixel_value = gray_image.get_pixel(x, y)[0] as f64;
                let value = if pixel_value > threshold { 255u8 } else { 0u8 };
                result.put_pixel(x, y, Luma([value]));
            }
        }

        Ok(result)
    }

    /// Niblack adaptive thresholding algorithm
    ///
    /// T = mean + k * std_dev
    /// Good for documents with variable lighting. k typically -0.1 to -0.2.
    fn niblack_threshold(gray_image: &GrayImage) -> Result<GrayImage> {
        let (width, height) = gray_image.dimensions();
        let window_size = 31u32;
        let k = -0.2f64;
        let w = width as usize;

        let mut result = GrayImage::new(width, height);

        let (integral, integral_sq) = Self::build_integral_images(gray_image);

        let half = (window_size / 2) as i32;

        for y in 0..height {
            for x in 0..width {
                let x0 = (x as i32 - half).max(0) as u32;
                let y0 = (y as i32 - half).max(0) as u32;
                let x1 = (x as i32 + half).min(width as i32 - 1) as u32;
                let y1 = (y as i32 + half).min(height as i32 - 1) as u32;

                let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
                let sum = Self::integral_rect(&integral, w, x0, y0, x1, y1) as f64;
                let sum_sq = Self::integral_rect(&integral_sq, w, x0, y0, x1, y1) as f64;

                let mean = sum / area;
                let variance = (sum_sq / area) - (mean * mean);
                let std_dev = variance.max(0.0).sqrt();

                let threshold = mean + k * std_dev;

                let pixel_value = gray_image.get_pixel(x, y)[0] as f64;
                let value = if pixel_value > threshold { 255u8 } else { 0u8 };
                result.put_pixel(x, y, Luma([value]));
            }
        }

        Ok(result)
    }

    /// Build integral images for fast windowed statistics
    fn build_integral_images(gray_image: &GrayImage) -> (Vec<u64>, Vec<u64>) {
        let (width, height) = gray_image.dimensions();
        let w = width as usize;
        let h = height as usize;

        let mut integral = vec![0u64; w * h];
        let mut integral_sq = vec![0u64; w * h];

        for y in 0..h {
            let mut row_sum: u64 = 0;
            let mut row_sum_sq: u64 = 0;
            for x in 0..w {
                let px = gray_image.get_pixel(x as u32, y as u32)[0] as u64;
                row_sum += px;
                row_sum_sq += px * px;

                let above = if y > 0 { integral[(y - 1) * w + x] } else { 0 };
                let above_sq = if y > 0 {
                    integral_sq[(y - 1) * w + x]
                } else {
                    0
                };

                integral[y * w + x] = above + row_sum;
                integral_sq[y * w + x] = above_sq + row_sum_sq;
            }
        }

        (integral, integral_sq)
    }

    /// Sum over rectangle [x0,x1] x [y0,y1] in a summed-area-table (SAT).
    ///
    /// The SAT is stored row-major with stride `w` (image width).
    /// SAT[y][x] = sum of pixels in [0,x] x [0,y] (inclusive).
    fn integral_rect(sat: &[u64], w: usize, x0: u32, y0: u32, x1: u32, y1: u32) -> u64 {
        // D = SAT[y1][x1]
        let d = sat[y1 as usize * w + x1 as usize];
        // B = SAT[y0-1][x1]  (0 if y0 == 0)
        let b = if y0 > 0 {
            sat[(y0 as usize - 1) * w + x1 as usize]
        } else {
            0
        };
        // C = SAT[y1][x0-1]  (0 if x0 == 0)
        let c = if x0 > 0 {
            sat[y1 as usize * w + (x0 as usize - 1)]
        } else {
            0
        };
        // A = SAT[y0-1][x0-1]  (0 if y0==0 or x0==0)
        let a = if y0 > 0 && x0 > 0 {
            sat[(y0 as usize - 1) * w + (x0 as usize - 1)]
        } else {
            0
        };
        d + a - b - c
    }

    /// Get the estimated resolution
    pub fn estimated_resolution(&self) -> u32 {
        self.estimated_res
    }

    /// Set the estimated resolution
    pub fn set_estimated_resolution(&mut self, res: u32) {
        self.estimated_res = res;
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Set the scale factor
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }
}

impl Default for ImageThresholder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    fn test_thresholder_creation() {
        let thresholder = ImageThresholder::new();
        assert!(thresholder.is_empty());
    }

    #[test]
    fn test_otsu_thresholding() {
        // Create a simple test image
        let mut test_image = GrayImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let value = if x < 50 { 50u8 } else { 200u8 };
                test_image.put_pixel(x, y, Luma([value]));
            }
        }

        let result = ImageThresholder::otsu_threshold(&test_image).unwrap();

        // Check that thresholding was applied
        assert_eq!(result.dimensions(), (100, 100));

        // Left side should be dark (0), right side should be light (255)
        assert_eq!(result.get_pixel(10, 10)[0], 0);
        assert_eq!(result.get_pixel(90, 10)[0], 255);
    }
}
