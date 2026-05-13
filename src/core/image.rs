//! Image data structures and operations for OCR

use crate::utils::Point2D;
use anyhow::Result;
use image::{DynamicImage, GenericImage, GenericImageView, GrayImage, Pixel, RgbImage, RgbaImage};
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

/// Image format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    /// Grayscale image (8 bits per pixel)
    Grayscale,
    /// RGB image (24 bits per pixel)
    Rgb,
    /// RGBA image (32 bits per pixel)
    Rgba,
    /// Binary image (1 bit per pixel)
    Binary,
}

/// Image data structure
#[derive(Debug, Clone)]
pub struct OcrImage {
    /// The actual image data
    pub data: DynamicImage,
    /// Image format
    pub format: ImageFormat,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Image resolution in DPI
    pub dpi: u32,
    /// Image metadata
    pub metadata: HashMap<String, String>,
}

impl Serialize for OcrImage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("OcrImage", 6)?;
        state.serialize_field("format", &self.format)?;
        state.serialize_field("width", &self.width)?;
        state.serialize_field("height", &self.height)?;
        state.serialize_field("dpi", &self.dpi)?;
        state.serialize_field("metadata", &self.metadata)?;

        // Serialize image as raw bytes
        let image_bytes = self.data.as_bytes();
        state.serialize_field("image_data", &image_bytes)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for OcrImage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct OcrImageVisitor;

        impl<'de> Visitor<'de> for OcrImageVisitor {
            type Value = OcrImage;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct OcrImage")
            }

            fn visit_map<V>(self, mut map: V) -> Result<OcrImage, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut format = None;
                let mut width = None;
                let mut height = None;
                let mut dpi = None;
                let mut metadata = None;
                let mut image_data = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "format" => {
                            if format.is_some() {
                                return Err(de::Error::duplicate_field("format"));
                            }
                            format = Some(map.next_value()?);
                        }
                        "width" => {
                            if width.is_some() {
                                return Err(de::Error::duplicate_field("width"));
                            }
                            width = Some(map.next_value()?);
                        }
                        "height" => {
                            if height.is_some() {
                                return Err(de::Error::duplicate_field("height"));
                            }
                            height = Some(map.next_value()?);
                        }
                        "dpi" => {
                            if dpi.is_some() {
                                return Err(de::Error::duplicate_field("dpi"));
                            }
                            dpi = Some(map.next_value()?);
                        }
                        "metadata" => {
                            if metadata.is_some() {
                                return Err(de::Error::duplicate_field("metadata"));
                            }
                            metadata = Some(map.next_value()?);
                        }
                        "image_data" => {
                            if image_data.is_some() {
                                return Err(de::Error::duplicate_field("image_data"));
                            }
                            image_data = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }

                let format = format.ok_or_else(|| de::Error::missing_field("format"))?;
                let width = width.ok_or_else(|| de::Error::missing_field("width"))?;
                let height = height.ok_or_else(|| de::Error::missing_field("height"))?;
                let dpi = dpi.ok_or_else(|| de::Error::missing_field("dpi"))?;
                let metadata = metadata.ok_or_else(|| de::Error::missing_field("metadata"))?;
                let image_data: Vec<u8> =
                    image_data.ok_or_else(|| de::Error::missing_field("image_data"))?;

                // Reconstruct DynamicImage from bytes
                let data = image::load_from_memory(&image_data)
                    .map_err(|e| de::Error::custom(format!("Failed to load image: {}", e)))?;

                Ok(OcrImage {
                    data,
                    format,
                    width,
                    height,
                    dpi,
                    metadata,
                })
            }
        }

        const FIELDS: &'static [&'static str] =
            &["format", "width", "height", "dpi", "metadata", "image_data"];
        deserializer.deserialize_struct("OcrImage", FIELDS, OcrImageVisitor)
    }
}

impl OcrImage {
    /// Create a new OCR image from dynamic image
    pub fn new(data: DynamicImage, dpi: u32) -> Self {
        let (width, height) = data.dimensions();
        let format = match data {
            DynamicImage::ImageLuma8(_) => ImageFormat::Grayscale,
            DynamicImage::ImageRgb8(_) => ImageFormat::Rgb,
            DynamicImage::ImageRgba8(_) => ImageFormat::Rgba,
            _ => ImageFormat::Grayscale, // Default fallback
        };

        Self {
            data,
            format,
            width,
            height,
            dpi,
            metadata: HashMap::new(),
        }
    }

    /// Create a new OCR image from raw pixel data
    pub fn from_raw_pixels(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        format: ImageFormat,
        dpi: u32,
    ) -> Result<Self> {
        let expected_size = match format {
            ImageFormat::Grayscale => (width * height) as usize,
            ImageFormat::Rgb => (width * height * 3) as usize,
            ImageFormat::Rgba => (width * height * 4) as usize,
            ImageFormat::Binary => ((width * height + 7) / 8) as usize,
        };

        if pixels.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Expected {} bytes, got {}",
                expected_size,
                pixels.len()
            ));
        }

        let data = match format {
            ImageFormat::Grayscale => DynamicImage::ImageLuma8(
                GrayImage::from_raw(width, height, pixels)
                    .ok_or_else(|| anyhow::anyhow!("Invalid grayscale image data"))?,
            ),
            ImageFormat::Rgb => DynamicImage::ImageRgb8(
                RgbImage::from_raw(width, height, pixels)
                    .ok_or_else(|| anyhow::anyhow!("Invalid RGB image data"))?,
            ),
            ImageFormat::Rgba => DynamicImage::ImageRgba8(
                RgbaImage::from_raw(width, height, pixels)
                    .ok_or_else(|| anyhow::anyhow!("Invalid RGBA image data"))?,
            ),
            ImageFormat::Binary => {
                // Convert binary data to grayscale
                let mut gray_pixels = Vec::new();
                for byte in pixels {
                    for bit in 0..8 {
                        let pixel = if (byte >> (7 - bit)) & 1 == 1 { 255 } else { 0 };
                        gray_pixels.push(pixel);
                    }
                }
                DynamicImage::ImageLuma8(
                    GrayImage::from_raw(width, height, gray_pixels)
                        .ok_or_else(|| anyhow::anyhow!("Invalid binary image data"))?,
                )
            }
        };

        Ok(Self {
            data,
            format,
            width,
            height,
            dpi,
            metadata: HashMap::new(),
        })
    }

    /// Convert image to grayscale
    pub fn to_grayscale(&self) -> Self {
        let gray_data = self.data.to_luma8();
        Self {
            data: DynamicImage::ImageLuma8(gray_data),
            format: ImageFormat::Grayscale,
            width: self.width,
            height: self.height,
            dpi: self.dpi,
            metadata: self.metadata.clone(),
        }
    }

    /// Convert image to RGB
    pub fn to_rgb(&self) -> Self {
        let rgb_data = self.data.to_rgb8();
        Self {
            data: DynamicImage::ImageRgb8(rgb_data),
            format: ImageFormat::Rgb,
            width: self.width,
            height: self.height,
            dpi: self.dpi,
            metadata: self.metadata.clone(),
        }
    }

    /// Convert image to binary (using fixed threshold)
    pub fn to_binary(&self, threshold: u8) -> Self {
        self.threshold(threshold)
    }

    /// Get pixel value at coordinates
    pub fn get_pixel(&self, x: u32, y: u32) -> Result<PixelValue> {
        if x >= self.width || y >= self.height {
            return Err(anyhow::anyhow!(
                "Coordinates ({}, {}) out of bounds ({}, {})",
                x,
                y,
                self.width,
                self.height
            ));
        }

        let pixel = self.data.get_pixel(x, y);
        Ok(PixelValue::from_rgba(
            pixel[0], pixel[1], pixel[2], pixel[3],
        ))
    }

    /// Set pixel value at coordinates
    pub fn set_pixel(&mut self, x: u32, y: u32, value: PixelValue) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Err(anyhow::anyhow!(
                "Coordinates ({}, {}) out of bounds ({}, {})",
                x,
                y,
                self.width,
                self.height
            ));
        }

        let rgba = value.to_rgba();
        self.data
            .put_pixel(x, y, image::Rgba([rgba[0], rgba[1], rgba[2], rgba[3]]));
        Ok(())
    }

    /// Crop image to specified rectangle
    pub fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> Result<Self> {
        if x + width > self.width || y + height > self.height {
            return Err(anyhow::anyhow!("Crop rectangle out of bounds"));
        }

        let cropped = self.data.crop_imm(x, y, width, height);
        Ok(Self {
            data: cropped,
            format: self.format,
            width,
            height,
            dpi: self.dpi,
            metadata: self.metadata.clone(),
        })
    }

    /// Get image statistics
    pub fn statistics(&self) -> ImageStatistics {
        let mut min_val = 255;
        let mut max_val = 0;
        let mut sum = 0u64;
        let mut pixel_count = 0u64;

        for (x, y, pixel) in self.data.pixels() {
            let gray = pixel.to_luma().0[0];
            min_val = min_val.min(gray);
            max_val = max_val.max(gray);
            sum += gray as u64;
            pixel_count += 1;
        }

        let mean = if pixel_count > 0 {
            sum as f32 / pixel_count as f32
        } else {
            0.0
        };

        ImageStatistics {
            min: min_val,
            max: max_val,
            mean,
            pixel_count,
        }
    }

    /// Apply threshold to create binary image
    pub fn threshold(&self, threshold: u8) -> Self {
        let gray = self.data.to_luma8();
        let binary: GrayImage = GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
            let pixel = gray.get_pixel(x, y);
            if pixel[0] > threshold {
                image::Luma([255])
            } else {
                image::Luma([0])
            }
        });

        Self {
            data: DynamicImage::ImageLuma8(binary),
            format: ImageFormat::Binary,
            width: self.width,
            height: self.height,
            dpi: self.dpi,
            metadata: self.metadata.clone(),
        }
    }

    /// Invert image colors (for dark background images)
    pub fn invert(&self) -> Result<Self> {
        use image::imageops;

        // Convert to grayscale first for consistent handling
        let gray = self.data.to_luma8();
        let mut inverted_gray = gray.clone();
        imageops::invert(&mut inverted_gray);

        Ok(Self {
            data: DynamicImage::ImageLuma8(inverted_gray),
            format: ImageFormat::Grayscale,
            width: self.width,
            height: self.height,
            dpi: self.dpi,
            metadata: self.metadata.clone(),
        })
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Rotate image by angle (in radians)
    pub fn rotate(&self, angle: f32) -> Result<Self> {
        let rotated = match self.format {
            ImageFormat::Grayscale | ImageFormat::Binary => {
                let gray = self.data.to_luma8();
                let rotated = rotate_about_center(
                    &gray,
                    angle,
                    Interpolation::Bilinear,
                    image::Luma([255]), // White background
                );
                DynamicImage::ImageLuma8(rotated)
            }
            ImageFormat::Rgb => {
                let rgb = self.data.to_rgb8();
                let rotated = rotate_about_center(
                    &rgb,
                    angle,
                    Interpolation::Bilinear,
                    image::Rgb([255, 255, 255]), // White background
                );
                DynamicImage::ImageRgb8(rotated)
            }
            ImageFormat::Rgba => {
                let rgba = self.data.to_rgba8();
                let rotated = rotate_about_center(
                    &rgba,
                    angle,
                    Interpolation::Bilinear,
                    image::Rgba([255, 255, 255, 0]), // Transparent background
                );
                DynamicImage::ImageRgba8(rotated)
            }
        };

        Ok(Self::new(rotated, self.dpi))
    }

    /// Resize image
    pub fn resize(&self, width: u32, height: u32) -> Result<Self> {
        let resized = self
            .data
            .resize(width, height, image::imageops::FilterType::Lanczos3);
        Ok(OcrImage {
            data: resized,
            format: self.format,
            width,
            height,
            dpi: self.dpi,
            metadata: self.metadata.clone(),
        })
    }

    /// Translate image
    pub fn translate(&self, tx: f32, ty: f32) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Adjust brightness
    pub fn adjust_brightness(&self, factor: f32) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Adjust contrast
    pub fn adjust_contrast(&self, factor: f32) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Add noise
    pub fn add_noise(&self, noise_level: f32) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Apply Gaussian blur
    pub fn gaussian_blur(&self, blur_radius: f32) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Apply perspective transform
    pub fn perspective_transform(
        &self,
        src_points: &[(f32, f32)],
        dst_points: &[(f32, f32)],
    ) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Apply elastic deformation
    pub fn elastic_deformation(&self, alpha: f32, sigma: f32) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }

    /// Apply color jitter
    pub fn color_jitter(
        &self,
        hue_shift: f32,
        saturation_factor: f32,
        value_factor: f32,
    ) -> Result<Self> {
        // Placeholder implementation
        Ok(self.clone())
    }
}

/// Pixel value representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelValue {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl PixelValue {
    /// Create a new pixel value from RGBA components
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create a pixel value from RGBA array
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to RGBA array
    pub fn to_rgba(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Convert to grayscale value
    pub fn to_grayscale(self) -> u8 {
        // Use standard luminance formula
        ((self.r as f32 * 0.299) + (self.g as f32 * 0.587) + (self.b as f32 * 0.114)) as u8
    }

    /// Check if pixel is white (for binary images)
    pub fn is_white(self) -> bool {
        self.r > 128 && self.g > 128 && self.b > 128
    }

    /// Check if pixel is black (for binary images)
    pub fn is_black(self) -> bool {
        self.r < 128 && self.g < 128 && self.b < 128
    }
}

/// Image statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageStatistics {
    pub min: u8,
    pub max: u8,
    pub mean: f32,
    pub pixel_count: u64,
}

/// Image region of interest
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ImageRegion {
    /// Create a new image region
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if region contains a point
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Get the area of the region
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Get the center point of the region
    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.x + self.width / 2) as f32,
            (self.y + self.height / 2) as f32,
        )
    }
}
