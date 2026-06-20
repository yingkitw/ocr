//! Image distortion pipeline for synthetic OCR data augmentation
//!
//! Applies realistic degradations to synthetic text-line images to simulate
//! scanner noise, camera blur, uneven lighting, and page warp.

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba};
use imageproc::geometric_transformations::{warp, Interpolation, Projection};
use rand::Rng;

/// Configuration for which distortions to apply and their intensity ranges
#[derive(Debug, Clone)]
pub struct DistortionConfig {
    /// Rotation angle in degrees (uniform random in [-max, max])
    pub rotation_degrees: f32,
    /// Gaussian blur sigma (uniform random in [0, max])
    pub blur_sigma: f32,
    /// Salt & pepper noise probability per pixel
    pub noise_probability: f32,
    /// Contrast factor multiplier (uniform random in [1/max, max])
    pub contrast_factor: f32,
    /// Brightness offset (uniform random in [-max, max])
    pub brightness_offset: i16,
    /// Perspective shear amount (uniform random in [-max, max])
    pub perspective_shear: f32,
    /// Probability of applying each distortion (0.0–1.0)
    pub apply_probability: f32,
}

impl Default for DistortionConfig {
    fn default() -> Self {
        Self {
            rotation_degrees: 2.0,
            blur_sigma: 1.0,
            noise_probability: 0.02,
            contrast_factor: 1.2,
            brightness_offset: 15,
            perspective_shear: 0.05,
            apply_probability: 0.8,
        }
    }
}

impl DistortionConfig {
    /// Mild distortions (clean synthetic data)
    pub fn mild() -> Self {
        Self {
            rotation_degrees: 1.0,
            blur_sigma: 0.5,
            noise_probability: 0.01,
            contrast_factor: 1.1,
            brightness_offset: 8,
            perspective_shear: 0.02,
            apply_probability: 0.5,
        }
    }

    /// Heavy distortions (challenging synthetic data)
    pub fn heavy() -> Self {
        Self {
            rotation_degrees: 5.0,
            blur_sigma: 2.0,
            noise_probability: 0.05,
            contrast_factor: 1.5,
            brightness_offset: 30,
            perspective_shear: 0.15,
            apply_probability: 1.0,
        }
    }
}

/// Apply a configured set of distortions to an image
pub fn apply_distortions(image: &DynamicImage, config: &DistortionConfig) -> DynamicImage {
    let mut rng = rand::thread_rng();
    let mut result = image.clone();

    // Rotation
    if rng.gen::<f32>() < config.apply_probability {
        let angle = rng.gen_range(-config.rotation_degrees..=config.rotation_degrees);
        result = rotate_image(&result, angle);
    }

    // Gaussian blur
    if rng.gen::<f32>() < config.apply_probability {
        let sigma = rng.gen_range(0.0..=config.blur_sigma);
        if sigma > 0.1 {
            result = blur_image(&result, sigma);
        }
    }

    // Salt & pepper noise
    if rng.gen::<f32>() < config.apply_probability {
        result = add_noise(&result, config.noise_probability);
    }

    // Contrast adjustment
    if rng.gen::<f32>() < config.apply_probability {
        let factor = rng.gen_range(1.0 / config.contrast_factor..=config.contrast_factor);
        result = adjust_contrast(&result, factor);
    }

    // Brightness offset
    if rng.gen::<f32>() < config.apply_probability {
        let offset = rng.gen_range(-config.brightness_offset..=config.brightness_offset);
        result = adjust_brightness(&result, offset);
    }

    // Perspective shear
    if rng.gen::<f32>() < config.apply_probability {
        let shear = rng.gen_range(-config.perspective_shear..=config.perspective_shear);
        if shear.abs() > 0.001 {
            result = apply_shear(&result, shear);
        }
    }

    result
}

/// Rotate an image around its center
fn rotate_image(image: &DynamicImage, degrees: f32) -> DynamicImage {
    let radians = degrees.to_radians();
    let (w, h) = (image.width(), image.height());
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    // Use imageproc rotate_about_center
    let rotated = imageproc::geometric_transformations::rotate_about_center(
        &image.to_luma8(),
        radians,
        Interpolation::Bilinear,
        Luma([255]),
    );

    DynamicImage::ImageLuma8(rotated)
}

/// Apply Gaussian blur
fn blur_image(image: &DynamicImage, sigma: f32) -> DynamicImage {
    let blurred = imageproc::filter::gaussian_blur_f32(&image.to_luma8(), sigma);
    DynamicImage::ImageLuma8(blurred)
}

/// Add salt & pepper noise
fn add_noise(image: &DynamicImage, probability: f32) -> DynamicImage {
    let mut rng = rand::thread_rng();
    let mut img = image.to_luma8();

    for pixel in img.pixels_mut() {
        let roll: f32 = rng.gen();
        if roll < probability / 2.0 {
            pixel.0[0] = 0; // pepper (black)
        } else if roll < probability {
            pixel.0[0] = 255; // salt (white)
        }
    }

    DynamicImage::ImageLuma8(img)
}

/// Adjust contrast by scaling around mean
fn adjust_contrast(image: &DynamicImage, factor: f32) -> DynamicImage {
    let img = image.to_luma8();
    let mean = compute_mean_brightness(&img);

    let adjusted: GrayImage = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let val = img.get_pixel(x, y).0[0] as f32;
        let adjusted = mean + (val - mean) * factor;
        Luma([adjusted.clamp(0.0, 255.0) as u8])
    });

    DynamicImage::ImageLuma8(adjusted)
}

/// Adjust brightness by adding an offset
fn adjust_brightness(image: &DynamicImage, offset: i16) -> DynamicImage {
    let img = image.to_luma8();

    let adjusted: GrayImage = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let val = img.get_pixel(x, y).0[0] as i16;
        let adjusted = val + offset;
        Luma([adjusted.clamp(0, 255) as u8])
    });

    DynamicImage::ImageLuma8(adjusted)
}

/// Apply perspective shear
fn apply_shear(image: &DynamicImage, shear: f32) -> DynamicImage {
    let (w, h) = (image.width() as f32, image.height() as f32);

    // Simple horizontal shear projection
    let projection = Projection::from_control_points(
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
        [
            (0.0, 0.0),
            (w, shear * h),
            (w + shear * w, h - shear * h),
            (shear * w, h),
        ],
    )
    .unwrap_or_else(|| Projection::translate(0.0, 0.0));

    let warped = warp(
        &image.to_luma8(),
        &projection,
        Interpolation::Bilinear,
        Luma([255]),
    );

    DynamicImage::ImageLuma8(warped)
}

fn compute_mean_brightness(img: &GrayImage) -> f32 {
    let sum: u64 = img.pixels().map(|p| p.0[0] as u64).sum();
    let count = img.width() * img.height();
    if count == 0 {
        128.0
    } else {
        (sum as f32) / (count as f32)
    }
}

/// Augment a batch of synthetic samples in-place
pub fn augment_batch(samples: &mut [crate::synthetic::SyntheticSample], config: &DistortionConfig) {
    for sample in samples.iter_mut() {
        sample.image = apply_distortions(&sample.image, config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::TextLineGenerator;

    #[test]
    fn test_distortions_dont_panic() {
        let gen = TextLineGenerator::default();
        let sample = gen.generate("Test");

        let configs = [
            DistortionConfig::mild(),
            DistortionConfig::default(),
            DistortionConfig::heavy(),
        ];

        for config in &configs {
            let distorted = apply_distortions(&sample.image, config);
            assert!(distorted.width() > 0);
            assert!(distorted.height() > 0);
        }
    }

    #[test]
    fn test_contrast_adjustment() {
        let img = GrayImage::from_pixel(10, 10, Luma([128]));
        let adjusted = adjust_contrast(&DynamicImage::ImageLuma8(img), 1.5);
        // Mean should stay around 128, but spread increases
        let mean = compute_mean_brightness(&adjusted.to_luma8());
        assert!((mean - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_brightness_adjustment() {
        let img = GrayImage::from_pixel(10, 10, Luma([100]));
        let adjusted = adjust_brightness(&DynamicImage::ImageLuma8(img), 50);
        let luma = adjusted.to_luma8();
        let pixel = luma.get_pixel(0, 0);
        assert_eq!(pixel.0[0], 150);
    }

    #[test]
    fn test_noise_changes_pixels() {
        let img = GrayImage::from_pixel(10, 10, Luma([128]));
        let noisy = add_noise(&DynamicImage::ImageLuma8(img), 0.5);
        // With 50% noise, almost certainly some pixels changed
        let changed = noisy.to_luma8().pixels().any(|p| p.0[0] != 128);
        assert!(changed);
    }
}
