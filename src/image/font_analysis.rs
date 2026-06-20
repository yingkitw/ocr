//! Font attribute detection from word image crops
//!
//! Provides simple heuristic-based detection of bold, italic, and monospace
//! text properties using pixel-level analysis.

use crate::core::{BoundingBox, WordResult};
use crate::utils::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Luma};

/// Detect font attributes from a word image crop
pub fn analyze_font_attributes(
    image: &DynamicImage,
    word: &WordResult,
) -> Result<(bool, bool, bool)> {
    let crop = crop_word(image, &word.bounding_box);
    let gray = crop.to_luma8();

    let is_bold = detect_bold(&gray);
    let is_italic = detect_italic(&gray);
    let is_monospace = detect_monospace(word);

    Ok((is_bold, is_italic, is_monospace))
}

/// Crop image to word bounding box
fn crop_word(image: &DynamicImage, bbox: &BoundingBox) -> DynamicImage {
    let mut img = image.clone();
    let x = bbox.left.min(img.width() - 1);
    let y = bbox.top.min(img.height() - 1);
    let width = (bbox.right - bbox.left).min(img.width() - x);
    let height = (bbox.bottom - bbox.top).min(img.height() - y);
    img.crop(x, y, width, height)
}

/// Detect bold text by measuring stroke density.
/// Bold text has a higher ratio of dark pixels to total area.
fn detect_bold(gray: &GrayImage) -> bool {
    let (width, height) = gray.dimensions();
    if width == 0 || height == 0 {
        return false;
    }

    let total_pixels = (width * height) as u32;
    let dark_pixels: u32 = gray.pixels().map(|p| if p[0] < 128 { 1 } else { 0 }).sum();

    let density = dark_pixels as f32 / total_pixels as f32;
    // Normal text density is typically ~0.15-0.25 for clean images
    // Bold text tends to be >0.30
    density > 0.30
}

/// Detect italic text by measuring the slant of the word.
/// Uses the centroid shift between the top and bottom halves.
fn detect_italic(gray: &GrayImage) -> bool {
    let (width, height) = gray.dimensions();
    if width < 3 || height < 3 {
        return false;
    }

    let half = height / 2;

    // Compute centroid x for top half
    let (mut top_sum, mut top_count) = (0.0, 0.0);
    for y in 0..half {
        for x in 0..width {
            if gray.get_pixel(x, y)[0] < 128 {
                top_sum += x as f32;
                top_count += 1.0;
            }
        }
    }

    // Compute centroid x for bottom half
    let (mut bot_sum, mut bot_count) = (0.0, 0.0);
    for y in half..height {
        for x in 0..width {
            if gray.get_pixel(x, y)[0] < 128 {
                bot_sum += x as f32;
                bot_count += 1.0;
            }
        }
    }

    if top_count < 5.0 || bot_count < 5.0 {
        return false;
    }

    let top_centroid = top_sum / top_count;
    let bot_centroid = bot_sum / bot_count;
    let shift = bot_centroid - top_centroid;

    // If bottom is shifted significantly to the right relative to top, it's italic
    let shift_ratio = shift.abs() / width as f32;
    shift_ratio > 0.15
}

/// Detect monospace by checking if character widths are uniform.
fn detect_monospace(word: &WordResult) -> bool {
    if word.characters.len() < 3 {
        return false;
    }

    let widths: Vec<u32> = word
        .characters
        .iter()
        .map(|c| c.bounding_box.right - c.bounding_box.left)
        .collect();

    let avg_width = widths.iter().sum::<u32>() as f32 / widths.len() as f32;
    if avg_width < 1.0 {
        return false;
    }

    // Check if all widths are within 20% of the average
    widths.iter().all(|&w| {
        let diff = (w as f32 - avg_width).abs();
        diff / avg_width < 0.20
    })
}
