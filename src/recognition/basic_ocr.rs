//! Basic OCR engine implementation inspired by Tesseract
//!
//! This module implements a basic OCR engine that follows Tesseract's approach:
//! 1. Connected component analysis (blob detection)
//! 2. Text line and word segmentation
//! 3. Character recognition using pattern matching
//! 4. Basic language modeling

use crate::core::image::OcrImage;
use crate::core::recognition::{
    CharacterRecognition, LineRecognition, RecognitionResult, TextRecognizer, WordRecognition,
};
use crate::core::text::BoundingBox;
use crate::recognition::tesseract_blob::{
    extract_outlines, outlines_to_blobs,
};
use crate::recognition::tesseract_features::extract_features;
use crate::utils::Result;
use image::GrayImage;

/// Basic OCR engine following Tesseract's approach
pub struct BasicOcrEngine {
    /// Character templates for pattern matching
    character_templates: std::collections::BTreeMap<char, CharacterTemplate>,
    /// Minimum blob size (in pixels)
    min_blob_size: u32,
    /// Maximum blob size (in pixels)
    max_blob_size: u32,
}

/// Character template for pattern matching
#[derive(Debug, Clone)]
#[allow(dead_code)] // template metadata; matching currently compares the `template` bitmap only
struct CharacterTemplate {
    /// Character this template represents
    character: char,
    /// Template image (normalized size)
    template: Vec<Vec<u8>>,
    /// Template width
    width: u32,
    /// Template height
    height: u32,
}

/// Connected component (blob) detected in image
#[derive(Debug, Clone)]
struct Blob {
    /// Bounding box
    bounding_box: BoundingBox,
    /// Pixels in this blob
    pixels: Vec<(u32, u32)>,
    /// Average brightness
    brightness: f32,
    /// Stroke width (average)
    stroke_width: f32,
    /// Aspect ratio (width/height)
    aspect_ratio: f32,
    /// Pixel density (black pixels / total pixels)
    density: f32,
}

impl BasicOcrEngine {
    /// Create a new basic OCR engine
    pub fn new() -> Self {
        Self {
            character_templates: Self::create_default_templates(),
            min_blob_size: 4,
            max_blob_size: 200_000,
        }
    }

    /// Create an engine using trained templates from synthetic font rendering
    pub fn with_trained_templates(templates: &std::collections::HashMap<char, crate::synthetic::template_trainer::TrainedTemplate>) -> Self {
        let mut character_templates = std::collections::BTreeMap::new();
        for (ch, tpl) in templates {
            character_templates.insert(
                *ch,
                CharacterTemplate {
                    character: *ch,
                    template: tpl.template.clone(),
                    width: tpl.width,
                    height: tpl.height,
                },
            );
        }
        Self {
            character_templates,
            min_blob_size: 4,
            max_blob_size: 200_000,
        }
    }

    /// Recognize text from image
    pub fn recognize_sync(&self, image: &OcrImage) -> Result<RecognitionResult> {
        // Step 1: Convert to binary image (thresholding)
        let binary_image = self.threshold_image(image)?;

        // Step 2: Detect connected components (blobs)
        let blobs = self.detect_blobs(&binary_image)?;

        // Step 3: Group blobs into lines
        let lines = self.group_blobs_into_lines(&blobs)?;

        // Step 4: Recognize characters in each line
        let mut recognized_lines = Vec::new();
        let mut all_words: Vec<WordRecognition> = Vec::new();
        let mut all_characters: Vec<CharacterRecognition> = Vec::new();
        let mut full_text = String::new();
        let _all_lines: Vec<LineRecognition> = Vec::new();

        for line_blobs in &lines {
            let line_result = self.recognize_line(&binary_image, line_blobs)?;
            recognized_lines.push(line_result.clone());

            for word in &line_result.words {
                all_words.push(word.clone());
                for char_result in &word.characters {
                    all_characters.push(char_result.clone());
                }
            }

            if !line_result.line.is_empty() {
                if !full_text.is_empty() {
                    full_text.push('\n');
                }
                full_text.push_str(&line_result.line);
            }
        }

        // Join hyphenated words across lines (e.g. "hel-\nlo" -> "hello")
        let confidence = if recognized_lines.is_empty() {
            0.0
        } else {
            recognized_lines.iter().map(|l| l.confidence).sum::<f32>()
                / recognized_lines.len() as f32
        };

        let mut result = RecognitionResult {
            text: full_text,
            confidence,
            characters: all_characters,
            words: all_words,
            lines: recognized_lines,
            metadata: Default::default(),
            model_type: Some(crate::core::recognition::ModelType::Custom(
                "BasicOCR".to_string(),
            )),
            processing_time_ms: None,
            language: Some("en".to_string()),
            character_results: Vec::new(),
            word_results: Vec::new(),
            line_results: Vec::new(),
        };

        self.join_hyphenated_words(&mut result);

        Ok(result)
    }

    /// Threshold image to binary
    fn threshold_image(&self, image: &OcrImage) -> Result<GrayImage> {
        use crate::image::{ImageThresholder, ThresholdMethod};

        let mut thresholder = ImageThresholder::new();
        thresholder.set_image(image.clone())?;
        let binary = thresholder.threshold(ThresholdMethod::Otsu)?;

        // Convert to GrayImage
        Ok(binary.data.to_luma8())
    }

    /// Detect connected components (blobs) in binary image
    fn detect_blobs(&self, image: &GrayImage) -> Result<Vec<Blob>> {
        let (width, height) = image.dimensions();
        let mut visited = vec![vec![false; height as usize]; width as usize];
        let mut blobs = Vec::new();
        let image_area = (width as u64) * (height as u64);

        // Find all connected components
        // Try both black-on-white and white-on-black (inverted) images
        for y in 0..height {
            for x in 0..width {
                if !visited[x as usize][y as usize] {
                    let pixel = image.get_pixel(x, y);
                    // In binary image, 0 is black (text), 255 is white (background)
                    // We're looking for dark pixels (text)
                    // Use threshold of 128 to handle both binary and grayscale images
                    if pixel[0] < 128 {
                        let blob = self.flood_fill(image, x, y, &mut visited)?;
                        let bbox_area = (blob.bounding_box.width() as u64)
                            * (blob.bounding_box.height() as u64);
                        let touches_border = blob.bounding_box.left == 0
                            || blob.bounding_box.top == 0
                            || blob.bounding_box.right >= width
                            || blob.bounding_box.bottom >= height;
                        if touches_border && bbox_area > image_area / 2 {
                            continue;
                        }
                        // Filter blobs by size - minimal filtering to avoid removing valid characters
                        if blob.pixels.len() >= self.min_blob_size as usize
                            && blob.pixels.len() <= self.max_blob_size as usize
                        {
                            // Only filter out obvious noise - be very permissive
                            // Filter out extremely thin horizontal lines (likely noise)
                            if blob.aspect_ratio > 100.0 {
                                continue;
                            }
                            // Filter out extremely tall thin lines (likely noise)
                            if blob.aspect_ratio < 0.01 && blob.bounding_box.height() > 300 {
                                continue;
                            }
                            blobs.push(blob);
                        }
                    }
                }
            }
        }

        // If no blobs found with dark pixels, try inverted (white-on-black)
        if blobs.is_empty() {
            let mut visited_inv = vec![vec![false; height as usize]; width as usize];
            for y in 0..height {
                for x in 0..width {
                    if !visited_inv[x as usize][y as usize] {
                        let pixel = image.get_pixel(x, y);
                        // Look for bright pixels (white text on dark background)
                        if pixel[0] >= 128 {
                            let blob = self.flood_fill_inverted(image, x, y, &mut visited_inv)?;
                            let bbox_area = (blob.bounding_box.width() as u64)
                                * (blob.bounding_box.height() as u64);
                            let touches_border = blob.bounding_box.left == 0
                                || blob.bounding_box.top == 0
                                || blob.bounding_box.right >= width
                                || blob.bounding_box.bottom >= height;
                            if touches_border && bbox_area > image_area / 2 {
                                continue;
                            }
                            if blob.pixels.len() >= self.min_blob_size as usize
                                && blob.pixels.len() <= self.max_blob_size as usize
                            {
                                // Only filter out obvious noise - be very permissive
                                if blob.aspect_ratio > 100.0 {
                                    continue;
                                }
                                if blob.aspect_ratio < 0.01 && blob.bounding_box.height() > 300 {
                                    continue;
                                }
                                blobs.push(blob);
                            }
                        }
                    }
                }
            }
        }

        Ok(blobs)
    }

    /// Flood fill to find connected component (dark pixels)
    fn flood_fill(
        &self,
        image: &GrayImage,
        start_x: u32,
        start_y: u32,
        visited: &mut Vec<Vec<bool>>,
    ) -> Result<Blob> {
        let (width, height) = image.dimensions();
        let mut pixels = Vec::new();
        let mut stack = vec![(start_x, start_y)];
        let mut min_x = start_x;
        let mut min_y = start_y;
        let mut max_x = start_x;
        let mut max_y = start_y;
        let mut total_brightness = 0u32;

        while let Some((x, y)) = stack.pop() {
            if x >= width || y >= height {
                continue;
            }

            if visited[x as usize][y as usize] {
                continue;
            }

            let pixel = image.get_pixel(x, y);
            if pixel[0] >= 128 {
                continue; // Bright pixel (background for dark text)
            }

            visited[x as usize][y as usize] = true;
            pixels.push((x, y));
            total_brightness += pixel[0] as u32;

            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);

            // Add neighbors to stack (8-connectivity)
            if x > 0 {
                stack.push((x - 1, y));
            }
            if x < width - 1 {
                stack.push((x + 1, y));
            }
            if y > 0 {
                stack.push((x, y - 1));
            }
            if y < height - 1 {
                stack.push((x, y + 1));
            }
            if x > 0 && y > 0 {
                stack.push((x - 1, y - 1));
            }
            if x > 0 && y < height - 1 {
                stack.push((x - 1, y + 1));
            }
            if x < width - 1 && y > 0 {
                stack.push((x + 1, y - 1));
            }
            if x < width - 1 && y < height - 1 {
                stack.push((x + 1, y + 1));
            }
        }

        let brightness = if !pixels.is_empty() {
            total_brightness as f32 / pixels.len() as f32
        } else {
            0.0
        };

        let width = (max_x + 1).saturating_sub(min_x);
        let height = (max_y + 1).saturating_sub(min_y);
        let aspect_ratio = if height > 0 {
            width as f32 / height as f32
        } else {
            1.0
        };
        let density = if width > 0 && height > 0 {
            pixels.len() as f32 / (width * height) as f32
        } else {
            0.0
        };

        // Estimate stroke width (simplified - average distance from edge)
        let stroke_width = Self::estimate_stroke_width(image, min_x, min_y, max_x, max_y, &pixels);

        Ok(Blob {
            bounding_box: BoundingBox::new(min_x, min_y, max_x + 1, max_y + 1),
            pixels,
            brightness,
            stroke_width,
            aspect_ratio,
            density,
        })
    }

    /// Flood fill to find connected component (bright pixels - for inverted images)
    fn flood_fill_inverted(
        &self,
        image: &GrayImage,
        start_x: u32,
        start_y: u32,
        visited: &mut Vec<Vec<bool>>,
    ) -> Result<Blob> {
        let (width, height) = image.dimensions();
        let mut pixels = Vec::new();
        let mut stack = vec![(start_x, start_y)];
        let mut min_x = start_x;
        let mut min_y = start_y;
        let mut max_x = start_x;
        let mut max_y = start_y;
        let mut total_brightness = 0u32;

        while let Some((x, y)) = stack.pop() {
            if x >= width || y >= height {
                continue;
            }

            if visited[x as usize][y as usize] {
                continue;
            }

            let pixel = image.get_pixel(x, y);
            if pixel[0] < 128 {
                continue; // Dark pixel (background for bright text)
            }

            visited[x as usize][y as usize] = true;
            pixels.push((x, y));
            total_brightness += pixel[0] as u32;

            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);

            // Add neighbors to stack (8-connectivity)
            if x > 0 {
                stack.push((x - 1, y));
            }
            if x < width - 1 {
                stack.push((x + 1, y));
            }
            if y > 0 {
                stack.push((x, y - 1));
            }
            if y < height - 1 {
                stack.push((x, y + 1));
            }
            if x > 0 && y > 0 {
                stack.push((x - 1, y - 1));
            }
            if x > 0 && y < height - 1 {
                stack.push((x - 1, y + 1));
            }
            if x < width - 1 && y > 0 {
                stack.push((x + 1, y - 1));
            }
            if x < width - 1 && y < height - 1 {
                stack.push((x + 1, y + 1));
            }
        }

        let brightness = if !pixels.is_empty() {
            total_brightness as f32 / pixels.len() as f32
        } else {
            0.0
        };

        let width = (max_x + 1).saturating_sub(min_x);
        let height = (max_y + 1).saturating_sub(min_y);
        let aspect_ratio = if height > 0 {
            width as f32 / height as f32
        } else {
            1.0
        };
        let density = if width > 0 && height > 0 {
            pixels.len() as f32 / (width * height) as f32
        } else {
            0.0
        };

        // Estimate stroke width (simplified - average distance from edge)
        let stroke_width = Self::estimate_stroke_width(image, min_x, min_y, max_x, max_y, &pixels);

        Ok(Blob {
            bounding_box: BoundingBox::new(min_x, min_y, max_x + 1, max_y + 1),
            pixels,
            brightness,
            stroke_width,
            aspect_ratio,
            density,
        })
    }

    /// Estimate stroke width of a blob (simplified method)
    fn estimate_stroke_width(
        _image: &GrayImage,
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        pixels: &[(u32, u32)],
    ) -> f32 {
        if pixels.is_empty() {
            return 1.0;
        }

        // Simple estimation: find average distance to nearest edge
        let mut total_distance = 0.0;
        let mut count = 0;

        for &(x, y) in pixels.iter().take(100) {
            // Sample up to 100 pixels for performance
            let dist_to_left = (x - min_x) as f32;
            let dist_to_right = (max_x - x) as f32;
            let dist_to_top = (y - min_y) as f32;
            let dist_to_bottom = (max_y - y) as f32;

            let min_dist = dist_to_left
                .min(dist_to_right)
                .min(dist_to_top)
                .min(dist_to_bottom);
            total_distance += min_dist;
            count += 1;
        }

        if count > 0 {
            (total_distance / count as f32).max(1.0)
        } else {
            1.0
        }
    }

    /// Group blobs into text lines using projection profile method (Tesseract-inspired)
    fn group_blobs_into_lines(&self, blobs: &[Blob]) -> Result<Vec<Vec<Blob>>> {
        if blobs.is_empty() {
            return Ok(Vec::new());
        }

        // Build vertical projection profile (like Tesseract's TextlineProjection)
        let _max_right = blobs
            .iter()
            .map(|b| b.bounding_box.right)
            .max()
            .unwrap_or(0);
        let max_bottom = blobs
            .iter()
            .map(|b| b.bounding_box.bottom)
            .max()
            .unwrap_or(0);
        let image_height = max_bottom.max(100); // Ensure minimum height

        // Create projection profile: for each y, count how many blobs overlap (weighted by width)
        let mut projection = vec![0u32; image_height as usize];
        for blob in blobs {
            let top = blob.bounding_box.top as usize;
            let bottom = blob.bounding_box.bottom as usize;
            let width = blob.bounding_box.width();

            for y in top..bottom.min(projection.len()) {
                // Weight by blob width (like Tesseract does)
                projection[y] += width;
            }
        }

        // Smooth the projection (like Tesseract's blockconv)
        let smoothed = Self::smooth_projection(&projection, 3);

        // Find peaks in projection (text lines)
        let line_centers = Self::find_projection_peaks(&smoothed, 0.2);

        // If no peaks found or too few peaks, fall back to simple grouping
        // Also fall back if projection method doesn't seem to work well
        if line_centers.is_empty() || line_centers.len() < blobs.len() / 10 {
            return self.group_blobs_into_lines_simple(blobs);
        }

        // Group blobs by nearest line center
        let mut lines: Vec<Vec<Blob>> = vec![Vec::new(); line_centers.len()];

        for blob in blobs {
            let blob_center_y = (blob.bounding_box.top + blob.bounding_box.bottom) / 2;

            // Find nearest line center
            let mut best_line_idx = 0;
            let mut min_distance = u32::MAX;

            for (idx, &line_y) in line_centers.iter().enumerate() {
                let distance = blob_center_y.abs_diff(line_y);
                if distance < min_distance {
                    min_distance = distance;
                    best_line_idx = idx;
                }
            }

            // Check if blob is close enough to the line
            let line_y = line_centers[best_line_idx];
            let blob_height = blob.bounding_box.height();
            let tolerance = blob_height.max(5) / 2;

            if blob_center_y.abs_diff(line_y) <= tolerance {
                lines[best_line_idx].push(blob.clone());
            }
        }

        // Remove empty lines and sort blobs horizontally within each line
        let mut result_lines = Vec::new();
        for mut line in lines {
            if !line.is_empty() {
                line.sort_by(|a, b| a.bounding_box.left.cmp(&b.bounding_box.left));
                result_lines.push(line);
            }
        }

        Ok(result_lines)
    }

    /// Simple line grouping fallback (original method)
    fn group_blobs_into_lines_simple(&self, blobs: &[Blob]) -> Result<Vec<Vec<Blob>>> {
        if blobs.is_empty() {
            return Ok(Vec::new());
        }

        // Sort blobs by y-position
        let mut sorted_blobs = blobs.to_vec();
        sorted_blobs.sort_by(|a, b| {
            let a_center_y = (a.bounding_box.top + a.bounding_box.bottom) / 2;
            let b_center_y = (b.bounding_box.top + b.bounding_box.bottom) / 2;
            a_center_y.cmp(&b_center_y)
        });

        // Group blobs that are on the same line
        let mut lines = Vec::new();
        let mut current_line = Vec::new();
        let mut last_y = None;

        for blob in sorted_blobs {
            let center_y = (blob.bounding_box.top + blob.bounding_box.bottom) / 2;
            let height = blob.bounding_box.height();
            let tolerance = height.max(5) / 2;

            if let Some(last_y_pos) = last_y {
                if center_y > last_y_pos + tolerance {
                    // New line
                    if !current_line.is_empty() {
                        lines.push(current_line);
                        current_line = Vec::new();
                    }
                }
            }

            current_line.push(blob);
            last_y = Some(center_y);
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        // Sort blobs within each line horizontally
        for line in &mut lines {
            line.sort_by(|a, b| a.bounding_box.left.cmp(&b.bounding_box.left));
        }

        Ok(lines)
    }

    /// Smooth projection profile using simple moving average
    fn smooth_projection(projection: &[u32], window_size: usize) -> Vec<u32> {
        if projection.is_empty() {
            return Vec::new();
        }

        let half_window = window_size / 2;
        let mut smoothed = vec![0u32; projection.len()];

        for i in 0..projection.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(projection.len());
            let sum: u32 = projection[start..end].iter().sum();
            smoothed[i] = sum / (end - start).max(1) as u32;
        }

        smoothed
    }

    /// Find peaks in projection profile (text line centers)
    fn find_projection_peaks(projection: &[u32], threshold_ratio: f32) -> Vec<u32> {
        if projection.is_empty() {
            return Vec::new();
        }

        let max_value = *projection.iter().max().unwrap_or(&1) as f32;
        let threshold = (max_value * threshold_ratio) as u32;

        let mut peaks = Vec::new();
        let mut in_peak = false;
        let mut peak_max = 0;
        let mut peak_max_y = 0;

        for (y, &value) in projection.iter().enumerate() {
            if value >= threshold {
                if !in_peak {
                    in_peak = true;
                    peak_max = value;
                    peak_max_y = y;
                } else if value > peak_max {
                    peak_max = value;
                    peak_max_y = y;
                }
            } else if in_peak {
                // End of peak - record the center
                peaks.push(peak_max_y as u32);
                in_peak = false;
            }
        }

        // Handle peak at end of projection
        if in_peak {
            peaks.push(peak_max_y as u32);
        }

        peaks
    }

    /// Recognize characters in a line
    fn recognize_line(&self, image: &GrayImage, blobs: &[Blob]) -> Result<LineRecognition> {
        let mut words = Vec::new();
        let mut characters = Vec::new();
        let mut line_text = String::new();

        let merged_blobs = self.merge_character_blobs(blobs);

        let word_groups = self.group_blobs_into_words(&merged_blobs);

        for word_blobs in word_groups {
            let word_result = self.recognize_word(image, &word_blobs)?;
            words.push(word_result.clone());

            for char_result in &word_result.characters {
                characters.push(char_result.clone());
            }

            if !line_text.is_empty() {
                line_text.push(' ');
            }
            line_text.push_str(&word_result.word);
        }

        let confidence = if !characters.is_empty() {
            characters.iter().map(|c| c.confidence).sum::<f32>() / characters.len() as f32
        } else {
            0.0
        };

        let bounding_box = if !merged_blobs.is_empty() {
            let mut min_x = u32::MAX;
            let mut min_y = u32::MAX;
            let mut max_x = 0u32;
            let mut max_y = 0u32;

            for blob in &merged_blobs {
                min_x = min_x.min(blob.bounding_box.left);
                min_y = min_y.min(blob.bounding_box.top);
                max_x = max_x.max(blob.bounding_box.right);
                max_y = max_y.max(blob.bounding_box.bottom);
            }

            BoundingBox::new(min_x, min_y, max_x, max_y)
        } else {
            BoundingBox::new(0, 0, 0, 0)
        };

        let mut line = LineRecognition::with_bounding_box(line_text, confidence, bounding_box);
        line.words = words;

        Ok(line)
    }

    fn merge_character_blobs(&self, blobs: &[Blob]) -> Vec<Blob> {
        if blobs.is_empty() {
            return Vec::new();
        }

        let mut widths: Vec<u32> = blobs
            .iter()
            .map(|b| b.bounding_box.width().max(1))
            .collect();
        widths.sort_unstable();
        let median_width = widths[widths.len() / 2].max(1);

        let mut heights: Vec<u32> = blobs
            .iter()
            .map(|b| b.bounding_box.height().max(1))
            .collect();
        heights.sort_unstable();
        let median_height = heights[heights.len() / 2].max(1);

        let merge_gap = ((median_width as f32) * 1.5).ceil() as u32;
        let max_char_width = ((median_height as f32) * 1.25).ceil() as u32;

        let mut sorted = blobs.to_vec();
        sorted.sort_by(|a, b| a.bounding_box.left.cmp(&b.bounding_box.left));

        let mut out: Vec<Blob> = Vec::new();
        let mut current = sorted[0].clone();

        for blob in sorted.into_iter().skip(1) {
            let gap = blob
                .bounding_box
                .left
                .saturating_sub(current.bounding_box.right);

            let overlap_top = current.bounding_box.top.max(blob.bounding_box.top);
            let overlap_bottom = current.bounding_box.bottom.min(blob.bounding_box.bottom);
            let overlap_h = overlap_bottom.saturating_sub(overlap_top);
            let min_h = current
                .bounding_box
                .height()
                .min(blob.bounding_box.height())
                .max(1);
            let overlap_ratio = (overlap_h as f32) / (min_h as f32);

            let min_x = current.bounding_box.left.min(blob.bounding_box.left);
            let min_y = current.bounding_box.top.min(blob.bounding_box.top);
            let max_x = current.bounding_box.right.max(blob.bounding_box.right);
            let max_y = current.bounding_box.bottom.max(blob.bounding_box.bottom);
            let merged_width = max_x.saturating_sub(min_x);

            if gap <= merge_gap && overlap_ratio >= 0.5 && merged_width <= max_char_width {
                let total_pixels = (current.pixels.len() + blob.pixels.len()).max(1) as f32;
                let current_w = current.pixels.len() as f32 / total_pixels;
                let blob_w = blob.pixels.len() as f32 / total_pixels;

                current.bounding_box = BoundingBox::new(min_x, min_y, max_x, max_y);

                current.pixels.extend(blob.pixels);
                current.brightness = current.brightness * current_w + blob.brightness * blob_w;
                current.stroke_width =
                    current.stroke_width * current_w + blob.stroke_width * blob_w;
                current.aspect_ratio = current.bounding_box.width() as f32
                    / (current.bounding_box.height().max(1) as f32);
                current.density = (current.pixels.len() as f32)
                    / ((current.bounding_box.width() * current.bounding_box.height()).max(1)
                        as f32);
            } else {
                out.push(current);
                current = blob;
            }
        }

        out.push(current);
        out
    }

    /// Group blobs into words
    fn group_blobs_into_words(&self, blobs: &[Blob]) -> Vec<Vec<Blob>> {
        if blobs.is_empty() {
            return Vec::new();
        }

        let mut widths: Vec<u32> = blobs
            .iter()
            .map(|b| b.bounding_box.width().max(1))
            .collect();
        widths.sort_unstable();
        let median_width = widths[widths.len() / 2].max(1);

        // Collect all gaps between consecutive blobs
        let mut gaps = Vec::new();
        for i in 1..blobs.len() {
            let gap = blobs[i]
                .bounding_box
                .left
                .saturating_sub(blobs[i - 1].bounding_box.right);
            if gap < (median_width * 10) {
                gaps.push(gap as f64);
            }
        }

        // Dynamic threshold: use Otsu on the gap distribution to separate
        // intra-word gaps (small) from inter-word gaps (large)
        let gap_threshold = if gaps.len() >= 4 {
            compute_gap_threshold(&gaps, median_width)
        } else {
            ((median_width as f32) * 1.5).ceil() as u32
        };

        let mut words = Vec::new();
        let mut current_word = vec![blobs[0].clone()];

        for i in 1..blobs.len() {
            let prev_blob = &blobs[i - 1];
            let curr_blob = &blobs[i];

            let gap = curr_blob
                .bounding_box
                .left
                .saturating_sub(prev_blob.bounding_box.right);

            if gap <= gap_threshold {
                current_word.push(curr_blob.clone());
            } else {
                words.push(current_word);
                current_word = vec![curr_blob.clone()];
            }
        }

        if !current_word.is_empty() {
            words.push(current_word);
        }

        words
    }

    /// Detect and join hyphenated words across lines
    ///
    /// Given the last word of a line and the first word of the next line,
    /// returns true if they should be joined (last word ends with "-").
    fn is_hyphenated_continuation(
        last_line: &LineRecognition,
        next_line: &LineRecognition,
    ) -> bool {
        if last_line.words.is_empty() || next_line.words.is_empty() {
            return false;
        }

        let last_word = last_line.words.last().unwrap();
        let last_char = last_word.word.chars().last();

        match last_char {
            Some('-') => {
                let next_word = &next_line.words[0].word;
                // Only join if next word is lowercase (not a new sentence)
                next_word
                    .chars()
                    .next()
                    .map(|c| c.is_lowercase())
                    .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Join hyphenated words across lines in recognition result
    fn join_hyphenated_words(&self, result: &mut RecognitionResult) {
        if result.lines.len() < 2 {
            return;
        }

        let mut i = 0;
        while i + 1 < result.lines.len() {
            if Self::is_hyphenated_continuation(&result.lines[i], &result.lines[i + 1]) {
                let joined_word = {
                    let left = result.lines[i].words.last().unwrap().word.clone();
                    let right = result.lines[i + 1].words[0].word.clone();
                    format!("{}{}", &left[..left.len() - 1], right)
                };

                // Merge line[i+1] into line[i]
                let mut chars: Vec<CharacterRecognition> = result.lines[i]
                    .words
                    .iter()
                    .flat_map(|w| w.characters.iter().cloned())
                    .collect();
                if let Some(right_chars) = result.lines[i + 1].words.get(0) {
                    chars.extend(right_chars.characters.iter().skip(1).cloned());
                }
                let mut merged_words = result.lines[i].words.clone();
                if let Some(last) = merged_words.last_mut() {
                    last.word = joined_word;
                }
                merged_words.extend(result.lines[i + 1].words.iter().skip(1).cloned());

                let new_line_text = merged_words
                    .iter()
                    .map(|w| w.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let mut new_line = LineRecognition::with_bounding_box(
                    new_line_text,
                    result.lines[i].confidence,
                    result.lines[i].bounding_box.unwrap_or(
                        result.lines[i + 1]
                            .bounding_box
                            .unwrap_or(crate::core::text::BoundingBox::new(0, 0, 0, 0)),
                    ),
                );
                new_line.words = merged_words;

                result.lines[i] = new_line;
                result.lines.remove(i + 1);
                // Don't increment i — check the merged line again
            } else {
                i += 1;
            }
        }

        // Rebuild full text
        result.text = result
            .lines
            .iter()
            .map(|l| l.line.as_str())
            .collect::<Vec<_>>()
            .join("\n");
    }

    /// Recognize characters in a word
    fn recognize_word(&self, image: &GrayImage, blobs: &[Blob]) -> Result<WordRecognition> {
        let mut characters = Vec::new();
        let mut word_text = String::new();

        for blob in blobs {
            let char_result = self.recognize_character(image, blob)?;
            characters.push(char_result.clone());
            word_text.push(char_result.character);
        }

        let confidence = if !characters.is_empty() {
            characters.iter().map(|c| c.confidence).sum::<f32>() / characters.len() as f32
        } else {
            0.0
        };

        let bounding_box = if !blobs.is_empty() {
            let mut min_x = u32::MAX;
            let mut min_y = u32::MAX;
            let mut max_x = 0u32;
            let mut max_y = 0u32;

            for blob in blobs {
                min_x = min_x.min(blob.bounding_box.left);
                min_y = min_y.min(blob.bounding_box.top);
                max_x = max_x.max(blob.bounding_box.right);
                max_y = max_y.max(blob.bounding_box.bottom);
            }

            BoundingBox::new(min_x, min_y, max_x, max_y)
        } else {
            BoundingBox::new(0, 0, 0, 0)
        };

        let mut word = WordRecognition::with_bounding_box(word_text, confidence, bounding_box);
        word.characters = characters;

        Ok(word)
    }

    /// Recognize a single character from a blob
    fn recognize_character(&self, image: &GrayImage, blob: &Blob) -> Result<CharacterRecognition> {
        let char_image = self.extract_blob_image(image, blob)?;

        let (character, confidence) = self.match_character_template(&char_image)?;

        let bbox = BoundingBox::new(
            blob.bounding_box.left,
            blob.bounding_box.top,
            blob.bounding_box.right,
            blob.bounding_box.bottom,
        );

        let features = self.extract_blob_features(image, blob).ok();

        let mut result = CharacterRecognition::with_bounding_box(character, confidence, bbox);
        result.features = features;

        Ok(result)
    }

    /// Extract INT_FEATURE from a blob's region in the binary image
    fn extract_blob_features(&self, image: &GrayImage, blob: &Blob) -> Result<(Vec<u8>, Vec<u8>)> {
        let bbox = &blob.bounding_box;
        let pad = ((bbox.width().min(bbox.height()) / 10).max(1)).min(3);
        let left = bbox.left.saturating_sub(pad);
        let top = bbox.top.saturating_sub(pad);
        let right = (bbox.right + pad).min(image.width());
        let bottom = (bbox.bottom + pad).min(image.height());

        let w = right.saturating_sub(left);
        let h = bottom.saturating_sub(top);

        if w < 3 || h < 3 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut sub_image = GrayImage::from_pixel(w, h, image::Luma([255u8]));
        for (x, y) in &blob.pixels {
            let local_x = x.saturating_sub(left);
            let local_y = y.saturating_sub(top);
            if local_x < w && local_y < h {
                sub_image.put_pixel(local_x, local_y, image::Luma([0u8]));
            }
        }

        let outlines = extract_outlines(&sub_image)?;
        let blobs = outlines_to_blobs(outlines);

        if blobs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let (bl_features, cn_features, _fx_result) = extract_features(&blobs[0], false)?;

        let bl_bytes: Vec<u8> = bl_features
            .iter()
            .flat_map(|f| [f.x, f.y, f.theta])
            .collect();
        let cn_bytes: Vec<u8> = cn_features
            .iter()
            .flat_map(|f| [f.x, f.y, f.theta])
            .collect();

        Ok((bl_bytes, cn_bytes))
    }

    /// Extract blob image and normalize
    fn extract_blob_image(&self, image: &GrayImage, blob: &Blob) -> Result<Vec<Vec<u8>>> {
        let bbox = &blob.bounding_box;
        let pad = ((bbox.width().min(bbox.height()) / 10).max(1)).min(3);
        let left = bbox.left.saturating_sub(pad);
        let top = bbox.top.saturating_sub(pad);
        let right = (bbox.right + pad).min(image.width());
        let bottom = (bbox.bottom + pad).min(image.height());

        let width = right.saturating_sub(left);
        let height = bottom.saturating_sub(top);

        let mut char_image = vec![vec![255u8; width as usize]; height as usize];

        for (x, y) in &blob.pixels {
            let local_x = x.saturating_sub(left);
            let local_y = y.saturating_sub(top);
            if local_x < width && local_y < height {
                let pixel = image.get_pixel(*x, *y);
                char_image[local_y as usize][local_x as usize] = pixel[0];
            }
        }

        Ok(char_image)
    }

    /// Match character against templates
    fn match_character_template(&self, char_image: &[Vec<u8>]) -> Result<(char, f32)> {
        if char_image.is_empty() || char_image[0].is_empty() {
            return Ok(('?', 0.0));
        }

        let template_candidate = self
            .match_from_templates(char_image)
            .filter(|(ch, conf)| *ch != '?' && *conf > 0.0);
        if let Some((ch, conf)) = template_candidate {
            if conf >= 0.55 {
                return Ok((ch, conf));
            }
        }

        let char_height = char_image.len();
        let char_width = char_image[0].len();

        // Very basic pattern matching based on dimensions
        if char_width < 2 || char_height < 2 {
            if let Some((ch, conf)) = template_candidate {
                return Ok((ch, conf));
            }
            return Ok(('?', 0.1));
        }

        // Count black pixels for density
        let mut black_pixels = 0;
        let total_pixels = char_width * char_height;
        for row in char_image {
            for &pixel in row {
                if pixel < 128 {
                    black_pixels += 1;
                }
            }
        }
        let density = black_pixels as f32 / total_pixels as f32;

        // Calculate aspect ratio
        let aspect_ratio = char_width as f32 / char_height as f32;

        // Analyze horizontal and vertical projections for better recognition
        let mut horizontal_proj = vec![0u32; char_height];
        let mut vertical_proj = vec![0u32; char_width];

        for (y, row) in char_image.iter().enumerate() {
            for (x, &pixel) in row.iter().enumerate() {
                if pixel < 128 {
                    horizontal_proj[y] += 1;
                    vertical_proj[x] += 1;
                }
            }
        }

        // Find peaks in projections
        let h_peaks = Self::count_peaks(&horizontal_proj);
        let v_peaks = Self::count_peaks(&vertical_proj);

        // Enhanced character recognition with better feature analysis
        // Analyze horizontal projection more carefully
        let h_max = horizontal_proj.iter().max().copied().unwrap_or(0);
        let h_center = char_height / 2;
        let h_top_density: u32 = horizontal_proj[..h_center].iter().sum();
        let h_bottom_density: u32 = horizontal_proj[h_center..].iter().sum();
        let h_top_heavy = h_top_density > h_bottom_density * 3 / 2;
        let h_bottom_heavy = h_bottom_density > h_top_density * 3 / 2;

        // Analyze vertical projection
        let v_max = vertical_proj.iter().max().copied().unwrap_or(0);
        let v_left_density: u32 = vertical_proj[..char_width / 2].iter().sum();
        let v_right_density: u32 = vertical_proj[char_width / 2..].iter().sum();
        let v_center_density: u32 = if char_width >= 3 {
            vertical_proj[char_width / 3..2 * char_width / 3]
                .iter()
                .sum()
        } else {
            0
        };

        // Check for holes (for 'O', '0', 'Q', 'D', etc.)
        let has_hole = Self::detect_hole(char_image);

        // Check for horizontal bars (for 'A', 'H', 'E', 'F', etc.)
        let has_horizontal_bar = h_max > char_width as u32 / 2;

        // Check for vertical bars
        let has_vertical_bar = v_max > char_height as u32 / 2;

        // Enhanced character recognition
        let (best_match, best_score) = if aspect_ratio < 0.4 {
            // Very narrow - likely 'I', 'l', '1', '|'
            if h_peaks >= 2 {
                ('1', 0.45)
            } else {
                ('I', 0.4)
            }
        } else if aspect_ratio > 1.8 {
            // Very wide - likely 'M', 'W', or multiple characters
            if v_peaks >= 3 {
                if v_left_density > v_right_density {
                    ('M', 0.4)
                } else {
                    ('W', 0.4)
                }
            } else if has_horizontal_bar {
                ('E', 0.35)
            } else {
                ('H', 0.3)
            }
        } else if has_hole {
            // Has hole - likely 'O', 'Q', '0', 'D', 'B', '8'
            if aspect_ratio > 1.1 && aspect_ratio < 1.3 {
                if h_bottom_heavy {
                    ('Q', 0.45)
                } else {
                    ('O', 0.5)
                }
            } else if aspect_ratio < 0.9 {
                ('0', 0.45)
            } else if v_peaks >= 2 {
                ('B', 0.4)
            } else {
                ('D', 0.4)
            }
        } else if density > 0.6 {
            // High density without hole - likely '8', '&', '@'
            if h_peaks >= 2 {
                ('8', 0.4)
            } else {
                ('&', 0.35)
            }
        } else if density < 0.2 {
            // Low density - likely '-', '_', or noise
            if aspect_ratio > 2.0 {
                ('-', 0.3)
            } else {
                ('_', 0.25)
            }
        } else if has_horizontal_bar && h_peaks >= 2 {
            // Multiple horizontal features
            if h_top_heavy {
                ('T', 0.4)
            } else if h_bottom_heavy {
                ('L', 0.35)
            } else if v_peaks >= 2 {
                ('E', 0.4)
            } else {
                ('F', 0.35)
            }
        } else if v_peaks >= 3 {
            // Multiple vertical features
            if v_center_density > v_left_density && v_center_density > v_right_density {
                ('H', 0.4)
            } else {
                ('M', 0.35)
            }
        } else if v_peaks >= 2 && has_horizontal_bar {
            // Both vertical and horizontal features
            if h_top_heavy {
                ('A', 0.45)
            } else {
                ('H', 0.4)
            }
        } else if aspect_ratio > 1.0 && aspect_ratio < 1.3 {
            // Square-ish - check if it's actually a hole or just dense
            if has_hole {
                if has_vertical_bar {
                    ('D', 0.35)
                } else {
                    ('O', 0.3)
                }
            } else {
                // No hole but square-ish - could be many things
                if density > 0.5 {
                    ('N', 0.3)
                } else if v_peaks >= 2 {
                    ('H', 0.3)
                } else {
                    ('U', 0.25)
                }
            }
        } else if v_peaks == 1 && h_peaks == 1 {
            // Simple shape
            if aspect_ratio > 1.5 {
                ('L', 0.3)
            } else if aspect_ratio < 0.7 {
                ('C', 0.3)
            } else {
                ('U', 0.25)
            }
        } else {
            // Default - try to match common patterns based on features
            if v_left_density > v_right_density * 2 {
                ('P', 0.3)
            } else if v_right_density > v_left_density * 2 {
                ('R', 0.3)
            } else if has_horizontal_bar {
                ('E', 0.25)
            } else if v_peaks >= 2 {
                ('N', 0.25)
            } else {
                // Last resort - use density and aspect ratio
                if density > 0.4 {
                    ('A', 0.2)
                } else if aspect_ratio > 1.2 {
                    ('S', 0.2)
                } else {
                    ('X', 0.15)
                }
            }
        };

        if let Some((ch, conf)) = template_candidate {
            if conf >= 0.45 && (conf + 0.1) >= best_score {
                return Ok((ch, conf));
            }
        }

        Ok((best_match, best_score))
    }

    fn match_from_templates(&self, char_image: &[Vec<u8>]) -> Option<(char, f32)> {
        const TEMPLATE_SIZE: usize = 16;
        if self.character_templates.is_empty() {
            return None;
        }

        let bin = Self::binarize_char_image(char_image);
        let trimmed = Self::trim_binary_image(&bin)?;
        let normalized = Self::resize_binary_image(&trimmed, TEMPLATE_SIZE, TEMPLATE_SIZE);
        let inverted = Self::invert_binary_image(&normalized);

        let mut best_char = '?';
        let mut best_score = 0.0f32;
        let mut second_char = '?';
        let mut second_score = 0.0f32;

        for (ch, template) in &self.character_templates {
            if template.template.is_empty() {
                continue;
            }

            let tmpl_trimmed = Self::trim_binary_image(&template.template)
                .unwrap_or_else(|| template.template.clone());
            let tmpl_norm = Self::resize_binary_image(&tmpl_trimmed, TEMPLATE_SIZE, TEMPLATE_SIZE);

            let iou_score = Self::similarity_with_small_shifts(&normalized, &tmpl_norm)
                .max(Self::similarity_with_small_shifts(&inverted, &tmpl_norm));
            let proj_score = Self::projection_similarity(&normalized, &tmpl_norm)
                .max(Self::projection_similarity(&inverted, &tmpl_norm));
            let score = (iou_score * 0.7 + proj_score * 0.3).clamp(0.0, 1.0);
            if score > best_score {
                second_score = best_score;
                second_char = best_char;
                best_score = score;
                best_char = *ch;
            } else if score > second_score {
                second_score = score;
                second_char = *ch;
            }
        }

        if (best_char == 'O' && second_char == '0') || (best_char == '0' && second_char == 'O') {
            if (best_score - second_score).abs() < 0.06 {
                let center_bg = {
                    let h = normalized.len();
                    let w = normalized[0].len();
                    let cy = h / 2;
                    let cx = w / 2;
                    let span = (h.min(w) / 4).max(2);
                    let y0 = cy.saturating_sub(span);
                    let y1 = (cy + span).min(h);
                    let x0 = cx.saturating_sub(span);
                    let x1 = (cx + span).min(w);
                    let mut bg = 0u32;
                    let mut total = 0u32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            total += 1;
                            if normalized[y][x] == 0 {
                                bg += 1;
                            }
                        }
                    }
                    if total == 0 {
                        0.0
                    } else {
                        (bg as f32) / (total as f32)
                    }
                };
                if center_bg >= 0.72 {
                    best_char = 'O';
                } else {
                    best_char = '0';
                }
            }
        }

        if best_char == '?' {
            None
        } else {
            Some((best_char, best_score.clamp(0.0, 1.0)))
        }
    }

    fn similarity_with_small_shifts(a: &[Vec<u8>], b: &[Vec<u8>]) -> f32 {
        if a.is_empty() || b.is_empty() || a[0].is_empty() || b[0].is_empty() {
            return 0.0;
        }
        let height = a.len().min(b.len());
        let width = a[0].len().min(b[0].len());
        if height == 0 || width == 0 {
            return 0.0;
        }

        let mut best = 0.0f32;
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                let score = Self::binary_similarity_shifted(a, b, dx, dy);
                if score > best {
                    best = score;
                }
            }
        }
        best
    }

    fn projection_similarity(a: &[Vec<u8>], b: &[Vec<u8>]) -> f32 {
        if a.is_empty() || b.is_empty() || a[0].is_empty() || b[0].is_empty() {
            return 0.0;
        }

        let height = a.len().min(b.len());
        let width = a[0].len().min(b[0].len());
        if height == 0 || width == 0 {
            return 0.0;
        }

        let mut ah = vec![0f32; height];
        let mut bh = vec![0f32; height];
        let mut av = vec![0f32; width];
        let mut bv = vec![0f32; width];

        for y in 0..height {
            for x in 0..width {
                ah[y] += a[y][x] as f32;
                bh[y] += b[y][x] as f32;
                av[x] += a[y][x] as f32;
                bv[x] += b[y][x] as f32;
            }
        }

        let h_sim = Self::normalized_vector_similarity(&ah, &bh);
        let v_sim = Self::normalized_vector_similarity(&av, &bv);
        ((h_sim + v_sim) * 0.5).clamp(0.0, 1.0)
    }

    fn normalized_vector_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let max_a = a.iter().take(n).copied().fold(0.0f32, f32::max);
        let max_b = b.iter().take(n).copied().fold(0.0f32, f32::max);
        let denom_a = if max_a <= 0.0 { 1.0 } else { max_a };
        let denom_b = if max_b <= 0.0 { 1.0 } else { max_b };

        let mut diff_sum = 0.0f32;
        for i in 0..n {
            let na = a[i] / denom_a;
            let nb = b[i] / denom_b;
            diff_sum += (na - nb).abs();
        }
        (1.0 - (diff_sum / (n as f32))).clamp(0.0, 1.0)
    }

    fn binary_similarity_shifted(a: &[Vec<u8>], b: &[Vec<u8>], dx: i32, dy: i32) -> f32 {
        let height = a.len();
        let width = a[0].len();
        if height == 0 || width == 0 {
            return 0.0;
        }

        let mut intersection = 0u32;
        let mut union = 0u32;

        for y in 0..height {
            for x in 0..width {
                let bx = x as i32 + dx;
                let by = y as i32 + dy;
                let b_val = if by >= 0
                    && (by as usize) < b.len()
                    && bx >= 0
                    && (bx as usize) < b[0].len()
                {
                    b[by as usize][bx as usize]
                } else {
                    0
                };

                let a_val = a[y][x];
                if a_val == 1 || b_val == 1 {
                    union += 1;
                    if a_val == 1 && b_val == 1 {
                        intersection += 1;
                    }
                }
            }
        }

        if union == 0 {
            0.0
        } else {
            (intersection as f32) / (union as f32)
        }
    }

    fn binarize_char_image(char_image: &[Vec<u8>]) -> Vec<Vec<u8>> {
        char_image
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&px| if px < 128 { 1u8 } else { 0u8 })
                    .collect()
            })
            .collect()
    }

    fn invert_binary_image(img: &[Vec<u8>]) -> Vec<Vec<u8>> {
        img.iter()
            .map(|row| {
                row.iter()
                    .map(|&px| if px == 0 { 1u8 } else { 0u8 })
                    .collect()
            })
            .collect()
    }

    fn trim_binary_image(img: &[Vec<u8>]) -> Option<Vec<Vec<u8>>> {
        if img.is_empty() || img[0].is_empty() {
            return None;
        }
        let height = img.len();
        let width = img[0].len();

        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut has_fg = false;

        for y in 0..height {
            for x in 0..width {
                if img[y][x] == 1 {
                    has_fg = true;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }

        if !has_fg {
            return None;
        }

        let out_w = max_x.saturating_sub(min_x) + 1;
        let out_h = max_y.saturating_sub(min_y) + 1;
        let mut out = vec![vec![0u8; out_w]; out_h];

        for y in 0..out_h {
            for x in 0..out_w {
                out[y][x] = img[min_y + y][min_x + x];
            }
        }

        Some(out)
    }

    fn resize_binary_image(img: &[Vec<u8>], target_w: usize, target_h: usize) -> Vec<Vec<u8>> {
        if img.is_empty() || img[0].is_empty() || target_w == 0 || target_h == 0 {
            return vec![vec![0u8; target_w]; target_h];
        }

        let src_h = img.len();
        let src_w = img[0].len();

        let mut out = vec![vec![0u8; target_w]; target_h];
        for y in 0..target_h {
            let sy = (y * src_h) / target_h;
            for x in 0..target_w {
                let sx = (x * src_w) / target_w;
                out[y][x] = img[sy][sx];
            }
        }
        out
    }

    /// Count peaks in a projection (local maxima)
    fn count_peaks(proj: &[u32]) -> usize {
        if proj.len() < 3 {
            return 0;
        }

        let mut peaks = 0;
        let max_val = proj.iter().max().copied().unwrap_or(0);
        let threshold = max_val / 3;

        for i in 1..proj.len() - 1 {
            if proj[i] > threshold && proj[i] > proj[i - 1] && proj[i] > proj[i + 1] {
                peaks += 1;
            }
        }

        peaks
    }

    /// Detect if character has a hole (for 'O', '0', 'Q', 'D', 'B', '8')
    fn detect_hole(char_image: &[Vec<u8>]) -> bool {
        if char_image.len() < 5 || char_image[0].len() < 5 {
            return false;
        }

        let height = char_image.len();
        let width = char_image[0].len();
        let center_y = height / 2;
        let center_x = width / 2;

        // Check if center region has background pixels (hole)
        let check_size = (height.min(width) / 3).max(2);
        let start_y = center_y.saturating_sub(check_size);
        let end_y = (center_y + check_size).min(height);
        let start_x = center_x.saturating_sub(check_size);
        let end_x = (center_x + check_size).min(width);

        let mut background_pixels = 0;
        let mut total_pixels = 0;

        for y in start_y..end_y {
            for x in start_x..end_x {
                total_pixels += 1;
                if char_image[y][x] >= 128 {
                    background_pixels += 1;
                }
            }
        }

        // If more than 50% of center region is background, likely has a hole
        // Make this stricter to avoid false positives
        if total_pixels > 0 {
            (background_pixels as f32 / total_pixels as f32) > 0.5
        } else {
            false
        }
    }

    /// Create default character templates
    fn create_default_templates() -> std::collections::BTreeMap<char, CharacterTemplate> {
        const TEMPLATE_SIZE: usize = 16;

        let mut templates = std::collections::BTreeMap::new();

        // Uppercase letters A-Z
        for ch in 'A'..='Z' {
            if let Some(rows) = Self::glyph_5x7_rows(ch) {
                let template = Self::glyph_5x7_to_template(rows, TEMPLATE_SIZE, TEMPLATE_SIZE);
                templates.insert(
                    ch,
                    CharacterTemplate {
                        character: ch,
                        template,
                        width: TEMPLATE_SIZE as u32,
                        height: TEMPLATE_SIZE as u32,
                    },
                );
            }
        }

        // Lowercase letters a-z
        for ch in 'a'..='z' {
            if let Some(rows) = Self::glyph_5x7_rows(ch) {
                let template = Self::glyph_5x7_to_template(rows, TEMPLATE_SIZE, TEMPLATE_SIZE);
                templates.insert(
                    ch,
                    CharacterTemplate {
                        character: ch,
                        template,
                        width: TEMPLATE_SIZE as u32,
                        height: TEMPLATE_SIZE as u32,
                    },
                );
            }
        }

        // Digits 0-9
        for ch in '0'..='9' {
            if let Some(rows) = Self::glyph_5x7_rows(ch) {
                let template = Self::glyph_5x7_to_template(rows, TEMPLATE_SIZE, TEMPLATE_SIZE);
                templates.insert(
                    ch,
                    CharacterTemplate {
                        character: ch,
                        template,
                        width: TEMPLATE_SIZE as u32,
                        height: TEMPLATE_SIZE as u32,
                    },
                );
            }
        }

        // Common punctuation
        for ch in [
            '.', '-', ':', '/', ',', '!', '?', ';', '(', ')', '[', ']', '{', '}', '"', '\'',
        ] {
            if let Some(rows) = Self::glyph_5x7_rows(ch) {
                let template = Self::glyph_5x7_to_template(rows, TEMPLATE_SIZE, TEMPLATE_SIZE);
                templates.insert(
                    ch,
                    CharacterTemplate {
                        character: ch,
                        template,
                        width: TEMPLATE_SIZE as u32,
                        height: TEMPLATE_SIZE as u32,
                    },
                );
            }
        }

        templates
    }

    fn glyph_5x7_to_template(
        rows: [&'static str; 7],
        target_w: usize,
        target_h: usize,
    ) -> Vec<Vec<u8>> {
        let src_h = rows.len();
        let src_w = rows[0].as_bytes().len();
        let mut out = vec![vec![0u8; target_w]; target_h];

        for y in 0..target_h {
            let sy = (y * src_h) / target_h;
            let row = rows[sy].as_bytes();
            for x in 0..target_w {
                let sx = (x * src_w) / target_w;
                out[y][x] = if row[sx] == b'1' { 1u8 } else { 0u8 };
            }
        }

        out
    }

    fn glyph_5x7_rows(ch: char) -> Option<[&'static str; 7]> {
        match ch {
            'A' => Some([
                "01110", "10001", "10001", "11111", "10001", "10001", "10001",
            ]),
            'B' => Some([
                "11110", "10001", "10001", "11110", "10001", "10001", "11110",
            ]),
            'C' => Some([
                "01111", "10000", "10000", "10000", "10000", "10000", "01111",
            ]),
            'D' => Some([
                "11110", "10001", "10001", "10001", "10001", "10001", "11110",
            ]),
            'E' => Some([
                "11111", "10000", "10000", "11110", "10000", "10000", "11111",
            ]),
            'F' => Some([
                "11111", "10000", "10000", "11110", "10000", "10000", "10000",
            ]),
            'G' => Some([
                "01111", "10000", "10000", "10111", "10001", "10001", "01111",
            ]),
            'H' => Some([
                "10001", "10001", "10001", "11111", "10001", "10001", "10001",
            ]),
            'I' => Some([
                "11111", "00100", "00100", "00100", "00100", "00100", "11111",
            ]),
            'J' => Some([
                "00111", "00010", "00010", "00010", "00010", "10010", "01100",
            ]),
            'K' => Some([
                "10001", "10010", "10100", "11000", "10100", "10010", "10001",
            ]),
            'L' => Some([
                "10000", "10000", "10000", "10000", "10000", "10000", "11111",
            ]),
            'M' => Some([
                "10001", "11011", "10101", "10101", "10001", "10001", "10001",
            ]),
            'N' => Some([
                "10001", "11001", "10101", "10011", "10001", "10001", "10001",
            ]),
            'O' => Some([
                "01110", "10001", "10001", "10001", "10001", "10001", "01110",
            ]),
            'P' => Some([
                "11110", "10001", "10001", "11110", "10000", "10000", "10000",
            ]),
            'Q' => Some([
                "01110", "10001", "10001", "10001", "10101", "10010", "01101",
            ]),
            'R' => Some([
                "11110", "10001", "10001", "11110", "10100", "10010", "10001",
            ]),
            'S' => Some([
                "01111", "10000", "10000", "01110", "00001", "00001", "11110",
            ]),
            'T' => Some([
                "11111", "00100", "00100", "00100", "00100", "00100", "00100",
            ]),
            'U' => Some([
                "10001", "10001", "10001", "10001", "10001", "10001", "01110",
            ]),
            'V' => Some([
                "10001", "10001", "10001", "10001", "10001", "01010", "00100",
            ]),
            'W' => Some([
                "10001", "10001", "10001", "10101", "10101", "10101", "01010",
            ]),
            'X' => Some([
                "10001", "10001", "01010", "00100", "01010", "10001", "10001",
            ]),
            'Y' => Some([
                "10001", "10001", "01010", "00100", "00100", "00100", "00100",
            ]),
            'Z' => Some([
                "11111", "00001", "00010", "00100", "01000", "10000", "11111",
            ]),
            '0' => Some([
                "01110", "10001", "10011", "10101", "11001", "10001", "01110",
            ]),
            '1' => Some([
                "00100", "01100", "00100", "00100", "00100", "00100", "01110",
            ]),
            '2' => Some([
                "01110", "10001", "00001", "00010", "00100", "01000", "11111",
            ]),
            '3' => Some([
                "11110", "00001", "00001", "01110", "00001", "00001", "11110",
            ]),
            '4' => Some([
                "00010", "00110", "01010", "10010", "11111", "00010", "00010",
            ]),
            '5' => Some([
                "11111", "10000", "10000", "11110", "00001", "00001", "11110",
            ]),
            '6' => Some([
                "01110", "10000", "10000", "11110", "10001", "10001", "01110",
            ]),
            '7' => Some([
                "11111", "00001", "00010", "00100", "01000", "01000", "01000",
            ]),
            '8' => Some([
                "01110", "10001", "10001", "01110", "10001", "10001", "01110",
            ]),
            '9' => Some([
                "01110", "10001", "10001", "01111", "00001", "00001", "01110",
            ]),
            '.' => Some([
                "00000", "00000", "00000", "00000", "00000", "00100", "00100",
            ]),
            '-' => Some([
                "00000", "00000", "00000", "11111", "00000", "00000", "00000",
            ]),
            ':' => Some([
                "00000", "00100", "00100", "00000", "00100", "00100", "00000",
            ]),
            '/' => Some([
                "00001", "00010", "00100", "01000", "10000", "00000", "00000",
            ]),
            // Lowercase letters a-z
            'a' => Some([
                "00000", "00000", "01110", "00001", "01111", "10001", "01111",
            ]),
            'b' => Some([
                "10000", "10000", "11110", "10001", "10001", "10001", "11110",
            ]),
            'c' => Some([
                "00000", "00000", "01111", "10000", "10000", "10000", "01111",
            ]),
            'd' => Some([
                "00001", "00001", "01111", "10001", "10001", "10001", "01111",
            ]),
            'e' => Some([
                "00000", "00000", "01110", "10001", "11111", "10000", "01111",
            ]),
            'f' => Some([
                "00111", "01000", "01000", "11110", "01000", "01000", "01000",
            ]),
            'g' => Some([
                "00000", "00000", "01111", "10001", "10001", "01111", "00001",
            ]),
            'h' => Some([
                "10000", "10000", "11110", "10001", "10001", "10001", "10001",
            ]),
            'i' => Some([
                "00100", "00000", "00100", "00100", "00100", "00100", "00100",
            ]),
            'j' => Some([
                "00010", "00000", "00010", "00010", "00010", "10010", "01100",
            ]),
            'k' => Some([
                "10000", "10000", "10010", "10100", "11000", "10100", "10010",
            ]),
            'l' => Some([
                "01000", "01000", "01000", "01000", "01000", "01000", "01110",
            ]),
            'm' => Some([
                "00000", "00000", "11110", "10101", "10101", "10101", "10101",
            ]),
            'n' => Some([
                "00000", "00000", "11110", "10001", "10001", "10001", "10001",
            ]),
            'o' => Some([
                "00000", "00000", "01110", "10001", "10001", "10001", "01110",
            ]),
            'p' => Some([
                "00000", "00000", "11110", "10001", "10001", "11110", "10000",
            ]),
            'q' => Some([
                "00000", "00000", "01111", "10001", "10001", "01111", "00001",
            ]),
            'r' => Some([
                "00000", "00000", "11010", "10001", "10000", "10000", "10000",
            ]),
            's' => Some([
                "00000", "00000", "01111", "10000", "01110", "00001", "11110",
            ]),
            't' => Some([
                "01000", "01000", "11110", "01000", "01000", "01001", "00110",
            ]),
            'u' => Some([
                "00000", "00000", "10001", "10001", "10001", "10001", "01110",
            ]),
            'v' => Some([
                "00000", "00000", "10001", "10001", "10001", "01010", "00100",
            ]),
            'w' => Some([
                "00000", "00000", "10001", "10001", "10101", "10101", "01010",
            ]),
            'x' => Some([
                "00000", "00000", "10001", "01010", "00100", "01010", "10001",
            ]),
            'y' => Some([
                "00000", "00000", "10001", "10001", "10001", "01111", "00001",
            ]),
            'z' => Some([
                "00000", "00000", "11111", "00010", "00100", "01000", "11111",
            ]),
            // Additional punctuation
            ',' => Some([
                "00000", "00000", "00000", "00000", "00111", "00100", "01000",
            ]),
            '!' => Some([
                "00100", "00100", "00100", "00100", "00100", "00000", "00100",
            ]),
            '?' => Some([
                "01110", "10001", "00001", "00010", "00100", "00000", "00100",
            ]),
            ';' => Some([
                "00000", "00100", "00100", "00000", "00100", "00100", "01000",
            ]),
            '(' => Some([
                "00010", "00100", "01000", "01000", "01000", "00100", "00010",
            ]),
            ')' => Some([
                "01000", "00100", "00010", "00010", "00010", "00100", "01000",
            ]),
            '[' => Some([
                "00110", "00100", "00100", "00100", "00100", "00100", "00110",
            ]),
            ']' => Some([
                "01100", "00100", "00100", "00100", "00100", "00100", "01100",
            ]),
            '{' => Some([
                "00010", "00100", "00100", "01000", "00100", "00100", "00010",
            ]),
            '}' => Some([
                "01000", "00100", "00100", "00010", "00100", "00100", "01000",
            ]),
            '"' => Some([
                "01010", "01010", "01010", "00000", "00000", "00000", "00000",
            ]),
            '\'' => Some([
                "00100", "00100", "00100", "00000", "00000", "00000", "00000",
            ]),
            _ => None,
        }
    }
}

impl Default for BasicOcrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRecognizer for BasicOcrEngine {
    async fn recognize(&self, image: &OcrImage) -> Result<RecognitionResult> {
        let result = self.recognize_sync(image)?;
        Ok(result)
    }
}

/// Compute gap threshold using Otsu's method on the gap distribution
///
/// Separates intra-word gaps (small, frequent) from inter-word gaps (large, infrequent).
/// Falls back to 1.5x median width if Otsu doesn't find a good split.
fn compute_gap_threshold(gaps: &[f64], median_width: u32) -> u32 {
    if gaps.is_empty() {
        return ((median_width as f32) * 1.5).ceil() as u32;
    }

    // Build histogram of gaps (bucket size = 1 pixel)
    let max_gap = gaps.iter().cloned().fold(0.0f64, f64::max).ceil() as usize;
    let mut histogram = vec![0usize; max_gap + 1];
    for &g in gaps {
        let idx = g.ceil() as usize;
        if idx < histogram.len() {
            histogram[idx] += 1;
        }
    }

    let total_count: usize = gaps.len();

    // Otsu: find threshold that maximizes between-class variance
    let mut sum = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        sum += i * count;
    }

    let mut sum_b = 0usize;
    let mut w_b = 0usize;
    let mut max_variance = 0f64;
    let mut threshold = ((median_width as f32) * 1.5).ceil() as usize;

    for i in 0..histogram.len() {
        w_b += histogram[i];
        if w_b == 0 {
            continue;
        }
        let w_f = total_count - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += i * histogram[i];
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let v = (w_b as f64) * (w_f as f64) * (m_b - m_f) * (m_b - m_f);
        if v > max_variance {
            max_variance = v;
            threshold = i;
        }
    }

    // Ensure minimum threshold of 2 to avoid merging touching characters
    threshold.max(2).max((median_width as f32 * 0.5) as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    fn render_text_from_engine(
        engine: &BasicOcrEngine,
        text: &str,
        scale: u32,
        char_spacing: u32,
        line_spacing: u32,
    ) -> GrayImage {
        let lines: Vec<&str> = text.lines().collect();
        let glyph_w = 16 * scale;
        let glyph_h = 16 * scale;

        let max_line_len = lines
            .iter()
            .map(|l| l.chars().count() as u32)
            .max()
            .unwrap_or(0);
        let width = if max_line_len == 0 {
            1
        } else {
            max_line_len * glyph_w + max_line_len.saturating_sub(1) * char_spacing + scale * 2
        };
        let height = if lines.is_empty() {
            1
        } else {
            (lines.len() as u32) * glyph_h
                + (lines.len() as u32).saturating_sub(1) * line_spacing
                + scale * 2
        };

        let mut img = GrayImage::from_pixel(width, height, Luma([255u8]));

        let mut y = scale;
        for line in lines {
            let mut x = scale;
            for ch in line.chars() {
                if ch == ' ' {
                    x += glyph_w + char_spacing;
                    continue;
                }

                let key = ch.to_ascii_uppercase();
                if let Some(template) = engine.character_templates.get(&key) {
                    for (ty, row) in template.template.iter().enumerate() {
                        for (tx, &v) in row.iter().enumerate() {
                            if v == 1 {
                                for dy in 0..scale {
                                    for dx in 0..scale {
                                        img.put_pixel(
                                            x + (tx as u32) * scale + dx,
                                            y + (ty as u32) * scale + dy,
                                            Luma([0u8]),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                x += glyph_w + char_spacing;
            }
            y += glyph_h + line_spacing;
        }

        img
    }

    #[tokio::test]
    async fn test_basic_ocr_recognition() {
        let engine = BasicOcrEngine::new();
        let img = render_text_from_engine(&engine, "HELLO\nWORLD", 4, 4, 10);
        let ocr_image = OcrImage::new(DynamicImage::ImageLuma8(img), 300);

        let result = engine.recognize(&ocr_image).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.text.trim(), "HELLO\nWORLD");
        assert!(result.confidence > 0.6);
    }
}
