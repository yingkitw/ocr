//! Template trainer: synthesize character images from fonts to build
//! high-quality pattern-matching templates for `BasicOcrEngine`.
//!
//! Each character is rendered individually, cropped to its glyph bounding
//! box, and normalized to a fixed size.  Multiple fonts can be averaged
//! for robustness.

use crate::core::image::OcrImage;
use crate::synthetic::generator::TextLineGenerator;
use image::{DynamicImage, GrayImage, Luma};
use std::collections::HashMap;

/// A trained character template (normalized binary image)
#[derive(Debug, Clone)]
pub struct TrainedTemplate {
    pub character: char,
    /// Normalized template: rows × cols, values 0–255
    pub template: Vec<Vec<u8>>,
    pub width: u32,
    pub height: u32,
}

/// Trainer that generates character templates from synthetic renders
pub struct TemplateTrainer {
    generator: TextLineGenerator,
    target_size: usize,
}

impl Default for TemplateTrainer {
    fn default() -> Self {
        Self {
            generator: TextLineGenerator::with_size(32.0, 64),
            target_size: 16,
        }
    }
}

impl TemplateTrainer {
    /// Create a trainer with a given font size and template target size
    pub fn new(font_size: f32, image_height: u32, target_size: usize) -> Self {
        Self {
            generator: TextLineGenerator::with_size(font_size, image_height),
            target_size,
        }
    }

    /// Add a TTF/OTF font for rendering
    pub fn add_font(&mut self, font_data: Vec<u8>) {
        self.generator.add_font(font_data);
    }

    /// Train templates for a set of characters, averaging across all loaded fonts
    pub fn train_templates(&self, chars: &[char]) -> HashMap<char, TrainedTemplate> {
        let mut map: HashMap<char, Vec<GrayImage>> = HashMap::new();

        // Render each character with every available font
        for &ch in chars {
            let font_count = self.generator_font_count();
            let mut renders = Vec::with_capacity(font_count.max(1));

            if font_count == 0 {
                // Bitmap fallback: render single char
                if let Some(img) = self.render_char_bitmap(ch) {
                    renders.push(img);
                }
            } else {
                for idx in 0..font_count {
                    if let Some(img) = self.render_char_with_font(ch, idx) {
                        renders.push(img);
                    }
                }
            }

            if !renders.is_empty() {
                map.insert(ch, renders);
            }
        }

        // Average across fonts and normalize
        let mut templates = HashMap::new();
        for (ch, renders) in map {
            if let Some(tpl) = self.average_and_normalize(&renders, ch) {
                templates.insert(ch, tpl);
            }
        }

        templates
    }

    /// Convenience: train ASCII printable characters
    pub fn train_ascii(&self) -> HashMap<char, TrainedTemplate> {
        let chars: Vec<char> = (' '..='~').collect();
        self.train_templates(&chars)
    }

    fn generator_font_count(&self) -> usize {
        self.generator.font_count()
    }

    /// Render a single character with a specific font index.
    /// Returns the cropped glyph image.
    fn render_char_with_font(&self, ch: char, font_index: usize) -> Option<GrayImage> {
        let text = ch.to_string();
        let sample = self.generator.generate_with_font(&text, font_index);
        let gray = sample.image.to_luma8();
        Self::crop_glyph(&gray)
    }

    /// Render a single character using the bitmap font fallback.
    fn render_char_bitmap(&self, ch: char) -> Option<GrayImage> {
        let text = ch.to_string();
        let sample = self.generator.generate(&text);
        let gray = sample.image.to_luma8();
        Self::crop_glyph(&gray)
    }

    /// Crop the glyph from the rendered line image by finding tight bounding box
    fn crop_glyph(img: &GrayImage) -> Option<GrayImage> {
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            return None;
        }

        let mut min_x = w;
        let mut max_x = 0u32;
        let mut min_y = h;
        let mut max_y = 0u32;
        let threshold = 200u8; // white-ish

        for y in 0..h {
            for x in 0..w {
                if img.get_pixel(x, y)[0] < threshold {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if min_x > max_x || min_y > max_y {
            return None;
        }

        let crop_w = max_x - min_x + 1;
        let crop_h = max_y - min_y + 1;
        let mut cropped = GrayImage::new(crop_w, crop_h);
        for y in 0..crop_h {
            for x in 0..crop_w {
                let px = img.get_pixel(min_x + x, min_y + y)[0];
                cropped.put_pixel(x, y, Luma([px]));
            }
        }
        Some(cropped)
    }

    /// Average multiple renders and normalize to target size
    fn average_and_normalize(
        &self,
        renders: &[GrayImage],
        ch: char,
    ) -> Option<TrainedTemplate> {
        if renders.is_empty() {
            return None;
        }

        // Resize all to the target size using simple nearest-neighbor
        let t = self.target_size;
        let mut accum = vec![vec![0u32; t]; t];

        for img in renders {
            let resized = Self::resize_nearest(img, t, t);
            for y in 0..t {
                for x in 0..t {
                    accum[y][x] += resized[y][x] as u32;
                }
            }
        }

        let n = renders.len() as u32;
        let mut template = vec![vec![0u8; t]; t];
        for y in 0..t {
            for x in 0..t {
                let avg = (accum[y][x] / n) as u8;
                // Binarize: dark = text
                template[y][x] = if avg < 128 { 1 } else { 0 };
            }
        }

        Some(TrainedTemplate {
            character: ch,
            template,
            width: t as u32,
            height: t as u32,
        })
    }

    /// Simple nearest-neighbor resize of a grayscale image
    fn resize_nearest(img: &GrayImage, new_w: usize, new_h: usize) -> Vec<Vec<u8>> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut out = vec![vec![0u8; new_w]; new_h];
        for y in 0..new_h {
            let sy = (y * h) / new_h;
            for x in 0..new_w {
                let sx = (x * w) / new_w;
                out[y][x] = img.get_pixel(sx as u32, sy as u32)[0];
            }
        }
        out
    }

    /// Evaluate trained templates against hand-coded ones on a synthetic test set.
    /// Returns (trained_correct, baseline_correct, total).
    pub fn evaluate_templates(
        trained: &HashMap<char, TrainedTemplate>,
        baseline: &HashMap<char, TrainedTemplate>,
        test_chars: &[char],
        generator: &TextLineGenerator,
    ) -> (usize, usize, usize) {
        let mut trained_correct = 0usize;
        let mut baseline_correct = 0usize;
        let mut total = 0usize;

        for &ch in test_chars {
            // Generate a clean render of this character
            let sample = generator.generate(&ch.to_string());
            let gray = sample.image.to_luma8();
            let Some(cropped) = Self::crop_glyph(&gray) else {
                continue;
            };
            let resized = Self::resize_nearest(&cropped, trained.values().next().map(|t| t.template.len()).unwrap_or(16), trained.values().next().map(|t| t.template[0].len()).unwrap_or(16));
            let flat: Vec<u8> = resized.iter().map(|row| row.iter().map(|&v| if v < 128 { 1 } else { 0 }).collect::<Vec<u8>>()).flatten().collect();

            let trained_best = Self::match_template_map(trained, &flat);
            let baseline_best = Self::match_template_map(baseline, &flat);

            total += 1;
            if trained_best == Some(ch) {
                trained_correct += 1;
            }
            if baseline_best == Some(ch) {
                baseline_correct += 1;
            }
        }

        (trained_correct, baseline_correct, total)
    }

    fn match_template_map(
        templates: &HashMap<char, TrainedTemplate>,
        sample: &[u8],
    ) -> Option<char> {
        let mut best_char = None;
        let mut best_score = 0u32;
        for (ch, tpl) in templates {
            let flat: Vec<u8> = tpl.template.iter().flatten().cloned().collect();
            if flat.len() != sample.len() {
                continue;
            }
            let score: u32 = flat.iter().zip(sample.iter()).map(|(a, b)| if a == b { 1 } else { 0 }).sum();
            if score > best_score {
                best_score = score;
                best_char = Some(*ch);
            }
        }
        best_char
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_ascii_templates() {
        let trainer = TemplateTrainer::default();
        let templates = trainer.train_ascii();
        // At minimum the bitmap fallback should produce printable ASCII
        assert!(!templates.is_empty(), "Should produce at least some templates");
    }

    #[test]
    fn test_crop_glyph_empty() {
        let img = GrayImage::from_pixel(10, 10, Luma([255]));
        assert!(TemplateTrainer::crop_glyph(&img).is_none());
    }

    #[test]
    fn test_crop_glyph_finds_dark_region() {
        let mut img = GrayImage::from_pixel(20, 20, Luma([255]));
        for y in 5..10 {
            for x in 3..8 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let cropped = TemplateTrainer::crop_glyph(&img).unwrap();
        assert_eq!(cropped.width(), 5);
        assert_eq!(cropped.height(), 5);
    }

    #[test]
    fn test_resize_nearest() {
        let img = GrayImage::from_pixel(2, 2, Luma([100]));
        let resized = TemplateTrainer::resize_nearest(&img, 4, 4);
        assert_eq!(resized.len(), 4);
        assert_eq!(resized[0].len(), 4);
    }

    #[test]
    fn test_trained_vs_baseline_accuracy() {
        // Train templates from synthetic renders
        let trainer = TemplateTrainer::default();
        let trained = trainer.train_ascii();

        // Build baseline (hand-coded 5x7 bitmaps) by mimicking BasicOcrEngine defaults
        let mut baseline = HashMap::new();
        // Use a simple subset for the comparison test
        let test_chars: Vec<char> = ('A'..='Z').chain('0'..='9').collect();
        for ch in &test_chars {
            if let Some(rows) = crate::synthetic::bitmap_font::glyph_5x7_rows(*ch) {
                let mut template = vec![vec![0u8; 16]; 16];
                let src_h = rows.len();
                let src_w = rows[0].as_bytes().len();
                for y in 0..16 {
                    let sy = (y * src_h) / 16;
                    let row = rows[sy].as_bytes();
                    for x in 0..16 {
                        let sx = (x * src_w) / 16;
                        template[y][x] = if row[sx] == b'1' { 1 } else { 0 };
                    }
                }
                baseline.insert(*ch, TrainedTemplate {
                    character: *ch,
                    template,
                    width: 16,
                    height: 16,
                });
            }
        }

        // Evaluate on clean synthetic renders
        let generator = TextLineGenerator::with_size(32.0, 64);
        let (trained_ok, baseline_ok, total) =
            TemplateTrainer::evaluate_templates(&trained, &baseline, &test_chars, &generator);

        // Trained templates should match at least as well as hand-coded bitmaps
        assert!(
            trained_ok >= baseline_ok || total < 5,
            "Trained templates ({trained_ok}/{total}) should be >= baseline ({baseline_ok}/{total})"
        );
    }
}
