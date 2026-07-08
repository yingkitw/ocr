//! Synthetic multi-column document generator for layout detection evaluation
//!
//! Creates document images with known text region bounding boxes,
//! useful for measuring detection recall and precision.

use crate::core::text::BoundingBox;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use rusttype::{Font, Scale};

/// A ground-truth text region in a synthetic document
#[derive(Debug, Clone)]
pub struct DocumentRegion {
    pub text: String,
    pub bbox: BoundingBox,
}

/// Synthetic document with known layout
#[derive(Debug, Clone)]
pub struct SyntheticDocument {
    pub image: DynamicImage,
    pub regions: Vec<DocumentRegion>,
}

/// Generates synthetic multi-column documents
pub struct DocumentGenerator {
    page_width: u32,
    page_height: u32,
    font_size: f32,
    line_height: u32,
    column_gap: u32,
    margin: u32,
}

impl Default for DocumentGenerator {
    fn default() -> Self {
        Self {
            page_width: 800,
            page_height: 1100,
            font_size: 18.0,
            line_height: 28,
            column_gap: 30,
            margin: 40,
        }
    }
}

impl DocumentGenerator {
    /// Generate a two-column document with random text lines
    pub fn generate_two_column(&self, lines_per_column: usize) -> SyntheticDocument {
        let mut img = RgbImage::from_pixel(self.page_width, self.page_height, Rgb([255, 255, 255]));
        let mut regions = Vec::new();
        let scale = Scale::uniform(self.font_size);

        let col_width = (self.page_width - 2 * self.margin - self.column_gap) / 2;
        let left_col_x = self.margin;
        let right_col_x = self.margin + col_width + self.column_gap;

        let font = Self::default_font();

        for col in 0..2 {
            let x = if col == 0 { left_col_x } else { right_col_x };
            for row in 0..lines_per_column {
                let y = self.margin + row as u32 * self.line_height;
                let text = format!("Line {} column {} with sample text here", row, col);
                let text_width = self.measure_text_width(&text, &font, scale);
                let w = text_width.min(col_width);

                if let Some(ref f) = font {
                    draw_text_mut(
                        &mut img,
                        Rgb([0, 0, 0]),
                        x as i32,
                        y as i32,
                        scale,
                        f,
                        &text,
                    );
                } else {
                    // Fallback: draw simple bars
                    for dx in 0..w {
                        for dy in 2..self.line_height.saturating_sub(2) {
                            img.put_pixel(x + dx, y + dy, Rgb([0, 0, 0]));
                        }
                    }
                }

                regions.push(DocumentRegion {
                    text: text.clone(),
                    bbox: BoundingBox::new(x, y, x + w, y + self.line_height),
                });
            }
        }

        SyntheticDocument {
            image: DynamicImage::ImageRgb8(img),
            regions,
        }
    }

    /// Generate a single-column document
    pub fn generate_single_column(&self, line_count: usize) -> SyntheticDocument {
        let mut img = RgbImage::from_pixel(self.page_width, self.page_height, Rgb([255, 255, 255]));
        let mut regions = Vec::new();
        let scale = Scale::uniform(self.font_size);
        let font = Self::default_font();

        let col_width = self.page_width - 2 * self.margin;
        let x = self.margin;

        for row in 0..line_count {
            let y = self.margin + row as u32 * self.line_height;
            let text = format!("This is sample line number {}", row);
            let text_width = self.measure_text_width(&text, &font, scale);
            let w = text_width.min(col_width);

            if let Some(ref f) = font {
                draw_text_mut(
                    &mut img,
                    Rgb([0, 0, 0]),
                    x as i32,
                    y as i32,
                    scale,
                    f,
                    &text,
                );
            } else {
                for dx in 0..w {
                    for dy in 2..self.line_height.saturating_sub(2) {
                        img.put_pixel(x + dx, y + dy, Rgb([0, 0, 0]));
                    }
                }
            }

            regions.push(DocumentRegion {
                text: text.clone(),
                bbox: BoundingBox::new(x, y, x + w, y + self.line_height),
            });
        }

        SyntheticDocument {
            image: DynamicImage::ImageRgb8(img),
            regions,
        }
    }

    fn measure_text_width(&self, text: &str, font: &Option<Font>, scale: Scale) -> u32 {
        if let Some(f) = font {
            let mut width = 0.0f32;
            for glyph in f.layout(text, scale, rusttype::point(0.0, 0.0)) {
                if let Some(bb) = glyph.pixel_bounding_box() {
                    width = width.max(bb.max.x as f32);
                }
            }
            (width as u32).max(1)
        } else {
            (text.len() as u32 * 8).min(200)
        }
    }

    fn default_font() -> Option<Font<'static>> {
        let candidates = [
            "/System/Library/Fonts/Monaco.dfont",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ];
        for path in &candidates {
            if let Ok(data) = std::fs::read(path) {
                if let Some(font) = Font::try_from_vec(data) {
                    return Some(font);
                }
            }
        }
        None
    }
}

/// Evaluate detection recall and precision against ground truth
pub fn evaluate_detection(
    detected: &[BoundingBox],
    ground_truth: &[BoundingBox],
    iou_threshold: f32,
) -> (f32, f32) {
    if detected.is_empty() && ground_truth.is_empty() {
        return (1.0, 1.0);
    }
    if detected.is_empty() {
        return (0.0, 1.0);
    }
    if ground_truth.is_empty() {
        return (1.0, 0.0);
    }

    let mut matched_gt = vec![false; ground_truth.len()];
    let mut true_positives = 0usize;

    for det in detected {
        let mut best_iou = 0.0f32;
        let mut best_idx = 0usize;
        for (i, gt) in ground_truth.iter().enumerate() {
            let iou = bbox_iou(det, gt);
            if iou > best_iou {
                best_iou = iou;
                best_idx = i;
            }
        }
        if best_iou >= iou_threshold && !matched_gt[best_idx] {
            matched_gt[best_idx] = true;
            true_positives += 1;
        }
    }

    let recall = true_positives as f32 / ground_truth.len() as f32;
    let precision = true_positives as f32 / detected.len() as f32;
    (recall, precision)
}

fn bbox_iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let x1 = a.left.max(b.left);
    let y1 = a.top.max(b.top);
    let x2 = a.right.min(b.right);
    let y2 = a.bottom.min(b.bottom);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let inter = ((x2 - x1) * (y2 - y1)) as f32;
    let area_a = ((a.right - a.left) * (a.bottom - a.top)) as f32;
    let area_b = ((b.right - b.left) * (b.bottom - b.top)) as f32;
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_single_column() {
        let gen = DocumentGenerator::default();
        let doc = gen.generate_single_column(5);
        assert_eq!(doc.regions.len(), 5);
    }

    #[test]
    fn test_generate_two_column() {
        let gen = DocumentGenerator::default();
        let doc = gen.generate_two_column(5);
        assert_eq!(doc.regions.len(), 10);
    }

    #[test]
    fn test_evaluate_detection_perfect() {
        let detected = vec![
            BoundingBox::new(0, 0, 100, 20),
            BoundingBox::new(0, 25, 100, 45),
        ];
        let gt = vec![
            BoundingBox::new(0, 0, 100, 20),
            BoundingBox::new(0, 25, 100, 45),
        ];
        let (recall, precision) = evaluate_detection(&detected, &gt, 0.5);
        assert!((recall - 1.0).abs() < 0.01);
        assert!((precision - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_detection_empty() {
        let detected: Vec<BoundingBox> = vec![];
        let gt: Vec<BoundingBox> = vec![];
        let (recall, precision) = evaluate_detection(&detected, &gt, 0.5);
        assert!((recall - 1.0).abs() < 0.01);
        assert!((precision - 1.0).abs() < 0.01);
    }
}
