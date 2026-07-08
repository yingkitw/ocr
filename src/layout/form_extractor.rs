//! Form field extraction
//!
//! Detects and extracts structured form fields from document layouts:
//! - Key-value pair detection (label → input region)
//! - Checkbox / radio button recognition
//! - Line-based fill-in fields

use crate::core::layout::{LayoutResult, TextRegion};
use crate::core::text::BoundingBox;
use crate::utils::Result;

/// Detected form field type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    TextInput,
    Checkbox,
    RadioButton,
    Signature,
    Date,
    Unknown,
}

/// A single form field with its location and content
#[derive(Debug, Clone)]
pub struct FormField {
    pub label: String,
    pub field_type: FieldType,
    pub bounding_box: BoundingBox,
    pub value: Option<String>,
    pub checked: bool,
    pub confidence: f32,
}

/// Form extraction result
#[derive(Debug, Clone, Default)]
pub struct FormExtractionResult {
    pub fields: Vec<FormField>,
    pub confidence: f32,
}

/// Extracts form fields from a page layout
pub struct FormExtractor;

impl FormExtractor {
    /// Detect form fields from layout and image
    pub fn extract(
        layout: &LayoutResult,
        image: &crate::core::image::OcrImage,
    ) -> Result<FormExtractionResult> {
        let mut fields = Vec::new();

        // 1. Detect checkboxes from image analysis
        let checkboxes = Self::detect_checkboxes(image)?;
        fields.extend(checkboxes);

        // 2. Detect key-value pairs from text regions
        let kv_pairs = Self::detect_key_value_pairs(&layout.text_regions);
        fields.extend(kv_pairs);

        // 3. Detect underline-style fill-in fields
        let underlines = Self::detect_underline_fields(image)?;
        fields.extend(underlines);

        let confidence = if fields.is_empty() {
            0.0
        } else {
            fields.iter().map(|f| f.confidence).sum::<f32>() / fields.len() as f32
        };

        Ok(FormExtractionResult { fields, confidence })
    }

    /// Detect checkboxes and radio buttons by looking for square/circular hollow regions
    fn detect_checkboxes(
        image: &crate::core::image::OcrImage,
    ) -> Result<Vec<FormField>> {
        let gray = image.to_grayscale();
        let luma = gray.data.to_luma8();
        let (width, height) = (luma.width(), luma.height());

        let mut fields = Vec::new();
        let mut visited = vec![false; (width * height) as usize];

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let idx = (y * width + x) as usize;
                if visited[idx] {
                    continue;
                }

                let px = luma.get_pixel(x, y)[0];
                // Look for dark pixels that could be box edges
                if px > 128 {
                    continue;
                }

                // Try to trace a small closed contour (checkbox is typically 10-30px)
                if let Some(bbox) = Self::trace_box_contour(&luma, x, y, width, height, &mut visited) {
                    let w = bbox.right - bbox.left;
                    let h = bbox.bottom - bbox.top;

                    // Check aspect ratio and size for checkbox/radio
                    if w >= 8 && h >= 8 && w <= 40 && h <= 40 {
                        let aspect = w as f32 / h as f32;
                        if aspect > 0.7 && aspect < 1.3 {
                            // Likely checkbox (square-ish)
                            let is_checked = Self::is_checkbox_checked(&luma, &bbox);
                            fields.push(FormField {
                                label: String::new(),
                                field_type: FieldType::Checkbox,
                                bounding_box: bbox,
                                value: None,
                                checked: is_checked,
                                confidence: 0.75,
                            });
                        } else if aspect > 0.85 && aspect < 1.15 {
                            // Radio button (circular-ish, harder to detect simply)
                            fields.push(FormField {
                                label: String::new(),
                                field_type: FieldType::RadioButton,
                                bounding_box: bbox,
                                value: None,
                                checked: false,
                                confidence: 0.6,
                            });
                        }
                    }
                }
            }
        }

        Ok(fields)
    }

    /// Flood-fill trace to find bounding box of connected dark pixels
    fn trace_box_contour(
        img: &image::GrayImage,
        start_x: u32,
        start_y: u32,
        width: u32,
        height: u32,
        visited: &mut [bool],
    ) -> Option<BoundingBox> {
        let mut stack = vec![(start_x, start_y)];
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;
        let mut count = 0;

        while let Some((cx, cy)) = stack.pop() {
            let idx = (cy * width + cx) as usize;
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            count += 1;

            min_x = min_x.min(cx);
            max_x = max_x.max(cx);
            min_y = min_y.min(cy);
            max_y = max_y.max(cy);

            if count > 400 {
                // Too large, not a checkbox
                return None;
            }

            // Check 4-connected neighbors
            for (nx, ny) in [(cx + 1, cy), (cx.saturating_sub(1), cy), (cx, cy + 1), (cx, cy.saturating_sub(1))] {
                if nx < width && ny < height {
                    let nidx = (ny * width + nx) as usize;
                    if !visited[nidx] && img.get_pixel(nx, ny)[0] < 128 {
                        stack.push((nx, ny));
                    }
                }
            }
        }

        if count >= 8 {
            Some(BoundingBox::new(min_x, min_y, max_x + 1, max_y + 1))
        } else {
            None
        }
    }

    /// Check if a checkbox region has an internal fill (checked)
    fn is_checkbox_checked(img: &image::GrayImage, bbox: &BoundingBox) -> bool {
        let mut dark_count = 0;
        let mut total = 0;
        // Sample interior (inset by 2px from edges)
        let inset = 2;
        let left = (bbox.left + inset).min(img.width() - 1);
        let right = bbox.right.saturating_sub(inset).max(left + 1);
        let top = (bbox.top + inset).min(img.height() - 1);
        let bottom = bbox.bottom.saturating_sub(inset).max(top + 1);

        for y in top..bottom {
            for x in left..right {
                total += 1;
                if img.get_pixel(x, y)[0] < 128 {
                    dark_count += 1;
                }
            }
        }

        if total == 0 {
            return false;
        }
        // If more than 30% of interior is dark, consider it checked
        (dark_count as f32 / total as f32) > 0.30
    }

    /// Detect key-value pairs from text regions
    /// A key-value pair is text ending with ":" followed by an empty region or underline
    fn detect_key_value_pairs(text_regions: &[TextRegion]) -> Vec<FormField> {
        let mut fields = Vec::new();

        for (i, region) in text_regions.iter().enumerate() {
            let text = region.text.trim();
            if text.ends_with(':') || text.ends_with("：") {
                // Look for a nearby region to the right that might be the value
                let label_bbox = &region.bounding_box;
                let mut best_value_region: Option<&TextRegion> = None;
                let mut best_distance = u32::MAX;

                for other in text_regions.iter().skip(i + 1) {
                    let other_bbox = &other.bounding_box;
                    // Value should be to the right and roughly same vertical level
                    if other_bbox.left > label_bbox.right
                        && other_bbox.top.abs_diff(label_bbox.top) < label_bbox.height() * 2
                    {
                        let dist = other_bbox.left - label_bbox.right;
                        if dist < best_distance && dist < 300 {
                            best_distance = dist;
                            best_value_region = Some(other);
                        }
                    }
                }

                let label_clean = text.trim_end_matches(':').trim_end_matches("：").to_string();
                let value = best_value_region.map(|r| r.text.clone());
                let field_bbox = best_value_region
                    .map(|r| r.bounding_box.clone())
                    .unwrap_or_else(|| {
                        // If no value region found, estimate one to the right
                        let est_width = 150u32;
                        BoundingBox::new(
                            label_bbox.right + 10,
                            label_bbox.top,
                            label_bbox.right + 10 + est_width,
                            label_bbox.bottom,
                        )
                    });

                fields.push(FormField {
                    label: label_clean,
                    field_type: FieldType::TextInput,
                    bounding_box: field_bbox,
                    value,
                    checked: false,
                    confidence: if best_value_region.is_some() { 0.8 } else { 0.5 },
                });
            }
        }

        fields
    }

    /// Detect underline-style fill-in fields (horizontal lines with blank space above)
    fn detect_underline_fields(
        image: &crate::core::image::OcrImage,
    ) -> Result<Vec<FormField>> {
        let gray = image.to_grayscale();
        let luma = gray.data.to_luma8();
        let (width, height) = (luma.width(), luma.height());

        let mut fields = Vec::new();
        let min_line_length = width / 6; // At least ~15% of page width

        for y in 0..height {
            let mut run_start = None;
            for x in 0..width {
                let px = luma.get_pixel(x, y)[0];
                if px < 128 {
                    if run_start.is_none() {
                        run_start = Some(x);
                    }
                } else if let Some(start) = run_start {
                    let len = x - start;
                    if len >= min_line_length && len < width / 2 {
                        // Check if there's mostly white space above the line
                        let mut white_above = 0u32;
                        let check_above = y.saturating_sub(1);
                        for check_x in start..x {
                            if luma.get_pixel(check_x, check_above)[0] > 200 {
                                white_above += 1;
                            }
                        }
                        let line_len = x - start;
                        if white_above as f32 / line_len as f32 > 0.7 {
                            fields.push(FormField {
                                label: String::new(),
                                field_type: FieldType::TextInput,
                                bounding_box: BoundingBox::new(start, y.saturating_sub(10), x, y + 2),
                                value: None,
                                checked: false,
                                confidence: 0.65,
                            });
                        }
                    }
                    run_start = None;
                }
            }
        }

        // Deduplicate overlapping underlines
        fields.sort_by_key(|f| (f.bounding_box.top, f.bounding_box.left));
        let mut deduped = Vec::new();
        for field in fields {
            let overlaps = deduped.iter().any(|existing: &FormField| {
                let e = &existing.bounding_box;
                let f = &field.bounding_box;
                // Check vertical overlap and horizontal proximity
                f.top < e.bottom && f.bottom > e.top
                    && f.left.abs_diff(e.left) < 20
                    && f.right.abs_diff(e.right) < 20
            });
            if !overlaps {
                deduped.push(field);
            }
        }

        Ok(deduped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image::OcrImage;
    use image::DynamicImage;

    fn make_test_image_with_checkbox() -> OcrImage {
        let mut img = image::RgbImage::new(200, 100);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        // Draw a checkbox (20x20 square at 10,10)
        for x in 10..30 {
            img.put_pixel(x, 10, image::Rgb([0, 0, 0]));
            img.put_pixel(x, 30, image::Rgb([0, 0, 0]));
        }
        for y in 10..31 {
            img.put_pixel(10, y, image::Rgb([0, 0, 0]));
            img.put_pixel(29, y, image::Rgb([0, 0, 0]));
        }
        OcrImage::new(DynamicImage::ImageRgb8(img), 300)
    }

    #[test]
    fn test_detect_checkboxes() {
        let img = make_test_image_with_checkbox();
        let fields = FormExtractor::detect_checkboxes(&img).unwrap();
        assert!(!fields.is_empty());
        assert_eq!(fields[0].field_type, FieldType::Checkbox);
    }

    #[test]
    fn test_detect_key_value_pairs() {
        let regions = vec![
            TextRegion {
                id: "1".to_string(),
                bounding_box: BoundingBox::new(10, 10, 100, 30),
                text: "Name:".to_string(),
                confidence: 1.0,
                properties: Default::default(),
            },
            TextRegion {
                id: "2".to_string(),
                bounding_box: BoundingBox::new(120, 10, 250, 30),
                text: "John Doe".to_string(),
                confidence: 1.0,
                properties: Default::default(),
            },
        ];
        let fields = FormExtractor::detect_key_value_pairs(&regions);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].label, "Name");
        assert_eq!(fields[0].value.as_ref().unwrap(), "John Doe");
    }

    #[test]
    fn test_checkbox_checked() {
        let mut img = image::RgbImage::new(50, 50);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        // Draw filled checkbox
        for x in 10..30 {
            for y in 10..30 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let luma = DynamicImage::ImageRgb8(img).to_luma8();
        let bbox = BoundingBox::new(8, 8, 32, 32);
        assert!(FormExtractor::is_checkbox_checked(&luma, &bbox));
    }
}
