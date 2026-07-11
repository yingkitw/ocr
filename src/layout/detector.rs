//! Region detection operations
//!
//! Provides a `TextDetector` trait with multiple implementations:
//! - `CclDetector`: Connected component labeling (baseline)
//! - `CnnDetector`: Lightweight CNN-based text detection (stub)

use crate::core::layout::*;
use crate::core::text::BoundingBox;
use crate::image::thresholder::{ImageThresholder, ThresholdMethod};
use crate::utils::Result;
use image::Luma;
use imageproc::distance_transform::Norm;
use imageproc::morphology::dilate;
use imageproc::region_labelling::{connected_components, Connectivity};

/// Trait for text region detection algorithms
pub trait TextDetector: Send + Sync {
    /// Detect text regions in an image
    fn detect(&self, image: &crate::core::image::OcrImage) -> Result<Vec<TextRegion>>;
    /// Detector name
    fn name(&self) -> &'static str;
}

/// CCL-based text detector using connected component labeling
pub struct CclDetector;

impl TextDetector for CclDetector {
    fn detect(&self, image: &crate::core::image::OcrImage) -> Result<Vec<TextRegion>> {
        TextRegionDetector::detect_text_regions(image)
    }

    fn name(&self) -> &'static str {
        "CCL"
    }
}

/// CNN-based text detector using a lightweight detection CNN
pub struct CnnDetector {
    cnn: crate::layout::detection_cnn::TextDetectionCNN,
}

impl Default for CnnDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CnnDetector {
    /// Create a new CNN detector with heuristic initial weights
    pub fn new() -> Self {
        Self {
            cnn: crate::layout::detection_cnn::TextDetectionCNN::new(),
        }
    }
}

impl TextDetector for CnnDetector {
    fn detect(&self, image: &crate::core::image::OcrImage) -> Result<Vec<TextRegion>> {
        let gray = image.to_grayscale();
        let luma = gray.data.to_luma8();
        let (w, h) = (luma.width() as usize, luma.height() as usize);

        // Convert to normalized ndarray [0,1]
        let mut arr = ndarray::Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                arr[(y, x)] = luma.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
            }
        }

        // Run CNN
        let heatmap = self.cnn.forward(&arr);

        // Post-process: threshold, connected components, bounding boxes
        let boxes = self.cnn.post_process(&heatmap, 0.5, 20);

        let mut regions = Vec::new();
        for (min_x, max_x, min_y, max_y) in boxes {
            let bbox = BoundingBox::new(min_x as u32, min_y as u32, (max_x + 1) as u32, (max_y + 1) as u32);
            regions.push(TextRegion::new(
                format!("cnn_{}_{}", min_x, min_y),
                bbox,
                String::new(),
            ));
        }

        // Fallback to CCL if CNN finds nothing (safety net)
        if regions.is_empty() {
            return TextRegionDetector::detect_text_regions(image);
        }

        Ok(regions)
    }

    fn name(&self) -> &'static str {
        "CNN"
    }
}

/// Multi-angle CCL detector: finds text that is not axis-aligned.
///
/// Sweeps candidate rotations, runs CCL on each, maps boxes back to the
/// original image, and keeps the highest-scoring non-overlapping set.
pub struct OrientedCclDetector {
    /// Angle step in degrees (e.g. 15 → tries -45,-30,…,45)
    pub angle_step_deg: f32,
    /// Maximum absolute angle to search
    pub max_angle_deg: f32,
}

impl Default for OrientedCclDetector {
    fn default() -> Self {
        Self {
            angle_step_deg: 15.0,
            max_angle_deg: 45.0,
        }
    }
}

impl OrientedCclDetector {
    pub fn new(angle_step_deg: f32, max_angle_deg: f32) -> Self {
        Self {
            angle_step_deg: angle_step_deg.max(1.0),
            max_angle_deg: max_angle_deg.max(0.0),
        }
    }

    fn candidate_angles(&self) -> Vec<f32> {
        let mut angles = Vec::new();
        let mut a = -self.max_angle_deg;
        while a <= self.max_angle_deg + 1e-3 {
            angles.push(a);
            a += self.angle_step_deg;
        }
        if !angles.iter().any(|x| x.abs() < 1e-3) {
            angles.push(0.0);
        }
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        angles.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        angles
    }
}

impl TextDetector for OrientedCclDetector {
    fn detect(&self, image: &crate::core::image::OcrImage) -> Result<Vec<TextRegion>> {
        let (ow, oh) = (image.width as f32, image.height as f32);
        let cx = ow / 2.0;
        let cy = oh / 2.0;

        let mut candidates: Vec<(TextRegion, f32)> = Vec::new();

        for angle_deg in self.candidate_angles() {
            let angle_rad = angle_deg.to_radians();
            let rotated = if angle_deg.abs() < 1e-3 {
                image.clone()
            } else {
                image.rotate(angle_rad)?
            };

            let regions = TextRegionDetector::detect_text_regions(&rotated)?;
            let (rw, rh) = (rotated.width as f32, rotated.height as f32);
            let rcx = rw / 2.0;
            let rcy = rh / 2.0;

            for mut region in regions {
                // Map axis-aligned box corners from rotated → original
                let bbox = &region.bounding_box;
                let corners = [
                    (bbox.left as f32, bbox.top as f32),
                    (bbox.right as f32, bbox.top as f32),
                    (bbox.right as f32, bbox.bottom as f32),
                    (bbox.left as f32, bbox.bottom as f32),
                ];
                let inv = -angle_rad;
                let mapped: Vec<(f32, f32)> = corners
                    .iter()
                    .map(|&(x, y)| {
                        let dx = x - rcx;
                        let dy = y - rcy;
                        let cos = inv.cos();
                        let sin = inv.sin();
                        (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
                    })
                    .collect();

                let min_x = mapped
                    .iter()
                    .map(|p| p.0)
                    .fold(f32::INFINITY, f32::min)
                    .clamp(0.0, ow - 1.0);
                let max_x = mapped
                    .iter()
                    .map(|p| p.0)
                    .fold(f32::NEG_INFINITY, f32::max)
                    .clamp(0.0, ow - 1.0);
                let min_y = mapped
                    .iter()
                    .map(|p| p.1)
                    .fold(f32::INFINITY, f32::min)
                    .clamp(0.0, oh - 1.0);
                let max_y = mapped
                    .iter()
                    .map(|p| p.1)
                    .fold(f32::NEG_INFINITY, f32::max)
                    .clamp(0.0, oh - 1.0);

                if max_x <= min_x || max_y <= min_y {
                    continue;
                }

                let mapped_bbox = BoundingBox::new(
                    min_x as u32,
                    min_y as u32,
                    max_x as u32 + 1,
                    max_y as u32 + 1,
                );
                region.bounding_box = mapped_bbox;
                region.properties.rotation_deg = angle_deg;
                region.id = format!("orient_{}_{}", angle_deg as i32, region.id);

                // Prefer wider text-like regions; penalize near-square blobs slightly
                let w = region.bounding_box.width() as f32;
                let h = region.bounding_box.height().max(1) as f32;
                let aspect = w / h;
                let score = (w * h) * aspect.clamp(0.5, 8.0);
                candidates.push((region, score));
            }
        }

        if candidates.is_empty() {
            return TextRegionDetector::detect_text_regions(image);
        }

        // Sort by score descending and greedy NMS
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<TextRegion> = Vec::new();
        for (region, _) in candidates {
            let overlaps = kept.iter().any(|k| {
                bbox_iou(&k.bounding_box, &region.bounding_box) > 0.3
            });
            if !overlaps {
                kept.push(region);
            }
        }

        Ok(kept)
    }

    fn name(&self) -> &'static str {
        "OrientedCCL"
    }
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
    let area_a = a.area() as f32;
    let area_b = b.area() as f32;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Text region detector
pub struct TextRegionDetector;

impl TextRegionDetector {
    /// Detect text regions
    pub fn detect_text_regions(img: &crate::core::image::OcrImage) -> Result<Vec<TextRegion>> {
        // 1. Binarize image
        let mut thresholder = ImageThresholder::new();
        thresholder.set_image(img.clone())?;
        let binary = thresholder.threshold(ThresholdMethod::Otsu)?;

        // 2. Invert if needed (we want white text on black background for morphological ops usually,
        // or just consider foreground as non-zero).
        // Imageproc assumes non-zero is foreground.
        // If binary is standard (text=0, bg=255), we need to invert.
        // Let's check typical output of thresholding. usually text is black (0).
        // So we invert to make text white (255).
        let mut gray = binary.data.to_luma8();
        image::imageops::invert(&mut gray);

        // 3. Dilate to merge characters into words/lines
        // Kernel size depends on resolution. Assuming 300 DPI.
        // 3x3 or 5x5 kernel.
        let dilated = dilate(&gray, Norm::LInf, 2); // k=2 means 5x5 kernel (2*k+1)

        // 4. Connected components
        let labeled = connected_components(&dilated, Connectivity::Eight, Luma([0u8]));

        // 5. Extract bounding boxes
        let mut components = std::collections::HashMap::new();

        for (x, y, pixel) in labeled.enumerate_pixels() {
            let label = pixel[0];
            if label > 0 {
                components
                    .entry(label)
                    .or_insert_with(Vec::new)
                    .push((x, y));
            }
        }

        let mut regions = Vec::new();

        for (label, pixels) in components {
            if pixels.is_empty() {
                continue;
            }

            let min_x = pixels.iter().map(|p| p.0).min().unwrap();
            let max_x = pixels.iter().map(|p| p.0).max().unwrap();
            let min_y = pixels.iter().map(|p| p.1).min().unwrap();
            let max_y = pixels.iter().map(|p| p.1).max().unwrap();

            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;

            // Filter noise
            if width < 5 || height < 5 {
                continue;
            }

            let bbox = BoundingBox::new(
                min_x as u32,
                min_y as u32,
                (max_x + 1) as u32,
                (max_y + 1) as u32,
            );

            regions.push(TextRegion::new(
                format!("region_{}", label),
                bbox,
                String::new(), // Content not yet recognized
            ));
        }

        Ok(regions)
    }

    /// Isolate characters in a text region
    pub fn isolate_characters(
        img: &crate::core::image::OcrImage,
        region: &TextRegion,
    ) -> Result<Vec<BoundingBox>> {
        // Extract region image
        let region_img = img.crop(
            region.bounding_box.left,
            region.bounding_box.top,
            region.bounding_box.width(),
            region.bounding_box.height(),
        )?;
        let gray = region_img.to_grayscale();

        // Binarize locally (simple threshold for now)
        let binary = gray.threshold(128);

        // Connected components to find characters
        // Characters are usually connected components
        // But for touching characters, we might need projection profile.
        // For now, use connected components as a baseline.

        // We need to invert if text is black (0).
        let mut luma = binary.data.to_luma8();
        image::imageops::invert(&mut luma);

        let labeled = connected_components(&luma, Connectivity::Eight, Luma([0u8]));

        let mut components = std::collections::HashMap::new();
        for (x, y, pixel) in labeled.enumerate_pixels() {
            let label = pixel[0];
            if label > 0 {
                components
                    .entry(label)
                    .or_insert_with(Vec::new)
                    .push((x, y));
            }
        }

        let mut char_boxes = Vec::new();
        for (_, pixels) in components {
            if pixels.is_empty() {
                continue;
            }

            let min_x = pixels.iter().map(|p| p.0).min().unwrap();
            let max_x = pixels.iter().map(|p| p.0).max().unwrap();
            let min_y = pixels.iter().map(|p| p.1).min().unwrap();
            let max_y = pixels.iter().map(|p| p.1).max().unwrap();

            // Filter noise (too small)
            if max_x - min_x < 2 || max_y - min_y < 2 {
                continue;
            }

            let char_bbox = BoundingBox::new(
                region.bounding_box.left + min_x as u32,
                region.bounding_box.top + min_y as u32,
                region.bounding_box.left + max_x as u32 + 1,
                region.bounding_box.top + max_y as u32 + 1,
            );

            char_boxes.push(char_bbox);
        }

        // Sort left to right
        char_boxes.sort_by_key(|b| b.left);

        Ok(char_boxes)
    }

    /// Check if a text region contains handwritten text
    pub fn is_handwritten(img: &crate::core::image::OcrImage, region: &TextRegion) -> bool {
        // Heuristic: Handwritten text has high variance in character height and vertical position
        // 1. Isolate components (characters)
        if let Ok(chars) = Self::isolate_characters(img, region) {
            if chars.len() < 3 {
                return false;
            } // Too few characters to judge

            // Calculate heights
            let heights: Vec<f32> = chars.iter().map(|b| b.height() as f32).collect();
            let count = heights.len() as f32;
            let mean_height: f32 = heights.iter().sum::<f32>() / count;

            // Calculate variance
            let variance: f32 = heights
                .iter()
                .map(|h| (h - mean_height).powi(2))
                .sum::<f32>()
                / count;
            let std_dev = variance.sqrt();
            let cv = std_dev / mean_height; // Coefficient of variation

            // Calculate vertical scatter (baseline irregularity)
            // Use bottom coordinate as baseline proxy
            let bottoms: Vec<f32> = chars.iter().map(|b| b.bottom as f32).collect();
            let mean_bottom: f32 = bottoms.iter().sum::<f32>() / count;
            let bottom_variance: f32 = bottoms
                .iter()
                .map(|b| (b - mean_bottom).powi(2))
                .sum::<f32>()
                / count;
            let bottom_std_dev = bottom_variance.sqrt();

            // Thresholds (tuned empirically)
            // High height variation (> 0.25) OR high baseline variation (> 5.0 pixels)
            // Printed text usually has CV < 0.1 and baseline std dev < 2.0

            if cv > 0.25 || bottom_std_dev > 5.0 {
                return true;
            }
        }
        false
    }
}

/// Image region detector
pub struct ImageRegionDetector;

impl ImageRegionDetector {
    /// Detect image regions
    pub fn detect_image_regions(img: &crate::core::image::OcrImage) -> Result<Vec<ImageRegion>> {
        // 1. Binarize
        let mut thresholder = ImageThresholder::new();
        thresholder.set_image(img.clone())?;
        let binary = thresholder.threshold(ThresholdMethod::Otsu)?;

        let mut gray = binary.data.to_luma8();
        image::imageops::invert(&mut gray);

        // 2. Connected components without dilation (to find solid image blocks)
        // or with small dilation to merge lines in line drawings
        let dilated = dilate(&gray, Norm::LInf, 1);
        let labeled = connected_components(&dilated, Connectivity::Eight, Luma([0u8]));

        // 3. Extract and filter
        let mut components = std::collections::HashMap::new();
        for (x, y, pixel) in labeled.enumerate_pixels() {
            let label = pixel[0];
            if label > 0 {
                components
                    .entry(label)
                    .or_insert_with(Vec::new)
                    .push((x, y));
            }
        }

        let mut regions = Vec::new();
        for (label, pixels) in components {
            if pixels.is_empty() {
                continue;
            }

            let min_x = pixels.iter().map(|p| p.0).min().unwrap();
            let max_x = pixels.iter().map(|p| p.0).max().unwrap();
            let min_y = pixels.iter().map(|p| p.1).min().unwrap();
            let max_y = pixels.iter().map(|p| p.1).max().unwrap();

            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;

            // Heuristic: Images are large
            if width > 100 && height > 100 {
                let bbox = BoundingBox::new(
                    min_x as u32,
                    min_y as u32,
                    (max_x + 1) as u32,
                    (max_y + 1) as u32,
                );

                regions.push(ImageRegion::new(format!("img_{}", label), bbox));
            }
        }

        Ok(regions)
    }
}

/// Table detector
pub struct TableDetector;

impl TableDetector {
    /// Detect tables
    pub fn detect_tables(img: &crate::core::image::OcrImage) -> Result<Vec<Table>> {
        // 1. Binarize
        let mut thresholder = ImageThresholder::new();
        thresholder.set_image(img.clone())?;
        let binary = thresholder.threshold(ThresholdMethod::Otsu)?;

        let gray = binary.data.to_luma8();
        let width = gray.width();
        let height = gray.height();

        // 2. Detect horizontal lines
        let h_line_threshold = width / 5; // Must be at least 20% of page width
        let mut h_lines = Vec::new();

        for y in 0..height {
            let mut run_start = None;
            for x in 0..width {
                let pixel = gray.get_pixel(x, y)[0];
                if pixel < 128 {
                    // Dark pixel
                    if run_start.is_none() {
                        run_start = Some(x);
                    }
                } else {
                    if let Some(start) = run_start {
                        let len = x - start;
                        if len > h_line_threshold {
                            h_lines.push(BoundingBox::new(start, y, x, y + 1));
                        }
                        run_start = None;
                    }
                }
            }
            if let Some(start) = run_start {
                let len = width - start;
                if len > h_line_threshold {
                    h_lines.push(BoundingBox::new(start, y, width, y + 1));
                }
            }
        }

        // 3. Detect vertical lines
        let v_line_threshold = height / 10; // Must be at least 10% of page height
        let mut v_lines = Vec::new();

        for x in 0..width {
            let mut run_start = None;
            for y in 0..height {
                let pixel = gray.get_pixel(x, y)[0];
                if pixel < 128 {
                    // Dark pixel
                    if run_start.is_none() {
                        run_start = Some(y);
                    }
                } else {
                    if let Some(start) = run_start {
                        let len = y - start;
                        if len > v_line_threshold {
                            v_lines.push(BoundingBox::new(x, start, x + 1, y));
                        }
                        run_start = None;
                    }
                }
            }
            if let Some(start) = run_start {
                let len = height - start;
                if len > v_line_threshold {
                    v_lines.push(BoundingBox::new(x, start, x + 1, height));
                }
            }
        }

        // 4. Find intersections and group into tables
        let mut tables = Vec::new();

        if !h_lines.is_empty() && !v_lines.is_empty() {
            // Find bounding box of all lines
            let min_x = h_lines
                .iter()
                .map(|b| b.left)
                .min()
                .unwrap_or(0)
                .min(v_lines.iter().map(|b| b.left).min().unwrap_or(0));
            let max_x = h_lines
                .iter()
                .map(|b| b.right)
                .max()
                .unwrap_or(0)
                .max(v_lines.iter().map(|b| b.right).max().unwrap_or(0));
            let min_y = h_lines
                .iter()
                .map(|b| b.top)
                .min()
                .unwrap_or(0)
                .min(v_lines.iter().map(|b| b.top).min().unwrap_or(0));
            let max_y = h_lines
                .iter()
                .map(|b| b.bottom)
                .max()
                .unwrap_or(0)
                .max(v_lines.iter().map(|b| b.bottom).max().unwrap_or(0));

            // Check if we have enough lines to call it a table (e.g. at least 2 H and 2 V)
            if h_lines.len() >= 2 && v_lines.len() >= 2 {
                let bbox = BoundingBox::new(min_x, min_y, max_x, max_y);
                let mut table = Table::new("table_1".to_string(), bbox);

                // 5. Structure analysis (Cells)
                // Find intersections
                let mut row_coords = Vec::new();
                let mut col_coords = Vec::new();

                for h in &h_lines {
                    row_coords.push(h.top); // Y coordinate
                }
                for v in &v_lines {
                    col_coords.push(v.left); // X coordinate
                }

                // Sort and dedup with tolerance
                row_coords.sort();
                col_coords.sort();

                let mut unique_rows = Vec::new();
                if !row_coords.is_empty() {
                    unique_rows.push(row_coords[0]);
                    for &r in &row_coords[1..] {
                        if r > unique_rows.last().unwrap() + 10 {
                            // 10px tolerance
                            unique_rows.push(r);
                        }
                    }
                }

                let mut unique_cols = Vec::new();
                if !col_coords.is_empty() {
                    unique_cols.push(col_coords[0]);
                    for &c in &col_coords[1..] {
                        if c > unique_cols.last().unwrap() + 10 {
                            unique_cols.push(c);
                        }
                    }
                }

                // Create cells
                let rows = if unique_rows.len() > 1 {
                    unique_rows.len() - 1
                } else {
                    0
                };
                let cols = if unique_cols.len() > 1 {
                    unique_cols.len() - 1
                } else {
                    0
                };

                if rows > 0 && cols > 0 {
                    table.structure.rows = rows;
                    table.structure.columns = cols;

                    let mut cells = Vec::new();
                    for r in 0..rows {
                        let mut row_cells = Vec::new();
                        for c in 0..cols {
                            let cell_bbox = BoundingBox::new(
                                unique_cols[c],
                                unique_rows[r],
                                unique_cols[c + 1],
                                unique_rows[r + 1],
                            );

                            let mut cell = TableCell {
                                content: String::new(),
                                bounding_box: cell_bbox,
                                row_span: 1,
                                column_span: 1,
                                properties: CellProperties::default(),
                            };

                            // Basic border detection
                            cell.properties.borders.top.width = 1;
                            cell.properties.borders.bottom.width = 1;
                            cell.properties.borders.left.width = 1;
                            cell.properties.borders.right.width = 1;

                            row_cells.push(cell);
                        }
                        cells.push(row_cells);
                    }

                    // 6. Span inference: check which internal grid lines are actually present
                    Self::infer_spans(&mut cells, &unique_rows, &unique_cols, &gray);

                    table.structure.cells = cells;
                }

                tables.push(table);
            }
        }

        Ok(tables)
    }

    /// Infer row and column spans by checking which internal grid lines are actually present.
    fn infer_spans(
        cells: &mut [Vec<TableCell>],
        unique_rows: &[u32],
        unique_cols: &[u32],
        gray: &image::GrayImage,
    ) {
        if cells.is_empty() || cells[0].is_empty() {
            return;
        }
        let rows = cells.len();
        let cols = cells[0].len();

        // Check horizontal line presence between each pair of rows
        // h_present[r][c] = true if line between row r and r+1 exists at column c
        let mut h_present = vec![vec![false; cols]; rows.saturating_sub(1)];
        for r in 0..rows.saturating_sub(1) {
            let y = unique_rows[r + 1];
            if y >= gray.height() {
                continue;
            }
            for c in 0..cols {
                let x_start = unique_cols[c];
                let x_end = unique_cols[(c + 1).min(unique_cols.len() - 1)];
                let dark_count = Self::count_dark_pixels_horizontal(gray, y, x_start, x_end);
                let length = (x_end - x_start).max(1);
                // Line is "present" if > 30% of pixels are dark
                h_present[r][c] = (dark_count as f32 / length as f32) > 0.30;
            }
        }

        // Check vertical line presence between each pair of columns
        // v_present[r][c] = true if line between col c and c+1 exists at row r
        let mut v_present = vec![vec![false; cols.saturating_sub(1)]; rows];
        for r in 0..rows {
            let y_start = unique_rows[r];
            let y_end = unique_rows[(r + 1).min(unique_rows.len() - 1)];
            for c in 0..cols.saturating_sub(1) {
                let x = unique_cols[c + 1];
                if x >= gray.width() {
                    continue;
                }
                let dark_count = Self::count_dark_pixels_vertical(gray, x, y_start, y_end);
                let length = (y_end - y_start).max(1);
                v_present[r][c] = (dark_count as f32 / length as f32) > 0.30;
            }
        }

        // Merge column spans: for each row, find consecutive cells without vertical lines between them
        for r in 0..rows {
            let mut c = 0;
            while c < cols {
                let mut span = 1;
                while c + span < cols && !v_present[r][c + span - 1] {
                    span += 1;
                }
                for k in 0..span {
                    cells[r][c + k].column_span = span - k;
                }
                // Widen bbox for the first cell in the span
                cells[r][c].bounding_box.right = cells[r][c + span - 1].bounding_box.right;
                c += span;
            }
        }

        // Merge row spans: for each col, find consecutive cells without horizontal lines between them
        for c in 0..cols {
            let mut r = 0;
            while r < rows {
                let mut span = 1;
                while r + span < rows && !h_present[r + span - 1][c] {
                    span += 1;
                }
                for k in 0..span {
                    cells[r + k][c].row_span = span - k;
                }
                // Widen bbox for the first cell in the span
                cells[r][c].bounding_box.bottom = cells[r + span - 1][c].bounding_box.bottom;
                r += span;
            }
        }
    }

    fn count_dark_pixels_horizontal(
        gray: &image::GrayImage,
        y: u32,
        x_start: u32,
        x_end: u32,
    ) -> u32 {
        let mut count = 0;
        for x in x_start..x_end.min(gray.width()) {
            if gray.get_pixel(x, y)[0] < 128 {
                count += 1;
            }
        }
        count
    }

    fn count_dark_pixels_vertical(
        gray: &image::GrayImage,
        x: u32,
        y_start: u32,
        y_end: u32,
    ) -> u32 {
        let mut count = 0;
        for y in y_start..y_end.min(gray.height()) {
            if gray.get_pixel(x, y)[0] < 128 {
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod oriented_tests {
    use super::*;
    use crate::core::image::OcrImage;
    use image::{DynamicImage, GrayImage, Luma};

    fn make_horizontal_bar() -> OcrImage {
        let mut img = GrayImage::from_pixel(80, 80, Luma([255]));
        for x in 10..70 {
            for y in 38..42 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        OcrImage::new(DynamicImage::ImageLuma8(img), 72)
    }

    #[test]
    fn test_oriented_detector_finds_upright_text() {
        let img = make_horizontal_bar();
        let det = OrientedCclDetector::new(15.0, 30.0);
        let regions = det.detect(&img).unwrap();
        assert!(!regions.is_empty(), "should find at least one region");
    }

    #[test]
    fn test_oriented_detector_on_rotated_image() {
        let img = make_horizontal_bar();
        let rotated = img.rotate(30.0_f32.to_radians()).unwrap();
        let det = OrientedCclDetector::new(15.0, 45.0);
        let regions = det.detect(&rotated).unwrap();
        assert!(!regions.is_empty());
        assert!(regions.iter().any(|r| r.id.contains("orient_")));
    }

    #[test]
    fn test_bbox_iou_identical() {
        let a = BoundingBox::new(0, 0, 10, 10);
        assert!((bbox_iou(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_candidate_angles_include_zero() {
        let det = OrientedCclDetector::new(15.0, 45.0);
        let angles = det.candidate_angles();
        assert!(angles.iter().any(|a| a.abs() < 1e-3));
        assert!(*angles.first().unwrap() <= -45.0);
        assert!(*angles.last().unwrap() >= 45.0);
    }
}
