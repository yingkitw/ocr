//! Region detection operations

use crate::core::layout::*;
use crate::core::text::BoundingBox;
use crate::image::thresholder::{ImageThresholder, ThresholdMethod};
use crate::utils::Result;
use image::{GenericImageView, Luma};
use imageproc::distance_transform::Norm;
use imageproc::morphology::dilate;
use imageproc::region_labelling::{Connectivity, connected_components};

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
                    table.structure.cells = cells;
                }

                tables.push(table);
            }
        }

        Ok(tables)
    }
}
