//! Column detection for multi-column document layout analysis
//!
//! Ported from Tesseract's colfind.h/cpp
//! Detects columns in documents to properly order text regions

use crate::core::image::OcrImage;
use crate::core::layout::TextRegion;
use crate::core::text::BoundingBox;
use crate::utils::Result;
use std::collections::HashMap;

/// Column detector for multi-column document analysis
///
/// Analyzes document layout to identify columns and determines
/// reading order for text regions
pub struct ColumnDetector {
    /// Grid size estimate (text size in pixels)
    grid_size: u32,
    /// Image resolution
    resolution: u32,
    /// Whether CJK script is being processed
    cjk_script: bool,
    /// Aligned gap fraction threshold
    aligned_gap_fraction: f64,
}

/// Column partition representing a detected column
#[derive(Debug, Clone)]
pub struct ColumnPartition {
    /// Bounding box of the column
    pub bounding_box: BoundingBox,
    /// Text regions in this column
    pub text_regions: Vec<TextRegion>,
    /// Column type
    pub column_type: ColumnType,
    /// Confidence score
    pub confidence: f32,
}

/// Type of column partition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// Text column
    Text,
    /// Image column
    Image,
    /// Table column
    Table,
    /// Unknown/mixed content
    Unknown,
}

impl ColumnDetector {
    /// Create a new column detector
    ///
    /// # Arguments
    /// * `grid_size` - Estimate of text size in pixels
    /// * `resolution` - Image resolution in DPI
    /// * `cjk_script` - Whether processing CJK script
    /// * `aligned_gap_fraction` - Fraction for aligned gap detection
    pub fn new(
        grid_size: u32,
        resolution: u32,
        cjk_script: bool,
        aligned_gap_fraction: f64,
    ) -> Self {
        Self {
            grid_size,
            resolution,
            cjk_script,
            aligned_gap_fraction,
        }
    }

    /// Find columns in the document
    ///
    /// Analyzes the layout to detect column boundaries and organize
    /// text regions into columns
    pub fn find_columns(
        &self,
        image: &OcrImage,
        text_regions: &[TextRegion],
    ) -> Result<Vec<ColumnPartition>> {
        if text_regions.is_empty() {
            return Ok(Vec::new());
        }

        // Group text regions by vertical position
        let region_groups = self.group_regions_by_position(text_regions);

        // Detect column boundaries
        let column_boundaries = self.detect_column_boundaries(image, &region_groups)?;

        // Create column partitions
        let partitions = self.create_column_partitions(text_regions, &column_boundaries)?;

        Ok(partitions)
    }

    /// Group text regions by their vertical position
    fn group_regions_by_position(&self, text_regions: &[TextRegion]) -> HashMap<u32, Vec<usize>> {
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();

        for (idx, region) in text_regions.iter().enumerate() {
            // Group by vertical center (rounded to grid_size)
            let y_center = (region.bounding_box.top + region.bounding_box.bottom) / 2;
            let group_key = (y_center / self.grid_size) * self.grid_size;

            groups.entry(group_key).or_insert_with(Vec::new).push(idx);
        }

        groups
    }

    /// Detect column boundaries based on gaps between text regions
    fn detect_column_boundaries(
        &self,
        image: &OcrImage,
        region_groups: &HashMap<u32, Vec<usize>>,
    ) -> Result<Vec<u32>> {
        let mut boundaries = Vec::new();

        // Analyze horizontal gaps to find column boundaries
        // For each horizontal slice, find significant gaps
        let slice_height = self.grid_size * 2;

        for y in (0..image.height).step_by(slice_height as usize) {
            let slice_end = (y + slice_height).min(image.height);
            let slice_gaps = self.find_gaps_in_slice(image, y, slice_end)?;

            // Merge gaps that are aligned across slices
            for gap in slice_gaps {
                if self.is_significant_gap(gap, image.width) {
                    boundaries.push(gap);
                }
            }
        }

        // Sort and deduplicate boundaries
        boundaries.sort();
        boundaries.dedup();

        Ok(boundaries)
    }

    /// Find gaps in a horizontal slice of the image
    fn find_gaps_in_slice(&self, image: &OcrImage, y_start: u32, y_end: u32) -> Result<Vec<u32>> {
        let mut gaps = Vec::new();

        // Use pixel projection to find gaps
        // We need a binary or grayscale view.
        // Assuming white background (255) and black text (0).

        let gray = image.to_grayscale(); // This creates a new image, might be expensive if called many times
        // Optimization: Pass the gray image to find_columns and propagate it down
        // But for now, we follow the signature.

        let width = image.width;
        let y_start = y_start.min(image.height);
        let y_end = y_end.min(image.height);

        if y_start >= y_end {
            return Ok(gaps);
        }

        // Calculate vertical projection profile for the slice
        // Sum of (255 - pixel) for each column. High sum = content. 0 = white space.
        let mut projection = vec![0u64; width as usize];

        if let Some(buf) = gray.data.as_luma8() {
            for y in y_start..y_end {
                for x in 0..width {
                    let pixel = buf.get_pixel(x, y)[0];
                    if pixel < 200 {
                        // Consider < 200 as content (not white)
                        projection[x as usize] += 1;
                    }
                }
            }
        }

        // Find runs of 0s (gaps)
        let mut in_gap = false;
        let mut gap_start = 0;

        for x in 0..width {
            let has_content = projection[x as usize] > 0;

            if !has_content {
                if !in_gap {
                    in_gap = true;
                    gap_start = x;
                }
            } else {
                if in_gap {
                    in_gap = false;
                    let gap_width = x - gap_start;
                    // Center of the gap
                    if gap_width > self.grid_size {
                        gaps.push(gap_start + gap_width / 2);
                    }
                }
            }
        }

        // Check last gap
        if in_gap {
            let gap_width = width - gap_start;
            if gap_width > self.grid_size {
                gaps.push(gap_start + gap_width / 2);
            }
        }

        Ok(gaps)
    }

    /// Check if a gap is significant enough to be a column boundary
    fn is_significant_gap(&self, gap: u32, image_width: u32) -> bool {
        // Gap should be at least a fraction of the grid size
        let min_gap = self.grid_size as f64 * self.aligned_gap_fraction;
        gap as f64 >= min_gap && gap < image_width
    }

    /// Create column partitions from text regions and boundaries
    fn create_column_partitions(
        &self,
        text_regions: &[TextRegion],
        boundaries: &[u32],
    ) -> Result<Vec<ColumnPartition>> {
        let mut partitions = Vec::new();

        if boundaries.is_empty() {
            // Single column - all regions in one partition
            let bbox = self.compute_bounding_box(text_regions);
            partitions.push(ColumnPartition {
                bounding_box: bbox,
                text_regions: text_regions.to_vec(),
                column_type: ColumnType::Text,
                confidence: 1.0,
            });
            return Ok(partitions);
        }

        // Create partitions for each column
        let mut start_x = 0u32;
        for &end_x in boundaries {
            let column_regions: Vec<TextRegion> = text_regions
                .iter()
                .filter(|region| {
                    let center_x = (region.bounding_box.left + region.bounding_box.right) / 2;
                    center_x >= start_x && center_x < end_x
                })
                .cloned()
                .collect();

            if !column_regions.is_empty() {
                let bbox = self.compute_bounding_box(&column_regions);
                partitions.push(ColumnPartition {
                    bounding_box: bbox,
                    text_regions: column_regions,
                    column_type: ColumnType::Text,
                    confidence: 0.8,
                });
            }

            start_x = end_x;
        }

        // Handle last column
        if let Some(&last_boundary) = boundaries.last() {
            let column_regions: Vec<TextRegion> = text_regions
                .iter()
                .filter(|region| {
                    let center_x = (region.bounding_box.left + region.bounding_box.right) / 2;
                    center_x >= last_boundary
                })
                .cloned()
                .collect();

            if !column_regions.is_empty() {
                let bbox = self.compute_bounding_box(&column_regions);
                partitions.push(ColumnPartition {
                    bounding_box: bbox,
                    text_regions: column_regions,
                    column_type: ColumnType::Text,
                    confidence: 0.8,
                });
            }
        }

        Ok(partitions)
    }

    /// Compute bounding box for a set of text regions
    fn compute_bounding_box(&self, regions: &[TextRegion]) -> BoundingBox {
        if regions.is_empty() {
            return BoundingBox::new(0, 0, 0, 0);
        }

        let mut min_left = u32::MAX;
        let mut min_top = u32::MAX;
        let mut max_right = 0u32;
        let mut max_bottom = 0u32;

        for region in regions {
            min_left = min_left.min(region.bounding_box.left);
            min_top = min_top.min(region.bounding_box.top);
            max_right = max_right.max(region.bounding_box.right);
            max_bottom = max_bottom.max(region.bounding_box.bottom);
        }

        BoundingBox::new(min_left, min_top, max_right, max_bottom)
    }

    /// Determine reading order for columns
    ///
    /// For multi-column layouts, determines the correct reading order
    /// (left-to-right, top-to-bottom for most languages)
    pub fn determine_reading_order(&self, partitions: &[ColumnPartition]) -> Vec<usize> {
        // Sort partitions by top position, then by left position
        let mut indices: Vec<usize> = (0..partitions.len()).collect();

        indices.sort_by(|&a, &b| {
            let part_a = &partitions[a];
            let part_b = &partitions[b];

            // First sort by top (top-to-bottom)
            part_a
                .bounding_box
                .top
                .cmp(&part_b.bounding_box.top)
                .then_with(|| {
                    // Then by left (left-to-right)
                    part_a.bounding_box.left.cmp(&part_b.bounding_box.left)
                })
        });

        indices
    }

    /// Set CJK script mode
    pub fn set_cjk_script(&mut self, cjk_script: bool) {
        self.cjk_script = cjk_script;
    }

    /// Recursive XY-cut algorithm for complex multi-column layouts
    ///
    /// Recursively partitions the page by alternating horizontal and vertical cuts
    /// at significant gaps. This handles nested column layouts better than
    /// simple gap analysis.
    pub fn find_columns_xycut(
        &self,
        text_regions: &[TextRegion],
        page_width: u32,
        page_height: u32,
    ) -> Vec<ColumnPartition> {
        if text_regions.is_empty() {
            return Vec::new();
        }

        let mut partitions = Vec::new();
        self.xycut_recursive(
            text_regions,
            0,
            0,
            page_width,
            page_height,
            0, // start with vertical cut (columns)
            &mut partitions,
        );

        if partitions.is_empty() {
            let bbox = self.compute_bounding_box(text_regions);
            partitions.push(ColumnPartition {
                bounding_box: bbox,
                text_regions: text_regions.to_vec(),
                column_type: ColumnType::Text,
                confidence: 1.0,
            });
        }

        partitions
    }

    fn xycut_recursive(
        &self,
        regions: &[TextRegion],
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        depth: u32,
        out: &mut Vec<ColumnPartition>,
    ) {
        if regions.is_empty() || depth > 10 {
            return;
        }

        let min_gap = self.grid_size * 3; // Minimum gap to consider a cut
        let max_depth = 4;

        if depth >= max_depth {
            // Leaf node: create partition
            let bbox = self.compute_bounding_box(regions);
            out.push(ColumnPartition {
                bounding_box: bbox,
                text_regions: regions.to_vec(),
                column_type: ColumnType::Text,
                confidence: 0.9,
            });
            return;
        }

        // Alternate between vertical (columns) and horizontal (rows/sections) cuts
        if depth % 2 == 0 {
            // Vertical cut: look for column gaps
            if let Some(cut_x) = self.find_best_vertical_cut(regions, x, w, min_gap) {
                let (left_regions, right_regions): (Vec<_>, Vec<_>) = regions
                    .iter()
                    .cloned()
                    .partition(|r| {
                        let cx = (r.bounding_box.left + r.bounding_box.right) / 2;
                        cx < cut_x
                    });

                if !left_regions.is_empty() && !right_regions.is_empty() {
                    self.xycut_recursive(&left_regions, x, y, cut_x - x, h, depth + 1, out);
                    self.xycut_recursive(
                        &right_regions,
                        cut_x,
                        y,
                        (x + w) - cut_x,
                        h,
                        depth + 1,
                        out,
                    );
                    return;
                }
            }
        } else {
            // Horizontal cut: look for section gaps
            if let Some(cut_y) = self.find_best_horizontal_cut(regions, y, h, min_gap) {
                let (top_regions, bottom_regions): (Vec<_>, Vec<_>) = regions
                    .iter()
                    .cloned()
                    .partition(|r| {
                        let cy = (r.bounding_box.top + r.bounding_box.bottom) / 2;
                        cy < cut_y
                    });

                if !top_regions.is_empty() && !bottom_regions.is_empty() {
                    self.xycut_recursive(&top_regions, x, y, w, cut_y - y, depth + 1, out);
                    self.xycut_recursive(
                        &bottom_regions,
                        x,
                        cut_y,
                        w,
                        (y + h) - cut_y,
                        depth + 1,
                        out,
                    );
                    return;
                }
            }
        }

        // No cut found: leaf node
        let bbox = self.compute_bounding_box(regions);
        out.push(ColumnPartition {
            bounding_box: bbox,
            text_regions: regions.to_vec(),
            column_type: ColumnType::Text,
            confidence: 0.9,
        });
    }

    fn find_best_vertical_cut(
        &self,
        regions: &[TextRegion],
        x_offset: u32,
        width: u32,
        min_gap: u32,
    ) -> Option<u32> {
        // Build projection profile
        let mut projection = vec![0u32; width as usize];

        for region in regions {
            let left = region.bounding_box.left.saturating_sub(x_offset);
            let right = region
                .bounding_box
                .right
                .saturating_sub(x_offset)
                .min(width - 1);
            for px in left..=right {
                if (px as usize) < projection.len() {
                    projection[px as usize] += 1;
                }
            }
        }

        // Find widest gap with zero projection
        let mut best_gap_start = 0u32;
        let mut best_gap_width = 0u32;
        let mut gap_start = None;

        for (x, &count) in projection.iter().enumerate() {
            let x = x as u32;
            if count == 0 {
                if gap_start.is_none() {
                    gap_start = Some(x);
                }
            } else if let Some(start) = gap_start {
                let gap_width = x - start;
                if gap_width > best_gap_width {
                    best_gap_width = gap_width;
                    best_gap_start = start;
                }
                gap_start = None;
            }
        }

        if best_gap_width >= min_gap {
            let cut = x_offset + best_gap_start + best_gap_width / 2;
            // Verify cut splits regions meaningfully
            let left_count = regions
                .iter()
                .filter(|r| (r.bounding_box.left + r.bounding_box.right) / 2 < cut)
                .count();
            let right_count = regions.len() - left_count;
            if left_count > 0 && right_count > 0 {
                return Some(cut);
            }
        }

        None
    }

    fn find_best_horizontal_cut(
        &self,
        regions: &[TextRegion],
        y_offset: u32,
        height: u32,
        min_gap: u32,
    ) -> Option<u32> {
        let mut projection = vec![0u32; height as usize];

        for region in regions {
            let top = region.bounding_box.top.saturating_sub(y_offset);
            let bottom = region
                .bounding_box
                .bottom
                .saturating_sub(y_offset)
                .min(height - 1);
            for py in top..=bottom {
                if (py as usize) < projection.len() {
                    projection[py as usize] += 1;
                }
            }
        }

        let mut best_gap_start = 0u32;
        let mut best_gap_width = 0u32;
        let mut gap_start = None;

        for (y, &count) in projection.iter().enumerate() {
            let y = y as u32;
            if count == 0 {
                if gap_start.is_none() {
                    gap_start = Some(y);
                }
            } else if let Some(start) = gap_start {
                let gap_width = y - start;
                if gap_width > best_gap_width {
                    best_gap_width = gap_width;
                    best_gap_start = start;
                }
                gap_start = None;
            }
        }

        if best_gap_width >= min_gap {
            let cut = y_offset + best_gap_start + best_gap_width / 2;
            let top_count = regions
                .iter()
                .filter(|r| (r.bounding_box.top + r.bounding_box.bottom) / 2 < cut)
                .count();
            let bottom_count = regions.len() - top_count;
            if top_count > 0 && bottom_count > 0 {
                return Some(cut);
            }
        }

        None
    }
}

impl Default for ColumnDetector {
    fn default() -> Self {
        Self::new(20, 300, false, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_detector_creation() {
        let detector = ColumnDetector::new(20, 300, false, 0.5);
        assert_eq!(detector.grid_size, 20);
        assert_eq!(detector.resolution, 300);
    }

    #[test]
    fn test_reading_order() {
        let partitions = vec![
            ColumnPartition {
                bounding_box: BoundingBox::new(200, 0, 300, 100),
                text_regions: Vec::new(),
                column_type: ColumnType::Text,
                confidence: 1.0,
            },
            ColumnPartition {
                bounding_box: BoundingBox::new(0, 0, 100, 100),
                text_regions: Vec::new(),
                column_type: ColumnType::Text,
                confidence: 1.0,
            },
        ];

        let detector = ColumnDetector::default();
        let order = detector.determine_reading_order(&partitions);

        // Should be sorted left-to-right
        assert_eq!(order, vec![1, 0]);
    }
}
