//! Line detection for text layout analysis
//!
//! Ported from Tesseract's linefind.h/cpp
//! Detects text lines in documents for proper text ordering

use crate::core::layout::TextRegion;
use crate::core::text::BoundingBox;
use crate::utils::Result;

/// Line detector for finding text lines in documents
///
/// Analyzes text regions to group them into lines and determine
/// line ordering for reading
pub struct LineDetector {
    /// Grid size estimate (text size in pixels)
    grid_size: u32,
    /// Maximum line spacing
    max_line_spacing: f32,
    /// Minimum line height
    min_line_height: u32,
}

/// Detected text line
#[derive(Debug, Clone)]
pub struct TextLine {
    /// Bounding box of the line
    pub bounding_box: BoundingBox,
    /// Text regions in this line
    pub text_regions: Vec<TextRegion>,
    /// Baseline y-coordinate
    pub baseline: f32,
    /// Line height
    pub line_height: u32,
    /// Confidence score
    pub confidence: f32,
}

impl LineDetector {
    /// Create a new line detector
    ///
    /// # Arguments
    /// * `grid_size` - Estimate of text size in pixels
    /// * `max_line_spacing` - Maximum spacing between lines (as multiple of grid_size)
    /// * `min_line_height` - Minimum height for a valid line
    pub fn new(grid_size: u32, max_line_spacing: f32, min_line_height: u32) -> Self {
        Self {
            grid_size,
            max_line_spacing,
            min_line_height,
        }
    }

    /// Detect lines from text regions
    ///
    /// Groups text regions into lines based on their vertical positions
    /// and horizontal alignment
    pub fn detect_lines(&self, text_regions: &[TextRegion]) -> Result<Vec<TextLine>> {
        if text_regions.is_empty() {
            return Ok(Vec::new());
        }

        // Group regions by vertical position (potential lines)
        let line_groups = self.group_regions_vertically(text_regions);

        // Refine line groups by horizontal alignment
        let mut lines = Vec::new();
        for group in line_groups {
            if let Some(line) = self.create_line_from_group(&group, text_regions)? {
                lines.push(line);
            }
        }

        // Sort lines by reading order (top to bottom)
        lines.sort_by(|a, b| {
            a.bounding_box
                .top
                .cmp(&b.bounding_box.top)
                .then_with(|| a.bounding_box.left.cmp(&b.bounding_box.left))
        });

        Ok(lines)
    }

    /// Group text regions by their vertical position
    fn group_regions_vertically(&self, text_regions: &[TextRegion]) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut used = vec![false; text_regions.len()];

        for i in 0..text_regions.len() {
            if used[i] {
                continue;
            }

            let mut group = vec![i];
            used[i] = true;

            // Find all regions that are on the same line
            let center_y =
                (text_regions[i].bounding_box.top + text_regions[i].bounding_box.bottom) / 2;
            let tolerance = self.grid_size / 2;

            for j in (i + 1)..text_regions.len() {
                if used[j] {
                    continue;
                }

                let other_center_y =
                    (text_regions[j].bounding_box.top + text_regions[j].bounding_box.bottom) / 2;
                let y_diff = if center_y > other_center_y {
                    center_y - other_center_y
                } else {
                    other_center_y - center_y
                };

                if y_diff <= tolerance {
                    group.push(j);
                    used[j] = true;
                }
            }

            if !group.is_empty() {
                groups.push(group);
            }
        }

        groups
    }

    /// Create a line from a group of region indices
    fn create_line_from_group(
        &self,
        group: &[usize],
        text_regions: &[TextRegion],
    ) -> Result<Option<TextLine>> {
        if group.is_empty() {
            return Ok(None);
        }

        // Get regions for this group
        let regions: Vec<TextRegion> = group.iter().map(|&idx| text_regions[idx].clone()).collect();

        // Sort regions horizontally (left to right)
        let mut sorted_regions = regions;
        sorted_regions.sort_by(|a, b| a.bounding_box.left.cmp(&b.bounding_box.left));

        // Compute line bounding box
        let bounding_box = self.compute_line_bounding_box(&sorted_regions);

        // Compute baseline (average of region centers)
        let baseline = self.compute_baseline(&sorted_regions);

        // Compute line height (average of region heights)
        let line_height = self.compute_line_height(&sorted_regions);

        // Validate line
        if line_height < self.min_line_height {
            return Ok(None);
        }

        Ok(Some(TextLine {
            bounding_box,
            text_regions: sorted_regions,
            baseline,
            line_height,
            confidence: 0.9,
        }))
    }

    /// Compute bounding box for a line
    fn compute_line_bounding_box(&self, regions: &[TextRegion]) -> BoundingBox {
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

    /// Compute baseline for a line
    fn compute_baseline(&self, regions: &[TextRegion]) -> f32 {
        if regions.is_empty() {
            return 0.0;
        }

        let sum: f32 = regions
            .iter()
            .map(|r| {
                // Use bottom of bounding box as baseline estimate
                r.bounding_box.bottom as f32
            })
            .sum();

        sum / regions.len() as f32
    }

    /// Compute average line height
    fn compute_line_height(&self, regions: &[TextRegion]) -> u32 {
        if regions.is_empty() {
            return 0;
        }

        let sum: u32 = regions.iter().map(|r| r.bounding_box.height()).sum();

        sum / regions.len() as u32
    }

    /// Detect line spacing
    ///
    /// Analyzes spacing between lines to determine paragraph breaks
    pub fn detect_line_spacing(&self, lines: &[TextLine]) -> Vec<f32> {
        if lines.len() < 2 {
            return Vec::new();
        }

        let mut spacings = Vec::new();
        for i in 0..lines.len() - 1 {
            let current_bottom = lines[i].bounding_box.bottom as f32;
            let next_top = lines[i + 1].bounding_box.top as f32;
            let spacing = next_top - current_bottom;
            spacings.push(spacing);
        }

        spacings
    }

    /// Identify paragraph breaks based on line spacing
    pub fn identify_paragraphs(&self, lines: &[TextLine]) -> Vec<usize> {
        let spacings = self.detect_line_spacing(lines);
        if spacings.is_empty() {
            return vec![0]; // Single paragraph
        }

        // Calculate average spacing
        let avg_spacing: f32 = spacings.iter().sum::<f32>() / spacings.len() as f32;
        let paragraph_threshold = avg_spacing * self.max_line_spacing;

        // Find paragraph breaks (large spacing)
        let mut breaks = vec![0]; // Start of first paragraph
        for (i, spacing) in spacings.iter().enumerate() {
            if *spacing > paragraph_threshold {
                breaks.push(i + 1);
            }
        }

        breaks
    }
}

impl Default for LineDetector {
    fn default() -> Self {
        Self::new(20, 1.5, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_detector_creation() {
        let detector = LineDetector::new(20, 1.5, 10);
        assert_eq!(detector.grid_size, 20);
    }

    #[test]
    fn test_line_detection() {
        let detector = LineDetector::default();

        // Create test regions on the same line
        let regions = vec![
            TextRegion::new(
                "1".to_string(),
                BoundingBox::new(0, 10, 50, 30),
                "Hello".to_string(),
            ),
            TextRegion::new(
                "2".to_string(),
                BoundingBox::new(60, 10, 110, 30),
                "World".to_string(),
            ),
        ];

        let lines = detector.detect_lines(&regions).unwrap();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text_regions.len(), 2);
    }
}
