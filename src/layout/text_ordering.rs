//! Text ordering for proper reading sequence
//!
//! Ported from Tesseract's textord.cpp
//! Determines the correct reading order for text regions in documents

use crate::core::image::OcrImage;
use crate::core::layout::{LayoutResult, ReadingOrder, TextRegion};
use crate::layout::{ColumnDetector, ColumnPartition, LineDetector, TextLine};
use crate::utils::Result;

/// Text ordering engine
///
/// Determines the correct reading order for text regions based on
/// document layout analysis
pub struct TextOrdering {
    line_detector: LineDetector,
    column_detector: ColumnDetector,
}

impl TextOrdering {
    /// Create a new text ordering engine
    pub fn new(grid_size: u32, resolution: u32) -> Self {
        Self {
            line_detector: LineDetector::new(grid_size, 1.5, 10),
            column_detector: ColumnDetector::new(grid_size, resolution, false, 0.5),
        }
    }

    /// Order text regions for reading
    ///
    /// Analyzes the layout and determines the correct reading order
    /// for text regions
    pub fn order_text_regions(
        &self,
        image: &OcrImage,
        text_regions: &[TextRegion],
    ) -> Result<Vec<TextRegion>> {
        if text_regions.is_empty() {
            return Ok(Vec::new());
        }

        // First, detect columns
        let columns = self.column_detector.find_columns(image, text_regions)?;

        // If multiple columns, order by column first
        if columns.len() > 1 {
            self.order_by_columns(columns)
        } else {
            // Single column - order by lines
            let lines = self.line_detector.detect_lines(text_regions)?;
            self.order_by_lines(lines)
        }
    }

    /// Order text regions by columns
    fn order_by_columns(&self, columns: Vec<ColumnPartition>) -> Result<Vec<TextRegion>> {
        // Get reading order for columns
        let column_order = self.column_detector.determine_reading_order(&columns);

        let mut ordered_regions = Vec::new();

        // Process columns in reading order
        for &col_idx in &column_order {
            let column = &columns[col_idx];

            // Order regions within column by lines
            let lines = self.line_detector.detect_lines(&column.text_regions)?;
            let line_ordered = self.order_by_lines(lines)?;

            ordered_regions.extend(line_ordered);
        }

        Ok(ordered_regions)
    }

    /// Order text regions by lines
    fn order_by_lines(&self, lines: Vec<TextLine>) -> Result<Vec<TextRegion>> {
        let mut ordered_regions = Vec::new();

        for line in lines {
            // Regions within a line are already sorted left-to-right
            ordered_regions.extend(line.text_regions);
        }

        Ok(ordered_regions)
    }

    /// Determine reading order for a layout result
    ///
    /// Updates the reading order in the layout result based on
    /// detected columns and lines
    pub fn determine_reading_order(
        &self,
        image: &OcrImage,
        layout: &mut LayoutResult,
    ) -> Result<()> {
        // Order text regions
        let ordered_regions = self.order_text_regions(image, &layout.text_regions)?;
        layout.text_regions = ordered_regions;

        // Detect columns to determine reading order type
        let columns = self
            .column_detector
            .find_columns(image, &layout.text_regions)?;

        if columns.len() > 1 {
            layout.reading_order = ReadingOrder::MultiColumn;
        } else {
            layout.reading_order = ReadingOrder::TopToBottom;
        }

        Ok(())
    }

    /// Set CJK script mode
    pub fn set_cjk_script(&mut self, cjk_script: bool) {
        self.column_detector.set_cjk_script(cjk_script);
    }
}

impl Default for TextOrdering {
    fn default() -> Self {
        Self::new(20, 300)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::text::BoundingBox;

    #[test]
    fn test_text_ordering_creation() {
        let _ordering = TextOrdering::new(20, 300);
        // Should not panic
    }

    #[test]
    fn test_single_column_ordering() {
        let ordering = TextOrdering::default();

        let regions = vec![
            TextRegion::new(
                "1".to_string(),
                BoundingBox::new(0, 0, 50, 20),
                "First".to_string(),
            ),
            TextRegion::new(
                "2".to_string(),
                BoundingBox::new(0, 30, 50, 50),
                "Second".to_string(),
            ),
        ];

        // Create a dummy image
        let image = OcrImage::new(image::DynamicImage::new_rgb8(100, 100), 300);

        let ordered = ordering.order_text_regions(&image, &regions).unwrap();
        assert_eq!(ordered.len(), 2);
        // Should be ordered top to bottom
        assert_eq!(ordered[0].id, "1");
        assert_eq!(ordered[1].id, "2");
    }
}
