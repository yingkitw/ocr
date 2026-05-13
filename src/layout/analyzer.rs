//! Layout analysis operations

use crate::core::layout::*;
use crate::layout::column_detector::ColumnDetector;
use crate::layout::detector::{ImageRegionDetector, TableDetector, TextRegionDetector};
use crate::utils::Result;

/// Layout analyzer
pub struct LayoutAnalyzer;

impl LayoutAnalyzer {
    /// Analyze page layout
    pub fn analyze_layout(img: &crate::core::image::OcrImage) -> Result<LayoutResult> {
        let page_size = PageSize::new(img.width, img.height, img.dpi);
        let mut result = LayoutResult::new(page_size);

        // Detect text regions
        let mut text_regions = TextRegionDetector::detect_text_regions(img)?;

        // Detect images
        let image_regions = ImageRegionDetector::detect_image_regions(img)?;
        result.images = image_regions;

        // Detect tables
        let tables = TableDetector::detect_tables(img)?;
        result.tables = tables;

        // Resolve conflicts (Mixed content processing)
        // 1. Remove text regions that are inside tables
        let mut text_regions_in_tables = Vec::new();
        text_regions.retain(|region| {
            let center = region.bounding_box.center();
            let mut inside_table = false;
            for table in &result.tables {
                if table
                    .bounding_box
                    .contains(center.x as u32, center.y as u32)
                {
                    inside_table = true;
                    break;
                }
            }
            if inside_table {
                text_regions_in_tables.push(region.clone());
            }
            !inside_table
        });

        // 2. Remove text regions that are inside images (optional, depends on requirement)
        // For now we keep them but maybe mark them? Or if they are very small compared to image?

        // Detect columns and reorder text regions
        let column_detector = ColumnDetector::default();
        let column_partitions = column_detector.find_columns(img, &text_regions)?;

        // Use column reading order
        let reading_order = column_detector.determine_reading_order(&column_partitions);

        // Reconstruct sorted text regions
        let mut sorted_text_regions = Vec::new();
        for &idx in &reading_order {
            if let Some(partition) = column_partitions.get(idx) {
                // Sort regions within the partition (Top-to-Bottom)
                let mut regions = partition.text_regions.clone();
                regions.sort_by(|a, b| a.bounding_box.top.cmp(&b.bounding_box.top));
                sorted_text_regions.extend(regions);
            }
        }

        // If column detection failed or returned empty (single column), fallback to simple sort
        if sorted_text_regions.is_empty() && !text_regions.is_empty() {
            sorted_text_regions = text_regions;
            sorted_text_regions.sort_by(|a, b| {
                a.bounding_box
                    .top
                    .cmp(&b.bounding_box.top)
                    .then_with(|| a.bounding_box.left.cmp(&b.bounding_box.left))
            });
        }

        result.text_regions = sorted_text_regions;

        // Convert all regions to blocks
        for region in &mut result.text_regions {
            // Check if handwritten
            let is_handwritten = TextRegionDetector::is_handwritten(img, region);

            if is_handwritten {
                region.properties.font_family = Some("handwritten".to_string());
            }

            let mut block = Block::new(
                format!("block_{}", region.id),
                BlockType::Text,
                region.bounding_box,
            );

            if is_handwritten {
                block.properties.background_color =
                    Some(crate::core::layout::Color::rgb(255, 255, 200)); // Mark with light yellow
            }

            result.blocks.push(block);
        }

        for region in &result.images {
            let block = Block::new(
                format!("block_{}", region.id),
                BlockType::Image,
                region.bounding_box,
            );
            result.blocks.push(block);
        }

        for table in &result.tables {
            let block = Block::new(
                format!("block_{}", table.id),
                BlockType::Table,
                table.bounding_box,
            );
            // We could populate block content with table structure here if we had it
            result.blocks.push(block);
        }

        // Final sort of all blocks (interleaving text, images, tables)
        // This is tricky with columns.
        // A simple approach is to sort by top, but that breaks columns.
        // Better: Keep text in its column order, and insert images/tables where they fit vertically?
        // For now, let's trust the text ordering we just did, and append others?
        // No, that puts images at the end.
        // Let's use a "reading order" sort that respects columns if possible.
        // But since Block structure is flat, we usually sort Top-Left.
        // If we want to preserve column order, we should assign "priority" or "index" to blocks.

        for (i, block) in result.blocks.iter_mut().enumerate() {
            block.properties.priority = i as u32;
        }

        // Actually, let's sort by Top-Left for the final block list,
        // but the `text_regions` list in `result` preserves the reading order.
        result.blocks.sort_by(|a, b| {
            a.bounding_box
                .top
                .cmp(&b.bounding_box.top)
                .then_with(|| a.bounding_box.left.cmp(&b.bounding_box.left))
        });

        Ok(result)
    }
}
