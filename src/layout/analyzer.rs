//! Layout analysis operations

use crate::core::layout::*;
use crate::layout::classifier::RegionClassifier;
use crate::layout::column_detector::ColumnDetector;
use crate::layout::detector::{ImageRegionDetector, TableDetector, TextDetector, TextRegionDetector};
use crate::utils::Result;

/// Layout analyzer
pub struct LayoutAnalyzer;

impl LayoutAnalyzer {
    /// Analyze page layout
    pub fn analyze_layout(img: &crate::core::image::OcrImage) -> Result<LayoutResult> {
        Self::analyze_layout_with_options(img, false)
    }

    /// Analyze page layout, optionally using multi-angle text detection.
    pub fn analyze_layout_with_options(
        img: &crate::core::image::OcrImage,
        arbitrary_angle: bool,
    ) -> Result<LayoutResult> {
        let page_size = PageSize::new(img.width, img.height, img.dpi);
        let mut result = LayoutResult::new(page_size);

        // Detect text regions (axis-aligned CCL or multi-angle oriented CCL)
        let mut text_regions = if arbitrary_angle {
            crate::layout::detector::OrientedCclDetector::default().detect(img)?
        } else {
            TextRegionDetector::detect_text_regions(img)?
        };

        // Detect images
        let image_regions = ImageRegionDetector::detect_image_regions(img)?;
        result.images = image_regions;

        // Detect tables
        let tables = TableDetector::detect_tables(img)?;
        result.tables = tables;

        // Resolve conflicts: remove text regions inside tables
        text_regions.retain(|region| {
            let center = region.bounding_box.center();
            !result.tables.iter().any(|table| {
                table
                    .bounding_box
                    .contains(center.x as u32, center.y as u32)
            })
        });

        // Classify text regions (heading, body, caption, etc.)
        let mut classifier = RegionClassifier::new(img.width, img.height);
        let classifications = classifier.classify(&text_regions);

        // Detect columns using recursive XY-cut for complex layouts
        let column_detector = ColumnDetector::default();
        let column_partitions =
            column_detector.find_columns_xycut(&text_regions, img.width, img.height);

        // Use column reading order
        let reading_order = column_detector.determine_reading_order(&column_partitions);

        // Reconstruct sorted text regions
        let mut sorted_text_regions = Vec::new();
        for &idx in &reading_order {
            if let Some(partition) = column_partitions.get(idx) {
                let mut regions = partition.text_regions.clone();
                regions.sort_by(|a, b| a.bounding_box.top.cmp(&b.bounding_box.top));
                sorted_text_regions.extend(regions);
            }
        }

        // Fallback to simple sort if column detection produced nothing
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

        // Set reading order based on column count
        if column_partitions.len() > 1 {
            result.reading_order = ReadingOrder::MultiColumn;
        } else {
            result.reading_order = ReadingOrder::TopToBottom;
        }

        // Convert regions to blocks with classification info
        for (i, region) in result.text_regions.iter_mut().enumerate() {
            let is_handwritten = TextRegionDetector::is_handwritten(img, region);
            if is_handwritten {
                region.properties.font_family = Some("handwritten".to_string());
            }

            let region_type = if i < classifications.len() {
                classifications[i].region_type
            } else {
                crate::layout::classifier::RegionType::Body
            };

            let mut block = Block::new(
                format!("block_{}", region.id),
                BlockType::Text,
                region.bounding_box,
            );

            if is_handwritten {
                block.properties.background_color =
                    Some(crate::core::layout::Color::rgb(255, 255, 200));
            }

            // Store classification in block metadata
            block.properties.priority = match region_type {
                crate::layout::classifier::RegionType::Heading => 1,
                crate::layout::classifier::RegionType::SubHeading => 2,
                crate::layout::classifier::RegionType::Body => 10,
                crate::layout::classifier::RegionType::ListItem => 8,
                crate::layout::classifier::RegionType::Caption => 20,
                crate::layout::classifier::RegionType::Footer => 90,
                crate::layout::classifier::RegionType::PageNumber => 91,
                crate::layout::classifier::RegionType::Header => 5,
                crate::layout::classifier::RegionType::Unknown => 50,
            };

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
            result.blocks.push(block);
        }

        // Sort blocks by priority then position
        result.blocks.sort_by(|a, b| {
            a.properties
                .priority
                .cmp(&b.properties.priority)
                .then_with(|| a.bounding_box.top.cmp(&b.bounding_box.top))
                .then_with(|| a.bounding_box.left.cmp(&b.bounding_box.left))
        });

        Ok(result)
    }
}
