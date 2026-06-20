//! Region classifier for document layout analysis
//!
//! Classifies text regions into semantic types:
//! heading, body, caption, footer, page_number, etc.

use crate::core::layout::{Block, BlockType, TextRegion};
use crate::core::text::BoundingBox;
use crate::utils::Result;

/// Semantic type of a text region
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionType {
    Heading,
    SubHeading,
    Body,
    ListItem,
    Caption,
    Footer,
    PageNumber,
    Header,
    Unknown,
}

/// Classification result with confidence
#[derive(Debug, Clone)]
pub struct RegionClassification {
    pub region_type: RegionType,
    pub confidence: f32,
}

/// Classifies text regions based on position, size, and font properties
pub struct RegionClassifier {
    page_width: u32,
    page_height: u32,
    avg_font_size: f32,
}

impl RegionClassifier {
    pub fn new(page_width: u32, page_height: u32) -> Self {
        Self {
            page_width,
            page_height,
            avg_font_size: 12.0,
        }
    }

    /// Classify all text regions in a page
    pub fn classify(&mut self, regions: &[TextRegion]) -> Vec<RegionClassification> {
        if regions.is_empty() {
            return Vec::new();
        }

        self.avg_font_size = self.estimate_avg_font_size(regions);

        regions.iter().map(|r| self.classify_region(r)).collect()
    }

    fn estimate_avg_font_size(&self, regions: &[TextRegion]) -> f32 {
        let heights: Vec<f32> = regions
            .iter()
            .map(|r| (r.bounding_box.bottom - r.bounding_box.top) as f32)
            .collect();

        if heights.is_empty() {
            return 12.0;
        }

        let sum: f32 = heights.iter().sum();
        sum / heights.len() as f32
    }

    fn classify_region(&self, region: &TextRegion) -> RegionClassification {
        let bbox = &region.bounding_box;
        let height = (bbox.bottom - bbox.top) as f32;
        let width = (bbox.right - bbox.left) as f32;
        let top_rel = bbox.top as f32 / self.page_height as f32;
        let bottom_rel = bbox.bottom as f32 / self.page_height as f32;
        let width_rel = width / self.page_width as f32;

        // Page number: small or normal, at top or bottom, narrow
        if height <= self.avg_font_size * 1.05
            && (top_rel < 0.08 || bottom_rel > 0.92)
            && width_rel < 0.2
        {
            return RegionClassification {
                region_type: RegionType::PageNumber,
                confidence: 0.85,
            };
        }

        // Footer: at very bottom of page, small text
        if bottom_rel > 0.93 && height < self.avg_font_size * 1.1 {
            return RegionClassification {
                region_type: RegionType::Footer,
                confidence: 0.8,
            };
        }

        // Header: at very top of page
        if top_rel < 0.07 && height < self.avg_font_size * 1.2 {
            return RegionClassification {
                region_type: RegionType::Header,
                confidence: 0.75,
            };
        }

        // Heading: larger than average
        if height > self.avg_font_size * 1.3 {
            return RegionClassification {
                region_type: RegionType::Heading,
                confidence: 0.85,
            };
        }

        // Sub-heading: slightly larger than average
        if height > self.avg_font_size * 1.1 {
            return RegionClassification {
                region_type: RegionType::SubHeading,
                confidence: 0.7,
            };
        }

        // Caption: small text that's relatively narrow
        if height < self.avg_font_size * 0.85 && width_rel < 0.6 {
            return RegionClassification {
                region_type: RegionType::Caption,
                confidence: 0.6,
            };
        }

        // List item: text starts with bullet or number marker
        let text = region.text.trim();
        if Self::is_list_item(text) {
            return RegionClassification {
                region_type: RegionType::ListItem,
                confidence: 0.8,
            };
        }

        // Default: body text
        RegionClassification {
            region_type: RegionType::Body,
            confidence: 0.9,
        }
    }

    fn is_list_item(text: &str) -> bool {
        let trimmed = text.trim_start();
        // Bullet markers
        if trimmed.starts_with("•") || trimmed.starts_with("-") || trimmed.starts_with("*") {
            return true;
        }
        // Numbered list: "1.", "1)", "(1)", "a.", "A.", "i.", "I."
        Self::is_numbered_list_item(trimmed)
    }

    fn is_numbered_list_item(text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        // Pattern: digits followed by . or )
        let mut chars = text.chars();
        let first = chars.next().unwrap();
        if first.is_numeric() {
            // digits followed by . or )
            let mut saw_digit = true;
            for ch in chars {
                if ch.is_numeric() {
                    saw_digit = true;
                } else if saw_digit && (ch == '.' || ch == ')') {
                    return true;
                } else {
                    break;
                }
            }
        }
        // Single letter followed by .
        if first.is_ascii_alphabetic() {
            if let Some(second) = text.chars().nth(1) {
                if second == '.' {
                    return true;
                }
            }
        }
        // Roman numerals i. I. v. V. x. X. followed by .
        if "iIvVxX".contains(first) {
            if let Some(second) = text.chars().nth(1) {
                if second == '.' {
                    return true;
                }
            }
        }
        // Parenthesized number: (1)
        if first == '(' {
            let mut found_digit = false;
            for ch in text.chars().skip(1) {
                if ch.is_numeric() {
                    found_digit = true;
                } else if found_digit && ch == ')' {
                    return true;
                } else {
                    break;
                }
            }
        }
        false
    }
}

/// Block classifier that uses region classification results
pub struct BlockClassifier;

impl BlockClassifier {
    /// Classify blocks based on their type and position
    pub fn classify_blocks(blocks: &[Block]) -> Result<Vec<Block>> {
        let mut classified = blocks.to_vec();
        for block in &mut classified {
            match block.block_type {
                BlockType::Text => {
                    // Text blocks keep their type
                }
                BlockType::Image => {
                    block.properties.priority = 100; // Images after text
                }
                BlockType::Table => {
                    block.properties.priority = 50; // Tables after text, before images
                }
                _ => {}
            }
        }
        Ok(classified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(top: u32, bottom: u32, left: u32, right: u32) -> TextRegion {
        TextRegion {
            id: 0.to_string(),
            bounding_box: BoundingBox::new(left, top, right, bottom),
            text: String::new(),
            confidence: 1.0,
            properties: Default::default(),
        }
    }

    #[test]
    fn test_classify_heading() {
        let mut classifier = RegionClassifier::new(600, 800);
        let regions = vec![
            make_region(50, 80, 100, 500),
            make_region(100, 115, 100, 500),
        ];
        let classes = classifier.classify(&regions);
        assert_eq!(classes[0].region_type, RegionType::Heading);
        assert_eq!(classes[1].region_type, RegionType::Body);
    }

    #[test]
    fn test_classify_page_number() {
        let mut classifier = RegionClassifier::new(600, 800);
        let regions = vec![make_region(770, 785, 270, 330)];
        let classes = classifier.classify(&regions);
        assert_eq!(classes[0].region_type, RegionType::PageNumber);
    }

    #[test]
    fn test_classify_footer() {
        let mut classifier = RegionClassifier::new(600, 800);
        let regions = vec![make_region(760, 775, 100, 500)];
        let classes = classifier.classify(&regions);
        assert_eq!(classes[0].region_type, RegionType::Footer);
    }

    #[test]
    fn test_classify_header() {
        let mut classifier = RegionClassifier::new(600, 800);
        let regions = vec![make_region(10, 25, 100, 500)];
        let classes = classifier.classify(&regions);
        assert_eq!(classes[0].region_type, RegionType::Header);
    }

    #[test]
    fn test_classify_list_item() {
        let mut classifier = RegionClassifier::new(600, 800);
        let mut region = make_region(100, 115, 100, 500);
        region.text = "• First item".to_string();
        let classes = classifier.classify(&[region]);
        assert_eq!(classes[0].region_type, RegionType::ListItem);
    }

    #[test]
    fn test_is_numbered_list_item() {
        assert!(RegionClassifier::is_numbered_list_item("1. Hello"));
        assert!(RegionClassifier::is_numbered_list_item("12) Hello"));
        assert!(RegionClassifier::is_numbered_list_item("(3) Hello"));
        assert!(RegionClassifier::is_numbered_list_item("a. Hello"));
        assert!(RegionClassifier::is_numbered_list_item("i. Hello"));
        assert!(!RegionClassifier::is_numbered_list_item("Hello"));
    }
}
