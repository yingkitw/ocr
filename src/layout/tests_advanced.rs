#[cfg(test)]
mod tests {
    use crate::core::layout::{Table, TextRegion};
    use crate::core::text::BoundingBox;

    #[test]
    fn test_conflict_resolution_logic() {
        // This test simulates the logic inside LayoutAnalyzer::analyze

        let table_bbox = BoundingBox::new(100, 100, 500, 500);
        let table = Table::new("1".to_string(), table_bbox);

        let text_inside = TextRegion::new(
            "1".to_string(),
            BoundingBox::new(150, 150, 200, 200),
            String::new(),
        );

        let text_outside = TextRegion::new(
            "2".to_string(),
            BoundingBox::new(600, 600, 700, 700),
            String::new(),
        );

        let tables = vec![table];
        let mut text_regions = vec![text_inside.clone(), text_outside.clone()];

        // The logic from analyzer.rs:
        text_regions.retain(|region| {
            let center = region.bounding_box.center();
            let mut inside_table = false;
            for table in &tables {
                if table
                    .bounding_box
                    .contains(center.x as u32, center.y as u32)
                {
                    inside_table = true;
                    break;
                }
            }
            !inside_table
        });

        assert_eq!(text_regions.len(), 1);
        assert_eq!(text_regions[0].id, "2");
    }
}
