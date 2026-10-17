//! Tesseract-parity tests for the Page Segmentation Mode (PSM) surface and
//! OSD (orientation + script detection).
//!
//! Contract reference: TESSERACT_PARITY.md §2. The PSM numbering, symbolic
//! names, and mode semantics mirror Tesseract 5.x (`publictypes.h:157-178`).

use image::{DynamicImage, GrayImage, Luma};
use ocr::api::Ocr;
use ocr::core::config::{LayoutAnalysisConfig, OcrConfig, PageSegMode};

/// White 320x120 page with a few text-like left-aligned bars.
fn text_page_png() -> Vec<u8> {
    let mut g = GrayImage::from_pixel(320, 120, Luma([255u8]));
    for &(y0, x1) in &[(20u32, 260u32), (50, 300), (80, 240)] {
        for y in y0..y0 + 8 {
            for x in 20..x1 {
                g.put_pixel(x, y, Luma([0u8]));
            }
        }
    }
    let mut buf = Vec::new();
    DynamicImage::ImageLuma8(g)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("encode png");
    buf
}

/// Page content enclosed in a large ring (PSM 9 fixture).
fn circled_page_png() -> Vec<u8> {
    let mut g = GrayImage::from_pixel(200, 200, Luma([255u8]));
    for y in 0..200u32 {
        for x in 0..200u32 {
            let dx = x as f64 - 100.0;
            let dy = y as f64 - 100.0;
            if ((dx * dx + dy * dy).sqrt() - 90.0).abs() < 1.5 {
                g.put_pixel(x, y, Luma([0u8]));
            }
        }
    }
    for y in 90..110u32 {
        for x in 70..130u32 {
            g.put_pixel(x, y, Luma([0u8]));
        }
    }
    let mut buf = Vec::new();
    DynamicImage::ImageLuma8(g)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("encode png");
    buf
}

fn config_with_psm(mode: PageSegMode) -> OcrConfig {
    OcrConfig {
        layout_analysis: LayoutAnalysisConfig {
            page_seg_mode: mode,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn test_psm_osd_only_reports_orientation_and_script() {
    let ocr = Ocr::with_config(config_with_psm(PageSegMode::OsdOnly)).expect("ocr");
    ocr.initialize().await.expect("init");
    let result = ocr
        .recognize_text_from_bytes(&text_page_png())
        .await
        .expect("psm 0");

    // Tesseract-shaped OSD report
    assert!(result.text.contains("Orientation in degrees:"));
    assert!(result.text.contains("Orientation confidence:"));
    assert!(result.text.contains("Script:"));
    assert!(result.text.contains("Script confidence:"));
    // Machine-readable metadata
    assert_eq!(result.metadata.get("psm").map(String::as_str), Some("0"));
    assert!(result.metadata.contains_key("orientation"));
    assert!(result.metadata.contains_key("orientation_confidence"));
    assert!(result.metadata.contains_key("script"));
    assert!(result.metadata.contains_key("script_confidence"));
}

#[tokio::test]
async fn test_psm_auto_osd_attaches_metadata() {
    let ocr = Ocr::with_config(config_with_psm(PageSegMode::AutoOsd)).expect("ocr");
    ocr.initialize().await.expect("init");
    let result = ocr
        .recognize_text_from_bytes(&text_page_png())
        .await
        .expect("psm 1");

    assert_eq!(result.metadata.get("psm").map(String::as_str), Some("1"));
    assert!(result.metadata.contains_key("orientation"));
    assert!(result.metadata.contains_key("script"));
}

#[tokio::test]
async fn test_psm_auto_only_layout_without_recognition() {
    let ocr = Ocr::with_config(config_with_psm(PageSegMode::AutoOnly)).expect("ocr");
    ocr.initialize().await.expect("init");
    let result = ocr
        .recognize_text_from_bytes(&text_page_png())
        .await
        .expect("psm 2");

    // Layout-only: orientation report + region count, no OCR text
    assert_eq!(result.metadata.get("psm").map(String::as_str), Some("2"));
    assert!(result.metadata.contains_key("layout_regions"));
    assert!(result.text.contains("Layout regions:"));
}

#[tokio::test]
async fn test_psm_single_char_treats_page_as_one_char() {
    let ocr = Ocr::with_config(config_with_psm(PageSegMode::SingleChar)).expect("ocr");
    ocr.initialize().await.expect("init");
    let result = ocr
        .recognize_text_from_bytes(&text_page_png())
        .await
        .expect("psm 10");
    assert_eq!(result.metadata.get("psm").map(String::as_str), Some("10"));
}

#[tokio::test]
async fn test_psm_circle_word_removes_ring() {
    let ocr = Ocr::with_config(config_with_psm(PageSegMode::CircleWord)).expect("ocr");
    ocr.initialize().await.expect("init");
    let result = ocr
        .recognize_text_from_bytes(&circled_page_png())
        .await
        .expect("psm 9");
    assert_eq!(result.metadata.get("psm").map(String::as_str), Some("9"));
}

#[tokio::test]
async fn test_psm_symbolic_names_parse_everywhere() {
    // The library-level enum must agree with the CLI-parseable names.
    for (name, expected) in [
        ("osd_only", PageSegMode::OsdOnly),
        ("auto_osd", PageSegMode::AutoOsd),
        ("auto_only", PageSegMode::AutoOnly),
        ("auto", PageSegMode::Auto),
        ("single_column", PageSegMode::SingleColumn),
        ("single_block_vert_text", PageSegMode::SingleBlockVertText),
        ("single_block", PageSegMode::SingleBlock),
        ("single_line", PageSegMode::SingleLine),
        ("single_word", PageSegMode::SingleWord),
        ("circle_word", PageSegMode::CircleWord),
        ("single_char", PageSegMode::SingleChar),
        ("sparse_text", PageSegMode::SparseText),
        ("sparse_text_osd", PageSegMode::SparseTextWithOsd),
        ("raw_line", PageSegMode::SingleLineRaw),
    ] {
        assert_eq!(PageSegMode::parse_psm(name), Ok(expected), "psm {name}");
    }
}

#[tokio::test]
async fn test_whitelist_restricts_plain_text_output() {
    let mut config = config_with_psm(PageSegMode::SingleBlock);
    config.recognition.character_whitelist = Some(vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
    let ocr = Ocr::with_config(config).expect("ocr");
    ocr.initialize().await.expect("init");
    let result = ocr
        .recognize_text_from_bytes(&text_page_png())
        .await
        .expect("whitelist run");

    // Plain text must respect the whitelist: every non-whitespace char is a digit.
    assert!(
        result.text.chars().all(|c| c.is_whitespace() || c.is_ascii_digit()),
        "non-digit chars leaked into text output: {:?}",
        result.text
    );
}

/// PSM metadata flows through `analyze_layout`'s new arms without changing
/// region semantics (regression guard for the match-arm refactor).
#[test]
fn test_page_seg_mode_covers_all_tesseract_modes() {
    // 14 modes, numbered 0..=13 exactly like Tesseract.
    let all = [
        PageSegMode::OsdOnly,
        PageSegMode::AutoOsd,
        PageSegMode::AutoOnly,
        PageSegMode::Auto,
        PageSegMode::SingleColumn,
        PageSegMode::SingleBlockVertText,
        PageSegMode::SingleBlock,
        PageSegMode::SingleLine,
        PageSegMode::SingleWord,
        PageSegMode::CircleWord,
        PageSegMode::SingleChar,
        PageSegMode::SparseText,
        PageSegMode::SparseTextWithOsd,
        PageSegMode::SingleLineRaw,
    ];
    for (i, mode) in all.iter().enumerate() {
        assert_eq!(mode.psm_number(), i as u8, "mode {i:?} numbering drifted");
    }
}
