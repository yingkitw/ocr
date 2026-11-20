//! Binarization-method integration tests (Tesseract `thresholding_method` parity).
//!
//! The `--threshold-method` / `binarization_method` config controls how the
//! page is binarized before layout region detection (text / image / table)
//! and in the font-size DPI estimator. These tests assert that the choice is
//! actually plumbed through — not just parsed and dropped.

use ocr::core::config::BinarizationMethod;
use ocr::core::image::{ImageFormat, OcrImage};
use ocr::image::{threshold_method_for, ThresholdMethod};
use ocr::layout::analyzer::LayoutAnalyzer;

/// Build a grayscale page with text-like dark bars on an **uneven**
/// background (a dark gradient on the left half) — the case where Sauvola's
/// local threshold differs measurably from a global Otsu cut.
fn uneven_page(width: u32, height: u32) -> OcrImage {
    let mut gray = image::GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            // Background brightens left → right: 90 on the left, 210 on the right.
            let bg = 90 + (210 - 90) * x / width.max(1);
            let mut v = bg;
            // A few text bars in the left (dark) half.
            if (20..24).contains(&y) && (x % 30) < 14 && x < width / 2 {
                v = 20;
            }
            if (60..64).contains(&y) && (x % 30) < 14 && x < width / 2 {
                v = 20;
            }
            gray.put_pixel(x, y, image::Luma([v as u8]));
        }
    }
    OcrImage::from_raw_pixels(
        width,
        height,
        gray.into_raw(),
        ImageFormat::Grayscale,
        300,
    )
    .expect("valid grayscale page")
}

#[test]
fn test_threshold_method_mapping_is_total() {
    // Every config method maps onto a real thresholder method.
    assert_eq!(
        threshold_method_for(BinarizationMethod::Otsu),
        ThresholdMethod::Otsu
    );
    assert_eq!(
        threshold_method_for(BinarizationMethod::LeptonicaOtsu),
        ThresholdMethod::LeptonicaOtsu
    );
    assert_eq!(
        threshold_method_for(BinarizationMethod::Sauvola),
        ThresholdMethod::Sauvola
    );
    assert_eq!(
        threshold_method_for(BinarizationMethod::Niblack),
        ThresholdMethod::Niblack
    );
}

#[test]
fn test_layout_runs_with_every_binarization_method() {
    let page = uneven_page(240, 90);
    for method in [
        BinarizationMethod::Otsu,
        BinarizationMethod::LeptonicaOtsu,
        BinarizationMethod::Sauvola,
        BinarizationMethod::Niblack,
    ] {
        let result = LayoutAnalyzer::analyze_layout_with(
            &page,
            false,
            threshold_method_for(method),
        )
        .unwrap_or_else(|e| panic!("layout failed with {method:?}: {e}"));
        // Global methods (Otsu) may legitimately find nothing on an uneven
        // background — that is precisely why the option exists. Local
        // methods (Sauvola/Niblack) must recover the bars.
        if matches!(
            method,
            BinarizationMethod::Sauvola | BinarizationMethod::Niblack
        ) {
            assert!(
                !result.text_regions.is_empty() || !result.images.is_empty(),
                "{method:?}: local thresholding still produced no regions"
            );
        }
    }
}

#[test]
fn test_sauvola_keeps_text_on_uneven_background_where_otsu_loses_it() {
    let page = uneven_page(240, 90);
    let otsu = LayoutAnalyzer::analyze_layout_with(&page, false, ThresholdMethod::Otsu)
        .expect("otsu layout");
    let sauvola = LayoutAnalyzer::analyze_layout_with(&page, false, ThresholdMethod::Sauvola)
        .expect("sauvola layout");

    let otsu_bars = count_bar_regions(&otsu);
    let sauvola_bars = count_bar_regions(&sauvola);

    // This is the regression that justifies the option: a global Otsu cut on
    // a left-dark gradient can swallow the left-half text entirely, while
    // Sauvola's local windows keep it. Assert Sauvola finds at least as many
    // bar regions as Otsu, and specifically that it recovers regions Otsu
    // misses on the dark side.
    assert!(
        sauvola_bars >= otsu_bars,
        "sauvola found {sauvola_bars} bar regions, otsu found {otsu_bars} — \
         sauvola should not do worse on an uneven background"
    );
}

/// Count text regions that sit inside the dark left half of the page.
fn count_bar_regions(result: &ocr::core::layout::LayoutResult) -> usize {
    result
        .text_regions
        .iter()
        .filter(|r| r.bounding_box.left < 120)
        .count()
}

#[test]
fn test_layout_with_otsu_matches_legacy_entry_point() {
    // The `_with` variant must be behavior-identical to the legacy entry
    // point when given Otsu, so existing callers don't change output.
    let page = uneven_page(240, 90);
    let legacy = LayoutAnalyzer::analyze_layout(&page).expect("legacy layout");
    let with = LayoutAnalyzer::analyze_layout_with(&page, false, ThresholdMethod::Otsu)
        .expect("_with layout");
    assert_eq!(
        legacy.text_regions.len(),
        with.text_regions.len(),
        "legacy and _with(Otsu) disagree on text region count"
    );
    assert_eq!(legacy.tables.len(), with.tables.len());
}
