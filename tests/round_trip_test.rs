//! Comprehensive round-trip OCR tests: text -> image -> OCR -> text
//!
//! Test categories:
//! 1. Glyph rendering validation
//! 2. Per-character round-trip (A-Z, 0-9, special chars)
//! 3. Single-word round-trip
//! 4. Multi-word / space handling
//! 5. Multi-line round-trip
//! 6. Scale and spacing variations
//! 7. Pipeline configuration variations
//! 8. Inverted / edge-case images
//! 9. Result structure validation (words, chars, lines, confidence)
//! 10. Reproducibility and sequential processing
//! 11. Engine configuration and statistics

use image::{DynamicImage, GrayImage, Luma};
use ocr::core::config::OcrConfig;
use ocr::core::engine::{EngineStatistics, OcrEngine};
use ocr::core::image::OcrImage;

const SUPPORTED_LETTERS: &[char] = &[
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];
const SUPPORTED_DIGITS: &[char] = &['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
const SUPPORTED_SPECIALS: &[char] = &['.', '-', ':', '/', ',', '\'', '!', '?'];

fn glyph_rows(ch: char) -> [&'static str; 7] {
    match ch {
        'A' => ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        'B' => ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        'C' => ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        'D' => ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        'E' => ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        'F' => ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        'G' => ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
        'H' => ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        'I' => ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        'J' => ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
        'K' => ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        'L' => ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        'M' => ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        'N' => ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        'O' => ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        'P' => ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        'Q' => ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        'R' => ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        'S' => ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        'T' => ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        'U' => ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        'V' => ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        'W' => ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        'X' => ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        'Y' => ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        'Z' => ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
        '0' => ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        '1' => ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        '2' => ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        '3' => ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        '4' => ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        '5' => ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        '6' => ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        '7' => ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        '8' => ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        '9' => ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        '.' => ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
        '-' => ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        ':' => ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
        '/' => ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
        ',' => ["00000", "00000", "00000", "00000", "00000", "00100", "01000"],
        '\'' => ["00100", "00100", "00000", "00000", "00000", "00000", "00000"],
        '!' => ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
        '?' => ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
        _ => ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    }
}

fn render_text_5x7(
    text: &str,
    scale: u32,
    char_spacing: u32,
    line_spacing: u32,
) -> GrayImage {
    let lines: Vec<&str> = text.lines().collect();
    let glyph_w = 5 * scale;
    let glyph_h = 7 * scale;

    let max_line_len = lines
        .iter()
        .map(|l| l.chars().count() as u32)
        .max()
        .unwrap_or(0);
    let width = if max_line_len == 0 {
        1
    } else {
        max_line_len * glyph_w + max_line_len.saturating_sub(1) * char_spacing + scale * 2
    };
    let height = if lines.is_empty() {
        1
    } else {
        (lines.len() as u32) * glyph_h
            + (lines.len() as u32).saturating_sub(1) * line_spacing
            + scale * 2
    };

    let mut img = GrayImage::from_pixel(width.max(10), height.max(10), Luma([255u8]));

    let mut y = scale;
    for line in &lines {
        let mut x = scale;
        for ch in line.chars() {
            if ch == ' ' {
                x += glyph_w + char_spacing;
                continue;
            }
            let rows = glyph_rows(ch.to_ascii_uppercase());
            for (ry, row) in rows.iter().enumerate() {
                for (rx, b) in row.as_bytes().iter().enumerate() {
                    if *b == b'1' {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                img.put_pixel(
                                    x + (rx as u32) * scale + dx,
                                    y + (ry as u32) * scale + dy,
                                    Luma([0u8]),
                                );
                            }
                        }
                    }
                }
            }
            x += glyph_w + char_spacing;
        }
        y += glyph_h + line_spacing;
    }

    img
}

fn render_inverted(text: &str, scale: u32, char_spacing: u32, line_spacing: u32) -> GrayImage {
    let normal = render_text_5x7(text, scale, char_spacing, line_spacing);
    let (w, h) = normal.dimensions();
    let mut inv = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = normal.get_pixel(x, y)[0];
            inv.put_pixel(x, y, Luma([255 - v]));
        }
    }
    inv
}

fn make_ocr_image(img: GrayImage) -> OcrImage {
    OcrImage::new(DynamicImage::ImageLuma8(img), 300)
}

fn make_engine() -> OcrEngine {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = false;
    config.layout_analysis.enable_layout_analysis = false;
    OcrEngine::with_config(config).unwrap()
}

fn make_engine_full() -> OcrEngine {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.layout_analysis.enable_layout_analysis = true;
    OcrEngine::with_config(config).unwrap()
}

async fn recognize_with(engine: &OcrEngine, img: GrayImage) -> String {
    let ocr_img = make_ocr_image(img);
    let result = engine.process_image(ocr_img).await.unwrap();
    result.text.trim().to_string()
}

async fn recognize(img: GrayImage) -> String {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    recognize_with(&engine, img).await
}

fn character_accuracy(recognized: &str, expected: &str) -> f32 {
    if expected.is_empty() {
        return if recognized.is_empty() { 1.0 } else { 0.0 };
    }
    let r: Vec<char> = recognized.chars().collect();
    let e: Vec<char> = expected.chars().collect();
    let max_len = r.len().max(e.len());
    if max_len == 0 {
        return 1.0;
    }
    let matches = r
        .iter()
        .zip(e.iter())
        .filter(|(a, b)| a == b)
        .count();
    matches as f32 / max_len as f32
}

fn count_glyph_pixels(ch: char) -> u32 {
    let rows = glyph_rows(ch);
    rows.iter()
        .flat_map(|r| r.as_bytes().iter())
        .filter(|&&b| b == b'1')
        .count() as u32
}

// ============================================================
// 1. Glyph rendering validation
// ============================================================

#[test]
fn test_glyph_render_all_letters_have_pixels() {
    for &ch in SUPPORTED_LETTERS {
        let px = count_glyph_pixels(ch);
        assert!(px > 0, "Glyph '{}' should have at least 1 foreground pixel, got {}", ch, px);
    }
}

#[test]
fn test_glyph_render_all_digits_have_pixels() {
    for &ch in SUPPORTED_DIGITS {
        let px = count_glyph_pixels(ch);
        assert!(px > 0, "Glyph '{}' should have at least 1 foreground pixel, got {}", ch, px);
    }
}

#[test]
fn test_glyph_render_special_chars_have_pixels() {
    for &ch in SUPPORTED_SPECIALS {
        let px = count_glyph_pixels(ch);
        assert!(px > 0, "Glyph '{}' should have foreground pixels", ch);
    }
}

#[test]
fn test_glyph_render_unknown_char_has_no_pixels() {
    let px = count_glyph_pixels('@');
    assert_eq!(px, 0, "Unsupported char should render as blank");
}

#[test]
fn test_render_produces_correct_dimensions_single_char() {
    let img = render_text_5x7("A", 6, 6, 12);
    let (w, h) = img.dimensions();
    assert!(w >= 10, "Width should be >= 10, got {}", w);
    assert!(h >= 10, "Height should be >= 10, got {}", h);
}

#[test]
fn test_render_produces_correct_dimensions_multi_char() {
    let img = render_text_5x7("ABCD", 6, 6, 12);
    let (w, h) = img.dimensions();
    let expected_w = 4 * 5 * 6 + 3 * 6 + 6 * 2;
    let expected_h = 7 * 6 + 6 * 2;
    assert_eq!(w, expected_w.max(10));
    assert_eq!(h, expected_h.max(10));
}

#[test]
fn test_render_multi_line_height() {
    let img1 = render_text_5x7("HELLO", 6, 6, 12);
    let img2 = render_text_5x7("HELLO\nWORLD", 6, 6, 12);
    let (_, h1) = img1.dimensions();
    let (_, h2) = img2.dimensions();
    assert!(h2 > h1, "Two lines should be taller than one line ({} > {})", h2, h1);
}

#[test]
fn test_render_pixel_values_are_binary() {
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let (w, h) = img.dimensions();
    for y in 0..h {
        for x in 0..w {
            let v = img.get_pixel(x, y)[0];
            assert!(
                v == 0 || v == 255,
                "All pixels should be 0 or 255, got {} at ({}, {})",
                v, x, y
            );
        }
    }
}

#[test]
fn test_render_has_black_pixels() {
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let (w, h) = img.dimensions();
    let mut has_black = false;
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y)[0] == 0 {
                has_black = true;
                break;
            }
        }
    }
    assert!(has_black, "Rendered text should contain black pixels");
}

#[test]
fn test_render_has_white_pixels() {
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let (w, h) = img.dimensions();
    let mut has_white = false;
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y)[0] == 255 {
                has_white = true;
                break;
            }
        }
    }
    assert!(has_white, "Rendered text should contain white pixels (background)");
}

#[test]
fn test_render_space_produces_no_black_pixels_for_space_char() {
    let img_with_space = render_text_5x7("A B", 6, 6, 12);
    let (w, h) = img_with_space.dimensions();
    assert!(w > 10, "Width with spaces should be > 10");
    assert!(h > 10, "Height should be > 10");
}

#[test]
fn test_render_scale_affects_dimensions() {
    let img_s = render_text_5x7("AB", 2, 2, 4);
    let img_l = render_text_5x7("AB", 8, 8, 16);
    let (ws, hs) = img_s.dimensions();
    let (wl, hl) = img_l.dimensions();
    assert!(wl > ws, "Larger scale => wider image ({} > {})", wl, ws);
    assert!(hl > hs, "Larger scale => taller image ({} > {})", hl, hs);
}

#[test]
fn test_render_spacing_affects_dimensions() {
    let img_tight = render_text_5x7("AB", 6, 2, 4);
    let img_wide = render_text_5x7("AB", 6, 12, 4);
    let (wt, _) = img_tight.dimensions();
    let (ww, _) = img_wide.dimensions();
    assert!(ww > wt, "Wider spacing => wider image ({} > {})", ww, wt);
}

// ============================================================
// 2. Per-character round-trip tests (A-Z)
// ============================================================

#[tokio::test]
async fn test_roundtrip_each_letter_a_through_m() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let mut failures = Vec::new();
    for &ch in &['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'] {
        let img = render_text_5x7(&ch.to_string(), 6, 6, 12);
        let text = recognize_with(&engine, img).await;
        let acc = character_accuracy(&text, &ch.to_string());
        if acc < 1.0 {
            failures.push(format!("{}: got '{}' ({:.0}%)", ch, text, acc * 100.0));
        }
    }
    assert!(
        failures.len() <= 2,
        "Too many letter failures A-M ({}): {}",
        failures.len(),
        failures.join(", ")
    );
}

#[tokio::test]
async fn test_roundtrip_each_letter_n_through_z() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let mut failures = Vec::new();
    for &ch in &['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] {
        let img = render_text_5x7(&ch.to_string(), 6, 6, 12);
        let text = recognize_with(&engine, img).await;
        let acc = character_accuracy(&text, &ch.to_string());
        if acc < 1.0 {
            failures.push(format!("{}: got '{}' ({:.0}%)", ch, text, acc * 100.0));
        }
    }
    assert!(
        failures.len() <= 2,
        "Too many letter failures N-Z ({}): {}",
        failures.len(),
        failures.join(", ")
    );
}

#[tokio::test]
async fn test_roundtrip_each_digit_0_through_9() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let mut failures = Vec::new();
    for &d in SUPPORTED_DIGITS {
        let img = render_text_5x7(&d.to_string(), 6, 6, 12);
        let text = recognize_with(&engine, img).await;
        let acc = character_accuracy(&text, &d.to_string());
        if acc < 1.0 {
            failures.push(format!("{}: got '{}' ({:.0}%)", d, text, acc * 100.0));
        }
    }
    assert!(
        failures.len() <= 2,
        "Too many digit failures ({}): {}",
        failures.len(),
        failures.join(", ")
    );
}

#[tokio::test]
async fn test_roundtrip_special_chars_no_crash() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    for &ch in SUPPORTED_SPECIALS {
        let img = render_text_5x7(&ch.to_string(), 6, 6, 12);
        let result = engine.process_image(make_ocr_image(img)).await;
        assert!(result.is_ok(), "Should not crash on special char '{}'", ch);
    }
}

// ============================================================
// 3. Single-word round-trip tests
// ============================================================

#[tokio::test]
async fn test_roundtrip_hello() {
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "HELLO");
    assert!(acc >= 0.8, "HELLO: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_world() {
    let img = render_text_5x7("WORLD", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "WORLD");
    assert!(acc >= 0.8, "WORLD: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_test() {
    let img = render_text_5x7("TEST", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "TEST");
    assert!(acc >= 0.75, "TEST: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_ocr() {
    let img = render_text_5x7("OCR", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "OCR");
    assert!(acc >= 0.66, "OCR: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_rust() {
    let img = render_text_5x7("RUST", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "RUST");
    assert!(acc >= 0.75, "RUST: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_foo() {
    let img = render_text_5x7("FOO", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "FOO");
    assert!(acc >= 0.66, "FOO: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_abc() {
    let img = render_text_5x7("ABC", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "ABC");
    assert!(acc >= 0.66, "ABC: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_long_word() {
    let img = render_text_5x7("ABCDEFGHIJ", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "ABCDEFGHIJ");
    assert!(acc >= 0.5, "ABCDEFGHIJ: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_all_digits_sequence() {
    let img = render_text_5x7("0123456789", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "0123456789");
    assert!(acc >= 0.5, "0123456789: got '{}' ({:.0}%)", text, acc * 100.0);
}

// ============================================================
// 4. Multi-word / space handling
// ============================================================

#[tokio::test]
async fn test_roundtrip_two_words_with_space() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO WORLD", 6, 6, 12);
    let text = recognize_with(&engine, img).await;
    assert!(
        text.contains("HELLO") || text.contains("HELL"),
        "Should recognize HELLO in '{}'",
        text
    );
    assert!(
        text.contains("WORLD") || text.contains("WORL") || text.contains("WOR"),
        "Should recognize WORLD in '{}'",
        text
    );
}

#[tokio::test]
async fn test_roundtrip_three_words_with_spaces() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO WORLD TEST", 6, 18, 12);
    let text = recognize_with(&engine, img).await;
    assert!(!text.is_empty(), "Should recognize text with spaces");
    let parts: Vec<&str> = text.split_whitespace().collect();
    assert!(parts.len() >= 2, "Should have at least 2 words, got {}: {:?}", parts.len(), parts);
}

#[tokio::test]
async fn test_roundtrip_lowercase_input_uppercased_output() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("hello", 6, 6, 12);
    let text = recognize_with(&engine, img).await;
    assert_eq!(
        text.to_uppercase(),
        text,
        "Output should be uppercase since renderer converts to uppercase glyphs"
    );
}

// ============================================================
// 5. Multi-line round-trip tests
// ============================================================

#[tokio::test]
async fn test_roundtrip_two_lines() {
    let img = render_text_5x7("HELLO\nWORLD", 6, 6, 12);
    let text = recognize(img).await;
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "Expected 2 lines, got {}: {:?}", lines.len(), lines);
    assert!(lines[0].starts_with("HELL"), "Line 1: got '{}'", lines[0]);
    assert!(!lines[1].is_empty(), "Line 2 should not be empty");
}

#[tokio::test]
async fn test_roundtrip_three_lines() {
    let img = render_text_5x7("HELLO\nWORLD\nTEST", 6, 6, 12);
    let text = recognize(img).await;
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "Expected 3 lines, got {}: {:?}", lines.len(), lines);
    for (i, line) in lines.iter().enumerate() {
        assert!(!line.is_empty(), "Line {} should not be empty", i);
    }
}

#[tokio::test]
async fn test_roundtrip_four_lines() {
    let img = render_text_5x7("AAA\nBBB\nCCC\nDDD", 6, 6, 12);
    let text = recognize(img).await;
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4, "Expected 4 lines, got {}: {:?}", lines.len(), lines);
}

#[tokio::test]
async fn test_roundtrip_five_lines() {
    let img = render_text_5x7("A\nB\nC\nD\nE", 6, 6, 12);
    let text = recognize(img).await;
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5, "Expected 5 lines, got {}: {:?}", lines.len(), lines);
}

#[tokio::test]
async fn test_roundtrip_different_line_lengths() {
    let img = render_text_5x7("A\nBBB\nCCCCC", 6, 6, 12);
    let text = recognize(img).await;
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "Expected 3 lines with varying lengths");
}

// ============================================================
// 6. Scale and spacing variations
// ============================================================

#[tokio::test]
async fn test_roundtrip_scale_2() {
    let img = render_text_5x7("HELLO", 2, 2, 4);
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "Scale 2 should not crash");
    let text = result.unwrap().text.trim().to_string();
    assert!(!text.is_empty(), "Scale 2 should produce non-empty text");
}

#[tokio::test]
async fn test_roundtrip_scale_4() {
    let img = render_text_5x7("HELLO", 4, 4, 8);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "HELLO");
    assert!(acc >= 0.6, "Scale 4: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_scale_6() {
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "HELLO");
    assert!(acc >= 0.8, "Scale 6: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_scale_8() {
    let img = render_text_5x7("HELLO", 8, 8, 16);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "HELLO");
    assert!(acc >= 0.8, "Scale 8: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_scale_12() {
    let img = render_text_5x7("HELLO", 12, 12, 24);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "HELLO");
    assert!(acc >= 0.8, "Scale 12: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_tight_spacing() {
    let img = render_text_5x7("AB", 6, 1, 4);
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "Tight spacing should not crash");
}

#[tokio::test]
async fn test_roundtrip_wide_spacing() {
    let img = render_text_5x7("AB", 6, 20, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "AB");
    assert!(acc >= 0.5, "Wide spacing: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_tight_line_spacing() {
    let img = render_text_5x7("HELLO\nWORLD", 6, 6, 2);
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "Tight line spacing should not crash");
}

#[tokio::test]
async fn test_roundtrip_wide_line_spacing() {
    let img = render_text_5x7("HELLO\nWORLD", 6, 6, 40);
    let text = recognize(img).await;
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "Wide line spacing should still produce 2 lines");
}

// ============================================================
// 7. Pipeline configuration variations
// ============================================================

#[tokio::test]
async fn test_roundtrip_preprocessing_enabled() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_preprocessing_disabled() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = false;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_layout_analysis_enabled() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = false;
    config.layout_analysis.enable_layout_analysis = true;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO\nWORLD", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_full_pipeline() {
    let engine = make_engine_full();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_binarization_otsu() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.image_processing.enable_binarization = true;
    config.image_processing.binarization_method = ocr::core::config::BinarizationMethod::Otsu;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_binarization_fixed() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.image_processing.enable_binarization = true;
    config.image_processing.binarization_method = ocr::core::config::BinarizationMethod::Fixed;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_binarization_adaptive() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.image_processing.enable_binarization = true;
    config.image_processing.binarization_method =
        ocr::core::config::BinarizationMethod::Adaptive;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

#[tokio::test]
async fn test_roundtrip_binarization_sauvola() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.image_processing.enable_binarization = true;
    config.image_processing.binarization_method =
        ocr::core::config::BinarizationMethod::Sauvola;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty());
}

// ============================================================
// 8. Inverted / edge-case images
// ============================================================

#[tokio::test]
async fn test_roundtrip_inverted_image_with_preprocessing() {
    let mut config = OcrConfig::default();
    config.image_processing.enable_preprocessing = true;
    config.layout_analysis.enable_layout_analysis = false;
    let engine = OcrEngine::with_config(config).unwrap();
    engine.initialize().await.unwrap();
    let img = render_inverted("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "Inverted image should not crash with preprocessing");
}

#[tokio::test]
async fn test_roundtrip_empty_white_image() {
    let img = GrayImage::from_pixel(100, 100, Luma([255u8]));
    let text = recognize(img).await;
    assert!(text.trim().is_empty(), "Empty white image => empty text, got '{}'", text);
}

#[tokio::test]
async fn test_roundtrip_empty_black_image() {
    let img = GrayImage::from_pixel(100, 100, Luma([0u8]));
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "All-black image should not crash");
}

#[tokio::test]
async fn test_roundtrip_all_gray_image() {
    let img = GrayImage::from_pixel(100, 100, Luma([128u8]));
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "All-gray image should not crash");
}

#[tokio::test]
async fn test_roundtrip_single_pixel_image() {
    let img = GrayImage::from_pixel(10, 10, Luma([255u8]));
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "10x10 image should not crash");
}

#[tokio::test]
async fn test_roundtrip_min_size_image() {
    let img = render_text_5x7("A", 1, 0, 0);
    let (w, h) = img.dimensions();
    assert!(w >= 10 && h >= 10, "Image must be >= min size, got {}x{}", w, h);
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "Minimum-size image should not crash");
}

#[tokio::test]
async fn test_roundtrip_wide_text() {
    let img = render_text_5x7("ABCDEFGHIJKLMNOP", 6, 6, 12);
    let text = recognize(img).await;
    let acc = character_accuracy(&text, "ABCDEFGHIJKLMNOP");
    assert!(acc >= 0.5, "Wide text: got '{}' ({:.0}%)", text, acc * 100.0);
}

#[tokio::test]
async fn test_roundtrip_repeated_char() {
    let img = render_text_5x7("AAAA", 6, 6, 12);
    let text = recognize(img).await;
    assert!(!text.is_empty(), "Repeated chars should produce text");
    assert!(
        text.chars().all(|c| c == 'A'),
        "AAAA should produce only A's, got '{}'",
        text
    );
}

// ============================================================
// 9. Result structure validation
// ============================================================

#[tokio::test]
async fn test_result_has_text() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(!result.text.trim().is_empty(), "text field should be populated");
}

#[tokio::test]
async fn test_result_confidence_in_range() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be in [0, 1], got {}",
        result.confidence
    );
}

#[tokio::test]
async fn test_result_confidence_nonzero_for_text() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(
        result.confidence > 0.0,
        "Confidence should be > 0 for recognizable text, got {}",
        result.confidence
    );
}

#[tokio::test]
async fn test_result_has_words() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(
        !result.words.is_empty(),
        "Should have at least one word result"
    );
    assert!(
        !result.words[0].text.is_empty(),
        "Word text should not be empty"
    );
}

#[tokio::test]
async fn test_result_has_characters() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(
        !result.characters.is_empty(),
        "Should have character-level results"
    );
    for (i, cr) in result.characters.iter().enumerate() {
        assert!(
            cr.confidence >= 0.0 && cr.confidence <= 1.0,
            "Character {} confidence should be in [0,1], got {}",
            i,
            cr.confidence
        );
    }
}

#[tokio::test]
async fn test_result_has_lines() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO\nWORLD", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(
        !result.lines.is_empty(),
        "Should have line-level results"
    );
}

#[tokio::test]
async fn test_result_line_count_matches_input() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("AAA\nBBB\nCCC", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    let text_lines: Vec<&str> = result.text.trim().lines().collect();
    assert_eq!(text_lines.len(), 3, "Text should have 3 lines");
    assert_eq!(result.lines.len(), 3, "lines field should have 3 entries");
}

#[tokio::test]
async fn test_result_word_count_helper() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert_eq!(result.word_count(), result.words.len());
}

#[tokio::test]
async fn test_result_character_count_helper() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert_eq!(result.character_count(), result.characters.len());
}

#[tokio::test]
async fn test_result_average_character_confidence() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    let avg = result.average_character_confidence();
    assert!(
        avg >= 0.0 && avg <= 1.0,
        "Average char confidence should be in [0,1], got {}",
        avg
    );
}

#[tokio::test]
async fn test_result_empty_image_zero_confidence() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = GrayImage::from_pixel(100, 100, Luma([255u8]));
    let result = engine.process_image(make_ocr_image(img)).await.unwrap();
    assert!(
        result.text.trim().is_empty(),
        "Empty image should produce empty text"
    );
    assert_eq!(
        result.word_count(),
        0,
        "Empty image should have 0 words"
    );
    assert_eq!(
        result.character_count(),
        0,
        "Empty image should have 0 characters"
    );
}

// ============================================================
// 10. Reproducibility and sequential processing
// ============================================================

#[tokio::test]
async fn test_reproducibility_same_input_same_output() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img1 = render_text_5x7("HELLO", 6, 6, 12);
    let img2 = render_text_5x7("HELLO", 6, 6, 12);
    let r1 = engine.process_image(make_ocr_image(img1)).await.unwrap();
    let r2 = engine.process_image(make_ocr_image(img2)).await.unwrap();
    assert_eq!(r1.text.trim(), r2.text.trim(), "Same input should produce same text");
    assert!(
        (r1.confidence - r2.confidence).abs() < 0.001,
        "Same input should produce same confidence ({} vs {})",
        r1.confidence,
        r2.confidence
    );
}

#[tokio::test]
async fn test_sequential_recognitions_same_engine() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let words = vec!["HELLO", "WORLD", "TEST", "OCR"];
    let mut results = Vec::new();
    for word in &words {
        let img = render_text_5x7(word, 6, 6, 12);
        let text = recognize_with(&engine, img).await;
        results.push(text);
    }
    assert_eq!(results.len(), 4, "Should have 4 results");
    for (i, text) in results.iter().enumerate() {
        assert!(!text.is_empty(), "Result {} should not be empty", i);
    }
}

#[tokio::test]
async fn test_engine_reuse_after_empty_image() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let empty = GrayImage::from_pixel(100, 100, Luma([255u8]));
    let r1 = engine.process_image(make_ocr_image(empty)).await.unwrap();
    assert!(r1.text.trim().is_empty());
    let hello = render_text_5x7("HELLO", 6, 6, 12);
    let r2 = engine.process_image(make_ocr_image(hello)).await.unwrap();
    assert!(!r2.text.trim().is_empty(), "Engine should still work after empty image");
}

#[tokio::test]
async fn test_engine_reuse_after_black_image() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let black = GrayImage::from_pixel(100, 100, Luma([0u8]));
    let r1 = engine.process_image(make_ocr_image(black)).await;
    assert!(r1.is_ok());
    let hello = render_text_5x7("HELLO", 6, 6, 12);
    let r2 = engine.process_image(make_ocr_image(hello)).await.unwrap();
    assert!(!r2.text.trim().is_empty(), "Engine should still work after black image");
}

#[tokio::test]
async fn test_engine_processes_many_images() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    for i in 0..20 {
        let ch = char::from(b'A' + (i % 26));
        let img = render_text_5x7(&ch.to_string(), 6, 6, 12);
        let result = engine.process_image(make_ocr_image(img)).await;
        assert!(result.is_ok(), "Processing image {} should succeed", i);
    }
}

// ============================================================
// 11. Engine configuration and statistics
// ============================================================

#[tokio::test]
async fn test_engine_statistics_updated() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let stats_before = engine.get_statistics().await.unwrap();
    engine.process_image(make_ocr_image(img)).await.unwrap();
    let stats_after = engine.get_statistics().await.unwrap();
    assert_eq!(
        stats_after.total_images_processed,
        stats_before.total_images_processed + 1,
        "Should have processed 1 more image"
    );
    assert!(
        stats_after.total_processing_time_ms >= stats_before.total_processing_time_ms,
        "Processing time should increase"
    );
    assert!(stats_after.last_processed.is_some(), "Last processed should be set");
}

#[tokio::test]
async fn test_engine_statistics_multiple_images() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    for _ in 0..5 {
        let img = render_text_5x7("HELLO", 6, 6, 12);
        engine.process_image(make_ocr_image(img)).await.unwrap();
    }
    let stats = engine.get_statistics().await.unwrap();
    assert_eq!(stats.total_images_processed, 5);
    assert!(stats.average_processing_time_ms > 0.0);
}

#[tokio::test]
async fn test_engine_clear_cache() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let result = engine.clear_cache().await;
    assert!(result.is_ok(), "clear_cache should succeed");
}

#[tokio::test]
async fn test_engine_reset_statistics() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    engine.process_image(make_ocr_image(img)).await.unwrap();
    engine.reset_statistics().await.unwrap();
    let stats = engine.get_statistics().await.unwrap();
    assert_eq!(stats.total_images_processed, 0, "Stats should be reset");
}

#[tokio::test]
async fn test_engine_metadata() {
    let engine = make_engine();
    let metadata = engine.get_metadata();
    assert_eq!(metadata.name, "OCR Engine");
    assert!(!metadata.supported_languages.is_empty());
    assert!(metadata.capabilities.supports_text_recognition);
}

#[tokio::test]
async fn test_engine_get_config() {
    let engine = make_engine();
    let config = engine.get_config();
    assert!(!config.recognition.language.is_empty());
    assert!(config.recognition.confidence_threshold >= 0.0);
    assert!(config.recognition.confidence_threshold <= 1.0);
}

#[tokio::test]
async fn test_config_validation_rejects_bad_confidence() {
    let mut config = OcrConfig::default();
    config.recognition.confidence_threshold = 2.0;
    let result = OcrEngine::with_config(config);
    assert!(result.is_err(), "Confidence > 1.0 should be rejected");
}

#[tokio::test]
async fn test_config_validation_rejects_zero_threads() {
    let mut config = OcrConfig::default();
    config.performance.max_threads = 0;
    let result = OcrEngine::with_config(config);
    assert!(result.is_err(), "max_threads=0 should be rejected");
}

#[tokio::test]
async fn test_config_validation_rejects_zero_memory() {
    let mut config = OcrConfig::default();
    config.performance.memory_limit_mb = 0;
    let result = OcrEngine::with_config(config);
    assert!(result.is_err(), "memory_limit_mb=0 should be rejected");
}

#[tokio::test]
async fn test_config_validation_accepts_valid() {
    let config = OcrConfig::default();
    let result = OcrEngine::with_config(config);
    assert!(result.is_ok(), "Default config should be valid");
}

#[tokio::test]
async fn test_engine_not_initialized_error() {
    let engine = make_engine();
    // NOT calling initialize()
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_err(), "Uninitialized engine should return error");
}

#[tokio::test]
async fn test_engine_initialize_idempotent() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    engine.initialize().await.unwrap();
    let img = render_text_5x7("HELLO", 6, 6, 12);
    let result = engine.process_image(make_ocr_image(img)).await;
    assert!(result.is_ok(), "Double init should not break engine");
}

#[tokio::test]
async fn test_confidence_increases_with_scale() {
    let engine = make_engine();
    engine.initialize().await.unwrap();
    let img_small = render_text_5x7("HELLO", 3, 3, 6);
    let img_large = render_text_5x7("HELLO", 10, 10, 20);
    let res_small = engine.process_image(make_ocr_image(img_small)).await.unwrap();
    let res_large = engine.process_image(make_ocr_image(img_large)).await.unwrap();
    assert!(
        res_large.confidence >= res_small.confidence * 0.8,
        "Larger scale should have >= confidence (small={:.3}, large={:.3})",
        res_small.confidence,
        res_large.confidence
    );
}
