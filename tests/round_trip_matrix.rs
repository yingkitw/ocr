//! Round-trip matrix — text → rendered page → template OCR → text.
//!
//! Complements `webocr_round_trip.rs` (engine scorecard) and
//! `round_trip_test.rs` (CRNN path, `#[ignore]`d pending a trained model)
//! with a broad fixture matrix for the WebOCR template engine, which is
//! where exact-text assertions can honestly live today.
//!
//! Generation complexity, raised over the sibling suite:
//! - real system fonts in three families (sans / serif / monospace)
//! - a size ladder (16–54 px) bracketing the template rasters
//! - multi-line paragraphs with varied line lengths
//! - deterministic degradations: speckle noise, box blur, skew, faded
//!   ink (low contrast), inverted polarity, uneven background, and
//!   a 2× downscale (effective half-size scan)
//!
//! Recognition is asserted on the space-aware `text` channel — the
//! engine's product — not on the `chars` vector (spaces are not
//! characters). Gates sit one notch under the accuracy measured when
//! the fixture was pinned (see the per-test comments), in the house
//! convention of the scorecard tests.

use ocr::webocr::{GrayImage, TemplateOcr};
use std::sync::OnceLock;

// ------------------------------------------------------------------
// metrics
// ------------------------------------------------------------------

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

fn accuracy(actual: &str, expected: &str) -> f32 {
    1.0 - (levenshtein(actual, expected) as f32 / expected.chars().count().max(1) as f32)
}

// ------------------------------------------------------------------
// fonts — one candidate list per family, first hit wins
// ------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Face {
    Sans,
    Serif,
    Mono,
}

fn face_data(face: Face) -> Option<Vec<u8>> {
    let paths: &[&str] = match face {
        Face::Sans => &[
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ],
        Face::Serif => &[
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/Times.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/times.ttf",
        ],
        Face::Mono => &[
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/System/Library/Fonts/Courier.ttc",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ],
    };
    paths.iter().find_map(|p| std::fs::read(p).ok())
}

// ------------------------------------------------------------------
// rendering — rusttype layout, one call per line
// ------------------------------------------------------------------

fn render_lines(
    lines: &[&str],
    font_data: &[u8],
    px: u32,
    line_gap: u32,
) -> Option<(Vec<u8>, usize, usize)> {
    use rusttype::{Font, Scale};
    let font = Font::try_from_bytes(font_data)?;
    let scale = Scale::uniform(px as f32);
    let pad = px / 2;
    let v = font.v_metrics(scale);
    let text_h = (v.ascent - v.descent).ceil() as u32;
    let mut width = 0u32;
    for line in lines {
        let mut w = 0u32;
        for g in font.layout(line, scale, rusttype::point(0.0, 0.0)) {
            if let Some(bb) = g.pixel_bounding_box() {
                w = w.max(bb.max.x as u32);
            }
        }
        width = width.max(w);
    }
    let w = (width + 2 * pad) as usize;
    let h = (text_h * lines.len() as u32
        + line_gap * (lines.len() as u32).saturating_sub(1)
        + 2 * pad) as usize;
    let mut data = vec![255u8; w * h];
    for (li, line) in lines.iter().enumerate() {
        let y0 = pad + li as u32 * (text_h + line_gap);
        for g in font.layout(line, scale, rusttype::point(pad as f32, y0 as f32 + v.ascent)) {
            if let Some(bb) = g.pixel_bounding_box() {
                // Signed offsets — clamping a negative y to 0 would draw
                // every glyph at the canvas top and lose the baseline.
                let (x_off, y_off) = (bb.min.x, bb.min.y);
                g.draw(|x, y, cov| {
                    let xi = (x_off + x as i32) as usize;
                    let yi = (y_off + y as i32) as usize;
                    if xi < w && yi < h {
                        let slot = &mut data[yi * w + xi];
                        *slot = 255u8
                            .saturating_sub((cov * 255.0).round() as u8)
                            .min(*slot);
                    }
                });
            }
        }
    }
    Some((data, w, h))
}

// ------------------------------------------------------------------
// deterministic degradations
// ------------------------------------------------------------------

/// xorshift64* — a fixed seed keeps the fixture reproducible run to run.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn speckle(buf: &mut [u8], prob: f32, seed: u64) {
    let mut s = seed.max(1);
    for p in buf.iter_mut() {
        let roll = (xorshift(&mut s) % 10000) as f32 / 10000.0;
        if roll < prob / 2.0 {
            *p = 0;
        } else if roll < prob {
            *p = 255;
        }
    }
}

fn box_blur3(buf: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut n) = (0u32, 0u32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xi, yi) = (x as i32 + dx, y as i32 + dy);
                    if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                        sum += buf[yi as usize * w + xi as usize] as u32;
                        n += 1;
                    }
                }
            }
            out[y * w + x] = (sum / n) as u8;
        }
    }
    out
}

fn rotate(buf: &[u8], w: usize, h: usize, deg: f32) -> (Vec<u8>, usize, usize) {
    let a = deg.to_radians();
    let (sin, cos) = a.sin_cos();
    let (sw, sh) = (w as f32, h as f32);
    let nw = (sw * cos + sh * sin).ceil() as usize;
    let nh = (sh * cos + sw * sin).ceil() as usize;
    let mut out = vec![255u8; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let dx = x as f32 - nw as f32 / 2.0;
            let dy = y as f32 - nh as f32 / 2.0;
            let sx = (dx * cos + dy * sin + sw / 2.0).round() as isize;
            let sy = (-dx * sin + dy * cos + sh / 2.0).round() as isize;
            if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                out[y * nw + x] = buf[sy as usize * w + sx as usize];
            }
        }
    }
    (out, nw, nh)
}

/// Fade the ink toward the background — a low-contrast scan.
fn fade(buf: &mut [u8], keep: f32) {
    for p in buf.iter_mut() {
        *p = 255u8.saturating_sub(((255 - *p) as f32 * keep).round() as u8);
    }
}

fn invert(buf: &mut [u8]) {
    for p in buf.iter_mut() {
        *p = 255 - *p;
    }
}

/// Uneven background: a dark-to-light horizontal gradient, the fixture
/// class that defeats global-threshold binarization.
fn gradient(buf: &mut [u8], w: usize, h: usize, dark: u8) {
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / w.max(1) as f32;
            let bg = dark as f32 + (255.0 - dark as f32) * t;
            let i = y * w + x;
            buf[i] = buf[i].min(bg.round() as u8);
        }
    }
}

/// Box-average downscale — the fixture for resolution loss (a page
/// scanned at half the needed DPI).
fn downscale(buf: &[u8], w: usize, h: usize, factor: usize) -> Vec<u8> {
    let (nw, nh) = (w / factor, h / factor);
    let mut small = vec![255u8; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    sum += buf[(y * factor + dy) * w + x * factor + dx] as u32;
                }
            }
            small[y * nw + x] = (sum / (factor * factor) as u32) as u8;
        }
    }
    small
}

// ------------------------------------------------------------------
// shared engine — the library build (every face × raster) is the
// expensive part, so it happens once per test binary
// ------------------------------------------------------------------

static OCR: OnceLock<Option<TemplateOcr>> = OnceLock::new();

fn shared_ocr() -> Option<&'static TemplateOcr> {
    OCR.get_or_init(|| TemplateOcr::new().ok()).as_ref()
}

fn recognize(ocr: &TemplateOcr, buf: &[u8], w: usize, h: usize) -> String {
    let r = ocr.recognize(&GrayImage {
        width: w,
        height: h,
        data: buf.to_vec(),
    });
    r.lines
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Assert a round-trip at a gate one notch under the measured value.
fn gate(actual: &str, expected: &str, min: f32, label: &str) {
    let acc = accuracy(actual.trim(), expected);
    assert!(
        acc >= min,
        "{label}: expected ≥{min}, got {acc:.3} (expected {expected:?}, got {actual:?})"
    );
}

macro_rules! skip {
    ($opt:expr, $what:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!("skipping: {}", $what);
                return;
            }
        }
    };
}

const PANGRAM: &str = "The quick brown fox jumps over the 42 lazy dogs";

// ==================================================================
// A · font matrix — same text, three type families
// ==================================================================

#[test]
fn matrix_sans_pangram() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    // Measured 1.000 post-veto.
    gate(&recognize(ocr, &b, w, h), PANGRAM, 0.95, "sans pangram");
}

#[test]
fn matrix_serif_pangram() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Serif), "no serif font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    // Measured 1.000 (Georgia); lower gate for cross-platform faces.
    gate(&recognize(ocr, &b, w, h), PANGRAM, 0.88, "serif pangram");
}

#[test]
fn matrix_mono_pangram() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Mono), "no mono font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    // Measured 0.979 (Courier New; residual "Over" — the veto protects
    // word-initial capitals, the price of not demoting real ones).
    gate(&recognize(ocr, &b, w, h), PANGRAM, 0.88, "mono pangram");
}

// ==================================================================
// B · size ladder — the template rasters bracket these sizes
// ==================================================================

#[test]
fn matrix_size_ladder() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    // (px, gate) — measured 0.936 / 0.979 / 1.000 / 1.000 / 1.000.
    // 16 px is below the smallest template raster (16 px templates cover
    // it but AA divergence costs ~6 %); the floor pin still guards it.
    for (px, min) in [(16u32, 0.78), (20, 0.90), (30, 0.95), (40, 0.95), (54, 0.95)] {
        let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, px, 12), "render");
        gate(
            &recognize(ocr, &b, w, h),
            PANGRAM,
            min,
            &format!("sans {px}px"),
        );
    }
}

// ==================================================================
// C · content variety — caps, punctuation, codes, mixed case
// ==================================================================

#[test]
fn matrix_all_caps_headline() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let text = "ALL CAPS HEADLINE 2026";
    let (b, w, h) = skip!(render_lines(&[text], &font, 30, 12), "render");
    // Measured 1.000.
    gate(&recognize(ocr, &b, w, h), text, 0.95, "all caps");
}

#[test]
fn matrix_punctuation_and_symbols() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    // Every glyph here is in LATIN_CHARSET (full printable ASCII).
    let text = "Order #1042: $3.50 (incl. 8% tax)";
    let (b, w, h) = skip!(render_lines(&[text], &font, 30, 12), "render");
    // Measured 1.000 — the dense punctuation advances exercise the
    // gap-cluster word splitter's biggest-jump split.
    gate(&recognize(ocr, &b, w, h), text, 0.90, "punctuation");
}

#[test]
fn matrix_mixed_case_with_digits() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let text = "Mixed Case Words With Numbers 99";
    let (b, w, h) = skip!(render_lines(&[text], &font, 30, 12), "render");
    // Measured 1.000 (was 0.938 pre-veto: "WordS"/"NumberS" flips).
    gate(&recognize(ocr, &b, w, h), text, 0.92, "mixed case");
}

#[test]
fn matrix_codes_and_abbreviations() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let text = "INV-2026-0342 XKCD-79 alpha";
    let (b, w, h) = skip!(render_lines(&[text], &font, 30, 12), "render");
    // Measured 1.000 — hyphenated codes and an all-caps acronym in a
    // lowercase context exercise the case veto's protect-the-caps side.
    gate(&recognize(ocr, &b, w, h), text, 0.90, "codes");
}

#[test]
fn matrix_abbreviations_with_periods() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let text = "e.g. i.e. etc. vs. Dr. Mr.";
    let (b, w, h) = skip!(render_lines(&[text], &font, 30, 12), "render");
    // Measured 1.000.
    gate(&recognize(ocr, &b, w, h), text, 0.90, "abbreviations");
}

// ==================================================================
// D · multi-line pages
// ==================================================================

#[test]
fn matrix_paragraph_three_lines() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let lines = ["The quick brown fox", "jumps over the lazy", "dogs in the park."];
    let expected = lines.join("\n");
    let (b, w, h) = skip!(render_lines(&lines, &font, 30, 24), "render");
    let r = ocr.recognize(&GrayImage {
        width: w,
        height: h,
        data: b,
    });
    assert_eq!(
        r.lines.len(),
        3,
        "expected 3 text lines, got {}: {:?}",
        r.lines.len(),
        r.lines.iter().map(|l| &l.text).collect::<Vec<_>>()
    );
    let actual: String = r.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    // Measured 1.000 (was 0.965 pre-veto).
    gate(&actual, &expected, 0.92, "3-line paragraph");
}

#[test]
fn matrix_paragraph_varied_lengths() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let lines = ["Short.", "A much longer second line of text here", "Mid-length line"];
    let expected = lines.join("\n");
    let (b, w, h) = skip!(render_lines(&lines, &font, 30, 24), "render");
    let r = ocr.recognize(&GrayImage {
        width: w,
        height: h,
        data: b,
    });
    assert_eq!(r.lines.len(), 3, "expected 3 lines");
    let actual: String = r.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    // Measured 1.000.
    gate(&actual, &expected, 0.90, "varied lengths");
}

// ==================================================================
// E · degraded scans — deterministic distortions
// ==================================================================

#[test]
fn matrix_speckle_noise() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    for (prob, min) in [(0.01f32, 0.93), (0.03, 0.88)] {
        let mut n = b.clone();
        speckle(&mut n, prob, 42);
        // Measured 1.000 at both levels (filter_noise drops the specks).
        gate(
            &recognize(ocr, &n, w, h),
            PANGRAM,
            min,
            &format!("speckle {prob}"),
        );
    }
}

#[test]
fn matrix_box_blur() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let blurred = box_blur3(&b, w, h);
    // Measured 1.000 — the normalization blur already softens edges.
    gate(&recognize(ocr, &blurred, w, h), PANGRAM, 0.90, "box blur");
}

#[test]
fn matrix_skew() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    for (deg, min) in [(2.0f32, 0.92), (5.0, 0.88)] {
        let (r, rw, rh) = rotate(&b, w, h, deg);
        // Measured 1.000 at both angles — the deskew stage earns it.
        gate(
            &recognize(ocr, &r, rw, rh),
            PANGRAM,
            min,
            &format!("skew {deg}°"),
        );
    }
}

#[test]
fn matrix_low_contrast() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let mut faded = b.clone();
    fade(&mut faded, 0.5);
    // Measured 1.000 — polarity + Otsu see through faded ink.
    gate(&recognize(ocr, &faded, w, h), PANGRAM, 0.90, "low contrast");
}

#[test]
fn matrix_inverted_polarity() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let mut inv = b.clone();
    invert(&mut inv);
    // Measured 1.000 — the binarize stage detects and flips polarity.
    gate(&recognize(ocr, &inv, w, h), PANGRAM, 0.95, "inverted");
}

#[test]
fn matrix_uneven_background() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let mut g = b.clone();
    gradient(&mut g, w, h, 170);
    // Measured 1.000 — the background floor dips to 170 but stays above
    // the ink, so Otsu still separates.
    gate(&recognize(ocr, &g, w, h), PANGRAM, 0.88, "gradient bg");
}

#[test]
fn matrix_downscaled_resolution() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    // Render at 60 px, halve to an effective-30 px scan.
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 60, 12), "render");
    let small = downscale(&b, w, h, 2);
    // Measured 0.894 on the 2× downscale of the 30 px render (o→0, ] flips);
    // the 60→30 px variant reads much higher. Gate on the floor.
    gate(
        &recognize(ocr, &small, w / 2, h / 2),
        PANGRAM,
        0.80,
        "downscale 2x",
    );
}

// ==================================================================
// F · invariants and result structure
// ==================================================================

#[test]
fn matrix_clean_beats_noisy() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let clean = accuracy(recognize(ocr, &b, w, h).trim(), PANGRAM);
    let mut n = b.clone();
    speckle(&mut n, 0.03, 7);
    let noisy = accuracy(recognize(ocr, &n, w, h).trim(), PANGRAM);
    assert!(
        clean >= noisy - 0.05,
        "noise should not improve the read: clean={clean:.3} noisy={noisy:.3}"
    );
}

#[test]
fn matrix_word_count_preserved() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let r = ocr.recognize(&GrayImage {
        width: w,
        height: h,
        data: b,
    });
    let words: Vec<&str> = r
        .lines
        .iter()
        .flat_map(|l| l.words.iter().map(|w| w.text.as_str()))
        .collect();
    assert_eq!(
        words.len(),
        PANGRAM.split_whitespace().count(),
        "word count mismatch: {words:?}"
    );
}

#[test]
fn matrix_deterministic_output() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let a = recognize(ocr, &b, w, h);
    let b2 = recognize(ocr, &b, w, h);
    assert_eq!(a, b2, "same input must read the same twice");
}

#[test]
fn matrix_result_structure_sane() {
    let ocr = skip!(shared_ocr(), "no fonts");
    let font = skip!(face_data(Face::Sans), "no sans font");
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let r = ocr.recognize(&GrayImage {
        width: w,
        height: h,
        data: b,
    });
    assert_eq!(r.lines.len(), 1);
    assert_eq!(r.char_count, r.lines.iter().map(|l| l.chars.len()).sum::<usize>());
    assert!(r.avg_score > 0.5 && r.avg_score <= 1.0);
    for line in &r.lines {
        for c in &line.chars {
            assert!(c.score > 0.0 && c.score <= 1.0, "char {:?} score {}", c.char, c.score);
            assert!(c.x1 <= w && c.y1 <= h, "char box escapes image");
        }
        // Words tile the line's chars — spaces live in `text`, not `chars`.
        let char_total: usize = line.words.iter().map(|wd| wd.chars.len()).sum();
        assert_eq!(char_total, line.chars.len());
    }
}

// ==================================================================
// G · the production path — OcrEngine + template engine end-to-end
// ==================================================================

fn template_engine() -> Option<ocr::core::engine::OcrEngine> {
    let mut config = ocr::core::config::OcrConfig::default();
    config.recognition.engine = ocr::core::config::RecognitionEngine::Template;
    config.image_processing.enable_preprocessing = false;
    config.layout_analysis.enable_layout_analysis = false;
    ocr::core::engine::OcrEngine::with_config(config).ok()
}

#[tokio::test]
async fn matrix_full_pipeline_paragraph() {
    let font = skip!(face_data(Face::Sans), "no sans font");
    if shared_ocr().is_none() {
        eprintln!("skipping: no fonts for template library");
        return;
    }
    let engine = skip!(template_engine(), "engine build");
    engine.initialize().await.unwrap();
    let lines = ["The quick brown fox", "jumps over the lazy dogs"];
    let expected = lines.join("\n");
    let (b, w, h) = skip!(render_lines(&lines, &font, 30, 24), "render");
    let img = image::DynamicImage::ImageLuma8(
        image::GrayImage::from_raw(w as u32, h as u32, b).unwrap(),
    );
    let res = engine
        .process_image(ocr::core::image::OcrImage::new(img, 300))
        .await
        .unwrap();
    gate(&res.text, &expected, 0.90, "full pipeline paragraph");
    assert!(!res.words.is_empty());
    assert!(res.confidence > 0.0 && res.confidence <= 1.0);
}

/// The dictionary-correction default must not rewrite good output.
///
/// Regression pin for the `enable_dictionary_correction` default flip:
/// the bundled common-words dictionary used to rewrite every word it
/// did not contain within edit distance 2 ("quick" → "quite",
/// "42" → "we"). With the default off the production path returns the
/// engine's text unmangled.
#[tokio::test]
async fn matrix_full_pipeline_unmangled_text() {
    let font = skip!(face_data(Face::Sans), "no sans font");
    if shared_ocr().is_none() {
        eprintln!("skipping: no fonts for template library");
        return;
    }
    let engine = skip!(template_engine(), "engine build");
    engine.initialize().await.unwrap();
    let (b, w, h) = skip!(render_lines(&[PANGRAM], &font, 30, 12), "render");
    let img = image::DynamicImage::ImageLuma8(
        image::GrayImage::from_raw(w as u32, h as u32, b).unwrap(),
    );
    let res = engine
        .process_image(ocr::core::image::OcrImage::new(img, 300))
        .await
        .unwrap();
    for word in ["quick", "brown", "fox", "42", "lazy"] {
        assert!(
            res.text.contains(word),
            "dictionary correction mangled the output: {word:?} missing from {:?}",
            res.text
        );
    }
}
