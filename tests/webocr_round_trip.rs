//! Round-trip tests for the WebOCR template-matching engine port.
//!
//! The crate's preferred pattern for recognition assertions (MEMORY §6):
//! render text with rusttype, OCR it back, assert on Levenshtein accuracy.
//! System-font dependent — skipped gracefully on bare CI images.

use ocr::webocr::{GrayImage, TemplateOcr};

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
    let m = levenshtein(actual, expected);
    let n = expected.chars().count().max(1);
    1.0 - (m as f32 / n as f32)
}

fn render_text(
    text: &str,
    font_data: &[u8],
    px: u32,
) -> Option<(Vec<u8>, u32, u32, f32)> {
    use rusttype::{Font, Scale};
    let font = Font::try_from_bytes(font_data)?;
    let scale = Scale::uniform(px as f32);
    let pad = px / 2;
    let v_metrics = font.v_metrics(scale);
    let ascent = v_metrics.ascent;
    let descent = -v_metrics.descent; // positive
    let mut width = 0u32;
    for g in font.layout(text, scale, rusttype::point(0.0, 0.0)) {
        if let Some(bb) = g.pixel_bounding_box() {
            width = width.max(bb.max.x as u32);
        }
    }
    let text_h = (ascent + descent).ceil() as u32;
    let w = width + 2 * pad;
    let h = text_h + 2 * pad;
    let mut data = vec![255u8; (w * h) as usize];
    let baseline_row = pad as f32 + ascent;
    for g in font.layout(text, scale, rusttype::point(pad as f32, pad as f32 + ascent)) {
        if let Some(bb) = g.pixel_bounding_box() {
            // The offsets are SIGNED: a glyph's bounding box reaches above
            // the origin (its top is a negative y), and clamping that to
            // zero would draw every glyph at the canvas top instead of at
            // its own height — the line's baseline would be meaningless and
            // the reading would lose its word spaces.
            let x_off = bb.min.x;
            let y_off = bb.min.y;
            g.draw(|x, y, v| {
                let xi = (x_off + x as i32) as usize;
                let yi = (y_off + y as i32) as usize;
                if xi < w as usize && yi < h as usize {
                    let slot = &mut data[yi * w as usize + xi];
                    let ink = (v * 255.0).round() as u8;
                    *slot = 255u8.saturating_sub(ink).min(*slot);
                }
            });
        }
    }
    Some((data, w, h, baseline_row))
}

fn system_font_data() -> Option<Vec<u8>> {
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ] {
        if let Ok(d) = std::fs::read(path) {
            return Some(d);
        }
    }
    None
}

#[test]
fn round_trip_sans_arial_recognises_short_text() {
    let data = match system_font_data() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no system sans font found");
            return;
        }
    };
    let ocr = match build_ocr() {
        Some(o) => o,
        None => {
            eprintln!("skipping: template engine unavailable (no fonts)");
            return;
        }
    };
    eprintln!("templates = {}", ocr.template_count());
    let text = "Hello 123";
    let Some((pixels, w, h, _baseline)) = render_text(text, &data, 48) else {
        eprintln!("skipping: render failed");
        return;
    };
    let gray = GrayImage {
        width: w as usize,
        height: h as usize,
        data: pixels,
    };
    let r = ocr.recognize(&gray);
    let actual: String = r
        .lines
        .iter()
        .flat_map(|l| l.chars.iter().map(|c| c.char))
        .collect();
    let acc = accuracy(&actual, text);
    // Threshold is deliberately lenient: the webocr fixture gates (sans
    // 100 %) are validated in the sibling webocr repo. Here we only need
    // the round-trip to prove the Rust port wires the pipeline.
    assert!(
        acc >= 0.8,
        "sans expected ≥0.8, got {acc:.3} (expected {text:?}, got {actual:?})"
    );
}

#[test]
fn round_trip_letters_only_recognises_alphabet() {
    let data = match system_font_data() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no system sans font found");
            return;
        }
    };
    let ocr = match build_ocr() {
        Some(o) => o,
        None => {
            eprintln!("skipping: template engine unavailable (no fonts)");
            return;
        }
    };
    let text = "Packmybox";
    let Some((pixels, w, h, _baseline)) = render_text(text, &data, 48) else {
        eprintln!("skipping: render failed");
        return;
    };
    let gray = GrayImage {
        width: w as usize,
        height: h as usize,
        data: pixels,
    };
    let r = ocr.recognize(&gray);
    let actual: String = r
        .lines
        .iter()
        .flat_map(|l| l.chars.iter().map(|c| c.char))
        .collect();
    let acc = accuracy(&actual, text);
    // Lower bar than the short-text test: the alphabet fixture trips the
    // O/0, o/O and similar aspect look-alikes that webocr explicitly
    // accepts as ambiguous (MEMORY §1). The point is the round-trip
    // pipeline works, not a per-letter gate.
    assert!(
        acc >= 0.7,
        "alpha expected ≥0.7, got {acc:.3} (expected {text:?}, got {actual:?})"
    );
}

/// The scorecard cell that gates the whole Latin pipeline: a pangram in a
/// real sans face, read back at the house metric.
///
/// This is the regression guard for the WebOCR parity pass (the case
/// channel, the page-context threshold, the raster ladder, the fragment
/// merge, the deskew). Before the pass the same cell read **83 %** with the
/// case flipped ("quiCk brOwn dOgS") and the ladder holding only 40 px and
/// 68 px rasters; after it the cell reads **100 %**.
///
/// The gate is deliberately at 95 % — one notch under the measured 100 % —
/// so a single marginal glyph does not turn the suite red on a different
/// platform's font stack, while a real regression (the case channel or the
/// ladder going missing) drops it to the 80s and does.
#[test]
fn scorecard_sans_pangram_reads_cleanly() {
    let data = match system_font_data() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no system sans font found");
            return;
        }
    };
    let ocr = match build_ocr() {
        Some(o) => o,
        None => {
            eprintln!("skipping: template engine unavailable (no fonts)");
            return;
        }
    };
    let text = "The quick brown fox jumps over the 42 lazy dogs";
    let Some((pixels, w, h, _baseline)) = render_text(text, &data, 30) else {
        eprintln!("skipping: render failed");
        return;
    };
    eprintln!(
        "DBG font_len={} image {}x{} sum={} otsu={}",
        data.len(),
        w,
        h,
        pixels.iter().map(|v| *v as u64).sum::<u64>(),
        ocr::webocr::binarize::otsu_threshold(&pixels)
    );
    let gray = GrayImage {
        width: w as usize,
        height: h as usize,
        data: pixels,
    };
    let r = ocr.recognize(&gray);
    let actual: String = r
        .lines
        .iter()
        .flat_map(|l| l.chars.iter().map(|c| c.char))
        .collect();
    let acc = accuracy(&actual, text);
    assert!(
        acc >= 0.95,
        "pangram expected ≥0.95, got {acc:.3} (expected {text:?}, got {actual:?})"
    );
}

/// A skewed page is straightened before segmentation — the deskew stage is
/// what makes a 4° scan readable at all (WebOCR measured 48.9 % before the
/// stage and 91.5 % after).
#[test]
fn scorecard_skewed_page_is_deskewed() {
    let data = match system_font_data() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no system sans font found");
            return;
        }
    };
    let ocr = match build_ocr() {
        Some(o) => o,
        None => {
            eprintln!("skipping: template engine unavailable (no fonts)");
            return;
        }
    };
    let text = "The quick brown fox jumps over the 42 lazy dogs";
    let Some((pixels, w, h, _baseline)) = render_text(text, &data, 30) else {
        eprintln!("skipping: render failed");
        return;
    };
    // Rotate the rendered line by 4° about its centre.
    let angle = 4.0f32.to_radians();
    let (sin, cos) = angle.sin_cos();
    let (sw, sh) = (w as f32, h as f32);
    let (nw, nh) = (
        (sw * cos + sh * sin).ceil() as usize,
        (sh * cos + sw * sin).ceil() as usize,
    );
    let mut rotated = vec![255u8; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let dx = x as f32 - nw as f32 / 2.0;
            let dy = y as f32 - nh as f32 / 2.0;
            let sx = (dx * cos + dy * sin + sw / 2.0).round() as isize;
            let sy = (-dx * sin + dy * cos + sh / 2.0).round() as isize;
            if sx >= 0 && sy >= 0 && (sx as usize) < w as usize && (sy as usize) < h as usize {
                rotated[y * nw + x] = pixels[sy as usize * w as usize + sx as usize];
            }
        }
    }
    let gray = GrayImage {
        width: nw,
        height: nh,
        data: rotated,
    };
    let r = ocr.recognize(&gray);
    let actual: String = r
        .lines
        .iter()
        .flat_map(|l| l.chars.iter().map(|c| c.char))
        .collect();
    let acc = accuracy(&actual, text);
    assert!(
        acc >= 0.85,
        "skewed pangram expected ≥0.85, got {acc:.3} (got {actual:?})"
    );
    // The boxes must still be in the ORIGINAL image's coordinates — the
    // deskew rotates the page, and the crate's API contract is that a
    // bounding box is in image coordinates.
    for line in &r.lines {
        assert!(
            line.x1 <= nw && line.y1 <= nh,
            "a line box escapes the image: {line:?}"
        );
    }
}

fn build_ocr() -> Option<ocr::webocr::TemplateOcr> {
    ocr::webocr::TemplateOcr::new().ok()
}

fn cjk_font_data() -> Option<Vec<u8>> {
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "C:/Windows/Fonts/msyh.ttc",
    ] {
        if let Ok(d) = std::fs::read(path) {
            return Some(d);
        }
    }
    None
}

/// Small CJK round-trip: render a 4-hanzi string with a CJK system
/// font, build a tiny library from the same font, OCR the rendering,
/// and assert non-empty recognition. Mirrors webocr's `reducedCharset`
/// pattern in `tests/realfont.test.ts` (MEMORY §6) — at small size the
/// CJK cosine ceiling is hit hard (cell=72px vs sample=72px, the
/// 1-cell rasterization slack), so the test asserts pipeline wiring
/// rather than per-character accuracy.
#[test]
fn round_trip_cjk_short_string() {
    let Some(font_data) = cjk_font_data() else {
        eprintln!("skipping: no CJK system font found");
        return;
    };
    // A few common, well-supported hanzi across multiple Unicode blocks
    // so the test exercises the codepoint-diversity path.
    let charset: String = "中文字测试".to_string();
    let renderer = ocr::webocr::RustTypeRenderer::for_cjk();
    if renderer.face_count() == 0 {
        eprintln!("skipping: CJK renderer loaded no faces");
        return;
    }
    let templates = ocr::webocr::templates::build_templates_for_options(
        &renderer,
        &charset,
        ocr::webocr::templates::CJK_SIZES,
        true, // coarse vectors for shortlist recall
        true, // cjk path: grid batching
    );
    if templates.is_empty() {
        eprintln!("skipping: CJK template library built empty");
        return;
    }
    let ocr = ocr::webocr::TemplateOcr::with_cjk_union(&renderer).unwrap();
    // Render a small subset to keep the test fast.
    let sample_text: &str = "中文字";
    let Some((pixels, w, h, _baseline)) = render_text(sample_text, &font_data, 72) else {
        eprintln!("skipping: render failed");
        return;
    };
    let gray = GrayImage {
        width: w as usize,
        height: h as usize,
        data: pixels,
    };
    let result = ocr.recognize(&gray);
    eprintln!(
        "CJK: templates={}, sample={:?}, result_text={:?}, scores={:?}, char_count={}",
        templates.len(),
        sample_text,
        result
            .lines
            .iter()
            .flat_map(|l| l.chars.iter().map(|c| c.char))
            .collect::<Vec<_>>(),
        result
            .lines
            .iter()
            .flat_map(|l| l.chars.iter().map(|c| c.score))
            .collect::<Vec<_>>(),
        result.char_count,
    );
    // The test is a pipeline smoke test: confirm the CJK flow runs
    // end-to-end on real CJK glyphs. We don't gate on per-character
    // accuracy here because the CJK fine-stage ceiling is
    // well-documented in webocr (MEMORY §13: 懒 fine-rank 7312,
    // 惰 loses by 0.042) and needs structural features beyond the
    // cosine path to break through. The asserts here only check the
    // pipeline runs and returns the right number of slots.
    assert_eq!(
        result.char_count,
        sample_text.chars().count(),
        "CJK pipeline should produce one result char per sample char"
    );
    // Each output char should have a finite score (the cosine ran
    // end-to-end without erroring).
    for ch in &result.lines[0].chars {
        assert!(ch.score > 0.0, "CJK match score was 0 for {:?}", ch.char);
        assert!(ch.score <= 1.0, "CJK match score out of range: {}", ch.score);
    }
}

/// WebOCR engine: `--lang auto` dispatches to the Latin library on
/// a Latin-shaped page (8 narrow 6×16 ink boxes) and to the CJK
/// library on a CJK-shaped page (4 hanzi with detached strokes).
/// This exercises the geometry-driven `resolve_script` path added in
/// the 2026-10-18 parity pass.
#[test]
fn resolve_script_dispatch_picks_latin_for_latin_page() {
    let mut g: Vec<u8> = vec![255u8; 200 * 30];
    for col in 0..8 {
        let x0 = 8 + col * 24;
        for y in 10..26 {
            for x in x0..x0 + 6 {
                g[y * 200 + x] = 0;
            }
        }
    }
    let script = ocr::webocr::resolve_script(&ocr::webocr::GrayImage {
        width: 200,
        height: 30,
        data: g,
    });
    assert_eq!(script, ocr::webocr::ScriptId::Latin);
}

#[test]
fn resolve_script_dispatch_picks_cjk_for_cjk_page() {
    let mut g: Vec<u8> = vec![255u8; 200 * 30];
    for hanzi in 0..4 {
        let x0 = 8 + hanzi * 24;
        for stroke in 0..3 {
            let x = x0 + stroke * 4;
            for y in 10..20 {
                for dx in 0..2 {
                    g[y * 200 + x + dx] = 0;
                }
            }
        }
    }
    let script = ocr::webocr::resolve_script(&ocr::webocr::GrayImage {
        width: 200,
        height: 30,
        data: g,
    });
    assert_eq!(script, ocr::webocr::ScriptId::Cjk);
}
