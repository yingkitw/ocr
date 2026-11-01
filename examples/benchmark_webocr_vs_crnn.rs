//! Release-mode benchmark: compare the WebOCR template engine against
//! the bundled CRNN. Renders a fixed set of test phrases with a system
//! font, runs each engine, reports per-engine wall time, characters
//! per second, and Levenshtein accuracy.
//!
//! Run: `cargo run --release --example benchmark_webocr_vs_crnn`.
//! Without `--release` the timings are dominated by debug codegen and
//! are not representative.

use std::time::Instant;

use ocr::core::config::OcrConfig;
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
) -> Option<(Vec<u8>, u32, u32)> {
    use rusttype::{Font, Scale};
    let font = match Font::try_from_bytes(font_data) {
        Some(f) => f,
        None => return None,
    };
    let scale = Scale::uniform(px as f32);
    let v_metrics = font.v_metrics(scale);
    let ascent = v_metrics.ascent;
    let descent = -v_metrics.descent;
    let pad = px / 2;
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
    for g in font.layout(text, scale, rusttype::point(pad as f32, pad as f32 + ascent)) {
        if let Some(bb) = g.pixel_bounding_box() {
            let x_off = (bb.min.x as i32).max(0) as u32;
            let y_off = (bb.min.y as i32).max(0) as u32;
            g.draw(|x, y, v| {
                let xi = (x_off + x) as usize;
                let yi = (y_off + y) as usize;
                if xi < w as usize && yi < h as usize {
                    let slot = &mut data[yi * w as usize + xi];
                    let ink = (v * 255.0).round() as u8;
                    *slot = 255u8.saturating_sub(ink).min(*slot);
                }
            });
        }
    }
    Some((data, w, h))
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

fn main() {
    let Some(data) = system_font_data() else {
        eprintln!("no system font found; nothing to benchmark");
        return;
    };

    let phrases: &[&str] = &[
        "Hello 123",
        "Packmybox",
        "The quick brown fox",
        "lorem ipsum dolor",
    ];
    let px = 48u32;
    let mut images: Vec<(String, Vec<u8>, u32, u32)> = Vec::new();
    for &p in phrases {
        if let Some((buf, w, h)) = render_text(p, &data, px) {
            images.push((p.to_string(), buf, w, h));
        }
    }
    if images.is_empty() {
        eprintln!("rendering produced no images; check font setup");
        return;
    }

    // ---- WebOCR template engine ----
    let template = match TemplateOcr::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("template engine unavailable: {e}");
            return;
        }
    };
    let mut webocr_total_chars = 0usize;
    let mut webocr_total_acc = 0f32;
    let webocr_start = Instant::now();
    for (text, pixels, w, h) in &images {
        let gray = GrayImage {
            width: *w as usize,
            height: *h as usize,
            data: pixels.clone(),
        };
        let r = template.recognize(&gray);
        let actual: String = r
            .lines
            .iter()
            .flat_map(|l| l.chars.iter().map(|c| c.char))
            .collect();
        let acc = accuracy(&actual, text);
        webocr_total_chars += r.char_count;
        webocr_total_acc += acc;
    }
    let webocr_elapsed = webocr_start.elapsed();
    eprintln!(
        "WebOCR template: {} phrases, {} chars, mean accuracy {:.3}, wall time {:.2?} ({:.0} chars/s)",
        images.len(),
        webocr_total_chars,
        webocr_total_acc / images.len() as f32,
        webocr_elapsed,
        webocr_total_chars as f64 / webocr_elapsed.as_secs_f64()
    );

    // ---- CRNN (bundled model) ----
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("tokio runtime: {e}");
            return;
        }
    };
    let api = rt.block_on(async {
        let api = ocr::api::Ocr::with_config(OcrConfig {
            recognition: ocr::core::config::RecognitionConfig {
                language: "en".to_string(),
                ..Default::default()
            },
            ..Default::default()
        })?;
        api.initialize().await?;
        Ok::<_, anyhow::Error>(api)
    });
    let Ok(api) = api else {
        eprintln!("CRNN unavailable");
        return;
    };
    let crnn_start = Instant::now();
    let mut crnn_total_chars = 0usize;
    let mut crnn_total_acc = 0f32;
    for (text, pixels, w, h) in &images {
        let r = rt.block_on(async { api.recognize_text(pixels, *w, *h).await });
        let Ok(r) = r else {
            eprintln!("CRNN recognize_text failed: {:?}", r.err());
            continue;
        };
        let acc = accuracy(&r.text.trim(), text);
        crnn_total_chars += r.text.chars().count();
        crnn_total_acc += acc;
    }
    let crnn_elapsed = crnn_start.elapsed();
    eprintln!(
        "CRNN (bundled): {} phrases, {} chars, mean accuracy {:.3}, wall time {:.2?} ({:.0} chars/s)",
        images.len(),
        crnn_total_chars,
        crnn_total_acc / images.len() as f32,
        crnn_elapsed,
        crnn_total_chars as f64 / crnn_elapsed.as_secs_f64().max(1e-6)
    );
}
