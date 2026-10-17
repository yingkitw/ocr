//! Pre-recognition Latin vs CJK hint from glyph geometry (WebOCR `mergeRatio` idea).
//!
//! Counts connected components on a binarized page, merges horizontally adjacent
//! blobs on the same text line, and compares `merge_ratio = cc_count / line_units`.
//! Latin body text tends to merge many strokes per word (high ratio); CJK tends
//! toward one blob per character (ratio near 1).

use crate::core::image::OcrImage;
use crate::lang::detector::LanguageGuess;
use crate::lang::unicode::Script;
use crate::layout::union_find_ccl::{connected_components_8connectivity, ComponentBoundingBox};
use image::{GrayImage, Luma};

/// Geometric script guess from CC clustering (no recognition required).
pub fn geometry_script_guess(image: &OcrImage) -> LanguageGuess {
    let gray = image.data.to_luma8();
    if gray.width() < 8 || gray.height() < 8 {
        return fallback_latin(0.0, 0.2);
    }

    let binary = otsu_foreground(&gray);
    let ccl = connected_components_8connectivity(&binary);
    if ccl.num_components == 0 {
        return fallback_latin(0.0, 0.15);
    }

    let page_h = gray.height().max(1);
    let min_h = (page_h / 40).max(4);
    let max_h = (page_h * 2 / 3).max(min_h + 1);

    let mut boxes = filter_glyph_boxes(&ccl, min_h, max_h);
    if boxes.len() < 2 {
        let relaxed_max = page_h.saturating_sub(2).max(min_h + 1);
        boxes = filter_glyph_boxes(&ccl, min_h, relaxed_max);
    }

    if boxes.is_empty() {
        return fallback_latin(0.0, 0.25);
    }
    if boxes.len() == 1 {
        let b = boxes[0];
        let aspect = b.width() as f32 / b.height().max(1) as f32;
        if aspect >= 0.72 && aspect <= 1.35 {
            return LanguageGuess {
                language: "zh".into(),
                confidence: 0.6,
                script: Script::CJK,
            };
        }
    }

    let mut heights: Vec<f32> = boxes.iter().map(|b| b.height() as f32).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = heights[heights.len() / 2].max(1.0);

    let mut aspects: Vec<f32> = boxes
        .iter()
        .map(|b| b.width() as f32 / b.height() as f32)
        .collect();
    aspects.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_aspect = aspects[aspects.len() / 2];

    boxes.sort_by(|a, b| {
        a.top
            .cmp(&b.top)
            .then(a.left.cmp(&b.left))
    });

    let mut widths: Vec<f32> = boxes.iter().map(|b| b.width() as f32).collect();
    widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_w = widths[widths.len() / 2].max(1.0);

    let line_units = merge_into_line_units(&boxes, med_h, med_w);
    let cc_count = boxes.len().max(1);
    let merge_ratio = cc_count as f32 / line_units.len().max(1) as f32;

    classify_from_metrics(merge_ratio, med_aspect)
}

/// Combine geometric hint with post-recognition text detection for `--lang auto`.
pub fn detect_auto(image: &OcrImage, recognized_text: &str) -> LanguageGuess {
    let geom = geometry_script_guess(image);
    let text_guess = crate::lang::detector::LanguageDetector::detect(recognized_text);

    let non_ws = recognized_text.chars().filter(|c| !c.is_whitespace()).count();
    if non_ws >= 16 && text_guess.confidence >= 0.55 {
        return text_guess;
    }
    if text_guess.confidence >= 0.82 {
        return text_guess;
    }

    if text_guess.script == geom.script {
        if text_guess.confidence >= geom.confidence {
            return text_guess;
        }
        return LanguageGuess {
            language: if text_guess.language.len() > 2 {
                text_guess.language
            } else {
                geom.language
            },
            confidence: (geom.confidence + text_guess.confidence) / 2.0,
            script: geom.script,
        };
    }

    if non_ws < 8 {
        return geom;
    }

    LanguageGuess {
        language: text_guess.language,
        confidence: text_guess.confidence * 0.85,
        script: text_guess.script,
    }
}

fn filter_glyph_boxes(
    ccl: &crate::layout::union_find_ccl::CclResult,
    min_h: u32,
    max_h: u32,
) -> Vec<ComponentBoundingBox> {
    let min_area = (min_h * min_h) as u64;
    let mut boxes: Vec<ComponentBoundingBox> = (1..=ccl.num_components)
        .filter_map(|i| {
            let b = ccl.bounding_boxes.get(i).copied()?;
            if b.is_empty() {
                return None;
            }
            let h = b.height();
            let w = b.width();
            if h < min_h || h > max_h || w == 0 {
                return None;
            }
            if b.area() < min_area {
                return None;
            }
            Some(b)
        })
        .collect();
    if boxes.len() > 64 {
        boxes.sort_by_key(|b| std::cmp::Reverse(b.area()));
        boxes.truncate(64);
    }
    boxes
}

fn classify_from_metrics(merge_ratio: f32, med_aspect: f32) -> LanguageGuess {
    // WebOCR-style thresholds (tuned on synthetic fixtures below).
    let cjk_like = merge_ratio <= 1.35 && med_aspect >= 0.72;
    let latin_like = merge_ratio >= 1.85 || (merge_ratio >= 1.55 && med_aspect <= 0.65);

    if cjk_like && !latin_like {
        return LanguageGuess {
            language: "zh".into(),
            confidence: (0.55 + (1.35 - merge_ratio).max(0.0) * 0.2).clamp(0.5, 0.88),
            script: Script::CJK,
        };
    }
    if latin_like && !cjk_like {
        return LanguageGuess {
            language: "en".into(),
            confidence: (0.55 + (merge_ratio - 1.5).max(0.0) * 0.15).clamp(0.5, 0.88),
            script: Script::Latin,
        };
    }

    if med_aspect >= 0.85 {
        LanguageGuess {
            language: "zh".into(),
            confidence: 0.45,
            script: Script::CJK,
        }
    } else {
        fallback_latin(merge_ratio, 0.4)
    }
}

fn fallback_latin(_merge_ratio: f32, confidence: f32) -> LanguageGuess {
    LanguageGuess {
        language: "en".into(),
        confidence,
        script: Script::Latin,
    }
}

fn merge_into_line_units(
    boxes: &[ComponentBoundingBox],
    med_h: f32,
    med_w: f32,
) -> Vec<ComponentBoundingBox> {
    let y_tol = (med_h * 0.55).max(2.0) as u32;
    // Stroke-level merge only (wide gaps between CJK chars must stay separate).
    let gap_tol = (med_h * 0.35)
        .min(med_w * 0.85)
        .max(2.0) as u32;

    let mut units: Vec<ComponentBoundingBox> = Vec::new();
    for b in boxes {
        let mut merged = false;
        for unit in &mut units {
            if vertical_overlap(*b, *unit, y_tol) {
                let gap = if b.left >= unit.right {
                    b.left.saturating_sub(unit.right)
                } else if unit.left >= b.right {
                    unit.left.saturating_sub(b.right)
                } else {
                    0
                };
                let narrow_pair =
                    b.width() <= (med_h * 0.55) as u32 && unit.width() <= (med_h * 0.55) as u32;
                if gap <= gap_tol && narrow_pair {
                    unit.left = unit.left.min(b.left);
                    unit.top = unit.top.min(b.top);
                    unit.right = unit.right.max(b.right);
                    unit.bottom = unit.bottom.max(b.bottom);
                    merged = true;
                    break;
                }
            }
        }
        if !merged {
            units.push(*b);
        }
    }
    units
}

fn vertical_overlap(a: ComponentBoundingBox, b: ComponentBoundingBox, y_tol: u32) -> bool {
    let a_mid = (a.top + a.bottom) / 2;
    let b_mid = (b.top + b.bottom) / 2;
    if a_mid.abs_diff(b_mid) <= y_tol {
        return true;
    }
    let overlap = a.bottom.min(b.bottom).saturating_sub(a.top.max(b.top));
    let min_h = a.height().min(b.height()).max(1);
    overlap * 2 >= min_h
}

fn otsu_foreground(gray: &GrayImage) -> GrayImage {
    let mut histogram = [0u32; 256];
    for p in gray.pixels() {
        histogram[p[0] as usize] += 1;
    }
    let total = gray.width() * gray.height();
    let mut sum = 0u64;
    for (i, &c) in histogram.iter().enumerate() {
        sum += i as u64 * c as u64;
    }
    let mut sum_b = 0u64;
    let mut w_b = 0u32;
    let mut best_var = 0f64;
    let mut threshold = 128u8;
    for (i, &c) in histogram.iter().enumerate() {
        w_b += c;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += i as u64 * c as u64;
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let var = w_b as f64 * w_f as f64 * (m_b - m_f).powi(2);
        if var > best_var {
            best_var = var;
            threshold = i as u8;
        }
    }

    let mut dark = 0u64;
    let mut light = 0u64;
    for p in gray.pixels() {
        if p[0] <= threshold {
            dark += 1;
        } else {
            light += 1;
        }
    }
    let text_is_dark = dark <= light;

    let mut out = GrayImage::new(gray.width(), gray.height());
    for (x, y, p) in gray.enumerate_pixels() {
        let is_text = if text_is_dark {
            p[0] <= threshold
        } else {
            p[0] >= threshold
        };
        // `connected_components_*` treats pixels < 128 as foreground.
        out.put_pixel(x, y, Luma([if is_text { 0 } else { 255 }]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image::{ImageFormat, OcrImage};
    use image::{DynamicImage, Rgb, RgbImage};

    fn white_page(w: u32, h: u32) -> OcrImage {
        let img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
        OcrImage {
            data: DynamicImage::ImageRgb8(img),
            format: ImageFormat::Rgb,
            width: w,
            height: h,
            dpi: 300,
            metadata: Default::default(),
        }
    }

    fn fill_rect(page: &mut OcrImage, left: u32, top: u32, w: u32, h: u32) {
        if let DynamicImage::ImageRgb8(ref mut img) = page.data {
            for y in top..top + h {
                for x in left..left + w {
                    if x < img.width() && y < img.height() {
                        img.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            }
        }
    }

    #[test]
    fn latin_word_many_strokes_high_merge_ratio() {
        let mut page = white_page(200, 80);
        let top = 20;
        let h = 40;
        // Simulate "hello" as five narrow vertical strokes close together.
        for i in 0..5 {
            fill_rect(&mut page, 20 + i * 8, top, 3, h);
        }
        for i in 0..5 {
            fill_rect(&mut page, 70 + i * 8, top, 3, h);
        }
        let g = geometry_script_guess(&page);
        assert_eq!(g.script, Script::Latin);
        assert_eq!(g.language, "en");
    }

    #[test]
    fn cjk_square_glyphs_low_merge_ratio() {
        let mut page = white_page(200, 80);
        let top = 18;
        let size = 44;
        for i in 0..4 {
            fill_rect(&mut page, 20 + i * (size + 10), top, size, size);
        }
        let g = geometry_script_guess(&page);
        assert_eq!(g.script, Script::CJK);
        assert_eq!(g.language, "zh");
    }

    #[test]
    fn detect_auto_prefers_geometry_when_text_empty() {
        let mut page = white_page(200, 80);
        fill_rect(&mut page, 30, 20, 40, 40);
        let g = detect_auto(&page, "   ");
        assert_eq!(g.script, Script::CJK);
    }
}
