//! Image-geometry script detection (WebOCR `detect.ts`).
//!
//! Reuses the webocr module's own binarize + CC pipeline (the purity
//! boundary stays intact — no external types). Returns Latin or CJK
//! based on the mergeRatio between raw connected components and
//! clustered glyph units, calibrated on WebOCR's real-font fixture
//! set (see `MEMORY.md §13`):
//!
//! - Latin ≈ 0.84–1.04 (each letter is one component; the 1.25·medH
//!   width cap in `cluster_components` stops letter pairs from
//!   chaining — the same cap that guards clustering becomes the
//!   detector).
//! - CJK ≈ 0.34–0.61 (disconnected strokes cluster into glyph boxes,
//!   halving the count).
//!
//! The 0.65 threshold is WebOCR's measured cutoff; below 5 glyph
//! units we fall back to squareness (CJK chars are typically square
//! box-shaped at the line's median height; Latin ascenders/descenders
//! are not).

use super::binarize::binarize;
use super::components::connected_components;
use super::segment::{extract_lines, filter_noise};
use super::types::{Comp, GrayImage};

/// Threshold: `mergeRatio = clustered_units / raw_components`. Above
/// this value the page is Latin; below it the page is CJK (WebOCR
/// `detect.ts` `MERGE_RATIO_LATIN_THRESHOLD = 0.65`).
const MERGE_RATIO_LATIN_THRESHOLD: f32 = 0.65;
/// Below this many glyph units the mergeRatio is too noisy; fall
/// back to squareness (WebOCR `MIN_UNITS = 5`).
const MIN_UNITS: usize = 5;

/// Resolved script id (WebOCR `ScriptId`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptId {
    Latin,
    Cjk,
}

/// Resolve a grayscale image to Latin or CJK based on glyph geometry.
pub fn resolve_script(gray: &GrayImage) -> ScriptId {
    let bin = binarize(gray).image;
    let comps = filter_noise(&connected_components(&bin));
    if comps.is_empty() {
        return ScriptId::Latin;
    }
    let med_h = median_height(&comps);
    if med_h < 2 {
        return ScriptId::Latin;
    }
    // CJK-mode line segmentation: cluster_components runs and merges
    // horizontally adjacent stroke components. For Latin letters this
    // is a no-op (the 1.25·medH width cap blocks chaining); for CJK
    // the count drops sharply.
    let lines = extract_lines(&bin, &comps, true);
    let raw_count = comps.len().max(1) as f32;
    let mut unit_count = 0usize;
    let mut square_count = 0usize;
    for line in &lines {
        for g in &line.glyphs {
            // Filter punctuation specks and small artifacts.
            if g.height() < (med_h as f32 * 0.45) as usize {
                continue;
            }
            unit_count += 1;
            let w = g.width() as f32;
            let h = g.height() as f32;
            let aspect = w / h.max(1.0);
            // Squareness: CJK hanzi boxes are ≈ 1:1; Latin letter
            // boxes are typically < 0.7 (narrow) or > 1.45 (wide 'm').
            if (0.7..=1.45).contains(&aspect) && h >= med_h as f32 * 0.78 {
                square_count += 1;
            }
        }
    }

    if unit_count < MIN_UNITS {
        // Short samples can't merge either way. Prefer CJK when the
        // page is squareness-dominant (CJK charset is the forgiving
        // side — it embeds full ASCII).
        let square_frac = if unit_count > 0 {
            square_count as f32 / unit_count as f32
        } else {
            0.0
        };
        return if square_frac > 0.5 {
            ScriptId::Cjk
        } else {
            ScriptId::Latin
        };
    }

    let ratio = unit_count as f32 / raw_count;
    if ratio > MERGE_RATIO_LATIN_THRESHOLD {
        ScriptId::Latin
    } else {
        ScriptId::Cjk
    }
}

fn median_height(comps: &[Comp]) -> usize {
    if comps.is_empty() {
        return 0;
    }
    let mut heights: Vec<usize> = comps.iter().map(|c| c.height()).collect();
    heights.sort_unstable();
    heights[heights.len() / 2].max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(width: usize, height: usize, fill: u8) -> GrayImage {
        GrayImage {
            width,
            height,
            data: vec![fill; width * height],
        }
    }

    /// Latin page fixture: a row of narrow 6×16 ink boxes (letters).
    /// 8 letters across, each a single component → mergeRatio ≈ 1.0.
    #[test]
    fn latin_letters_resolve_to_latin() {
        let mut g = gray(200, 30, 255);
        for col in 0..8 {
            let x0 = 8 + col * 24;
            for y in 10..26 {
                for x in x0..x0 + 6 {
                    g.data[y * 200 + x] = 0;
                }
            }
        }
        assert_eq!(resolve_script(&g), ScriptId::Latin);
    }

    /// CJK page fixture: 4 hanzi-shaped glyphs, each made of 3
    /// detached vertical strokes (close enough to chain under the
    /// 1.25·medH + 0.35·medH cluster rules). The 12 raw components
    /// collapse to 4 clustered units → mergeRatio = 4/12 = 0.33, below
    /// the 0.65 Latin threshold → CJK.
    ///
    /// This deliberately bypasses the "render CJK with rusttype"
    /// path: the detector is purely geometric (it looks at CC
    /// clusters, not character semantics), so we test the
    /// geometry it actually inspects.
    #[test]
    fn cjk_detached_strokes_resolve_to_cjk() {
        let w = 200;
        let h = 30;
        let mut g = gray(w, h, 255);
        // 4 hanzi, each made of 3 vertical strokes 2px wide × 8px tall
        // (width=2 satisfies `filter_noise` and gives mergeRatio a
        // clear signal). Strokes within a hanzi are 2px apart (within
        // the 0.35·medH cluster gap); hanzi are 20px apart (above it).
        for hanzi in 0..4 {
            let x0 = 8 + hanzi * 24;
            for stroke in 0..3 {
                let x = x0 + stroke * 4; // 2px stroke, 2px gap
                for y in 10..20 {
                    for dx in 0..2 {
                        g.data[y * w + x + dx] = 0;
                    }
                }
            }
        }
        assert_eq!(resolve_script(&g), ScriptId::Cjk);
    }

    /// Empty / uniform pages default to Latin.
    #[test]
    fn empty_page_defaults_to_latin() {
        let g = gray(64, 64, 255);
        assert_eq!(resolve_script(&g), ScriptId::Latin);
    }
}
