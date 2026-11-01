//! Glyph normalization shared by templates and samples (WebOCR port).
//!
//! Port of `webocr/src/ocr/normalize.ts`. The contract: a glyph is scaled
//! uniformly so its own height becomes `SCALE_REF` pixels (upscale capped so
//! a period never inflates into a bar), aspect ratio is preserved, and the
//! result is centered in a `FRAME`×`FRAME` field with the line baseline
//! anchored at row `baseline_row_for(frame)`.
//!
//! The rule depends only on the glyph itself — no line statistics — so a
//! template and a sample of the same character always land identically.
//! Aspect ratio carries the size information: "o" fills its frame wider than
//! "O", a period stays a small blob while "I" is a full-height bar.

use super::components::median;
use super::types::{BinaryImage, Comp};

/// Latin normalization height (px).
pub const SCALE_REF: f32 = 20.0;
/// CJK normalization height — sweep-calibrated on WebOCR fixtures (24 lifts
/// ja 89.7→100 %; 26 regresses; 32 clips inside the 30-frame). Kept for the
/// future CJK library.
pub const CJK_SCALE_REF: f32 = 24.0;
/// Maximum upscale factor: a 3 px period must not blow up into a 20 px bar.
pub const MAX_UPSCALE: f32 = 2.2;
/// Normalized frame side.
pub const FRAME: usize = 30;

/// Frame row where the line baseline lands after normalization.
pub fn baseline_row_for(frame: usize) -> f32 {
    (frame * 2).div_ceil(3) as f32
}

/// Glyph normalization height: `min(MAX_UPSCALE, ref / height)`.
pub fn glyph_scale(g: &Comp, reference: f32) -> f32 {
    let h = g.height() as f32;
    (reference / h.max(1.0)).min(MAX_UPSCALE)
}

/// Resample a glyph from the ink mask into its normalized frame using
/// area averaging (sub-pixel accurate in both directions). The baseline is
/// anchored at `baseline_row_for(frame)`, so punctuation hangs below it and
/// quotes sit above it — vertical position survives normalization.
pub fn place_glyph(
    bin: &BinaryImage,
    g: &Comp,
    scale: f32,
    baseline_y: f32,
    frame: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; frame * frame];
    let target_w = g.width() as f32 * scale;
    let target_h = g.height() as f32 * scale;
    let ox = ((frame as f32 - target_w) / 2.0).max(0.0);
    let desired = baseline_row_for(frame) - (baseline_y - g.y0 as f32) * scale;
    let max_oy = (frame as f32 - target_h).max(0.0);
    let oy = desired.round().clamp(0.0, max_oy);

    for ty in 0..frame {
        let sy0 = g.y0 as f32 + (ty as f32 - oy) / scale;
        let sy1 = g.y0 as f32 + (ty as f32 + 1.0 - oy) / scale;
        if sy1 <= g.y0 as f32 || sy0 >= g.y1 as f32 {
            continue;
        }
        for tx in 0..frame {
            let sx0 = g.x0 as f32 + (tx as f32 - ox) / scale;
            let sx1 = g.x0 as f32 + (tx as f32 + 1.0 - ox) / scale;
            if sx1 <= g.x0 as f32 || sx0 >= g.x1 as f32 {
                continue;
            }

            let mut sum = 0.0f32;
            let mut weight = 0.0f32;
            let y_a = (g.y0 as f32).max(sy0.floor()) as usize;
            let y_b = (g.y1 as f32).min(sy1.ceil()) as usize;
            let x_a = (g.x0 as f32).max(sx0.floor()) as usize;
            let x_b = (g.x1 as f32).min(sx1.ceil()) as usize;
            for y in y_a..y_b.min(g.y1) {
                let wy = sy1.min(y as f32 + 1.0) - sy0.max(y as f32);
                if wy <= 0.0 {
                    continue;
                }
                for x in x_a..x_b.min(g.x1) {
                    let wx = sx1.min(x as f32 + 1.0) - sx0.max(x as f32);
                    if wx <= 0.0 {
                        continue;
                    }
                    sum += bin.get(x, y) as f32 * wx * wy;
                    weight += wx * wy;
                }
            }
            out[ty * frame + tx] = if weight > 0.0 { sum / weight } else { 0.0 };
        }
    }
    out
}

/// Separable 1-2-1 blur. Softens aliasing so near-misses still score well.
pub fn blur3(src: &[f32], grid: usize) -> Vec<f32> {
    let last = grid - 1;
    let mut tmp = vec![0.0f32; grid * grid];
    for y in 0..grid {
        for x in 0..grid {
            let c = src[y * grid + x];
            let l = src[y * grid + if x == 0 { 0 } else { x - 1 }];
            let r = src[y * grid + if x == last { last } else { x + 1 }];
            tmp[y * grid + x] = (l + 2.0 * c + r) / 4.0;
        }
    }
    let mut out = vec![0.0f32; grid * grid];
    for y in 0..grid {
        for x in 0..grid {
            let c = tmp[y * grid + x];
            let u = tmp[if y == 0 { 0 } else { y - 1 } * grid + x];
            let d = tmp[if y == last { last } else { y + 1 } * grid + x];
            out[y * grid + x] = (u + 2.0 * c + d) / 4.0;
        }
    }
    out
}

/// Estimate the line baseline as the median glyph bottom — most glyphs sit
/// on it, so the median is robust even with descenders.
pub fn line_baseline(glyphs: &[Comp]) -> f32 {
    let bottoms: Vec<usize> = glyphs.iter().map(|g| g.y1).collect();
    median(&bottoms) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_row_generalizes_the_old_constant() {
        assert_eq!(baseline_row_for(30) as usize, 20); // the old BASELINE_ROW
        assert_eq!(baseline_row_for(15) as usize, 10);
    }

    #[test]
    fn upscale_is_clamped_so_a_period_stays_small() {
        // 3×3 dot
        let dot = Comp { x0: 0, y0: 0, x1: 3, y1: 3, area: 9 };
        assert!((glyph_scale(&dot, SCALE_REF) - MAX_UPSCALE).abs() < 1e-6);
        // 40 px tall glyph scales below 1
        let tall = Comp { x0: 0, y0: 0, x1: 10, y1: 40, area: 1 };
        assert!((glyph_scale(&tall, SCALE_REF) - 0.5).abs() < 1e-6);
    }

    fn frame_bin(w: usize, h: usize, rect: Comp) -> BinaryImage {
        let mut data = vec![0u8; w * h];
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                data[y * w + x] = 1;
            }
        }
        BinaryImage { width: w, height: h, data }
    }

    #[test]
    fn place_glyph_anchors_baseline_and_preserves_aspect() {
        // 10×23 bar whose baseline (y=25) cuts through it: 20 rows of the
        // bar sit above the baseline, 3 hang below (descender-like).
        let g = Comp { x0: 10, y0: 5, x1: 20, y1: 28, area: 230 };
        let bin = frame_bin(40, 40, g);
        let vec = place_glyph(&bin, &g, 1.0, 25.0, FRAME);
        // rows 0..=19 are above the baseline; the bar must END at the
        // baseline row 20 and continue 3 rows past it (hang below).
        let inked_rows: Vec<usize> = (0..FRAME)
            .filter(|&y| (0..FRAME).any(|x| vec[y * FRAME + x] > 0.5))
            .collect();
        assert_eq!(inked_rows.first(), Some(&0));
        assert_eq!(inked_rows.last(), Some(&22), "3 rows hang below the baseline row 20");
        assert!(inked_rows.contains(&20), "baseline row itself is inked");
        // horizontal centering: ink starts near the frame center minus half width
        let first_ink_x = (0..FRAME).find(|&x| vec[10 * FRAME + x] > 0.5).unwrap();
        assert!(first_ink_x >= 8 && first_ink_x <= 12, "x0 = {first_ink_x}");
    }

    #[test]
    fn blur3_preserves_a_constant_field_and_smooths_edges() {
        let frame = 6;
        let flat = vec![1.0f32; frame * frame];
        let blurred = blur3(&flat, frame);
        assert!(blurred.iter().all(|&v| (v - 1.0).abs() < 1e-6));

        let mut spike = vec![0.0f32; frame * frame];
        spike[3 * frame + 3] = 1.0;
        let b = blur3(&spike, frame);
        assert!(b[3 * frame + 3] < 1.0, "center mass spreads to neighbours");
        assert!(b[3 * frame + 2] > 0.0 && b[2 * frame + 3] > 0.0);
    }
}
