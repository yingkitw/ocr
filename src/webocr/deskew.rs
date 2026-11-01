//! Skew estimation and correction (WebOCR port of `webocr/src/ocr/deskew.ts`).
//!
//! Estimates the skew angle of an ink mask by a coarse-to-fine
//! projection-profile search, then rotates the mask to straight.
//!
//! The signal is the **row** projection, not the column one. A page of text
//! is a stack of horizontal LINES: when the page is straight, the ink per
//! row alternates sharply between "line" and "gap", and a 1-4° skew smears
//! every line into its neighbours and flattens that profile. The column
//! projection (letter stems) looks nearly identical at every angle on real
//! text — measured variance gain 1.02-1.08 for a 4° skew, far below the
//! 1.3 gate, so a column-projection deskew never corrects a real scan. The
//! row projection measures 1.000 on straight pages and 3.9-9.9 on skewed
//! ones — a clean separation, and the angle it picks is the correct one.
//!
//! Decision rule (WebOCR, measured on Arial/Georgia/Courier/Verdana/
//! Trebuchet at 24px and a 32px Arial test page rotated 0/±1/±3/±5/±7/±10°):
//! - clean pages: `var_best / var0` ∈ [1.00, 1.18] — fonts whose tall
//!   ascenders or small descenders create a tiny intrinsic bias (Georgia
//!   1.14, Verdana 1.17, Trebuchet 1.06) — close to 1.0 means "already
//!   straight".
//! - rotated pages: `var_best / var0` ∈ [1.92, 2.47] for ≥1° of skew.
//!
//! Threshold 1.3 separates the two cleanly with margin. Below it the input
//! is returned unchanged — no rotation, no growth of the bounding box, no
//! cost beyond the search.

use super::types::BinaryImage;

/// Variance ratio below this means the page is already straight.
const STRAIGHT_RATIO: f64 = 1.3;

/// The result of a deskew pass.
#[derive(Debug, Clone)]
pub struct DeskewResult {
    /// The (possibly rotated) ink mask — same as the input if no rotation
    /// was needed.
    pub bin: BinaryImage,
    /// The angle that was applied, in degrees (0 when nothing changed).
    pub angle: f32,
    /// The size the mask had **before** the rotation, so the caller can map
    /// the deskewed geometry back to the original image.
    pub original_width: usize,
    pub original_height: usize,
}

/// Downsample an ink mask with a majority rule per output pixel.
fn downsample_bin(src: &BinaryImage, max_w: usize) -> BinaryImage {
    let (sw, sh) = (src.width, src.height);
    let nw = max_w.min(sw);
    let nh = ((sh as f64 * nw as f64 / sw as f64).round() as usize).max(1);
    let mut data = vec![0u8; nw * nh];
    let x_ratio = sw as f64 / nw as f64;
    let y_ratio = sh as f64 / nh as f64;
    for y in 0..nh {
        for x in 0..nw {
            let sx0 = (x as f64 * x_ratio).floor() as usize;
            let sx1 = ((x as f64 + 1.0) * x_ratio).ceil() as usize;
            let sy0 = (y as f64 * y_ratio).floor() as usize;
            let sy1 = ((y as f64 + 1.0) * y_ratio).ceil() as usize;
            let mut sum = 0u32;
            let mut n = 0u32;
            for sy in sy0..sy1.min(sh) {
                for sx in sx0..sx1.min(sw) {
                    sum += src.get(sx, sy) as u32;
                    n += 1;
                }
            }
            data[y * nw + x] = u8::from(n > 0 && sum as f64 / n as f64 > 0.5);
        }
    }
    BinaryImage {
        width: nw,
        height: nh,
        data,
    }
}

/// Row sums of an ink mask — the deskew signal.
fn project_rows(bin: &BinaryImage) -> Vec<f64> {
    let mut proj = vec![0.0f64; bin.height];
    for y in 0..bin.height {
        let mut sum = 0.0f64;
        for x in 0..bin.width {
            sum += bin.get(x, y) as f64;
        }
        proj[y] = sum;
    }
    proj
}

/// Variance of a projection (a quick proxy for "peakiness").
fn variance(proj: &[f64]) -> f64 {
    let n = proj.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = proj.iter().sum::<f64>() / n;
    proj.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n
}

/// Rotate an ink mask counter-clockwise by `angle_deg`. Nearest-neighbour
/// sampling, exactly like WebOCR.
pub fn rotate_bin(src: &BinaryImage, angle_deg: f32) -> BinaryImage {
    let rad = (angle_deg as f64).to_radians();
    let (sin, cos) = rad.sin_cos();
    let (sw, sh) = (src.width, src.height);
    let cx = sw as f64 / 2.0;
    let cy = sh as f64 / 2.0;
    let nw = (sw as f64 * cos).abs() + (sh as f64 * sin).abs();
    let nh = (sw as f64 * sin).abs() + (sh as f64 * cos).abs();
    let nw = nw.ceil() as usize;
    let nh = nh.ceil() as usize;
    let mut data = vec![0u8; nw * nh];
    let ncx = nw as f64 / 2.0;
    let ncy = nh as f64 / 2.0;
    for y in 0..nh {
        for x in 0..nw {
            let dx = x as f64 - ncx;
            let dy = y as f64 - ncy;
            let sx = (dx * cos - dy * sin + cx).round() as isize;
            let sy = (dx * sin + dy * cos + cy).round() as isize;
            if sx >= 0 && sy >= 0 && (sx as usize) < sw && (sy as usize) < sh {
                data[y * nw + x] = src.get(sx as usize, sy as usize);
            }
        }
    }
    BinaryImage {
        width: nw,
        height: nh,
        data,
    }
}

/// Search the skew angle on a downsampled copy (60 % cheaper than a
/// full-resolution search). Returns the angle (rounded to 0.1°) whose
/// projection has the highest variance, plus the zero-angle variance.
fn search_angle(bin: &BinaryImage) -> (f32, f64, f64) {
    let small = downsample_bin(bin, 600);
    let mut best_angle = 0.0f32;
    let mut best_variance = 0.0f64;
    let mut var0 = 0.0f64;
    for a in -10..=10i32 {
        let r = rotate_bin(&small, a as f32);
        let v = variance(&project_rows(&r));
        if a == 0 {
            var0 = v;
        }
        if v > best_variance {
            best_variance = v;
            best_angle = a as f32;
        }
    }
    // Refine ±1° around the coarse winner in 0.1° steps.
    let coarse = best_angle;
    for i in -10..=10i32 {
        let a = coarse + i as f32 * 0.1;
        let r = rotate_bin(&small, a);
        let v = variance(&project_rows(&r));
        if v > best_variance {
            best_variance = v;
            best_angle = a;
        }
    }
    (best_angle, var0, best_variance)
}

/// Estimate and correct the skew of an ink mask.
///
/// Returns the (possibly rotated) mask, the angle that was applied, and the
/// mask's original size — the last so the caller can map the deskewed
/// geometry back into the original image's coordinates.
pub fn deskew(bin: &BinaryImage) -> DeskewResult {
    let original_width = bin.width;
    let original_height = bin.height;
    let unchanged = || DeskewResult {
        bin: bin.clone(),
        angle: 0.0,
        original_width,
        original_height,
    };
    if bin.width < 16 || bin.height < 16 {
        return unchanged();
    }
    let (angle, var0, var_best) = search_angle(bin);
    if var0 <= 0.0 || var_best / var0 < STRAIGHT_RATIO || angle.abs() < 0.2 {
        // Already straight — don't pay the rotation cost or grow the
        // bounding box.
        return unchanged();
    }
    DeskewResult {
        bin: rotate_bin(bin, angle),
        angle,
        original_width,
        original_height,
    }
}

/// Map a point from the deskewed frame back to the original image.
///
/// The rotation is a pure nearest-neighbour resample about the canvas
/// centre, and `rotate_bin` computes the **destination → source** mapping
/// directly, so the inverse of "rotate the page" is the same formula with
/// the two canvases' centres swapped.
pub fn map_point_back(
    angle_deg: f32,
    original: (usize, usize),
    deskewed: (usize, usize),
    x: f64,
    y: f64,
) -> (f64, f64) {
    let rad = (angle_deg as f64).to_radians();
    let (sin, cos) = rad.sin_cos();
    let (sw, sh) = original;
    let (nw, nh) = deskewed;
    let cx = sw as f64 / 2.0;
    let cy = sh as f64 / 2.0;
    let ncx = nw as f64 / 2.0;
    let ncy = nh as f64 / 2.0;
    let dx = x - ncx;
    let dy = y - ncy;
    (dx * cos - dy * sin + cx, dx * sin + dy * cos + cy)
}

impl DeskewResult {
    /// Map a box from the deskewed frame back to the original image's
    /// coordinates. The four corners are mapped and the bounding box of the
    /// results is returned, so the box stays axis-aligned.
    #[allow(clippy::type_complexity)]
    pub fn map_box_back(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> (f64, f64, f64, f64) {
        if self.angle == 0.0 {
            return (x0, y0, x1, y1);
        }
        let original = (self.original_width, self.original_height);
        let deskewed = (self.bin.width, self.bin.height);
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for (px, py) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let (sx, sy) = map_point_back(self.angle, original, deskewed, px, py);
            min_x = min_x.min(sx);
            min_y = min_y.min(sy);
            max_x = max_x.max(sx);
            max_y = max_y.max(sy);
        }
        (min_x, min_y, max_x, max_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straight_page_is_left_alone() {
        // A grid of horizontal bars — the row projection is already peaky.
        let (w, h) = (200usize, 60usize);
        let mut data = vec![0u8; w * h];
        for band in [5usize..12, 25usize..32, 45usize..52] {
            for y in band {
                for x in 5..195 {
                    data[y * w + x] = 1;
                }
            }
        }
        let bin = BinaryImage { width: w, height: h, data };
        let r = deskew(&bin);
        assert_eq!(r.angle, 0.0, "a straight page must not be rotated");
        assert_eq!(r.bin.width, w);
    }

    #[test]
    fn a_skewed_page_is_rotated_back() {
        // Build a straight page, rotate it by 4°, and check the deskew finds
        // the opposite angle.
        let (w, h) = (240usize, 80usize);
        let mut data = vec![0u8; w * h];
        for band in [10usize..18, 35usize..43, 60usize..68] {
            for y in band {
                for x in 10..230 {
                    data[y * w + x] = 1;
                }
            }
        }
        let straight = BinaryImage { width: w, height: h, data };
        let skewed = rotate_bin(&straight, 4.0);
        let r = deskew(&skewed);
        assert!(
            (r.angle + 4.0).abs() < 0.6,
            "expected about -4°, got {}",
            r.angle
        );
        assert_ne!(r.bin.width, skewed.width, "the mask was rotated");
    }

    #[test]
    fn rotate_bin_preserves_the_ink_mass() {
        let (w, h) = (40usize, 20usize);
        let mut data = vec![0u8; w * h];
        for y in 5..15 {
            for x in 5..35 {
                data[y * w + x] = 1;
            }
        }
        let bin = BinaryImage { width: w, height: h, data };
        let rotated = rotate_bin(&bin, 3.0);
        let before = bin.data.iter().filter(|&&v| v == 1).count();
        let after = rotated.data.iter().filter(|&&v| v == 1).count();
        let ratio = after as f64 / before as f64;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "nearest-neighbour rotation should keep the ink mass, got {ratio}"
        );
    }
}
