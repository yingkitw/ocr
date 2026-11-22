//! Text and background color extraction for OCR results.
//!
//! Recognition runs on a grayscale copy of the page, but the RGB source
//! still carries the ink and paper colors. For each word box the crop's
//! pixels are split at the crop's own Otsu luminance threshold; the
//! *minority* cluster is the ink (text strokes cover well under half of a
//! word box, even in white-on-dark badges), and each cluster reports its
//! per-channel median — robust against anti-aliased edges, which mix ink
//! and background at the rim and would drag a mean toward the middle.

use image::RgbImage;

/// Per-region text and background colors (sRGB triplets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionColors {
    /// Median color of the ink cluster — the glyph pixels.
    pub text: [u8; 3],
    /// Median color of the majority cluster — the paper/background.
    pub background: [u8; 3],
}

/// Sample text and background colors inside a bounding box of `rgb`.
///
/// Coordinates are clamped to the image. Returns `None` when the box is
/// degenerate (<2 px on a side) or the minority cluster holds under 2% of
/// the crop — a box that thin is a segmentation artifact, not text.
pub fn sample_region_colors(
    rgb: &RgbImage,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> Option<RegionColors> {
    let (w, h) = rgb.dimensions();
    let x0 = x0.min(w);
    let y0 = y0.min(h);
    let x1 = x1.min(w).max(x0);
    let y1 = y1.min(h).max(y0);
    if x1 - x0 < 2 || y1 - y0 < 2 {
        return None;
    }
    let n = ((x1 - x0) * (y1 - y0)) as usize;
    let mut lums = Vec::with_capacity(n);
    let mut px = Vec::with_capacity(n);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = *rgb.get_pixel(x, y);
            lums.push(((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8);
            px.push(p.0);
        }
    }
    // `otsu_threshold` returns the top bin of the dark cluster (the first
    // variance maximum), so the boundary value belongs to the dark side.
    let thr = crate::webocr::binarize::otsu_threshold(&lums);
    let dark = lums.iter().filter(|&&l| l <= thr).count();
    let bright = lums.len() - dark;
    // Ink is the minority cluster: text covers a fraction of its box,
    // whichever side of the threshold that lands on.
    let ink_is_dark = dark <= bright;
    let mut ink = Vec::new();
    let mut bg = Vec::new();
    for (&l, p) in lums.iter().zip(px.iter()) {
        if (l <= thr) == ink_is_dark {
            ink.push(*p);
        } else {
            bg.push(*p);
        }
    }
    if ink.len() < 3 || ink.len() * 50 < px.len() {
        return None;
    }
    Some(RegionColors {
        text: median_rgb(&ink),
        background: median_rgb(&bg),
    })
}

/// Per-channel median — much more robust than a mean when the cluster
/// still contains blended edge pixels.
fn median_rgb(px: &[[u8; 3]]) -> [u8; 3] {
    fn med(px: &[[u8; 3]], ch: usize) -> u8 {
        let mut v: Vec<u8> = px.iter().map(|p| p[ch]).collect();
        v.sort_unstable();
        v[v.len() / 2]
    }
    [med(px, 0), med(px, 1), med(px, 2)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, image::Rgb(rgb))
    }

    fn paint(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, rgb: [u8; 3]) {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, image::Rgb(rgb));
            }
        }
    }

    #[test]
    fn red_text_on_white_reports_red_on_white() {
        // "Text" drawn as an ink strip covering ~20% of the box.
        let mut img = solid(100, 40, [255, 255, 255]);
        paint(&mut img, 10, 10, 70, 30, [200, 20, 20]);
        let c = sample_region_colors(&img, 0, 0, 100, 40).unwrap();
        assert_eq!(c.text, [200, 20, 20]);
        assert_eq!(c.background, [255, 255, 255]);
    }

    #[test]
    fn white_text_on_navy_reports_white_on_navy() {
        // Minority-cluster rule: bright ink on a dark field still reads
        // correctly without a page-polarity hint.
        let mut img = solid(100, 40, [10, 10, 60]);
        paint(&mut img, 20, 12, 55, 30, [240, 240, 240]);
        let c = sample_region_colors(&img, 0, 0, 100, 40).unwrap();
        assert_eq!(c.text, [240, 240, 240]);
        assert_eq!(c.background, [10, 10, 60]);
    }

    #[test]
    fn empty_box_and_no_ink_return_none() {
        let img = solid(20, 20, [128, 128, 128]);
        assert!(sample_region_colors(&img, 0, 0, 1, 10).is_none()); // 1 px wide
        assert!(sample_region_colors(&img, 0, 0, 20, 20).is_none()); // uniform
    }

    #[test]
    fn aa_rim_does_not_drag_the_median() {
        // Black strokes with a mid-gray AA rim on white: the median of the
        // ink cluster must stay black even though mean would land on gray.
        let mut img = solid(60, 30, [255, 255, 255]);
        paint(&mut img, 8, 8, 40, 22, [140, 140, 140]); // rim
        paint(&mut img, 10, 10, 36, 20, [0, 0, 0]); // core
        let c = sample_region_colors(&img, 0, 0, 60, 30).unwrap();
        assert!(c.text.iter().all(|&v| v < 80), "ink too light: {:?}", c.text);
        assert!(c.background.iter().all(|&v| v > 200));
    }
}
