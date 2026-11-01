//! Otsu binarization with polarity detection and halo bias (WebOCR port).
//!
//! Faithful port of `webocr/src/ocr/binarize.ts`. The input here is already
//! grayscale (the crate decodes images to luma before dispatch), so the
//! RGBA luma pass lives at the integration boundary instead.

use super::types::{BinaryImage, GrayImage};

/// Otsu's method: exhaustively pick the threshold that maximizes
/// between-class variance. One pass over a 256-bin histogram.
pub fn otsu_threshold(gray: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &g in gray {
        hist[g as usize] += 1;
    }

    let total = gray.len() as u64;
    let mut sum: u64 = 0;
    for (t, &h) in hist.iter().enumerate() {
        sum += (t as u64) * (h as u64);
    }

    let mut sum_b: u64 = 0;
    let mut w_b: u64 = 0;
    let mut best: f64 = -1.0;
    let mut threshold: u8 = 127;
    for (t, &h) in hist.iter().enumerate() {
        w_b += h as u64;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += (t as u64) * (h as u64);
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        // wB·wF·(mB−mF)² in f64 — the u64 product can overflow on huge images.
        let between = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            threshold = t as u8;
        }
    }
    threshold
}

/// Result of binarization (WebOCR `BinarizeResult`).
#[derive(Debug, Clone)]
pub struct BinarizeResult {
    pub image: BinaryImage,
    /// The applied threshold.
    pub threshold: u8,
    /// true when the source was light ink on dark paper.
    pub inverted: bool,
}

/// Grayscale → Otsu threshold → 1-bit ink mask.
///
/// Polarity is detected automatically: documents are ink-on-paper, so if
/// "dark" pixels are the clear majority we flip and treat light pixels as
/// ink. The 0.55 margin (not 0.5) avoids flipping on photo-gray images
/// where the split is ambiguous.
///
/// Light ink on dark paper: anti-aliasing leaves a bright halo that fattens
/// strokes and closes counters (an "s" starts reading as "S"). Bias the
/// threshold toward the light end so only near-ink pixels count.
pub fn binarize(gray: &GrayImage) -> BinarizeResult {
    binarize_with_threshold(gray, None)
}

/// Grayscale → 1-bit ink mask, with an optional **forced** threshold.
///
/// The forced threshold is the way to give a bitmap the threshold of the
/// CONTEXT it stands for. Otsu reads the histogram it is given, and a single
/// glyph's histogram is not a page's: measured on a 16px Trebuchet "i", the
/// threshold of the glyph alone is 88 while a page of the same type gives
/// 167. The template is then fatter than the sample it is supposed to
/// mirror (its stem and its dot merge), and the equal-footing rule breaks
/// exactly where the ink is thinnest. See `templates::page_threshold`.
pub fn binarize_with_threshold(gray: &GrayImage, forced: Option<u8>) -> BinarizeResult {
    let otsu = forced.unwrap_or_else(|| otsu_threshold(&gray.data));

    let dark = gray.data.iter().filter(|&&g| g <= otsu).count();
    let inverted = dark > gray.data.len() * 55 / 100;

    let threshold: u8 = if inverted {
        // t' = t + 0.25·(255 − t), clamped so near-white pixels become ink.
        let t = otsu as u32 + (255 - otsu as u32) / 4;
        t.min(250) as u8
    } else {
        otsu
    };

    let data = gray
        .data
        .iter()
        .map(|&g| u8::from(if inverted { g > threshold } else { g <= threshold }))
        .collect();

    BinarizeResult {
        image: BinaryImage {
            width: gray.width,
            height: gray.height,
            data,
        },
        threshold,
        inverted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: usize, h: usize, fill: u8) -> GrayImage {
        GrayImage {
            width: w,
            height: h,
            data: vec![fill; w * h],
        }
    }

    #[test]
    fn otsu_splits_bimodal_histogram() {
        let mut g = gray(10, 10, 240);
        for y in 0..3 {
            for x in 0..10 {
                g.data[y * 10 + x] = 20;
            }
        }
        let t = otsu_threshold(&g.data);
        // For a perfectly bimodal image every threshold between the modes
        // maximizes between-class variance; ties keep the first, so the
        // result is the dark mode itself. It must never land ON a mode's
        // far side.
        assert!((20..=240).contains(&t), "threshold {t} must sit within the modes");
        let r = binarize(&g);
        assert_eq!(r.image.data.iter().filter(|&&v| v == 1).count(), 30);
    }

    #[test]
    fn dark_on_light_is_normal_polarity() {
        let mut g = gray(20, 20, 255);
        for y in 5..10 {
            for x in 5..10 {
                g.data[y * 20 + x] = 10;
            }
        }
        let r = binarize(&g);
        assert!(!r.inverted);
        assert_eq!(r.image.get(7, 7), 1);
        assert_eq!(r.image.get(0, 0), 0);
    }

    /// WebOCR halo-bias fixture: light text on dark ground. Plain Otsu would
    /// count halo pixels as ink; the biased threshold must flip polarity and
    /// keep only near-white pixels.
    #[test]
    fn light_on_dark_flips_and_biases_threshold() {
        let mut g = gray(30, 30, 25); // dark paper
        // light strokes
        for x in 4..26 {
            g.data[10 * 30 + x] = 250;
            g.data[20 * 30 + x] = 250;
        }
        let r = binarize(&g);
        assert!(r.inverted);
        assert_eq!(r.image.get(10, 10), 1, "stroke pixel (row 10) is ink");
        assert_eq!(r.image.get(10, 20), 1, "stroke pixel (row 20) is ink");
        assert_eq!(r.image.get(2, 2), 0, "paper pixel is background");
        // threshold biased toward the light end (≥ otsu + (255-otsu)/4 − 1)
        let plain_otsu = otsu_threshold(&g.data);
        let expected = (plain_otsu as u32 + (255 - plain_otsu as u32) / 4).min(250) as u8;
        assert_eq!(r.threshold, expected);
    }

    #[test]
    fn uniform_image_does_not_flip() {
        let g = gray(8, 8, 128);
        let r = binarize(&g);
        assert!(!r.inverted);
    }
}
