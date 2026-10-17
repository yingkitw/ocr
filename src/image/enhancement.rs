//! Image enhancement operations

use crate::core::image::OcrImage;
use crate::utils::{OcrError, Result, SimdImageOps};
use image::{imageops, DynamicImage, GrayImage, Luma};

/// Image enhancement operations
pub struct ImageEnhancer;

impl ImageEnhancer {
    /// Enhance image contrast
    pub fn enhance_contrast(img: &OcrImage, factor: f32) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let pixels: Vec<u8> = gray.pixels().map(|p| p[0]).collect();
        let adjusted = SimdImageOps::contrast_adjust(&pixels, factor);
        let mut result = GrayImage::new(gray.width(), gray.height());
        for (i, pixel) in adjusted.iter().enumerate() {
            let x = (i as u32) % gray.width();
            let y = (i as u32) / gray.width();
            result.put_pixel(x, y, Luma([*pixel]));
        }
        Ok(OcrImage::new(DynamicImage::ImageLuma8(result), img.dpi))
    }

    /// Reduce image noise (using Gaussian blur)
    pub fn reduce_noise(img: &OcrImage) -> Result<OcrImage> {
        let blurred = imageops::blur(&img.data, 0.5);
        Ok(OcrImage::new(DynamicImage::ImageRgba8(blurred), img.dpi))
    }

    /// Sharpen image
    pub fn sharpen(img: &OcrImage) -> Result<OcrImage> {
        let sharpened = img.data.clone();
        imageops::unsharpen(&sharpened, 1.0, 1);
        Ok(OcrImage::new(sharpened, img.dpi))
    }

    /// Deskew image by searching ±15° (handles typical scanner skew including ~10–15°).
    ///
    /// Coarse search at 1° steps, then refine ±1° at 0.1° around the best.
    /// Larger in-page rotations are handled by `OrientedCclDetector`.
    pub fn deskew(img: &OcrImage) -> Result<OcrImage> {
        let gray = img.to_grayscale();
        let binary = gray.threshold(200);

        let mut best_angle = 0.0f32;
        let mut max_variance = 0.0f64;

        for i in -15..=15 {
            let angle_rad = (i as f32).to_radians();
            let rotated = binary.rotate(angle_rad)?;
            let variance = Self::calculate_projection_variance(&rotated);
            if variance > max_variance {
                max_variance = variance;
                best_angle = angle_rad;
            }
        }

        let best_deg = best_angle.to_degrees();
        for i in -10..=10 {
            let angle_deg = best_deg + i as f32 * 0.1;
            let angle_rad = angle_deg.to_radians();
            let rotated = binary.rotate(angle_rad)?;
            let variance = Self::calculate_projection_variance(&rotated);
            if variance > max_variance {
                max_variance = variance;
                best_angle = angle_rad;
            }
        }

        if best_angle.abs() > 0.001 {
            img.rotate(best_angle).map_err(OcrError::from)
        } else {
            Ok(img.clone())
        }
    }

    /// Estimate DPI from image content by analyzing stroke widths
    ///
    /// Returns an estimated DPI. Falls back to the image's stored DPI if
    /// estimation is not possible. Typical printed text has strokes ~1pt wide,
    /// which at 300 DPI is ~4px.
    pub fn estimate_dpi(img: &OcrImage) -> u32 {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width < 50 || height < 50 {
            return img.dpi.max(72);
        }

        let mid_y = height / 2;
        let mut run_lengths = Vec::new();
        let mut current_run = 0u32;
        let mut in_dark = gray.get_pixel(0, mid_y)[0] < 128;

        for x in 0..width {
            let dark = gray.get_pixel(x, mid_y)[0] < 128;
            if dark == in_dark {
                current_run += 1;
            } else {
                if in_dark && current_run > 0 && current_run < width / 4 {
                    run_lengths.push(current_run);
                }
                current_run = 1;
                in_dark = dark;
            }
        }

        if run_lengths.len() < 3 {
            return img.dpi.max(72);
        }

        run_lengths.sort_unstable();
        let median_stroke = run_lengths[run_lengths.len() / 2];

        let estimated = (median_stroke as f32 * 72.0) as u32;
        estimated.clamp(72, 1200).max(img.dpi)
    }

    /// Orientation projection scores for the four rotations.
    ///
    /// Returns `[score_0, score_90, score_180, score_270]` — upright text has
    /// high horizontal-projection variance (distinct dark text-line bands).
    fn orientation_scores(img: &OcrImage) -> [f64; 4] {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width < 50 || height < 50 {
            return [0.0; 4];
        }

        let binary = Self::simple_threshold(&gray);

        [
            Self::orientation_score(&binary, width, height),
            Self::orientation_score(&Self::rotate_90_cw(&binary), height, width),
            Self::orientation_score(&Self::rotate_180(&binary), width, height),
            Self::orientation_score(&Self::rotate_90_ccw(&binary), height, width),
        ]
    }

    /// Detect image orientation (0, 90, 180, 270 degrees)
    ///
    /// Delegates to [`Self::detect_orientation_with_confidence`] and drops the
    /// confidence. Uses horizontal projection variance for the axis and a
    /// left-alignment cue for the polarity (see that method's docs).
    pub fn detect_orientation(img: &OcrImage) -> u32 {
        Self::detect_orientation_with_confidence(img).0
    }

    /// Detect image orientation with a confidence estimate.
    ///
    /// Returns `(orientation_degrees, confidence)` where confidence is in
    /// `0.0..=100.0`. Two-stage scoring, because per-row projection variance
    /// is invariant under 180° rotation (row sums are flip-symmetric), so a
    /// pure projection method can pick the text-line *axis* but not the
    /// polarity:
    ///
    /// 1. **Axis** (0/180 vs 90/270): horizontal-projection variance across
    ///    the four rotations (`orientation_scores`), mirroring Tesseract's
    ///    OSD confidence = gap between the best two orientation votes
    ///    (`ccmain/osdetect.cpp:51-71`), rescaled to `0..=100`.
    /// 2. **Polarity** (0 vs 180, 90 vs 270): vertical mass bias — where the
    ///    dark-pixel centroid sits inside the content band. The statistic
    ///    flips sign exactly under a 180° rotation, but on short or
    ///    symmetric lines it is noise, so a flip is applied **only** when
    ///    |bias| exceeds `POLARITY_EVIDENCE_MIN`; weaker evidence keeps the
    ///    lower angle. This mirrors Tesseract's "upside-down Latin text is
    ///    improbable" guard (`pagesegmain.cpp:390-405`): a wrong 180° flip
    ///    destroys recognition, so commission errors are avoided at the cost
    ///    of occasional missed corrections.
    pub fn detect_orientation_with_confidence(img: &OcrImage) -> (u32, f32) {
        /// Minimum |vertical mass bias| required to flip polarity.
        const POLARITY_EVIDENCE_MIN: f64 = 0.25;

        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();
        if width < 50 || height < 50 {
            return (0, 0.0);
        }
        let binary = Self::simple_threshold(&gray);

        let scores = Self::orientation_scores(img);
        let s_h = scores[0].max(scores[2]);
        let s_v = scores[1].max(scores[3]);

        // Stage 1: axis.
        let axis_conf = if s_h >= s_v {
            Self::normalized_margin(s_h, s_v)
        } else {
            Self::normalized_margin(s_v, s_h)
        };

        // Stage 2: polarity via vertical mass bias. Positive bias means the
        // candidate variant is upright, where the variant is the original
        // image for the horizontal axis and the CW-90° rotation for the
        // vertical axis — matching `correct_orientation`'s mapping
        // (90 → rotate(+π/2), 270 → rotate(−π/2)).
        let (orientation, pair_conf) = if s_h >= s_v {
            let b = Self::vertical_mass_bias(&binary, width, height);
            if b > POLARITY_EVIDENCE_MIN {
                (0, b)
            } else if b < -POLARITY_EVIDENCE_MIN {
                (180, -b)
            } else {
                (0, 0.0) // weak evidence: don't flip (improbability guard)
            }
        } else {
            let cw = Self::rotate_90_cw(&binary);
            let b = Self::vertical_mass_bias(&cw, height, width);
            if b > POLARITY_EVIDENCE_MIN {
                (90, b)
            } else if b < -POLARITY_EVIDENCE_MIN {
                (270, -b)
            } else {
                (90, 0.0)
            }
        };

        let confidence = ((axis_conf + pair_conf) * 0.5 * 100.0).clamp(0.0, 100.0);
        (orientation, confidence as f32)
    }

    /// Normalized top-1/top-2 margin in `0.0..=1.0` (0 when the best score
    /// is non-positive or the scores tie).
    fn normalized_margin(best: f64, second: f64) -> f64 {
        if best <= 0.0 {
            0.0
        } else {
            ((best - second) / best).clamp(0.0, 1.0)
        }
    }

    /// Vertical mass bias: where the dark-pixel centroid sits inside the
    /// content band, normalized to `−1..=1`. Positive = centroid above the
    /// band center (ascender/cap-heavy upright layout); negative = below.
    /// The sign flips exactly under a 180° rotation; ≈0 for symmetric or
    /// single-line content where polarity is genuinely ambiguous.
    fn vertical_mass_bias(binary: &GrayImage, width: u32, height: u32) -> f64 {
        let mut row_counts = vec![0u64; height as usize];
        for y in 0..height {
            let mut count = 0u64;
            for x in 0..width {
                if binary.get_pixel(x, y)[0] < 128 {
                    count += 1;
                }
            }
            row_counts[y as usize] = count;
        }
        let Some(first) = row_counts.iter().position(|&c| c > 0) else {
            return 0.0;
        };
        let last = row_counts.iter().rposition(|&c| c > 0).unwrap_or(first);
        if last <= first {
            return 0.0;
        }
        let mut mass = 0u64;
        let mut weighted = 0u64;
        for (i, &c) in row_counts.iter().enumerate().skip(first) {
            mass += c;
            weighted += c * (i as u64);
        }
        if mass == 0 {
            return 0.0;
        }
        let centroid = weighted as f64 / mass as f64;
        let center = (first + last) as f64 / 2.0;
        let half_band = (last - first) as f64 / 2.0;
        ((center - centroid) / half_band).clamp(-1.0, 1.0)
    }

    /// Rotate image to correct orientation based on detected text direction
    pub fn correct_orientation(img: &OcrImage) -> Result<OcrImage> {
        let orientation = Self::detect_orientation(img);

        match orientation {
            0 => Ok(img.clone()),
            90 => img
                .rotate(std::f32::consts::FRAC_PI_2)
                .map_err(OcrError::from),
            180 => img.rotate(std::f32::consts::PI).map_err(OcrError::from),
            270 => img
                .rotate(-std::f32::consts::FRAC_PI_2)
                .map_err(OcrError::from),
            _ => Ok(img.clone()),
        }
    }

    /// Remove an enclosing circle/ring around the page content (PSM 9,
    /// Tesseract `--psm circle_word`).
    ///
    /// Tesseract's `RemoveEnclosingCircle` (`pagesegmain.cpp:61-94`) erodes
    /// the binary image up to 8× so glyph strokes vanish before the ring,
    /// reconstructs the ring at full scale, and subtracts it. We use a
    /// simpler connected-component heuristic: find a large, sparse
    /// (ring-shaped) dark component whose bounding box spans most of the
    /// page and erase its pixels. Returns the image unchanged when no such
    /// ring is present. The returned image is grayscale.
    pub fn remove_enclosing_circle(img: &OcrImage) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();
        if width < 20 || height < 20 {
            return Ok(img.clone());
        }

        let binary = Self::simple_threshold(&gray);

        // Connected components over dark pixels (4-connectivity BFS).
        let mut label = vec![0u32; (width as usize) * (height as usize)];
        let mut bounds: Vec<(u32, u32, u32, u32, usize)> = Vec::new(); // min_x, min_y, max_x, max_y, count
        let mut next_label = 0u32;
        let mut queue = std::collections::VecDeque::new();
        for sy in 0..height {
            for sx in 0..width {
                let start = (sy * width + sx) as usize;
                if binary.get_pixel(sx, sy)[0] == 0 || label[start] != 0 {
                    continue;
                }
                next_label += 1;
                label[start] = next_label;
                queue.push_back((sx, sy));
                let mut min_x = sx;
                let mut min_y = sy;
                let mut max_x = sx;
                let mut max_y = sy;
                let mut count = 0usize;
                while let Some((x, y)) = queue.pop_front() {
                    count += 1;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    for (nx, ny) in [
                        (x.wrapping_sub(1), y),
                        (x + 1, y),
                        (x, y.wrapping_sub(1)),
                        (x, y + 1),
                    ] {
                        if nx >= width || ny >= height {
                            continue;
                        }
                        let idx = (ny * width + nx) as usize;
                        if binary.get_pixel(nx, ny)[0] == 0 && label[idx] == 0 {
                            label[idx] = next_label;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                bounds.push((min_x, min_y, max_x, max_y, count));
            }
        }

        // Ring candidate: spans ≥60% of both dimensions and is sparse
        // (a ring encloses mostly background; a solid blob would be dense).
        let ring_label = bounds
            .iter()
            .enumerate()
            .filter(|&(_, &(min_x, min_y, max_x, max_y, count))| {
                let bw = (max_x - min_x + 1) as f64;
                let bh = (max_y - min_y + 1) as f64;
                let fill = count as f64 / (bw * bh);
                bw >= 0.6 * width as f64 && bh >= 0.6 * height as f64 && fill <= 0.30 && count >= 100
            })
            .map(|(i, _)| i as u32 + 1)
            .next();

        let Some(rl) = ring_label else {
            return Ok(img.clone());
        };

        // Erase ring pixels (set to white) in a grayscale copy.
        let mut out = img.clone().to_grayscale();
        let mut gray_out = out.data.to_luma8();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if label[idx] == rl {
                    gray_out.put_pixel(x, y, image::Luma([255u8]));
                }
            }
        }
        out.data = image::DynamicImage::ImageLuma8(gray_out);
        Ok(out)
    }

    /// Remove document borders/margins
    ///
    /// Scans inward from each edge to find the first row/column containing
    /// dark pixels, then crops with a small margin.
    pub fn remove_borders(img: &OcrImage) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width < 20 || height < 20 {
            return Ok(img.clone());
        }

        let threshold = 200u8;
        let margin = 5u32;

        let mut left = margin;
        let mut right = width - margin - 1;
        let mut top = margin;
        let mut bottom = height - margin - 1;

        for x in margin..(width - margin) {
            let mut has_content = false;
            for y in margin..(height - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                left = x.saturating_sub(margin);
                break;
            }
        }

        for x in (margin..(width - margin)).rev() {
            let mut has_content = false;
            for y in margin..(height - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                right = (x + margin).min(width);
                break;
            }
        }

        for y in margin..(height - margin) {
            let mut has_content = false;
            for x in margin..(width - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                top = y.saturating_sub(margin);
                break;
            }
        }

        for y in (margin..(height - margin)).rev() {
            let mut has_content = false;
            for x in margin..(width - margin) {
                if gray.get_pixel(x, y)[0] < threshold {
                    has_content = true;
                    break;
                }
            }
            if has_content {
                bottom = (y + margin).min(height);
                break;
            }
        }

        let crop_width = right.saturating_sub(left);
        let crop_height = bottom.saturating_sub(top);

        if crop_width < 20 || crop_height < 20 || crop_width >= width || crop_height >= height {
            return Ok(img.clone());
        }

        img.crop(left, top, crop_width, crop_height)
            .map_err(OcrError::from)
    }

    /// Remove speckle noise (small connected components of dark pixels)
    ///
    /// Finds connected components of dark pixels using two-pass labeling
    /// and removes those whose area is smaller than `max_speckle_area`.
    pub fn remove_speckle(img: &OcrImage, max_speckle_area: u32) -> Result<OcrImage> {
        let gray = img.data.to_luma8();
        let (width, height) = gray.dimensions();

        if width == 0 || height == 0 {
            return Ok(img.clone());
        }

        let total_pixels = (width * height) as usize;
        let mut labels = vec![0u32; total_pixels];
        let mut next_label = 1u32;
        let mut areas = vec![0u32; total_pixels + 1];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if gray.get_pixel(x, y)[0] >= 128 {
                    continue;
                }

                let up = if y > 0 {
                    labels[((y - 1) * width + x) as usize]
                } else {
                    0
                };
                let left_label = if x > 0 {
                    labels[(y * width + x - 1) as usize]
                } else {
                    0
                };

                if up == 0 && left_label == 0 {
                    labels[idx] = next_label;
                    next_label += 1;
                } else if up > 0 && left_label == 0 {
                    labels[idx] = up;
                } else if left_label > 0 && up == 0 {
                    labels[idx] = left_label;
                } else {
                    labels[idx] = up.min(left_label);
                }
            }
        }

        for &label in &labels {
            if label > 0 && (label as usize) < areas.len() {
                areas[label as usize] += 1;
            }
        }

        let mut result = gray.clone();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let label = labels[idx];
                if label > 0
                    && (label as usize) < areas.len()
                    && areas[label as usize] < max_speckle_area
                {
                    result.put_pixel(x, y, Luma([255u8]));
                }
            }
        }

        Ok(OcrImage::new(DynamicImage::ImageLuma8(result), img.dpi))
    }

    fn simple_threshold(gray: &GrayImage) -> GrayImage {
        let (width, height) = gray.dimensions();

        let mut histogram = [0u32; 256];
        for pixel in gray.pixels() {
            histogram[pixel[0] as usize] += 1;
        }

        let total = width * height;
        let mut sum = 0u32;
        for (i, &count) in histogram.iter().enumerate() {
            sum += count * (i as u32);
        }

        let mut sum_b = 0u32;
        let mut w_b = 0u32;
        let mut max_variance = 0f64;
        let mut threshold = 128u8;

        for (i, &count) in histogram.iter().enumerate() {
            w_b += count;
            if w_b == 0 {
                continue;
            }
            let w_f = total - w_b;
            if w_f == 0 {
                break;
            }
            sum_b += (i as u32) * count;
            let m_b = sum_b as f64 / w_b as f64;
            let m_f = (sum - sum_b) as f64 / w_f as f64;
            let v = (w_b as f64) * (w_f as f64) * (m_b - m_f) * (m_b - m_f);
            if v > max_variance {
                max_variance = v;
                threshold = i as u8;
            }
        }

        let mut result = GrayImage::new(width, height);
        for (x, y, pixel) in gray.enumerate_pixels() {
            let value = if pixel[0] > threshold { 255u8 } else { 0u8 };
            result.put_pixel(x, y, Luma([value]));
        }
        result
    }

    fn orientation_score(binary: &GrayImage, width: u32, height: u32) -> f64 {
        let mut row_sums = Vec::with_capacity(height as usize);
        for y in 0..height {
            let mut sum = 0u64;
            for x in 0..width {
                if binary.get_pixel(x, y)[0] < 128 {
                    sum += 1;
                }
            }
            row_sums.push(sum as f64);
        }

        if row_sums.is_empty() {
            return 0.0;
        }

        let mean = row_sums.iter().sum::<f64>() / row_sums.len() as f64;
        

        row_sums
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>()
            / row_sums.len() as f64
    }

    fn rotate_90_cw(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        let mut result = GrayImage::new(height, width);
        for y in 0..height {
            for x in 0..width {
                let src_pixel = img.get_pixel(x, y);
                result.put_pixel(height - 1 - y, x, *src_pixel);
            }
        }
        result
    }

    fn rotate_90_ccw(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        let mut result = GrayImage::new(height, width);
        for y in 0..height {
            for x in 0..width {
                let src_pixel = img.get_pixel(x, y);
                result.put_pixel(y, width - 1 - x, *src_pixel);
            }
        }
        result
    }

    fn rotate_180(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        let mut result = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let src_pixel = img.get_pixel(x, y);
                result.put_pixel(width - 1 - x, height - 1 - y, *src_pixel);
            }
        }
        result
    }

    fn calculate_projection_variance(img: &OcrImage) -> f64 {
        let (width, height) = img.dimensions();
        let mut row_sums = Vec::with_capacity(height as usize);

        if let Some(buf) = img.data.as_luma8() {
            for y in 0..height {
                let mut sum = 0u64;
                for x in 0..width {
                    sum += buf.get_pixel(x, y)[0] as u64;
                }
                row_sums.push(sum as f64);
            }
        } else {
            return 0.0;
        }

        if row_sums.is_empty() {
            return 0.0;
        }

        let mean = row_sums.iter().sum::<f64>() / row_sums.len() as f64;
        

        row_sums
            .iter()
            .map(|s| {
                let diff = s - mean;
                diff * diff
            })
            .sum::<f64>()
            / row_sums.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three horizontal bars whose lengths shrink top-to-bottom (90/70/30):
    /// top-heavy like upright Latin text (cap/ascender mass above the band
    /// center), so the vertical-mass polarity cue is unambiguous.
    fn bars_image() -> OcrImage {
        let mut g = GrayImage::from_pixel(100, 100, Luma([255u8]));
        for &(y0, x1) in &[(20u32, 100u32), (45, 80), (70, 40)] {
            for y in y0..y0 + 5 {
                for x in 10..x1 {
                    g.put_pixel(x, y, Luma([0u8]));
                }
            }
        }
        OcrImage::new(DynamicImage::ImageLuma8(g), 300)
    }

    fn rotate180(img: &OcrImage) -> OcrImage {
        img.rotate(std::f32::consts::PI).unwrap()
    }

    #[test]
    fn test_detect_orientation_upright_and_upside_down() {
        let upright = bars_image();
        let (deg, conf) = ImageEnhancer::detect_orientation_with_confidence(&upright);
        assert_eq!(deg, 0, "left-aligned bars should read as upright");
        assert!(conf > 0.0, "asymmetric bars should give nonzero confidence");

        let flipped = rotate180(&upright);
        let (deg180, _) = ImageEnhancer::detect_orientation_with_confidence(&flipped);
        assert_eq!(deg180, 180, "180°-rotated bars should be detected");
    }

    #[test]
    fn test_orientation_confidence_blank_is_zero() {
        let blank = OcrImage::new(
            DynamicImage::ImageLuma8(GrayImage::from_pixel(100, 100, Luma([255u8]))),
            300,
        );
        let (deg, conf) = ImageEnhancer::detect_orientation_with_confidence(&blank);
        assert_eq!(deg, 0);
        assert_eq!(conf, 0.0, "no content → no orientation confidence");
    }

    #[test]
    fn test_remove_enclosing_circle_keeps_inner_content() {
        let mut g = GrayImage::from_pixel(100, 100, Luma([255u8]));
        // Ring: circle of radius 40 ± 1.2 px around (50, 50).
        for y in 0..100u32 {
            for x in 0..100u32 {
                let dx = x as f64 - 50.0;
                let dy = y as f64 - 50.0;
                let r = (dx * dx + dy * dy).sqrt();
                if (r - 40.0).abs() < 1.2 {
                    g.put_pixel(x, y, Luma([0u8]));
                }
            }
        }
        // Inner "text" rectangle at the center.
        for y in 44..56u32 {
            for x in 40..60u32 {
                g.put_pixel(x, y, Luma([0u8]));
            }
        }
        let img = OcrImage::new(DynamicImage::ImageLuma8(g), 300);
        let cleaned = ImageEnhancer::remove_enclosing_circle(&img).unwrap();

        let out = cleaned.data.to_luma8();
        // Ring stroke pixels become white…
        assert_eq!(out.get_pixel(50, 10)[0], 255, "ring top should be erased");
        assert_eq!(out.get_pixel(90, 50)[0], 255, "ring right should be erased");
        // …while the inner rectangle survives.
        assert_eq!(out.get_pixel(50, 50)[0], 0, "inner content must survive");
    }

    #[test]
    fn test_remove_enclosing_circle_noop_without_ring() {
        let mut g = GrayImage::from_pixel(100, 100, Luma([255u8]));
        for y in 40..60u32 {
            for x in 20..80u32 {
                g.put_pixel(x, y, Luma([0u8]));
            }
        }
        let img = OcrImage::new(DynamicImage::ImageLuma8(g), 300);
        let cleaned = ImageEnhancer::remove_enclosing_circle(&img).unwrap();
        // A solid rectangle is not a ring: image must be unchanged.
        assert_eq!(
            cleaned.data.to_luma8().get_pixel(50, 50)[0],
            0,
            "plain rect must not be erased"
        );
    }
}
