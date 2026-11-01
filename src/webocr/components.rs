//! Connected components and small statistics helpers (WebOCR port).
//!
//! Port of `webocr/src/ocr/components.ts`: 8-neighbourhood iterative flood
//! fill. 8-connectivity keeps italic strokes and most anti-aliased joins in
//! one piece.

use super::types::{BinaryImage, Comp};

/// 8-connected components of the ink mask, in scan order of their first pixel.
pub fn connected_components(bin: &BinaryImage) -> Vec<Comp> {
    let (w, h) = (bin.width, bin.height);
    let mut seen = vec![0u8; w * h];
    let mut comps = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..w * h {
        if bin.data[start] == 0 || seen[start] == 1 {
            continue;
        }
        seen[start] = 1;
        stack.clear();
        stack.push(start);

        let mut x0 = start % w;
        let mut x1 = x0 + 1;
        let mut y0 = start / w;
        let mut y1 = y0 + 1;
        let mut area = 0usize;

        while let Some(p) = stack.pop() {
            let px = p % w;
            let py = p / w;
            area += 1;
            if px < x0 {
                x0 = px;
            }
            if px + 1 > x1 {
                x1 = px + 1;
            }
            if py < y0 {
                y0 = py;
            }
            if py + 1 > y1 {
                y1 = py + 1;
            }

            for dy in -1i32..=1 {
                let ny = py as i32 + dy;
                if ny < 0 || ny >= h as i32 {
                    continue;
                }
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = px as i32 + dx;
                    if nx < 0 || nx >= w as i32 {
                        continue;
                    }
                    let n = ny as usize * w + nx as usize;
                    if bin.data[n] == 1 && seen[n] == 0 {
                        seen[n] = 1;
                        stack.push(n);
                    }
                }
            }
        }
        comps.push(Comp {
            x0,
            y0,
            x1,
            y1,
            area,
        });
    }
    comps
}

/// Bounding-box union of components (area is summed, not part of the contract).
pub fn union_box(comps: &[Comp]) -> Option<Comp> {
    let mut it = comps.iter();
    let first = *it.next()?;
    let mut out = first;
    for c in it {
        out.x0 = out.x0.min(c.x0);
        out.y0 = out.y0.min(c.y0);
        out.x1 = out.x1.max(c.x1);
        out.y1 = out.y1.max(c.y1);
        out.area += c.area;
    }
    Some(out)
}

/// Median (upper-middle element; matches WebOCR `median` exactly).
pub fn median(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted: Vec<usize> = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_img(w: usize, h: usize, ink: &[(usize, usize)]) -> BinaryImage {
        let mut data = vec![0u8; w * h];
        for &(x, y) in ink {
            data[y * w + x] = 1;
        }
        BinaryImage { width: w, height: h, data }
    }

    #[test]
    fn diagonal_pixels_are_one_component() {
        let img = bin_img(5, 5, &[(0, 0), (1, 1), (2, 2)]);
        let comps = connected_components(&img);
        assert_eq!(comps.len(), 1, "8-connectivity joins the diagonal");
        assert_eq!(comps[0].area, 3);
        assert_eq!((comps[0].x0, comps[0].y0, comps[0].x1, comps[0].y1), (0, 0, 3, 3));
    }

    #[test]
    fn separate_blobs_stay_separate() {
        let img = bin_img(10, 4, &[(1, 1), (5, 1), (9, 1)]);
        let comps = connected_components(&img);
        assert_eq!(comps.len(), 3);
    }

    #[test]
    fn union_box_and_median() {
        let comps = vec![
            Comp { x0: 2, y0: 3, x1: 5, y1: 6, area: 4 },
            Comp { x0: 0, y0: 8, x1: 1, y1: 9, area: 1 },
        ];
        let u = union_box(&comps).unwrap();
        assert_eq!((u.x0, u.y0, u.x1, u.y1), (0, 3, 5, 9));
        assert_eq!(u.area, 5);
        assert_eq!(median(&[5, 1, 9]), 5);
        assert_eq!(median(&[]), 0);
    }
}
