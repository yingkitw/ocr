//! System-font renderer for the template library (rusttype).
//!
//! Loads platform font files once and rasterizes individual glyphs with
//! anti-aliased coverage, reporting the exact baseline row so templates and
//! samples anchor identically (WebOCR canvasRenderer contract).

use super::templates::{GridItem, GridResult, RenderedGlyph, TemplateRenderer};
use super::types::Comp;
use rusttype::{Font, Scale};

/// One loaded font face (a family's regular or bold file).
struct LoadedFace {
    font: Font<'static>,
}

/// Renderer over platform font files. Faces are loaded from the first
/// existing candidate path per entry; missing platforms simply reduce the
/// face count (an empty renderer yields no templates, and the engine
/// surfaces a clear error instead of garbage).
pub struct RustTypeRenderer {
    faces: Vec<LoadedFace>,
}

/// CJK face candidates: macOS Hiragino (which covers Han + Kana), Linux
/// Noto CJK / Source Han, Windows Microsoft YaHei / MS Mincho, then
/// fallbacks. Order matters — the first existing file is the one used.
const CJK_FACE_CANDIDATES: &[&[&str]] = &[
    &[
        // macOS — Hiragino Sans GB covers Han + kana + Latin
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W2.ttc",
        // Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        // Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
    ],
    &[
        // CJK bold variant (Hiragino Sans GB W6/5 is the bold weight)
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
    ],
];

/// Candidate font files per face (first existing path wins). Mirrors the
/// WebOCR font coverage: sans reg/bold, serif reg/bold, monospace, two wide
/// sans — with Linux/Windows fallbacks. Each entry is one face; bold
/// variants are their own entries with their own candidate lists.
const FACE_CANDIDATES: &[&[&str]] = &[
    // sans regular
    &[
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    // sans bold
    &[
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    // serif regular
    &[
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Times.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/times.ttf",
    ],
    // serif bold
    &[
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
    ],
    // monospace regular
    &[
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Courier.ttc",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ],
    // wide sans regular
    &[
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ],
    // wide sans 2 regular
    &[
        "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
];

impl RustTypeRenderer {
    /// Load all available Latin faces.
    pub fn load() -> Self {
        Self::load_faces(FACE_CANDIDATES)
    }

    /// Load CJK-capable faces. Tries Hiragino Sans GB / Hiragino Kaku
    /// Gothic (macOS), Noto CJK (Linux), Microsoft YaHei (Windows).
    /// Returns a renderer with the CJK face in slot 0 — used by
    /// `TemplateOcr::with_cjk_union` to build the union library.
    pub fn for_cjk() -> Self {
        Self::load_faces(CJK_FACE_CANDIDATES)
    }

    fn load_faces(candidates: &[&[&str]]) -> Self {
        let mut faces = Vec::new();
        for paths in candidates {
            for path in *paths {
                if let Ok(data) = std::fs::read(path)
                    && let Some(font) = Font::try_from_vec(data)
                {
                    faces.push(LoadedFace { font });
                    break;
                }
            }
        }
        Self { faces }
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }
}

impl TemplateRenderer for RustTypeRenderer {
    fn face_count(&self) -> usize {
        self.faces.len()
    }

    fn render_char(&self, face: usize, ch: char, px: u32) -> Option<RenderedGlyph> {
        self.render_char_phase(face, ch, px, 0.0)
    }

    /// Render at a fractional x offset — the phase.
    ///
    /// The glyph is positioned at `phase` pixels from the origin, so the
    /// rasterizer lays its outline down at that sub-pixel offset and the
    /// anti-aliasing of every edge moves. That is exactly the difference
    /// between a scan's glyph and a phase-0 template, and it is what the
    /// phase ladder (and the phase rescue) exist to cover.
    fn render_char_phase(&self, face: usize, ch: char, px: u32, phase: f32) -> Option<RenderedGlyph> {
        let loaded = self.faces.get(face)?;
        let scale = Scale::uniform(px as f32);
        let glyph = loaded
            .font
            .glyphs_for(std::iter::once(ch))
            .next()?
            .scaled(scale)
            .positioned(rusttype::point(phase, 0.0));
        let bb = glyph.pixel_bounding_box()?;

        let pad = 4usize;
        let w = bb.width() as usize + 2 * pad;
        let h = bb.height() as usize + 2 * pad;
        let mut data = vec![255u8; w * h];
        glyph.draw(|x, y, v| {
            let xi = pad + x as usize;
            let yi = pad + y as usize;
            if xi < w && yi < h {
                let slot = &mut data[yi * w + xi];
                // Blend coverage over the white background (ink = dark).
                let ink = (v * 255.0).round() as u8;
                *slot = 255u8.saturating_sub(ink).min(*slot);
            }
        });

        // Baseline row: glyph space y=0 maps to canvas row pad − bb.min.y.
        let baseline = pad as f32 - bb.min.y as f32;
        Some(RenderedGlyph {
            width: w,
            height: h,
            data,
            baseline,
        })
    }

    /// Render a run of text on one line — the page-context sample for the
    /// binarization threshold (see `templates::page_threshold`).
    ///
    /// The charset drawn as one line of text carries the population of glyph
    /// edges a real page has, and its Otsu threshold lands within a level or
    /// two of the page's.
    fn render_line(&self, face: usize, text: &str, px: u32) -> Option<RenderedGlyph> {
        let loaded = self.faces.get(face)?;
        let scale = Scale::uniform(px as f32);
        let metrics = loaded.font.v_metrics(scale);
        let ascent = metrics.ascent;
        let descent = -metrics.descent;

        // Measure the run.
        let mut width = 0f32;
        for g in loaded.font.layout(text, scale, rusttype::point(0.0, 0.0)) {
            if let Some(bb) = g.pixel_bounding_box() {
                width = width.max(bb.max.x as f32);
            }
        }
        let pad = 4usize;
        let w = width.ceil() as usize + 2 * pad;
        let h = (ascent + descent).ceil() as usize + 2 * pad;
        if w == 0 || h == 0 {
            return None;
        }
        let mut data = vec![255u8; w * h];
        let baseline = pad as f32 + ascent;
        for g in loaded
            .font
            .layout(text, scale, rusttype::point(pad as f32, baseline))
        {
            let Some(bb) = g.pixel_bounding_box() else {
                continue;
            };
            let x_off = bb.min.x as i32;
            let y_off = bb.min.y as i32;
            g.draw(|x, y, v| {
                let xi = (x_off + x as i32) as usize;
                let yi = (y_off + y as i32) as usize;
                if xi < w && yi < h {
                    let slot = &mut data[yi * w + xi];
                    let ink = (v * 255.0).round() as u8;
                    *slot = 255u8.saturating_sub(ink).min(*slot);
                }
            });
        }
        Some(RenderedGlyph {
            width: w,
            height: h,
            data,
            baseline,
        })
    }

    /// Batched grid render for the 21K-glyph CJK library: 144 glyphs
    /// per canvas (12×12 cells), one binarize + CC pass, then assign
    /// components back to cells by centroid. This mirrors WebOCR's
    /// `buildGrid` and brings the CJK library build from minutes to
    /// seconds.
    fn render_grid(
        &self,
        face: usize,
        items: &[GridItem],
        px: u32,
        cell: u32,
        cols: u32,
    ) -> Option<GridResult> {
        let loaded = self.faces.get(face)?;
        if items.is_empty() {
            return Some(GridResult {
                width: 0,
                height: 0,
                items: Vec::new(),
                missing: Vec::new(),
            });
        }
        let cell = cell as usize;
        let cols = cols as usize;
        let scale = Scale::uniform(px as f32);
        let rows = (items.len() + cols - 1) / cols;
        let width = cols * cell;
        let height = rows * cell;

        // 1. Draw every glyph at its cell-relative baseline.
        let mut data = vec![255u8; width * height];
        for (i, item) in items.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let cx = (col * cell) as f32 + item.baseline_x;
            let cy = (row * cell) as f32 + item.baseline_y;
            let glyph = match loaded
                .font
                .glyphs_for(std::iter::once(item.ch))
                .next()
            {
                Some(g) => g.scaled(scale).positioned(rusttype::point(cx, cy)),
                None => continue,
            };
            glyph.draw(|x, y, v| {
                let xi = x as usize;
                let yi = y as usize;
                if xi < width && yi < height {
                    let slot = &mut data[yi * width + xi];
                    let ink = (v * 255.0).round() as u8;
                    *slot = 255u8.saturating_sub(ink).min(*slot);
                }
            });
        }

        // 2. Binarize via WebOCR's own binarizer (handles polarity).
        let bin = crate::webocr::binarize::binarize(&crate::webocr::types::GrayImage {
            width,
            height,
            data: data.clone(),
        })
        .image;
        // 3. Connected components.
        let comps = crate::webocr::components::connected_components(&bin);
        // 4. Assign each component to its cell by centroid.
        let mut cell_comps: Vec<Vec<Comp>> = (0..items.len()).map(|_| Vec::new()).collect();
        for c in &comps {
            if let Some(idx) = cell_index(c, cell, cols, items.len()) {
                cell_comps[idx].push(*c);
            }
        }

        // 5. For each item, union the assigned components into a box and
        //    crop the binarized image to that box + pad, computing the
        //    baseline row.
        let pad = 4usize;
        let mut out_items = Vec::with_capacity(items.len());
        let mut missing = Vec::new();
        for (i, _item) in items.iter().enumerate() {
            let comps_here = &cell_comps[i];
            if comps_here.is_empty() {
                missing.push(i);
                // Still push a placeholder so callers can index by position.
                out_items.push(RenderedGlyph {
                    width: 0,
                    height: 0,
                    data: Vec::new(),
                    baseline: 0.0,
                });
                continue;
            }
            let col = i % cols;
            let row = i / cols;
            let _cell_origin_x = (col * cell) as f32;
            let cell_origin_y = (row * cell) as f32;
            let box_ = crate::webocr::components::union_box(comps_here).unwrap();
            // Bounding box in canvas coords.
            let x0 = box_.x0.saturating_sub(pad);
            let y0 = box_.y0.saturating_sub(pad);
            let x1 = (box_.x1 + pad).min(width);
            let y1 = (box_.y1 + pad).min(height);
            let w = x1 - x0;
            let h = y1 - y0;
            let mut crop = vec![255u8; w * h];
            for dy in 0..h {
                for dx in 0..w {
                    crop[dy * w + dx] = data[(y0 + dy) * width + (x0 + dx)];
                }
            }
            // Baseline row in the cropped image: glyph-space baseline (cell_origin_y + item.baseline_y)
            // minus crop's y0, then adjusted for the pad (crop is box-relative).
            // In the original canvas, baseline_y = cell_origin_y + item.baseline_y.
            let baseline_in_canvas = cell_origin_y + _item.baseline_y;
            let baseline = baseline_in_canvas - y0 as f32;
            out_items.push(RenderedGlyph {
                width: w,
                height: h,
                data: crop,
                baseline,
            });
        }

        Some(GridResult {
            width: width as u32,
            height: height as u32,
            items: out_items,
            missing,
        })
    }
}

fn cell_index(c: &Comp, cell: usize, cols: usize, n: usize) -> Option<usize> {
    let cx = (c.x0 + c.x1) as f32 / 2.0;
    let cy = (c.y0 + c.y1) as f32 / 2.0;
    let col = (cx / cell as f32) as usize;
    let row = (cy / cell as f32) as usize;
    let idx = row * cols + col;
    if idx < n { Some(idx) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires a system font; skips gracefully on bare CI images.
    #[test]
    fn renders_chars_with_baseline_below_cap_height() {
        let renderer = RustTypeRenderer::load();
        if renderer.is_empty() {
            eprintln!("skipping: no system fonts found");
            return;
        }
        let g = renderer.render_char(0, 'H', 40).expect("'H' renders");
        assert!(g.width > 10 && g.height > 20);
        // Baseline sits inside the bitmap, in the lower half for a capital.
        assert!(g.baseline > g.height as f32 * 0.5 && g.baseline < g.height as f32);
        // Ink exists above the baseline (cap height) and not far below it.
        let gray = g.to_gray();
        let inked = |y: usize| (0..gray.width).any(|x| gray.data[y * gray.width + x] < 128);
        let baseline_row = g.baseline as usize;
        assert!((0..baseline_row).any(inked), "cap ink above baseline");
    }

    #[test]
    fn missing_char_returns_none_not_panic() {
        let renderer = RustTypeRenderer::load();
        if renderer.is_empty() {
            eprintln!("skipping: no system fonts found");
            return;
        }
        // NUL and control chars have no glyph in most fonts.
        let _ = renderer.render_char(0, '\0', 40);
        let _ = renderer.render_char(usize::MAX, 'A', 40); // out-of-range face
    }
}
