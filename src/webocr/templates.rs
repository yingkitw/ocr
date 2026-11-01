//! Template-library construction (WebOCR port of `webocr/src/ocr/templates.ts`).
//!
//! Templates MUST pass through the exact same pipeline as scan glyphs
//! (binarize → box → own-height scale → baseline anchor → blur). Any
//! asymmetry (templates squashed, samples anchored) is a latent accuracy
//! bug that only shows up as unexplained low scores — so [`make_template`]
//! calls the same [`place_glyph`]/[`blur3`]/[`glyph_scale`] the engine uses
//! (WebOCR "equal footing" rule).

use super::binarize::{binarize, binarize_with_threshold};
use super::classify::{l2, pool_coarse, PreparedTemplate};
use super::components::{connected_components, union_box};
use super::normalize::{blur3, glyph_scale, place_glyph, FRAME, SCALE_REF};
use super::segment::filter_noise;
use super::types::{BinaryImage, Comp, GrayImage};

/// A rendered glyph: grayscale pixels + the baseline row (WebOCR
/// `RenderedGlyph` / `TemplateRenderer.renderText`). Black ink on white.
#[derive(Debug, Clone)]
pub struct RenderedGlyph {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    /// Row of the text baseline within the bitmap.
    pub baseline: f32,
}

impl RenderedGlyph {
    pub fn to_gray(&self) -> GrayImage {
        GrayImage {
            width: self.width,
            height: self.height,
            data: self.data.clone(),
        }
    }
}

/// One position in a batched grid render (WebOCR `renderGrid`).
/// `baseline_x` and `baseline_y` are baseline positions in the
/// shared canvas's coordinate system.
#[derive(Debug, Clone, Copy)]
pub struct GridItem {
    pub ch: char,
    pub baseline_x: f32,
    pub baseline_y: f32,
}

/// Result of a grid render: the full canvas plus the per-item ink masks.
/// WebOCR's `renderGrid` returns a single image; the consumer binarizes
/// + CCs + assigns components back to cells by centroid. We do the
/// binarize + CC assignment inside the renderer so callers get a clean
/// `Vec<RenderedGlyph>`-equivalent (one per grid item).
#[derive(Debug, Clone)]
pub struct GridResult {
    pub width: u32,
    pub height: u32,
    /// Per-item: the ink mask cropped to that cell's bounding box.
    /// Same shape as `RenderedGlyph` (grayscale, baseline row).
    pub items: Vec<RenderedGlyph>,
    /// Characters that didn't produce a component (missing glyph, etc.).
    /// Indexes correspond to the input `items` slice.
    pub missing: Vec<usize>,
}

/// Font-agnostic glyph renderer (WebOCR `TemplateRenderer`). A *face* is
/// one loaded font variant (family × weight); templates fan out over all
/// faces for raster diversity. Implementors draw black text on a white
/// background and report the exact baseline, so templates and samples
/// anchor identically.
pub trait TemplateRenderer {
    /// Number of available faces (0 → no library can be built).
    fn face_count(&self) -> usize;

    /// Render one character with face `face` at `px` size. Returns `None`
    /// when the face cannot rasterize the character.
    fn render_char(&self, face: usize, ch: char, px: u32) -> Option<RenderedGlyph>;

    /// Render one character at a **fractional x offset** — the "phase".
    ///
    /// A glyph drawn at x=10 and the same glyph drawn at x=10.4 are
    /// rasterized differently (the anti-aliasing of every edge moves), and a
    /// scan's glyphs sit at whatever fraction the text put them at. The
    /// phase is what lets the template ladder cover that spread (see
    /// [`LATIN_PHASES`]).
    ///
    /// The default implementation ignores the phase — a renderer that cannot
    /// draw at a fractional offset still contributes the phase-0 templates.
    fn render_char_phase(&self, face: usize, ch: char, px: u32, phase: f32) -> Option<RenderedGlyph> {
        let _ = phase;
        self.render_char(face, ch, px)
    }

    /// Render a **run of text on one line** — the page-context sample.
    ///
    /// The charset rendered as one line of text is "a page" for the purpose
    /// of the binarization threshold: its histogram is the population of
    /// glyph edges a real page has, and nothing else (no glyph's own pixels
    /// to speak of). Renderers that cannot draw a run return `None`, and the
    /// templates fall back to their own Otsu threshold.
    fn render_line(&self, face: usize, text: &str, px: u32) -> Option<RenderedGlyph> {
        let _ = (face, text, px);
        None
    }

    /// Batch render many glyphs at once into a shared grid canvas
    /// (WebOCR `renderGrid`). The default implementation returns
    /// `None`; renderers that can batch should override for speed.
    ///
    /// `cell` is the per-glyph cell size in pixels; `cols` is the number
    /// of cells per row. The returned canvas is `cell*cols` wide and
    /// `cell*ceil(items.len()/cols)` tall.
    fn render_grid(
        &self,
        face: usize,
        items: &[GridItem],
        px: u32,
        cell: u32,
        cols: u32,
    ) -> Option<GridResult> {
        let _ = (face, items, px, cell, cols);
        None
    }
}

/// The default Latin charset: printable ASCII sans space.
pub const LATIN_CHARSET: &str = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

/// Default CJK raster sizes (WebOCR: 72 px preserves kana AA detail;
/// single size keeps the 21K-glyph library build manageable).
pub const CJK_SIZES: &[u32] = &[72];

/// CJK Unified Ideographs block (U+4E00–U+9FFF) — 20,902 hanzi.
/// The most common CJK characters in printed text. WebOCR uses the same
/// range for the CJK union's "han" portion; kana and hangul are future
/// work (see TODO.md).
pub fn cjk_han_charset() -> String {
    let mut s = String::with_capacity(20_902 * 3);
    for cp in 0x4E00..=0x9FFF {
        if let Some(c) = char::from_u32(cp) {
            s.push(c);
        }
    }
    s
}

/// CJK union charset: ASCII + Han + Hiragana + Katakana + Hangul
/// Syllables. Generated lazily (it's a ~33K-char string). The
/// partition in `classify::shortlist` separates the two script
/// families at codepoint 0x2E80 so both have recall under one library.
pub fn cjk_union_charset() -> String {
    let mut s = String::with_capacity(33_000 * 3);
    // ASCII
    for cp in 0x0020..=0x007E {
        s.push(char::from_u32(cp).unwrap());
    }
    // Hiragana
    for cp in 0x3040..=0x309F {
        if let Some(c) = char::from_u32(cp) {
            s.push(c);
        }
    }
    // Katakana
    for cp in 0x30A0..=0x30FF {
        if let Some(c) = char::from_u32(cp) {
            s.push(c);
        }
    }
    // CJK Unified Ideographs
    for cp in 0x4E00..=0x9FFF {
        if let Some(c) = char::from_u32(cp) {
            s.push(c);
        }
    }
    // Hangul Syllables
    for cp in 0xAC00..=0xD7A3 {
        if let Some(c) = char::from_u32(cp) {
            s.push(c);
        }
    }
    s
}

/// Codepoint threshold separating the two shortlist partitions
/// (WebOCR `engine.ts`): codepoints < 0x2E80 are routed to the Latin
/// shortlist, the rest to the CJK shortlist. 0x2E80 = CJK Radicals
/// Supplement start, a clean boundary that keeps kana in the CJK set.
pub const SCRIPT_PARTITION_CP: u32 = 0x2E80;

/// Coarse-vector resolution for shortlisting.
///
/// 12×12, not 8×8: characters whose discrimination lives in mid-frequency
/// stroke layout rank in the thousands at 8×8 and never reach the fine
/// stage (WebOCR measured 惰 at rank 4410 with pool8±1 — its full-res fine
/// rank is 1! — vs rank 5 at pool12±1, with ja/ko recall unchanged).
const COARSE_CELLS: usize = 12;

/// Fractional x offsets the Latin ladder is drawn at. A scan's glyphs sit at
/// whatever fraction the text put them at, and the anti-aliasing of a
/// one-pixel stem changes with it: at 16px a Georgia "u" scores 1.000
/// against a template drawn at phase 0 and 0.867 against one drawn at phase
/// 0.8.
pub const LATIN_PHASES: &[f32] = &[0.0, 1.0 / 3.0, 2.0 / 3.0];

/// The rasters that get the extra phases. The spread is widest at the small
/// rasters (13pp at 16px against 3pp at 40px), so covering only those is
/// most of the benefit for a fraction of the library.
pub const LATIN_PHASE_SIZES: &[u32] = &[16, 22];

/// Normalize one rendered glyph box into a `PreparedTemplate` through the
/// exact sample pipeline (equal footing).
#[allow(clippy::too_many_arguments)]
pub fn make_template(
    bin: &BinaryImage,
    box_: Comp,
    baseline: f32,
    char: char,
    use_blur: bool,
    blur_passes: usize,
    scale_ref: f32,
    em: f32,
    phase: f32,
) -> PreparedTemplate {
    let mut vec = place_glyph(bin, &box_, glyph_scale(&box_, scale_ref), baseline, FRAME);
    if use_blur {
        for _ in 0..blur_passes {
            vec = blur3(&vec, FRAME);
        }
    }
    PreparedTemplate {
        char,
        norm: l2(&vec).max(1.0),
        aspect: box_.width() as f32 / box_.height().max(1) as f32,
        vec,
        coarse: None,
        coarse_norm: 0.0,
        rel_h: box_.height() as f32 / em.max(1.0),
        // The case signal: how far the ink's top sits above the baseline, in
        // em units. A lowercase tops out at the x-height, a capital at the
        // cap height, a descender letter at the x-height again.
        top_rel: (baseline - box_.y0 as f32) / em.max(1.0),
        raster: em.max(1.0) as u32,
        phase,
    }
}

/// The binarization threshold a TEMPLATE is built with.
///
/// Otsu's method reads the histogram it is handed, and a single glyph's
/// histogram is not a page's. Measured on a 16px Trebuchet "i": binarized
/// ALONE the glyph's threshold is 88, and a page of the same type gives
/// 167. The template is therefore FATTER than the sample it is meant to
/// mirror — at 88 the "i"'s stem and its dot merge into one blob, while the
/// page (at 167) keeps them apart — and the equal-footing rule breaks
/// exactly where the ink is thinnest. The symptoms were the ones the
/// scorecard measured: the "i" read as "'", the "f" as "t" or "I", the "j"
/// as "]", and — the case confusion this port's scorecard reproduced — a
/// lowercase "s" read as "S".
///
/// The fix is to give the templates the threshold of the context they stand
/// for: a PAGE of the same type. The charset rendered as one line of text
/// is a page for this purpose — its histogram is the population of glyph
/// edges a real page has — and its Otsu threshold lands within a level or
/// two of the page's (WebOCR measured 167 against the page's 168).
///
/// The threshold is per (font, size): the ink's edge distribution moves with
/// the stroke width, so a 16px page and a 54px page do not share one.
fn page_threshold(
    renderer: &dyn TemplateRenderer,
    face: usize,
    size: u32,
    charset: &str,
) -> Option<u8> {
    let sample: String = charset.chars().filter(|c| *c != ' ' && *c != '\n').collect();
    if sample.is_empty() {
        return None;
    }
    let img = renderer.render_line(face, &sample, size)?;
    let gray = img.to_gray();
    Some(super::binarize::otsu_threshold(&gray.data))
}

/// Latin path: render every (character × face × size) individually with the
/// platform's own rasterizer, then push each glyph through the same
/// normalization a scan glyph will see.
///
/// The template raster must track the **sample's** raster: own-height
/// normalization scales every glyph into the same frame, so a lowercase and
/// an uppercase of the same letter become near-identical shapes and the tie
/// is broken by rasterization blur. A 30px scan upscaled into the frame
/// matches the softened 68px CAPITAL template better than the sharp 40px
/// lowercase one (WebOCR measured 0.922 vs 0.969 — the case flips in
/// "quiCk … dOgS"). Rendering the ladder at the sizes scans actually use
/// restores the distinction (a same-size template wins 1.000 vs 0.918).
#[allow(clippy::too_many_arguments)]
pub fn build_per_char(
    renderer: &dyn TemplateRenderer,
    charset: &str,
    sizes: &[u32],
    scale_ref: f32,
    use_blur: bool,
    blur_passes: usize,
    phases: &[f32],
    phase_sizes: &[u32],
) -> Vec<PreparedTemplate> {
    let mut templates = Vec::new();
    for face in 0..renderer.face_count() {
        for &size in sizes {
            // Only the rasters that need the phase cover pay for it.
            let size_phases: &[f32] = if phase_sizes.contains(&size) {
                phases
            } else {
                &[0.0]
            };
            // The threshold a page of this type would give (see
            // `page_threshold`).
            let threshold = if std::env::var("OCR_NOPAGETHRESHOLD").is_ok() { None } else { page_threshold(renderer, face, size, charset) };
            for ch in charset.chars() {
                if ch == ' ' {
                    continue;
                }
                for &phase in size_phases {
                    let Some(img) = renderer.render_char_phase(face, ch, size, phase) else {
                        continue;
                    };
                    let r = binarize_with_threshold(&img.to_gray(), threshold);
                    let comps = filter_noise(&connected_components(&r.image));
                    let Some(box_) = union_box(&comps) else {
                        continue;
                    };
                    templates.push(make_template(
                        &r.image,
                        box_,
                        img.baseline,
                        ch,
                        use_blur,
                        blur_passes,
                        scale_ref,
                        size as f32,
                        phase,
                    ));
                }
            }
        }
    }
    templates
}

/// CJK path: batched grid rendering (12×12 cells per canvas) so the
/// 21K-glyph library builds in seconds instead of minutes. WebOCR's
/// `buildGrid` in `templates.ts` uses canvas for this; we use a
/// 12×12-cell layout with each cell `cell = ceil(px * 1.5)` and
/// baseline at `baseline_y = ceil(px * 1.15)` from the cell top.
#[allow(clippy::too_many_arguments)]
pub fn build_grid(
    renderer: &dyn TemplateRenderer,
    charset: &str,
    sizes: &[u32],
    scale_ref: f32,
    use_blur: bool,
    blur_passes: usize,
    coarse: bool,
) -> Vec<PreparedTemplate> {
    const GRID_COLS: usize = 12;
    const GRID_ROWS: usize = 12;
    let per_grid = GRID_COLS * GRID_ROWS;

    let mut templates = Vec::new();
    let chars: Vec<char> = charset.chars().filter(|&c| c != ' ' && c != '\n').collect();

    for face in 0..renderer.face_count() {
        for &size in sizes {
            for chunk in chars.chunks(per_grid) {
                let cell = (size as f32 * 1.5).ceil() as u32;
                let baseline_y = (size as f32 * 1.15).ceil() as f32;
                let items: Vec<GridItem> = chunk
                    .iter()
                    .map(|&ch| GridItem {
                        ch,
                        // Within a cell: center the glyph horizontally, baseline at
                        // baseline_y. The renderer places the baseline at this y
                        // within the cell; rusttype then offsets by the glyph's
                        // own metric.
                        baseline_x: (cell as f32 - size as f32) / 2.0,
                        baseline_y,
                    })
                    .collect();

                let grid = renderer.render_grid(face, &items, size, cell, GRID_COLS as u32);
                let grid = match grid {
                    Some(g) => g,
                    None => {
                        // Renderer doesn't support grid: fall back to per-char for this chunk.
                        for it in &items {
                            if let Some(img) = renderer.render_char(face, it.ch, size) {
                                let r = binarize(&img.to_gray());
                                let comps = filter_noise(&connected_components(&r.image));
                                if let Some(b) = union_box(&comps) {
                                    templates.push(make_template(
                                        &r.image,
                                        b,
                                        img.baseline,
                                        it.ch,
                                        use_blur,
                                        blur_passes,
                                        scale_ref,
                                        size as f32,
                                        0.0,
                                    ));
                                }
                            }
                        }
                        continue;
                    }
                };

                for (i, img) in grid.items.into_iter().enumerate() {
                    if img.width == 0 {
                        continue;
                    }
                    let ch = chunk[i];
                    let r = binarize(&img.to_gray());
                    let comps = filter_noise(&connected_components(&r.image));
                    let Some(box_) = union_box(&comps) else {
                        continue;
                    };
                    templates.push(make_template(
                        &r.image,
                        box_,
                        img.baseline,
                        ch,
                        use_blur,
                        blur_passes,
                        scale_ref,
                        size as f32,
                        0.0,
                    ));
                }
            }
        }
    }

    if coarse {
        for t in &mut templates {
            let c = pool_coarse(&t.vec, FRAME, COARSE_CELLS);
            t.coarse_norm = l2(&c).max(1.0);
            t.coarse = Some(c);
        }
    }

    templates
}

/// Build the template library. Coarse vectors are attached when `coarse`
/// is set so large libraries can shortlist. For libraries > 1500
/// templates the engine automatically switches to the CJK-aware matching
/// path (shift search + shortlist + relH penalty); the caller doesn't
/// need to opt in beyond passing `coarse: true`.
pub fn build_templates(
    renderer: &dyn TemplateRenderer,
    charset: &str,
    sizes: &[u32],
    coarse: bool,
) -> Vec<PreparedTemplate> {
    build_templates_for_options(renderer, charset, sizes, coarse, false)
}

/// Like `build_templates` but with an explicit CJK switch. The CJK path
/// uses grid batching (per-char is correct but slow) and the
/// WebOCR-calibrated `CJK_SCALE_REF` (24) instead of Latin's 20.
#[allow(clippy::too_many_arguments)]
pub fn build_templates_for_options(
    renderer: &dyn TemplateRenderer,
    charset: &str,
    sizes: &[u32],
    coarse: bool,
    cjk: bool,
) -> Vec<PreparedTemplate> {
    let scale_ref = if cjk { super::normalize::CJK_SCALE_REF } else { SCALE_REF };
    let mut templates = if cjk {
        build_grid(
            renderer,
            charset,
            sizes,
            scale_ref,
            true,
            1,
            coarse,
        )
    } else {
        build_per_char(
            renderer,
            charset,
            sizes,
            scale_ref,
            true,
            1,
            LATIN_PHASES,
            LATIN_PHASE_SIZES,
        )
    };
    // Per-char fallback may not have attached coarse vectors.
    if coarse {
        for t in &mut templates {
            if t.coarse.is_none() {
                let c = pool_coarse(&t.vec, FRAME, COARSE_CELLS);
                t.coarse_norm = l2(&c).max(1.0);
                t.coarse = Some(c);
            }
        }
    }
    templates
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic 5×7 pixel-font renderer — the whole pipeline testable with
    /// zero system-font dependence (WebOCR "synthetic renderer doubles as a
    /// parity check"). Two faces with different cell sizes for fan-out.
    struct PixelFont;

    fn glyph_bits(ch: char) -> Option<[&'static str; 7]> {
        Some(match ch {
            'A' => ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
            'B' => ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
            'C' => ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
            'l' => ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
            'I' => ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
            'o' => ["00000", "00000", "01110", "10001", "10001", "10001", "01110"],
            'O' => ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
            '.' => ["00000", "00000", "00000", "00000", "00000", "00000", "00100"],
            _ => return None,
        })
    }

    impl TemplateRenderer for PixelFont {
        fn face_count(&self) -> usize {
            2
        }

        fn render_char(&self, face: usize, ch: char, _px: u32) -> Option<RenderedGlyph> {
            let rows = glyph_bits(ch)?;
            let cell = if face == 0 { 2usize } else { 3usize }; // raster diversity
            let pad = 4usize;
            let w = 5 * cell + 2 * pad;
            let h = 7 * cell + 2 * pad;
            let mut data = vec![255u8; w * h];
            for (ry, row) in rows.iter().enumerate() {
                for (rx, c) in row.chars().enumerate() {
                    if c == '1' {
                        for y in 0..cell {
                            for x in 0..cell {
                                data[(pad + ry * cell + y) * w + pad + rx * cell + x] = 0;
                            }
                        }
                    }
                }
            }
            Some(RenderedGlyph {
                width: w,
                height: h,
                data,
                baseline: (pad + 7 * cell) as f32,
            })
        }
    }

    #[test]
    fn pixel_font_fans_out_over_faces_and_sizes() {
        let renderer = PixelFont;
        let templates = build_templates(&renderer, "ABClo.", &[20, 24], false);
        // 6 chars × 2 faces × 2 sizes = 24 templates
        assert_eq!(templates.len(), 24);
        assert!(templates.iter().all(|t| t.vec.len() == FRAME * FRAME));
        assert!(templates.iter().all(|t| t.norm > 0.0 && t.rel_h > 0.0));
        assert!(templates.iter().all(|t| !t.vec.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn equal_footing_template_matches_its_own_glyph() {
        let renderer = PixelFont;
        let templates = build_templates(&renderer, "ABC", &[20], false);
        // Re-render 'B' (face 0) as a "sample" and match — it must hit itself.
        let img = renderer.render_char(0, 'B', 20).unwrap();
        let r = binarize(&img.to_gray());
        let comps = filter_noise(&connected_components(&r.image));
        let box_ = union_box(&comps).unwrap();
        let mut vec = place_glyph(&r.image, &box_, glyph_scale(&box_, SCALE_REF), img.baseline, FRAME);
        vec = blur3(&vec, FRAME);
        let refs: Vec<&PreparedTemplate> = templates.iter().collect();
        let m = crate::webocr::classify::match_glyph(
            &vec,
            box_.width() as f32 / box_.height() as f32,
            &refs,
            crate::webocr::classify::MatchOptions { frame: FRAME, ..Default::default() },
        );
        assert_eq!(m.char, 'B');
        assert!(m.score > 0.95, "identical glyph should match near-perfectly (score {:.3})", m.score);
    }

    #[test]
    fn coarse_vectors_enable_shortlisting() {
        let renderer = PixelFont;
        let templates = build_templates(&renderer, "ABCo.", &[20], true);
        assert!(templates.iter().all(|t| t.coarse.is_some() && t.coarse_norm > 0.0));
        // 12×12 pooled cells: 8×8 ranks characters whose discrimination
        // lives in mid-frequency stroke layout in the thousands, and they
        // never reach the fine stage (WebOCR measured 惰 at rank 4410 with
        // pool8±1 against rank 5 with pool12±1).
        assert_eq!(templates[0].coarse.as_ref().unwrap().len(), 144);
    }

    #[test]
    fn cjk_han_charset_covers_unified_ideographs() {
        let s = cjk_han_charset();
        // U+4E00..=U+9FFF contains 20,992 codepoints (matches the
        // canonical "CJK Unified Ideographs" block size).
        assert_eq!(s.chars().count(), 20_992);
        assert!(s.starts_with('\u{4E00}'));
        assert!(s.contains('\u{4E2D}')); // 中
        assert!(s.ends_with('\u{9FFF}'));
    }

    #[test]
    fn cjk_union_charset_includes_all_three_blocks() {
        let s = cjk_union_charset();
        // ASCII range
        assert!(s.chars().any(|c| c == 'A'));
        // Hiragana
        assert!(s.chars().any(|c| c == '\u{3042}')); // あ
        // Katakana
        assert!(s.chars().any(|c| c == '\u{30A2}')); // ア
        // Han
        assert!(s.chars().any(|c| c == '\u{4E2D}')); // 中
        // Hangul
        assert!(s.chars().any(|c| c == '\u{AC00}')); // 가
        // ≥ ASCII + Hiragana + Katakana + CJK Unified + Hangul Syllables
        assert!(s.chars().count() > 30_000);
    }

    #[test]
    fn script_partition_threshold_splits_at_cjk_radicals() {
        // ASCII + Latin-1 + general punctuation → Latin shortlist
        assert!(('A' as u32) < SCRIPT_PARTITION_CP);
        assert!(('~' as u32) < SCRIPT_PARTITION_CP);
        // Hiragana → CJK shortlist
        assert!(('\u{3042}' as u32) >= SCRIPT_PARTITION_CP);
        // CJK Unified → CJK shortlist
        assert!(('\u{4E2D}' as u32) >= SCRIPT_PARTITION_CP);
    }
}
