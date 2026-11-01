//! WebOCR engine — a Rust port of the sibling WebOCR template-matching
//! OCR (`../webocr`, browser JS/TS).
//!
//! WebOCR's measured accuracy on its real-font fixtures: sans 100 %,
//! monospace 100 %, serif ≈97.8 %, dark/inverted 100 % — with **no ML
//! model**: templates are rendered from system fonts at build time and
//! matched by penalized cosine similarity. This port mirrors its module
//! layout 1:1 so the two codebases stay cross-referenceable:
//!
//! | WebOCR          | this port                |
//! |-----------------|--------------------------|
//! | `binarize.ts`   | [`binarize`]             |
//! | `components.ts` | [`components`]           |
//! | `segment.ts`    | [`segment`]              |
//! | `normalize.ts`  | [`normalize`]            |
//! | `classify.ts`   | [`classify`]             |
//! | `templates.ts`  | [`templates`]            |
//! | `engine.ts`     | [`engine`]               |
//! | `correct.ts`    | *(not ported — the crate's `lang::confusion_correct` is the same design)* |
//!
//! Deliberate deviations:
//! - Input is already-grayscale pixel buffers (the crate decodes images).
//! - Post-hoc text correction lives in the crate's post-processing stage.
//! - The CJK union charset is future work (see TODO.md); `detect.ts`
//!   mergeRatio script detection already exists crate-side as
//!   `lang::script_geometry`.
//!
//! The module is pure: no I/O, no DOM, no ML deps. [`RustTypeRenderer`]
//! (system fonts) is the only platform touchpoint, behind the
//! [`TemplateRenderer`] trait.

pub mod binarize;
pub mod classify;
pub mod components;
pub mod deskew;
pub mod detect;
pub mod engine;
pub mod font_renderer;
pub mod normalize;
pub mod segment;
pub mod templates;
pub mod types;

pub use classify::PreparedTemplate;
pub use detect::{resolve_script, ScriptId};
pub use engine::{run_ocr, RunOptions};
pub use font_renderer::RustTypeRenderer;
pub use segment::split_words_by_gap_cluster;
pub use templates::TemplateRenderer;
pub use types::{GrayImage, TemplateOcrResult};

/// Default Latin raster sizes (WebOCR: anti-aliasing differs by size, so the
/// library carries a ladder that brackets the sizes a scan is likely to meet).
///
/// WebOCR's ladder is `[16, 22, 30, 40, 54]` and it matters: the template
/// must be drawn at the raster the scan was drawn at, because the AA of a
/// 16px stem differs from a 40px one (WebOCR measured 0.922 vs 0.969 cosine
/// for the same glyph at the two rasters). A library that only holds 40px and
/// 68px templates reads a 16px line at ~47 % — measured on the Rust port's
/// own scorecard before the ladder landed.
pub const DEFAULT_SIZES: &[u32] = &[16, 22, 30, 40, 54];

/// Default CJK raster size (WebOCR: 72 px preserves kana AA detail; a
/// single size keeps the 21K-glyph library build manageable).
/// Re-exported from `templates::CJK_SIZES` for convenience.
pub use templates::CJK_SIZES;

/// Ready-to-run template OCR engine: a built template library + the
/// pipeline. Thread-safe (immutable after construction).
pub struct TemplateOcr {
    templates: Vec<PreparedTemplate>,
    charset_len: usize,
    /// True if the library contains CJK codepoints. The engine reads
    /// this to switch on CJK-aware line segmentation, ideographic-space
    /// indents, and the no-inter-word-spaces policy for zh/ja.
    cjk: bool,
}

impl TemplateOcr {
    /// Build the default Latin library from system fonts via rusttype.
    /// Fails with a descriptive error when no usable fonts are found
    /// (e.g. bare containers) — callers surface this instead of guessing.
    pub fn new() -> Result<Self, String> {
        let renderer = RustTypeRenderer::load();
        Self::with_renderer(&renderer)
    }

    /// Build the library from any renderer (tests use a synthetic font).
    pub fn with_renderer(renderer: &dyn TemplateRenderer) -> Result<Self, String> {
        if renderer.face_count() == 0 {
            return Err(
                "template engine: no system fonts found — cannot build the template library"
                    .to_string(),
            );
        }
        // Coarse vectors: the library is well over the 1500-template
        // threshold, so the engine shortlists before the fine stage. Without
        // them every glyph is scored against every template (thousands of
        // full-frame dot products) and the fragment merge — which re-scores
        // a candidate pair against the union of their pools — becomes
        // unaffordable.
        let templates =
            templates::build_templates(renderer, templates::LATIN_CHARSET, DEFAULT_SIZES, true);
        if templates.is_empty() {
            return Err("template engine: template library built empty".to_string());
        }
        Ok(Self {
            templates,
            charset_len: templates::LATIN_CHARSET.chars().count(),
            cjk: false,
        })
    }

    /// Build the **CJK union library** (ASCII + Han + Kana + Hangul).
    /// Uses grid-batched rendering for speed; templates carry coarse
    /// vectors and the engine uses the codepoint-partitioned shortlist
    /// to guarantee both script families have recall.
    pub fn with_cjk_union(renderer: &dyn TemplateRenderer) -> Result<Self, String> {
        if renderer.face_count() == 0 {
            return Err(
                "template engine: no system fonts found — cannot build the CJK template library"
                    .to_string(),
            );
        }
        let charset = templates::cjk_union_charset();
        let templates = templates::build_templates_for_options(
            renderer,
            &charset,
            templates::CJK_SIZES,
            true, // coarse vectors for shortlist recall
            true, // cjk: grid batching + CJK_SCALE_REF
        );
        if templates.is_empty() {
            return Err("template engine: CJK template library built empty".to_string());
        }
        Ok(Self {
            templates,
            charset_len: charset.chars().count(),
            cjk: true,
        })
    }

    /// Convenience: build the CJK union library from the platform's
    /// CJK system font (Hiragino / Noto CJK / YaHei).
    pub fn with_system_cjk() -> Result<Self, String> {
        let renderer = RustTypeRenderer::for_cjk();
        Self::with_cjk_union(&renderer)
    }

    /// Number of templates in the library.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Charset size the library was built from.
    pub fn charset_len(&self) -> usize {
        self.charset_len
    }

    /// Recognize a grayscale image (WebOCR `runOcr`). CJK mode is
    /// auto-detected from the library so callers don't have to
    /// remember to set the flag.
    pub fn recognize(&self, gray: &GrayImage) -> TemplateOcrResult {
        let options = RunOptions { cjk: self.cjk, ..RunOptions::default() };
        run_ocr(gray, &self.templates, options)
    }

    /// Recognize with custom run options (caller can override cjk).
    pub fn recognize_with(
        &self,
        gray: &GrayImage,
        options: RunOptions,
    ) -> TemplateOcrResult {
        run_ocr(gray, &self.templates, options)
    }
}

impl std::fmt::Debug for TemplateOcr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateOcr")
            .field("templates", &self.templates.len())
            .field("charset_len", &self.charset_len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use templates::TemplateRenderer;

    /// 5×7 pixel font at cell=3 (cell=1 dies in `filter_noise`; cell=2 has
    /// too little resolution for the post-blur cosine to discriminate
    /// 'H' from 'i' at 30-px frame). Minimal end-to-end proof on synthetic
    /// glyphs with no system-font dependence.
    struct PixelFont;

    fn glyph_bits(ch: char) -> Option<[&'static str; 7]> {
        Some(match ch {
            'H' => ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
            'i' => ["00100", "00000", "00100", "00100", "00100", "00100", "00100"],
            _ => return None,
        })
    }

    impl TemplateRenderer for PixelFont {
        fn face_count(&self) -> usize {
            1
        }
        fn render_char(&self, _face: usize, ch: char, _px: u32) -> Option<RenderedGlyphLike> {
            let rows = glyph_bits(ch)?;
            let cell = 3usize;
            let pad = 4usize;
            let w = 5 * cell + 2 * pad; // 23
            let h = 7 * cell + 2 * pad + 1; // 26
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
            Some(RenderedGlyphLike {
                width: w,
                height: h,
                data,
                baseline: (pad + 7 * cell) as f32,
            })
        }
    }

    // Local alias so the test doesn't import the trait's associated name.
    type RenderedGlyphLike = templates::RenderedGlyph;

    #[test]
    fn facade_builds_library_and_reads_a_word() {
        let ocr = TemplateOcr::with_renderer(&PixelFont).unwrap();
        // 2 chars × the 5 rasters of DEFAULT_SIZES, with the two smallest
        // rasters carrying the three phase variants (the synthetic renderer
        // ignores the phase, so the variants are duplicates — the count is
        // what matters here).
        assert_eq!(ocr.template_count(), 2 * (3 + 3 + 1 + 1 + 1));

        // End-to-end proof: render two characters on a page, run the
        // pipeline, and check both characters are recognized. Word
        // grouping is tested separately in segment::tests — a 2-glyph
        // fixture is too fragile to assert on.
        const W: usize = 50;
        const H: usize = 30; // matches PixelFont bitmap height (cell=3: 7·3+2·4+1)
        let mut page = vec![255u8; W * H];
        let hi = [
            (PixelFont.render_char(0, 'H', 10).unwrap(), 0usize),
            (PixelFont.render_char(0, 'i', 10).unwrap(), 20usize),
        ];
        for (img, ox) in hi {
            for y in 0..img.height {
                for x in 0..img.width {
                    page[y * W + ox + x] = img.data[y * img.width + x];
                }
            }
        }
        let gray = GrayImage {
            width: W,
            height: H,
            data: page,
        };
        let result = ocr.recognize(&gray);
        assert_eq!(result.lines.len(), 1);
        let chars: String = result.lines[0].chars.iter().map(|c| c.char).collect();
        assert!(chars.contains('H') && chars.contains('i'), "got {chars:?}");
        // Per-glyph scores must be above the noise floor.
        for ch in &result.lines[0].chars {
            assert!(ch.score > 0.5, "char {:?} scored {:.3}", ch.char, ch.score);
        }
    }

    #[test]
    fn empty_renderer_is_an_error_not_a_panic() {
        struct NoFonts;
        impl TemplateRenderer for NoFonts {
            fn face_count(&self) -> usize {
                0
            }
            fn render_char(&self, _: usize, _: char, _: u32) -> Option<RenderedGlyphLike> {
                None
            }
        }
        assert!(TemplateOcr::with_renderer(&NoFonts).is_err());
    }
}
