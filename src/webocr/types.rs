//! Core data types for the template-matching OCR engine (WebOCR port).
//!
//! Mirrors `webocr/src/ocr/types.ts` + the image contracts of
//! `binarize.ts`/`components.ts`. The engine is pure: it consumes plain
//! grayscale pixel buffers and owns no I/O.

/// Grayscale image, one `u8` per pixel.
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// 1-bit ink mask, one `u8` (0 or 1) per pixel.
#[derive(Debug, Clone)]
pub struct BinaryImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl BinaryImage {
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }
}

/// Bounding box of a connected component or glyph.
/// `x0`/`y0` are inclusive; `x1`/`y1` are exclusive (same as WebOCR `Comp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comp {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub area: usize,
}

impl Comp {
    pub fn width(&self) -> usize {
        self.x1 - self.x0
    }

    pub fn height(&self) -> usize {
        self.y1 - self.y0
    }
}

/// A line of glyph boxes produced by segmentation (WebOCR `LineData`).
#[derive(Debug, Clone)]
pub struct LineData {
    /// Union of all glyph boxes in the line.
    pub box_: Comp,
    /// Left-to-right ordered glyph boxes.
    pub glyphs: Vec<Comp>,
    /// Median glyph height — the line's size reference.
    pub med_h: usize,
    /// Index of the text block this line was extracted from.
    pub block: usize,
}

/// One recognized character with its box and match score (WebOCR `CharBox`).
#[derive(Debug, Clone)]
pub struct CharBox {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub char: char,
    pub score: f32,
}

/// One recognized word: chars grouped at word gaps (word boundaries come
/// from the gap-cluster threshold).
#[derive(Debug, Clone)]
pub struct WordSpan {
    pub text: String,
    pub chars: Vec<CharBox>,
    /// Mean char score.
    pub score: f32,
}

/// A recognized text line (WebOCR `LineResult`).
#[derive(Debug, Clone)]
pub struct LineResult {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub text: String,
    pub chars: Vec<CharBox>,
    /// Chars grouped at word gaps.
    pub words: Vec<WordSpan>,
    /// Text-block index — drives paragraph/column assembly.
    pub block: usize,
}

/// Stage timings, in milliseconds (WebOCR `OcrResult.timing`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timing {
    pub binarize_ms: f64,
    pub segment_ms: f64,
    pub recognize_ms: f64,
    pub total_ms: f64,
}

/// Full result of a template-OCR run (WebOCR `OcrResult`).
#[derive(Debug, Clone)]
pub struct TemplateOcrResult {
    /// Layout-assembled plain text (paragraph gaps, indents, word spaces).
    pub text: String,
    /// Per-line results in reading order.
    pub lines: Vec<LineResult>,
    pub width: usize,
    pub height: usize,
    pub char_count: usize,
    /// Mean match score across all emitted characters, 0–1.
    pub avg_score: f32,
    /// Script the run was executed with (always `"latin"` in this cut).
    pub script: &'static str,
    pub timing: Timing,
}
