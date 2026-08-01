//! Text recognition engines for OCR

pub mod basic_ocr;
pub mod confidence;
pub mod crnn;
pub mod ctc_decoder;
pub mod engine;
pub mod font_attributes;
pub mod lstm_model;
pub mod pattern_model;
pub mod tesseract_blob;
pub mod tesseract_features;

pub use confidence::*;
pub use crnn::*;
pub use engine::*;
pub use lstm_model::*;
pub use pattern_model::*;

#[cfg(test)]
mod tests_recognition;

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
