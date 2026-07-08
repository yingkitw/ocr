//! Text recognition engines for OCR

pub mod basic_ocr;
pub mod cnn_model;
pub mod crnn;
pub mod ctc_decoder;
pub mod end_to_end_model;
pub mod engine;
pub mod font_attributes;
pub mod hybrid_model;
pub mod lstm;
pub mod lstm_model;
pub mod pattern;
pub mod pattern_model;
pub mod tesseract_blob;
pub mod tesseract_features;
pub mod tesseract_textline;
pub mod transformer_model;
pub mod vit_model;

pub use cnn_model::*;
pub use crnn::*;
pub use end_to_end_model::*;
pub use engine::*;
pub use hybrid_model::*;
pub use lstm::*;
pub use lstm_model::*;
pub use pattern::*;
pub use pattern_model::*;
pub use transformer_model::*;
pub use vit_model::*;

#[cfg(test)]
mod tests_recognition;

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
