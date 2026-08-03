//! Text recognition engines for OCR

pub mod backward;
pub mod confidence;
pub mod crnn;
pub mod ctc_decoder;
pub mod engine;
pub mod font_attributes;
pub mod loader;
pub mod model_format;

pub use confidence::*;
pub use crnn::*;
pub use engine::*;
pub use loader::*;

#[cfg(test)]
mod tests_recognition;

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
