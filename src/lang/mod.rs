//! Language support and character sets for OCR

pub mod cjk;
pub mod confusion_correct;
pub mod detector;
pub mod script_geometry;
pub mod dictionary;
mod ngram;
pub mod unicode;

pub use cjk::*;
pub use confusion_correct::*;
pub use detector::*;
pub use script_geometry::{detect_auto, geometry_script_guess};
pub use dictionary::*;
pub use ngram::*;
pub use unicode::*;

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
