//! High-level API for OCR

pub mod config;
pub mod error;
pub mod ocr;
pub mod text;

pub use config::*;
pub use error::*;
pub use ocr::*;
pub use text::*;

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
