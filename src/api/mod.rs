//! High-level API for MiniOCR

pub mod config;
pub mod error;
pub mod image;
pub mod ocr;
pub mod text;

pub use config::*;
pub use error::*;
pub use image::*;
pub use ocr::*;
pub use text::*;

/// Re-export commonly used types
pub use crate::utils::{MiniOcrError, Result};
