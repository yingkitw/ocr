//! Page layout analysis and text ordering for OCR

pub mod analyzer;
pub mod classifier;
pub mod column_detector;
pub mod detection_cnn;
pub mod detector;
pub mod form_extractor;
pub mod line_detector;
mod text_line_features;
pub mod text_ordering;
pub mod union_find_ccl;

pub use analyzer::*;
pub use classifier::*;
pub use column_detector::*;
pub use detection_cnn::*;
pub use detector::*;
pub use form_extractor::*;
pub use line_detector::*;
pub use text_line_features::*;
pub use text_ordering::*;
pub use union_find_ccl::*;

#[cfg(test)]
mod tests_advanced;

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
