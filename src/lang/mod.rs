//! Language support and character sets for MiniOCR

pub mod cjk;
pub mod detector;
pub mod dictionary;
mod ngram;
pub mod unicharset;
pub mod unicode;

pub use cjk::*;
pub use detector::*;
pub use dictionary::*;
pub use ngram::*;
pub use unicharset::*;
pub use unicode::*;

/// Re-export commonly used types
pub use crate::utils::{MiniOcrError, Result};
