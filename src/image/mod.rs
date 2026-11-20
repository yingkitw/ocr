//! Image processing and preprocessing for OCR

pub mod dewarp;
pub mod enhancement;
pub mod font_analysis;
pub mod pipeline;
pub mod processor;
pub mod quality;
pub mod super_resolution;
pub mod thresholder;

pub use dewarp::*;
pub use enhancement::*;
pub use font_analysis::*;
pub use pipeline::*;
pub use processor::*;
pub use quality::*;
pub use super_resolution::*;
pub use thresholder::*;

/// Map the user-facing [`BinarizationMethod`] onto the thresholder's
/// [`ThresholdMethod`].
///
/// `BinarizationMethod` mirrors Tesseract's `thresholding_method` numbering
/// (0 = legacy Otsu, 1 = Leptonica Otsu, 2 = Sauvola, 3 = Niblack), so this
/// is the single place where the CLI/config vocabulary meets the algorithms.
pub fn threshold_method_for(method: crate::core::config::BinarizationMethod) -> ThresholdMethod {
    match method {
        crate::core::config::BinarizationMethod::Otsu => ThresholdMethod::Otsu,
        crate::core::config::BinarizationMethod::LeptonicaOtsu => ThresholdMethod::LeptonicaOtsu,
        crate::core::config::BinarizationMethod::Sauvola => ThresholdMethod::Sauvola,
        crate::core::config::BinarizationMethod::Niblack => ThresholdMethod::Niblack,
    }
}

/// Re-export commonly used types
pub use crate::utils::{OcrError, Result};
