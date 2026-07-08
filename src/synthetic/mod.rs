//! Synthetic training data generation for OCR
//!
//! Generates text-line images from fonts with realistic distortions.
//! This allows training and benchmarking OCR engines without manually
//! collecting labeled real-world data.
//!
//! # Example
//! ```
//! use ocr::synthetic::{TextLineGenerator, DistortionConfig};
//!
//! let generator = TextLineGenerator::default();
//! let sample = generator.generate("Hello World");
//! ```

pub mod benchmark;
pub mod bitmap_font;
pub mod distortion;
pub mod document_generator;
pub mod generator;
pub mod multi_script;
pub mod template_trainer;

pub use benchmark::*;
pub use bitmap_font::*;
pub use distortion::*;
pub use document_generator::*;
pub use generator::*;
pub use multi_script::*;
pub use template_trainer::*;

