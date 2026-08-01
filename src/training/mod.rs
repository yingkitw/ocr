//! OCR Training Pipeline
//!
//! Currently exposes only the CRNN trainer (FC-only backprop on synthetic data).
//! Loss / optimizer / augmentation / checkpoint abstractions were speculative
//! and have been removed until a concrete second consumer arrives.

pub mod crnn_trainer;

pub use crnn_trainer::*;
