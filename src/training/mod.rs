//! MiniOCR Training Pipeline
//!
//! This crate provides neural network training capabilities for MiniOCR,
//! including data loading, training loops, loss functions, and model serialization.

pub mod augmentation;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod losses;
pub mod metrics;
pub mod optimizers;
pub mod training;

pub use augmentation::*;
pub use checkpoint::*;
pub use config::*;
pub use data::*;
pub use losses::*;
pub use metrics::*;
pub use optimizers::*;
pub use training::*;
