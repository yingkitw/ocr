//! Shared utilities and common functionality for OCR

pub mod async_utils;
pub mod error;
pub mod hash;
pub mod math;
pub mod quantization;
pub mod simd;
pub mod simd_advanced;
pub mod time;

pub use async_utils::*;
pub use error::*;
pub use hash::*;
pub use math::*;
pub use quantization::*;
pub use simd::*;
pub use simd_advanced::*;
pub use time::*;

/// Re-export commonly used types
pub use thiserror::Error;
