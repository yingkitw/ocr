#![allow(ambiguous_glob_reexports)]

pub mod config;
pub mod engine;
pub mod geometry;
pub mod image;
pub mod layout;
pub mod output;
pub mod recognition;
pub mod text;

pub use config::*;
pub use engine::*;
pub use geometry::*;
pub use image::*;
pub use layout::*;
pub use output::*;
pub use recognition::*;
pub use text::*;

pub use crate::utils::{OcrError, Result};
