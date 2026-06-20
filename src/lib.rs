#![allow(ambiguous_glob_reexports)]

pub mod api;
pub mod compute;
pub mod core;
pub mod image;
pub mod lang;
pub mod layout;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod recognition;
#[cfg(feature = "web-api")]
pub mod server;
pub mod synthetic;
pub mod training;
pub mod utils;

pub use api::*;
pub use utils::{OcrError, Result};
