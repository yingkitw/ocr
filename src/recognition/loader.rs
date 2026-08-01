//! Model loading from binary blobs (file or embedded bytes).
//!
//! Bundled models live in `models/*.bin` and are embedded at compile time via
//! `include_bytes!`. To replace a bundled model, drop a new file with the same
//! name in `models/` and rebuild.

use crate::recognition::crnn::CrnnModel;
use crate::recognition::model_format;
use crate::utils::{OcrError, Result};

/// Load a CRNN model from raw bytes (the binary format defined in
/// `model_format`).
pub fn load_from_bytes(bytes: &[u8]) -> Result<CrnnModel> {
    model_format::read_model(bytes)
}

/// Load a CRNN model from a file path.
pub fn load_from_path(path: &std::path::Path) -> Result<CrnnModel> {
    let bytes = std::fs::read(path).map_err(|e| {
        OcrError::Configuration(format!("failed to read model at {path:?}: {e}"))
    })?;
    load_from_bytes(&bytes)
}

/// The bundled English/Latin CRNN model.
///
/// This is the v0.2.0 trained model, embedded at compile time from
/// `models/crnn-en.bin`. To regenerate, run `cargo run --release -- train
/// --output models/crnn-en.bin` and rebuild the crate.
pub const BUNDLED_MODEL_BYTES: &[u8] = include_bytes!("../../models/crnn-en.bin");

/// Load the bundled English/Latin model.
pub fn load_bundled() -> Result<CrnnModel> {
    load_from_bytes(BUNDLED_MODEL_BYTES)
}
