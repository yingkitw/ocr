//! Generate a placeholder bundled model file at `models/crnn-en.bin`.
//!
//! Run with: `cargo run --example generate_default_model`
//!
//! This produces a freshly-initialised CRNN model that satisfies
//! `include_bytes!("../../models/crnn-en.bin")` so the crate compiles
//! before any real training run has happened. Replace the file with
//! the output of `cargo run --release -- train` to ship a trained model.

use ocr::recognition::crnn::{CrnnConfig, CrnnModel};
use ocr::recognition::model_format::write_model;
use std::io::Write;

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models/crnn-en.bin");
    let model = CrnnModel::new(CrnnConfig::default());
    let bytes = write_model(&model).expect("write_model");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(&bytes).expect("write");
    println!("Wrote {} bytes to {}", bytes.len(), path.display());
}
