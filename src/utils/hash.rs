//! Hashing utilities for OCR

use blake3::Hasher as Blake3Hasher;
use sha2::{Digest, Sha256};
use std::hash::{Hash, Hasher};

/// Hash a slice of bytes using BLAKE3
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Blake3Hasher::new();
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Hash a slice of bytes using SHA-256
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Calculate a simple hash for a value
pub fn simple_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Hash a string using BLAKE3
pub fn hash_string(s: &str) -> [u8; 32] {
    blake3_hash(s.as_bytes())
}

/// Hash a file's contents using BLAKE3
pub async fn hash_file(path: &std::path::Path) -> Result<[u8; 32]> {
    let data = std::fs::read(path)?;
    Ok(blake3_hash(&data))
}

/// Hash an image's pixel data
pub fn hash_image_pixels(pixels: &[u8]) -> [u8; 32] {
    blake3_hash(pixels)
}

/// Create a hash from multiple values
pub fn hash_multiple<T: Hash>(values: &[T]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in values {
        value.hash(&mut hasher);
    }
    hasher.finish()
}

use crate::utils::error::*;
