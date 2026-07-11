//! Hashing utilities for OCR (stdlib only)

use std::hash::{Hash, Hasher};

/// Calculate a simple hash for a value
pub fn simple_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Create a hash from multiple values
pub fn hash_multiple<T: Hash>(values: &[T]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in values {
        value.hash(&mut hasher);
    }
    hasher.finish()
}
