//! Image Processing Domain Module
//!
//! Provides image preprocessing and enhancement capabilities.

pub use service::ImageProcessingService;

mod service;

/// Image processing domain error types
#[derive(Debug, thiserror::Error)]
pub enum ImageProcessingError {
    #[error("Failed to load image: {0}")]
    LoadFailed(String),
    
    #[error("Failed to process image: {0}")]
    ProcessingFailed(String),
    
    #[error("Invalid image format: {0}")]
    InvalidFormat(String),
    
    #[error("Image dimensions too large: {width}x{height}")]
    DimensionsTooLarge { width: u32, height: u32 },
}

/// Image processing result
#[derive(Debug, Clone)]
pub struct ImageProcessingResult {
    /// Original image data
    pub original_data: Vec<u8>,
    /// Processed image data
    pub processed_data: Vec<u8>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Applied operations
    pub applied_operations: Vec<String>,
}
