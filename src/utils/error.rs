//! Error types and utilities for OCR

use thiserror::Error;

/// Main error type for OCR operations
#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    #[error("Recognition error: {0}")]
    Recognition(String),

    #[error("Layout analysis error: {0}")]
    LayoutAnalysis(String),

    #[error("Language support error: {0}")]
    LanguageSupport(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Semaphore acquire error: {0}")]
    SemaphoreAcquire(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model load error: {0}")]
    ModelLoad(String),
}

/// Result type alias for OCR operations
pub type Result<T> = std::result::Result<T, OcrError>;

/// Convert anyhow::Error to OcrError
impl From<anyhow::Error> for OcrError {
    fn from(err: anyhow::Error) -> Self {
        OcrError::Internal(err.to_string())
    }
}

/// Convert tokio::sync::AcquireError to OcrError
impl From<tokio::sync::AcquireError> for OcrError {
    fn from(err: tokio::sync::AcquireError) -> Self {
        OcrError::SemaphoreAcquire(err.to_string())
    }
}

/// Convert image::ImageError to OcrError
impl From<image::ImageError> for OcrError {
    fn from(err: image::ImageError) -> Self {
        OcrError::ImageProcessing(err.to_string())
    }
}

#[cfg(feature = "opencl")]
/// Convert ocl::Error to OcrError
impl From<ocl::Error> for OcrError {
    fn from(err: ocl::Error) -> Self {
        OcrError::Internal(format!("OpenCL error: {}", err))
    }
}
