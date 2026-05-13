//! Error types and utilities for MiniOCR

use thiserror::Error;

/// Main error type for MiniOCR operations
#[derive(Error, Debug)]
pub enum MiniOcrError {
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
}

/// Result type alias for MiniOCR operations
pub type Result<T> = std::result::Result<T, MiniOcrError>;

/// Convert anyhow::Error to MiniOcrError
impl From<anyhow::Error> for MiniOcrError {
    fn from(err: anyhow::Error) -> Self {
        MiniOcrError::Internal(err.to_string())
    }
}

/// Convert tokio::sync::AcquireError to MiniOcrError
impl From<tokio::sync::AcquireError> for MiniOcrError {
    fn from(err: tokio::sync::AcquireError) -> Self {
        MiniOcrError::SemaphoreAcquire(err.to_string())
    }
}

/// Convert image::ImageError to MiniOcrError
impl From<image::ImageError> for MiniOcrError {
    fn from(err: image::ImageError) -> Self {
        MiniOcrError::ImageProcessing(err.to_string())
    }
}
