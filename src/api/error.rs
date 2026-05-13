//! Error types for OCR API

use crate::utils::OcrError;
use thiserror::Error;

/// API-specific error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("OCR processing error: {0}")]
    OcrProcessing(OcrError),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    #[error("Text processing error: {0}")]
    TextProcessing(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias for API operations
pub type ApiResult<T> = std::result::Result<T, ApiError>;

/// Convert ApiError to OcrError
impl From<ApiError> for OcrError {
    fn from(err: ApiError) -> Self {
        match err {
            ApiError::OcrProcessing(e) => e,
            ApiError::Configuration(msg) => OcrError::Configuration(msg),
            ApiError::ImageProcessing(msg) => OcrError::ImageProcessing(msg),
            ApiError::TextProcessing(msg) => OcrError::Internal(msg),
            ApiError::Validation(msg) => OcrError::InvalidInput(msg),
            ApiError::Serialization(e) => OcrError::Serialization(e),
            ApiError::Io(e) => OcrError::Io(e),
            ApiError::Timeout(msg) => OcrError::Internal(msg),
            ApiError::RateLimitExceeded(msg) => OcrError::Internal(msg),
            ApiError::Authentication(msg) => OcrError::Internal(msg),
            ApiError::Internal(msg) => OcrError::Internal(msg),
        }
    }
}

/// Convert anyhow::Error to ApiError
impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::Internal(err.to_string())
    }
}

/// Convert tokio::sync::AcquireError to ApiError
impl From<tokio::sync::AcquireError> for ApiError {
    fn from(err: tokio::sync::AcquireError) -> Self {
        ApiError::Internal(format!("Semaphore acquire failed: {}", err))
    }
}

/// Convert OcrError to ApiError
impl From<OcrError> for ApiError {
    fn from(err: OcrError) -> Self {
        match err {
            OcrError::ImageProcessing(msg) => ApiError::ImageProcessing(msg),
            OcrError::Recognition(msg) => {
                ApiError::OcrProcessing(OcrError::Recognition(msg))
            }
            OcrError::LayoutAnalysis(msg) => {
                ApiError::OcrProcessing(OcrError::LayoutAnalysis(msg))
            }
            OcrError::LanguageSupport(msg) => {
                ApiError::OcrProcessing(OcrError::LanguageSupport(msg))
            }
            OcrError::Io(e) => ApiError::Io(e),
            OcrError::Serialization(e) => ApiError::Serialization(e),
            OcrError::InvalidInput(msg) => ApiError::Validation(msg),
            OcrError::Configuration(msg) => ApiError::Configuration(msg),
            OcrError::Internal(msg) => ApiError::Internal(msg),
            OcrError::SemaphoreAcquire(msg) => {
                ApiError::Internal(format!("Semaphore acquire failed: {}", msg))
            }
            OcrError::ModelNotFound(msg) => {
                ApiError::Internal(format!("Model not found: {}", msg))
            }
        }
    }
}
