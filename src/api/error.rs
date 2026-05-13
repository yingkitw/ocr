//! Error types for MiniOCR API

use crate::utils::MiniOcrError;
use thiserror::Error;

/// API-specific error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("OCR processing error: {0}")]
    OcrProcessing(MiniOcrError),

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

/// Convert ApiError to MiniOcrError
impl From<ApiError> for MiniOcrError {
    fn from(err: ApiError) -> Self {
        match err {
            ApiError::OcrProcessing(e) => e,
            ApiError::Configuration(msg) => MiniOcrError::Configuration(msg),
            ApiError::ImageProcessing(msg) => MiniOcrError::ImageProcessing(msg),
            ApiError::TextProcessing(msg) => MiniOcrError::Internal(msg),
            ApiError::Validation(msg) => MiniOcrError::InvalidInput(msg),
            ApiError::Serialization(e) => MiniOcrError::Serialization(e),
            ApiError::Io(e) => MiniOcrError::Io(e),
            ApiError::Timeout(msg) => MiniOcrError::Internal(msg),
            ApiError::RateLimitExceeded(msg) => MiniOcrError::Internal(msg),
            ApiError::Authentication(msg) => MiniOcrError::Internal(msg),
            ApiError::Internal(msg) => MiniOcrError::Internal(msg),
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

/// Convert MiniOcrError to ApiError
impl From<MiniOcrError> for ApiError {
    fn from(err: MiniOcrError) -> Self {
        match err {
            MiniOcrError::ImageProcessing(msg) => ApiError::ImageProcessing(msg),
            MiniOcrError::Recognition(msg) => {
                ApiError::OcrProcessing(MiniOcrError::Recognition(msg))
            }
            MiniOcrError::LayoutAnalysis(msg) => {
                ApiError::OcrProcessing(MiniOcrError::LayoutAnalysis(msg))
            }
            MiniOcrError::LanguageSupport(msg) => {
                ApiError::OcrProcessing(MiniOcrError::LanguageSupport(msg))
            }
            MiniOcrError::Io(e) => ApiError::Io(e),
            MiniOcrError::Serialization(e) => ApiError::Serialization(e),
            MiniOcrError::InvalidInput(msg) => ApiError::Validation(msg),
            MiniOcrError::Configuration(msg) => ApiError::Configuration(msg),
            MiniOcrError::Internal(msg) => ApiError::Internal(msg),
            MiniOcrError::SemaphoreAcquire(msg) => {
                ApiError::Internal(format!("Semaphore acquire failed: {}", msg))
            }
            MiniOcrError::ModelNotFound(msg) => {
                ApiError::Internal(format!("Model not found: {}", msg))
            }
        }
    }
}
