//! Configuration Domain Module
//!
//! Provides configuration management and validation for OCR operations.

pub use service::ConfigurationService;

mod service;

/// Configuration domain error types
#[derive(Debug, thiserror::Error)]
pub enum ConfigurationError {
    #[error("Failed to load configuration: {0}")]
    LoadFailed(String),
    
    #[error("Failed to save configuration: {0}")]
    SaveFailed(String),
    
    #[error("Invalid configuration: {0}")]
    ValidationFailed(String),
    
    #[error("Missing required configuration: {0}")]
    MissingRequired(String),
}
