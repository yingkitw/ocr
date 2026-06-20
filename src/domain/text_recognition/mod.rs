//! Text Recognition Domain Module
//!
//! Provides high-level text recognition capabilities and orchestration.
//! This module serves as facade for all text recognition operations,
//! coordinating between lower-level components.

use anyhow::Result;
use std::sync::Arc;

use crate::api::error::ApiError;
use crate::core::text::TextResult;
use crate::core::{OcrImage, OcrEngine, config::OcrConfig, layout::LayoutResult};

pub use service::TextRecognitionService;

mod service;

/// Text recognition domain error types
#[derive(Debug, thiserror::Error)]
pub enum TextRecognitionError {
    #[error("Recognition engine initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Image processing failed: {0}")]
    ImageProcessingFailed(String),
    
    #[error("Layout analysis failed: {0}")]
    LayoutAnalysisFailed(String),
    
    #[error("Recognition failed: {0}")]
    RecognitionFailed(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

impl From<TextRecognitionError> for ApiError {
    fn from(err: TextRecognitionError) -> Self {
        ApiError::OcrProcessing(crate::utils::OcrError::Recognition(err.to_string()))
    }
}

impl From<TextRecognitionError> for ApiError {
    fn from(err: TextRecognitionError) -> Self {
        ApiError::OcrProcessing(crate::utils::OcrError::Recognition(err.to_string()))
    }
}

/// Text recognition result with metadata
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Text result
    pub text_result: TextResult,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Confidence score
    pub confidence: f32,
    /// Engine used
    pub engine: String,
}

impl RecognitionResult {
    pub fn new(text_result: TextResult, processing_time_ms: u64, engine: String) -> Self {
        let confidence = text_result.confidence;
        Self {
            text_result,
            processing_time_ms,
            confidence,
            engine,
        }
    }
}

/// Builder for text recognition operations
pub struct TextRecognitionBuilder {
    config: Option<OcrConfig>,
}

impl TextRecognitionBuilder {
    pub fn new() -> Self {
        Self { config: None }
    }

    pub fn with_config(mut self, config: OcrConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> Result<TextRecognitionService, TextRecognitionError> {
        let config = self.config.unwrap_or_default();
        let engine = OcrEngine::with_config(config.clone())
            .map_err(|e| TextRecognitionError::InitializationFailed(e.to_string()))?;
        
        Ok(TextRecognitionService::new(engine, config))
    }
}

impl Default for TextRecognitionBuilder {
    fn default() -> Self {
        Self::new()
    }
}
