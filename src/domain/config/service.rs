//! Configuration Service
//!
//! High-level service for configuration management and validation.
//! Handles loading, saving, and validating OCR configuration.

use super::ConfigurationError;
use crate::core::config::OcrConfig;
use std::path::Path;

pub struct ConfigurationService {
    config: OcrConfig,
}

impl ConfigurationService {
    pub fn new() -> Self {
        Self {
            config: OcrConfig::default(),
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<OcrConfig, ConfigurationError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigurationError::LoadFailed(format!("Failed to read config file: {}", e)))?;
        
        let config: OcrConfig = serde_json::from_str(&content)
            .map_err(|e| ConfigurationError::LoadFailed(format!("Failed to parse config: {}", e)))?;
        
        self.validate(&config)?;
        Ok(config)
    }

    pub fn save_to_file<P: AsRef<Path>>(
        &self,
        config: &OcrConfig,
        path: P,
    ) -> Result<(), ConfigurationError> {
        let content = serde_json::to_string_pretty(config)
            .map_err(|e| ConfigurationError::SaveFailed(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| ConfigurationError::SaveFailed(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    pub fn validate(&self, config: &OcrConfig) -> Result<(), ConfigurationError> {
        config.validate()
            .map_err(|e| ConfigurationError::ValidationFailed(e.to_string()))?;
        Ok(())
    }

    pub fn create_default(&self) -> OcrConfig {
        OcrConfig::default()
    }

    pub fn get_config(&self) -> &OcrConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: OcrConfig) {
        self.config = config;
    }

    pub fn merge_configs(
        &self,
        base: &OcrConfig,
        override_config: &OcrConfig,
    ) -> OcrConfig {
        let mut merged = base.clone();
        
        if override_config.recognition.language != "en" {
            merged.recognition.language = override_config.recognition.language.clone();
        }
        if override_config.recognition.confidence_threshold != 0.5 {
            merged.recognition.confidence_threshold = override_config.recognition.confidence_threshold;
        }
        if override_config.recognition.engine != crate::core::config::RecognitionEngine::LSTM {
            merged.recognition.engine = override_config.recognition.engine;
        }
        if override_config.performance.device != "auto" {
            merged.performance.device = override_config.performance.device.clone();
        }
        
        merged
    }
}

impl Default for ConfigurationService {
    fn default() -> Self {
        Self::new()
    }
}
