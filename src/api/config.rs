//! Configuration utilities for OCR API

use crate::api::error::{ApiError, ApiResult};
use crate::core::config::OcrConfig;
use serde::{Deserialize, Serialize};

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// OCR configuration
    pub ocr: OcrConfig,
    /// API-specific settings
    pub api: ApiSettings,
    /// Performance settings
    pub performance: PerformanceSettings,
    /// Logging settings
    pub logging: LoggingSettings,
}

/// API-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSettings {
    /// API version
    pub version: String,
    /// API timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum request size in MB
    pub max_request_size_mb: usize,
    /// Enable CORS
    pub enable_cors: bool,
    /// CORS origins
    pub cors_origins: Vec<String>,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Rate limit requests per minute
    pub rate_limit_per_minute: u32,
    /// Enable authentication
    pub enable_authentication: bool,
    /// API key
    pub api_key: Option<String>,
}

impl Default for ApiSettings {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            timeout_seconds: 300,
            max_request_size_mb: 100,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            enable_rate_limiting: false,
            rate_limit_per_minute: 1000,
            enable_authentication: false,
            api_key: None,
        }
    }
}

/// Performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Enable connection pooling
    pub enable_connection_pooling: bool,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size in MB
    pub cache_size_mb: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            max_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            enable_connection_pooling: true,
            connection_pool_size: 10,
            enable_caching: true,
            cache_size_mb: 512,
            cache_ttl_seconds: 3600,
        }
    }
}

/// Logging settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Log level
    pub log_level: LogLevel,
    /// Enable structured logging
    pub enable_structured_logging: bool,
    /// Log format
    pub log_format: LogFormat,
    /// Enable request logging
    pub enable_request_logging: bool,
    /// Enable response logging
    pub enable_response_logging: bool,
    /// Log file path
    pub log_file_path: Option<String>,
    /// Enable console logging
    pub enable_console_logging: bool,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            enable_structured_logging: true,
            log_format: LogFormat::Json,
            enable_request_logging: true,
            enable_response_logging: false,
            log_file_path: None,
            enable_console_logging: true,
        }
    }
}

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    /// Error level
    Error,
    /// Warning level
    Warning,
    /// Info level
    Info,
    /// Debug level
    Debug,
    /// Trace level
    Trace,
}

/// Log format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    /// Plain text format
    Plain,
    /// JSON format
    Json,
    /// Compact format
    Compact,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            ocr: OcrConfig::default(),
            api: ApiSettings::default(),
            performance: PerformanceSettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

impl ApiConfig {
    /// Create a new API configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> ApiResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> ApiResult<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> ApiResult<()> {
        // Validate OCR configuration
        self.ocr.validate()?;

        // Validate API settings
        if self.api.timeout_seconds == 0 {
            return Err(ApiError::Configuration(
                "API timeout must be greater than 0".to_string(),
            ));
        }

        if self.api.max_request_size_mb == 0 {
            return Err(ApiError::Configuration(
                "Maximum request size must be greater than 0".to_string(),
            ));
        }

        if self.api.rate_limit_per_minute == 0 {
            return Err(ApiError::Configuration(
                "Rate limit must be greater than 0".to_string(),
            ));
        }

        // Validate performance settings
        if self.performance.max_concurrent_requests == 0 {
            return Err(ApiError::Configuration(
                "Maximum concurrent requests must be greater than 0".to_string(),
            ));
        }

        if self.performance.max_threads == 0 {
            return Err(ApiError::Configuration(
                "Maximum threads must be greater than 0".to_string(),
            ));
        }

        if self.performance.connection_pool_size == 0 {
            return Err(ApiError::Configuration(
                "Connection pool size must be greater than 0".to_string(),
            ));
        }

        if self.performance.cache_size_mb == 0 {
            return Err(ApiError::Configuration(
                "Cache size must be greater than 0".to_string(),
            ));
        }

        if self.performance.cache_ttl_seconds == 0 {
            return Err(ApiError::Configuration(
                "Cache TTL must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get a configuration parameter
    pub fn get_parameter(&self, key: &str) -> Option<String> {
        // Check OCR parameters
        if let Some(value) = self.ocr.get_parameter(key) {
            return Some(value.clone());
        }

        // Check API parameters
        match key {
            "api.version" => Some(self.api.version.clone()),
            "api.timeout_seconds" => Some(self.api.timeout_seconds.to_string()),
            "api.max_request_size_mb" => Some(self.api.max_request_size_mb.to_string()),
            "api.enable_cors" => Some(self.api.enable_cors.to_string()),
            "api.enable_rate_limiting" => Some(self.api.enable_rate_limiting.to_string()),
            "api.rate_limit_per_minute" => Some(self.api.rate_limit_per_minute.to_string()),
            "api.enable_authentication" => Some(self.api.enable_authentication.to_string()),
            _ => None,
        }
    }

    /// Set a configuration parameter
    pub fn set_parameter(&mut self, key: &str, value: String) -> ApiResult<()> {
        match key {
            "api.version" => {
                self.api.version = value;
            }
            "api.timeout_seconds" => {
                self.api.timeout_seconds = value
                    .parse()
                    .map_err(|_| ApiError::Configuration("Invalid timeout value".to_string()))?;
            }
            "api.max_request_size_mb" => {
                self.api.max_request_size_mb = value.parse().map_err(|_| {
                    ApiError::Configuration("Invalid max request size value".to_string())
                })?;
            }
            "api.enable_cors" => {
                self.api.enable_cors = value
                    .parse()
                    .map_err(|_| ApiError::Configuration("Invalid CORS value".to_string()))?;
            }
            "api.enable_rate_limiting" => {
                self.api.enable_rate_limiting = value.parse().map_err(|_| {
                    ApiError::Configuration("Invalid rate limiting value".to_string())
                })?;
            }
            "api.rate_limit_per_minute" => {
                self.api.rate_limit_per_minute = value
                    .parse()
                    .map_err(|_| ApiError::Configuration("Invalid rate limit value".to_string()))?;
            }
            "api.enable_authentication" => {
                self.api.enable_authentication = value.parse().map_err(|_| {
                    ApiError::Configuration("Invalid authentication value".to_string())
                })?;
            }
            _ => {
                // Set in OCR parameters
                self.ocr.set_parameter(key, value);
            }
        }

        Ok(())
    }
}

/// Configuration builder
pub struct ConfigBuilder {
    config: ApiConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: ApiConfig::default(),
        }
    }

    /// Set OCR configuration
    pub fn with_ocr_config(mut self, ocr_config: OcrConfig) -> Self {
        self.config.ocr = ocr_config;
        self
    }

    /// Set API version
    pub fn with_api_version(mut self, version: String) -> Self {
        self.config.api.version = version;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.config.api.timeout_seconds = timeout_seconds;
        self
    }

    /// Set maximum request size
    pub fn with_max_request_size(mut self, max_size_mb: usize) -> Self {
        self.config.api.max_request_size_mb = max_size_mb;
        self
    }

    /// Enable CORS
    pub fn with_cors(mut self, origins: Vec<String>) -> Self {
        self.config.api.enable_cors = true;
        self.config.api.cors_origins = origins;
        self
    }

    /// Enable rate limiting
    pub fn with_rate_limiting(mut self, requests_per_minute: u32) -> Self {
        self.config.api.enable_rate_limiting = true;
        self.config.api.rate_limit_per_minute = requests_per_minute;
        self
    }

    /// Enable authentication
    pub fn with_authentication(mut self, api_key: String) -> Self {
        self.config.api.enable_authentication = true;
        self.config.api.api_key = Some(api_key);
        self
    }

    /// Set maximum concurrent requests
    pub fn with_max_concurrent_requests(mut self, max_requests: usize) -> Self {
        self.config.performance.max_concurrent_requests = max_requests;
        self
    }

    /// Set maximum threads
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.config.performance.max_threads = max_threads;
        self
    }

    /// Enable caching
    pub fn with_caching(mut self, cache_size_mb: usize, ttl_seconds: u64) -> Self {
        self.config.performance.enable_caching = true;
        self.config.performance.cache_size_mb = cache_size_mb;
        self.config.performance.cache_ttl_seconds = ttl_seconds;
        self
    }

    /// Set log level
    pub fn with_log_level(mut self, log_level: LogLevel) -> Self {
        self.config.logging.log_level = log_level;
        self
    }

    /// Set log format
    pub fn with_log_format(mut self, log_format: LogFormat) -> Self {
        self.config.logging.log_format = log_format;
        self
    }

    /// Enable file logging
    pub fn with_file_logging(mut self, log_file_path: String) -> Self {
        self.config.logging.log_file_path = Some(log_file_path);
        self
    }

    /// Build the configuration
    pub fn build(self) -> ApiResult<ApiConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
