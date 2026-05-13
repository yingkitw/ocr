//! Training configuration and hyperparameters

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Data configuration
    pub data: DataConfig,
    /// Training hyperparameters
    pub training: TrainingHyperparams,
    /// Optimizer configuration
    pub optimizer: OptimizerConfig,
    /// Learning rate scheduler configuration
    pub scheduler: SchedulerConfig,
    /// Logging and monitoring configuration
    pub logging: LoggingConfig,
    /// Checkpoint configuration
    pub checkpoint: CheckpointConfig,
    /// Hardware configuration
    pub hardware: HardwareConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type (LSTM, Transformer, ViT, CNN, Hybrid, EndToEnd)
    pub model_type: String,
    /// Model architecture parameters
    pub architecture: HashMap<String, serde_json::Value>,
    /// Input image size (width, height, channels)
    pub input_size: (u32, u32, u32),
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Number of classes for classification
    pub num_classes: usize,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Dataset path
    pub dataset_path: String,
    /// Dataset format
    pub dataset_format: String,
    /// Batch size for training
    pub batch_size: usize,
    /// Batch size for validation
    pub val_batch_size: usize,
    /// Number of data loading workers
    pub num_workers: usize,
    /// Whether to shuffle training data
    pub shuffle: bool,
    /// Data augmentation configuration
    pub augmentation: AugmentationConfig,
    /// Data preprocessing configuration
    pub preprocessing: PreprocessingConfig,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Whether to enable augmentation
    pub enabled: bool,
    /// Random rotation range in degrees
    pub rotation_range: (f32, f32),
    /// Random scaling range
    pub scale_range: (f32, f32),
    /// Random translation range
    pub translation_range: (f32, f32),
    /// Random brightness adjustment range
    pub brightness_range: (f32, f32),
    /// Random contrast adjustment range
    pub contrast_range: (f32, f32),
    /// Random noise level
    pub noise_level: f32,
    /// Random blur probability
    pub blur_probability: f32,
    /// Random crop probability
    pub crop_probability: f32,
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Image normalization mean
    pub mean: Vec<f32>,
    /// Image normalization std
    pub std: Vec<f32>,
    /// Whether to convert to grayscale
    pub grayscale: bool,
    /// Target image size
    pub target_size: (u32, u32),
    /// Whether to preserve aspect ratio
    pub preserve_aspect_ratio: bool,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparams {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Gradient clipping threshold
    pub gradient_clip_norm: Option<f32>,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Label smoothing factor
    pub label_smoothing: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Weight decay
    pub weight_decay: f32,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type (SGD, Adam, RMSprop)
    pub optimizer_type: String,
    /// Learning rate
    pub learning_rate: f32,
    /// Optimizer-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type (StepLR, ExponentialLR, CosineAnnealingLR)
    pub scheduler_type: String,
    /// Scheduler-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Logging and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    /// Log directory
    pub log_dir: String,
    /// Whether to log to console
    pub console_logging: bool,
    /// Whether to log to file
    pub file_logging: bool,
    /// Logging interval in steps
    pub log_interval: usize,
    /// Metrics to track
    pub metrics: Vec<String>,
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Checkpoint directory
    pub checkpoint_dir: String,
    /// Save interval in epochs
    pub save_interval: usize,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Whether to save best model
    pub save_best: bool,
    /// Metric to use for best model selection
    pub best_metric: String,
    /// Whether to save optimizer state
    pub save_optimizer: bool,
    /// Whether to save scheduler state
    pub save_scheduler: bool,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Device type (CPU, CUDA, OpenCL)
    pub device: String,
    /// Number of GPUs to use
    pub num_gpus: usize,
    /// GPU memory fraction to use
    pub gpu_memory_fraction: f32,
    /// Whether to use mixed precision
    pub use_mixed_precision: bool,
    /// Number of CPU threads
    pub num_threads: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            data: DataConfig::default(),
            training: TrainingHyperparams::default(),
            optimizer: OptimizerConfig::default(),
            scheduler: SchedulerConfig::default(),
            logging: LoggingConfig::default(),
            checkpoint: CheckpointConfig::default(),
            hardware: HardwareConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: "LSTM".to_string(),
            architecture: HashMap::new(),
            input_size: (224, 224, 3),
            vocab_size: 1000,
            max_seq_length: 128,
            num_classes: 1000,
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            dataset_path: "./data".to_string(),
            dataset_format: "synthetic".to_string(),
            batch_size: 32,
            val_batch_size: 64,
            num_workers: 4,
            shuffle: true,
            augmentation: AugmentationConfig::default(),
            preprocessing: PreprocessingConfig::default(),
        }
    }
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rotation_range: (-5.0, 5.0),
            scale_range: (0.9, 1.1),
            translation_range: (-10.0, 10.0),
            brightness_range: (0.8, 1.2),
            contrast_range: (0.8, 1.2),
            noise_level: 0.01,
            blur_probability: 0.1,
            crop_probability: 0.1,
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            grayscale: false,
            target_size: (224, 224),
            preserve_aspect_ratio: true,
        }
    }
}

impl Default for TrainingHyperparams {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            early_stopping_patience: 10,
            gradient_clip_norm: Some(1.0),
            mixed_precision: false,
            label_smoothing: 0.0,
            dropout_rate: 0.1,
            weight_decay: 1e-4,
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert(
            "momentum".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()),
        );
        parameters.insert(
            "weight_decay".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-4).unwrap()),
        );

        Self {
            optimizer_type: "Adam".to_string(),
            learning_rate: 1e-3,
            parameters,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert(
            "step_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(30)),
        );
        parameters.insert(
            "gamma".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.1).unwrap()),
        );

        Self {
            scheduler_type: "StepLR".to_string(),
            parameters,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            log_dir: "./logs".to_string(),
            console_logging: true,
            file_logging: true,
            log_interval: 100,
            metrics: vec![
                "loss".to_string(),
                "accuracy".to_string(),
                "learning_rate".to_string(),
            ],
        }
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: "./checkpoints".to_string(),
            save_interval: 5,
            max_checkpoints: 5,
            save_best: true,
            best_metric: "val_accuracy".to_string(),
            save_optimizer: true,
            save_scheduler: true,
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device: "CPU".to_string(),
            num_gpus: 0,
            gpu_memory_fraction: 0.8,
            use_mixed_precision: false,
            num_threads: 4, // Default to 4 threads
        }
    }
}

impl TrainingConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: TrainingConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate model configuration
        if self.model.vocab_size == 0 {
            return Err(anyhow::anyhow!("Vocabulary size must be greater than 0"));
        }

        if self.model.max_seq_length == 0 {
            return Err(anyhow::anyhow!(
                "Maximum sequence length must be greater than 0"
            ));
        }

        // Validate data configuration
        if self.data.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be greater than 0"));
        }

        if self.data.val_batch_size == 0 {
            return Err(anyhow::anyhow!(
                "Validation batch size must be greater than 0"
            ));
        }

        // Validate training hyperparameters
        if self.training.num_epochs == 0 {
            return Err(anyhow::anyhow!("Number of epochs must be greater than 0"));
        }

        if self.optimizer.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be greater than 0"));
        }

        // Validate optimizer configuration
        if self.optimizer.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!(
                "Optimizer learning rate must be greater than 0"
            ));
        }

        // Validate hardware configuration
        if self.hardware.num_gpus > 0 && self.hardware.device == "CPU" {
            return Err(anyhow::anyhow!("Cannot use GPUs with CPU device"));
        }

        Ok(())
    }

    /// Get learning rate for given epoch and step
    pub fn get_learning_rate(&self, epoch: usize, step: usize) -> f32 {
        match self.scheduler.scheduler_type.as_str() {
            "StepLR" => {
                let step_size = self
                    .scheduler
                    .parameters
                    .get("step_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(30) as usize;
                let gamma = self
                    .scheduler
                    .parameters
                    .get("gamma")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.1) as f32;
                self.optimizer.learning_rate * gamma.powi((step / step_size) as i32)
            }
            "ExponentialLR" => {
                let gamma = self
                    .scheduler
                    .parameters
                    .get("gamma")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.95) as f32;
                self.optimizer.learning_rate * gamma.powi(step as i32)
            }
            "CosineAnnealingLR" => {
                let t_max = self
                    .scheduler
                    .parameters
                    .get("t_max")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(100) as usize;
                let eta_min = self
                    .scheduler
                    .parameters
                    .get("eta_min")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let step = step % t_max;
                eta_min
                    + (self.optimizer.learning_rate - eta_min)
                        * (1.0 + (std::f32::consts::PI * step as f32 / t_max as f32).cos())
                        / 2.0
            }
            _ => self.optimizer.learning_rate,
        }
    }

    /// Create a minimal configuration for testing
    pub fn test_config() -> Self {
        Self {
            model: ModelConfig {
                model_type: "LSTM".to_string(),
                architecture: HashMap::new(),
                input_size: (64, 64, 1),
                vocab_size: 100,
                max_seq_length: 32,
                num_classes: 100,
            },
            data: DataConfig {
                dataset_path: "./test_data".to_string(),
                dataset_format: "synthetic".to_string(),
                batch_size: 4,
                val_batch_size: 8,
                num_workers: 1,
                shuffle: true,
                augmentation: AugmentationConfig {
                    enabled: false,
                    ..Default::default()
                },
                preprocessing: PreprocessingConfig {
                    grayscale: true,
                    target_size: (64, 64),
                    ..Default::default()
                },
            },
            training: TrainingHyperparams {
                num_epochs: 2,
                early_stopping_patience: 1,
                ..Default::default()
            },
            optimizer: OptimizerConfig {
                optimizer_type: "Adam".to_string(),
                learning_rate: 1e-2,
                parameters: HashMap::new(),
            },
            scheduler: SchedulerConfig {
                scheduler_type: "StepLR".to_string(),
                parameters: HashMap::new(),
            },
            logging: LoggingConfig {
                log_level: "debug".to_string(),
                log_dir: "./test_logs".to_string(),
                console_logging: true,
                file_logging: false,
                log_interval: 1,
                metrics: vec!["loss".to_string()],
            },
            checkpoint: CheckpointConfig {
                checkpoint_dir: "./test_checkpoints".to_string(),
                save_interval: 1,
                max_checkpoints: 2,
                save_best: false,
                best_metric: "loss".to_string(),
                save_optimizer: false,
                save_scheduler: false,
            },
            hardware: HardwareConfig {
                device: "CPU".to_string(),
                num_gpus: 0,
                gpu_memory_fraction: 0.8,
                use_mixed_precision: false,
                num_threads: 1,
            },
        }
    }
}
