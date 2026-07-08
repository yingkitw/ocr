//! Checkpoint management for training

use crate::training::config::CheckpointConfig;
use crate::training::metrics::TrainingMetrics;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Training checkpoint containing model state and training progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Epoch number
    pub epoch: usize,
    /// Model state (parameters, weights, etc.)
    pub model_state: HashMap<String, Vec<f32>>,
    /// Optimizer state
    pub optimizer_state: crate::training::optimizers::OptimizerState,
    /// Training metrics at this checkpoint
    pub metrics: Option<TrainingMetrics>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Checkpoint manager for saving and loading training checkpoints
pub struct CheckpointManager {
    config: CheckpointConfig,
    checkpoint_dir: PathBuf,
    checkpoints: Vec<CheckpointInfo>,
}

#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub path: PathBuf,
    pub epoch: usize,
    pub metrics: Option<TrainingMetrics>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: &CheckpointConfig) -> Result<Self> {
        let checkpoint_dir = PathBuf::from(&config.checkpoint_dir);

        // Create checkpoint directory if it doesn't exist
        if !checkpoint_dir.exists() {
            std::fs::create_dir_all(&checkpoint_dir)?;
        }

        let mut manager = Self {
            config: config.clone(),
            checkpoint_dir,
            checkpoints: Vec::new(),
        };

        // Load existing checkpoints
        manager.load_existing_checkpoints()?;

        Ok(manager)
    }

    /// Save a checkpoint
    pub async fn save_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        let filename = format!("checkpoint_epoch_{:04}.json", checkpoint.epoch);
        let filepath = self.checkpoint_dir.join(&filename);

        // Serialize checkpoint to JSON
        let json = serde_json::to_string_pretty(checkpoint)?;
        fs::write(&filepath, json).await?;

        // Add to checkpoint list
        let checkpoint_info = CheckpointInfo {
            path: filepath.clone(),
            epoch: checkpoint.epoch,
            metrics: checkpoint.metrics.clone(),
            created_at: chrono::Utc::now(),
        };
        self.checkpoints.push(checkpoint_info);

        // Clean up old checkpoints if necessary
        self.cleanup_old_checkpoints().await?;

        tracing::info!("Checkpoint saved: {}", filepath.display());
        Ok(())
    }

    /// Load the latest checkpoint
    pub fn load_latest_checkpoint(&self) -> Result<Option<Checkpoint>> {
        if let Some(latest) = self.checkpoints.last() {
            self.load_checkpoint(&latest.path)
        } else {
            Ok(None)
        }
    }

    /// Load a specific checkpoint by epoch
    pub fn load_checkpoint_by_epoch(&self, epoch: usize) -> Result<Option<Checkpoint>> {
        if let Some(checkpoint_info) = self.checkpoints.iter().find(|c| c.epoch == epoch) {
            self.load_checkpoint(&checkpoint_info.path)
        } else {
            Ok(None)
        }
    }

    /// Load the best checkpoint based on metrics
    pub fn load_best_checkpoint(&self) -> Result<Option<Checkpoint>> {
        if let Some(best) = self.find_best_checkpoint() {
            self.load_checkpoint(&best.path)
        } else {
            Ok(None)
        }
    }

    /// Load checkpoint from file
    fn load_checkpoint(&self, filepath: &Path) -> Result<Option<Checkpoint>> {
        let json = std::fs::read_to_string(filepath)?;
        let checkpoint: Checkpoint = serde_json::from_str(&json)?;
        Ok(Some(checkpoint))
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Vec<CheckpointInfo> {
        self.checkpoints.clone()
    }

    /// Get checkpoint directory
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Load existing checkpoints from disk
    fn load_existing_checkpoints(&mut self) -> Result<()> {
        let mut entries = std::fs::read_dir(&self.checkpoint_dir)?;

        while let Some(entry) = entries.next() {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    if filename.starts_with("checkpoint_epoch_") {
                        // Try to parse epoch number
                        if let Some(epoch_str) = filename.strip_prefix("checkpoint_epoch_") {
                            if let Ok(epoch) = epoch_str.parse::<usize>() {
                                let checkpoint_info = CheckpointInfo {
                                    path: path.clone(),
                                    epoch,
                                    metrics: None, // Will be loaded when needed
                                    created_at: chrono::Utc::now(),
                                };
                                self.checkpoints.push(checkpoint_info);
                            }
                        }
                    }
                }
            }
        }

        // Sort by epoch
        self.checkpoints.sort_by_key(|c| c.epoch);
        Ok(())
    }

    /// Clean up old checkpoints to stay within max_checkpoints limit
    async fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        if self.checkpoints.len() > self.config.max_checkpoints {
            let to_remove = self.checkpoints.len() - self.config.max_checkpoints;

            for _ in 0..to_remove {
                if let Some(checkpoint_info) = self.checkpoints.pop() {
                    if checkpoint_info.path.exists() {
                        fs::remove_file(&checkpoint_info.path).await?;
                        tracing::info!(
                            "Removed old checkpoint: {}",
                            checkpoint_info.path.display()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Find the best checkpoint based on metrics
    fn find_best_checkpoint(&self) -> Option<&CheckpointInfo> {
        if self.checkpoints.is_empty() {
            return None;
        }

        match self.config.best_metric.as_str() {
            "val_accuracy" | "accuracy" => self.checkpoints.iter().max_by(|a, b| {
                let a_acc = a.metrics.as_ref().map(|m| m.accuracy()).unwrap_or(0.0);
                let b_acc = b.metrics.as_ref().map(|m| m.accuracy()).unwrap_or(0.0);
                a_acc
                    .partial_cmp(&b_acc)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            "val_loss" | "loss" => self.checkpoints.iter().min_by(|a, b| {
                let a_loss = a
                    .metrics
                    .as_ref()
                    .map(|m| m.loss())
                    .unwrap_or(f32::INFINITY);
                let b_loss = b
                    .metrics
                    .as_ref()
                    .map(|m| m.loss())
                    .unwrap_or(f32::INFINITY);
                a_loss
                    .partial_cmp(&b_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            _ => self.checkpoints.last(),
        }
    }
}

/// Checkpoint utilities
pub struct CheckpointUtils;

impl CheckpointUtils {
    /// Create a checkpoint from model state
    pub fn create_checkpoint(
        epoch: usize,
        model_state: HashMap<String, Vec<f32>>,
        optimizer_state: crate::training::optimizers::OptimizerState,
        metrics: Option<TrainingMetrics>,
    ) -> Checkpoint {
        Checkpoint {
            epoch,
            model_state,
            optimizer_state,
            metrics,
            metadata: HashMap::new(),
        }
    }

    /// Validate checkpoint integrity
    pub fn validate_checkpoint(checkpoint: &Checkpoint) -> Result<()> {
        // Check if model state is not empty
        if checkpoint.model_state.is_empty() {
            return Err(anyhow::anyhow!("Checkpoint model state is empty"));
        }

        // Check if optimizer state is valid
        if checkpoint.optimizer_state.optimizer_type.is_empty() {
            return Err(anyhow::anyhow!("Checkpoint optimizer state is invalid"));
        }

        // Check if metrics are reasonable (if present)
        if let Some(ref metrics) = checkpoint.metrics {
            if metrics.loss() < 0.0 {
                return Err(anyhow::anyhow!("Invalid loss value in checkpoint"));
            }
            if metrics.accuracy() < 0.0 || metrics.accuracy() > 1.0 {
                return Err(anyhow::anyhow!("Invalid accuracy value in checkpoint"));
            }
        }

        Ok(())
    }

    /// Compare two checkpoints
    pub fn compare_checkpoints(a: &Checkpoint, b: &Checkpoint) -> CheckpointComparison {
        CheckpointComparison {
            epoch_diff: a.epoch as i32 - b.epoch as i32,
            loss_diff: a.metrics.as_ref().map(|m| m.loss()).unwrap_or(0.0)
                - b.metrics.as_ref().map(|m| m.loss()).unwrap_or(0.0),
            accuracy_diff: a.metrics.as_ref().map(|m| m.accuracy()).unwrap_or(0.0)
                - b.metrics.as_ref().map(|m| m.accuracy()).unwrap_or(0.0),
        }
    }
}

/// Checkpoint comparison result
#[derive(Debug, Clone)]
pub struct CheckpointComparison {
    pub epoch_diff: i32,
    pub loss_diff: f32,
    pub accuracy_diff: f32,
}

/// Checkpoint backup manager
pub struct CheckpointBackup {
    backup_dir: PathBuf,
    max_backups: usize,
}

impl CheckpointBackup {
    pub fn new(backup_dir: PathBuf, max_backups: usize) -> Self {
        Self {
            backup_dir,
            max_backups,
        }
    }

    /// Create a backup of a checkpoint
    pub async fn backup_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        if !self.backup_dir.exists() {
            fs::create_dir_all(&self.backup_dir).await?;
        }

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!(
            "checkpoint_backup_epoch_{}_at_{}.json",
            checkpoint.epoch, timestamp
        );
        let filepath = self.backup_dir.join(filename);

        let json = serde_json::to_string_pretty(checkpoint)?;
        fs::write(&filepath, json).await?;

        tracing::info!("Checkpoint backed up to: {}", filepath.display());
        Ok(())
    }

    /// List all backups
    pub async fn list_backups(&self) -> Result<Vec<PathBuf>> {
        if !self.backup_dir.exists() {
            return Ok(Vec::new());
        }

        let mut backups = Vec::new();
        let mut entries = fs::read_dir(&self.backup_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                backups.push(path);
            }
        }

        // Sort by creation time (newest first)
        backups.sort_by_key(|p| p.metadata().unwrap().created().unwrap());
        backups.reverse();

        Ok(backups)
    }

    /// Clean up old backups
    pub async fn cleanup_old_backups(&self) -> Result<()> {
        let backups = self.list_backups().await?;

        if backups.len() > self.max_backups {
            let _to_remove = backups.len() - self.max_backups;

            for backup in backups.into_iter().skip(self.max_backups) {
                fs::remove_file(&backup).await?;
                tracing::info!("Removed old backup: {}", backup.display());
            }
        }

        Ok(())
    }
}
