//! Main training pipeline implementation

use crate::core::image::OcrImage;
use crate::recognition::engine::OcrModel;
use anyhow::Result;
use ndarray::s;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::training::checkpoint::{Checkpoint, CheckpointManager};
use crate::training::config::TrainingConfig;
use crate::training::data::{DataLoader, DatasetConfig, DatasetFormat, TrainingBatch};
use crate::training::losses::{CrossEntropyLoss, LossFunction};
use crate::training::metrics::{MetricTracker, TrainingMetrics};
use crate::training::optimizers::{Adam, Optimizer, RMSprop, SGD};

/// Training pipeline for OCR models
pub struct TrainingPipeline {
    config: TrainingConfig,
    model: Arc<RwLock<Box<dyn OcrModel + Send + Sync>>>,
    optimizer: Box<dyn Optimizer + Send + Sync>,
    loss_function: Box<dyn LossFunction + Send + Sync>,
    metrics: MetricTracker,
    checkpoint_manager: CheckpointManager,
    data_loader: DataLoader,
}

impl TrainingPipeline {
    /// Create a new training pipeline
    pub fn new(config: TrainingConfig, model: Box<dyn OcrModel + Send + Sync>) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Create optimizer
        let optimizer = Self::create_optimizer(&config)?;

        // Create loss function
        let loss_function = Self::create_loss_function(&config)?;

        // Create metrics tracker
        let metrics = MetricTracker::new();

        // Create checkpoint manager
        let checkpoint_manager = CheckpointManager::new(&config.checkpoint)?;

        // Create data loader
        let dataset_config = DatasetConfig {
            dataset_path: config.data.dataset_path.clone(),
            format: DatasetFormat::from_string(&config.data.dataset_format),
            train_split: 0.8, // Default values since DataConfig doesn't have these fields
            val_split: 0.1,
            test_split: 0.1,
            max_samples: None,
            shuffle: config.data.shuffle,
            seed: Some(42), // Default seed
        };
        let data_loader = DataLoader::new(dataset_config);

        Ok(Self {
            config,
            model: Arc::new(RwLock::new(model)),
            optimizer,
            loss_function,
            metrics,
            checkpoint_manager,
            data_loader,
        })
    }

    /// Start training
    pub async fn train(&mut self) -> Result<()> {
        info!("Starting training with config: {:?}", self.config);

        // Load dataset
        self.data_loader.load_dataset().await?;
        let splits = self
            .data_loader
            .get_splits()
            .ok_or_else(|| anyhow::anyhow!("Failed to load dataset"))?;

        info!(
            "Dataset loaded - Train: {}, Val: {}, Test: {}",
            splits.train.len(),
            splits.validation.len(),
            splits.test.len()
        );

        // Initialize metrics
        self.metrics.reset();

        // Training loop
        for epoch in 0..self.config.training.num_epochs {
            info!(
                "Starting epoch {}/{}",
                epoch + 1,
                self.config.training.num_epochs
            );

            // Train for one epoch
            let train_metrics = self.train_epoch(epoch).await?;
            info!("Epoch {} training metrics: {:?}", epoch + 1, train_metrics);

            // Validate
            let val_metrics = self.validate_epoch().await?;
            info!("Epoch {} validation metrics: {:?}", epoch + 1, val_metrics);

            // Update learning rate
            self.update_learning_rate(epoch).await?;

            // Save checkpoint
            if (epoch + 1) % self.config.checkpoint.save_interval == 0 {
                self.save_checkpoint(epoch).await?;
            }

            // Check for early stopping
            if self.should_early_stop(&val_metrics).await? {
                warn!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }

        // Final evaluation
        let test_metrics = self.evaluate().await?;
        info!("Final test metrics: {:?}", test_metrics);

        info!("Training completed successfully");
        Ok(())
    }

    /// Train for one epoch
    async fn train_epoch(&mut self, epoch: usize) -> Result<TrainingMetrics> {
        let mut epoch_metrics = TrainingMetrics::new();
        let train_samples = self
            .data_loader
            .get_train_samples()
            .ok_or_else(|| anyhow::anyhow!("No training samples available"))?;

        let batch_iterator = crate::training::data::BatchIterator::new(
            train_samples.to_vec(),
            self.config.data.batch_size,
        );

        let mut batch_count = 0;
        for batch in batch_iterator {
            // Convert batch to tensor
            let input = self.batch_to_tensor(&batch)?;

            // Forward pass
            let predictions = self.forward_pass_train(&input).await?;

            // Compute loss
            let targets = self.prepare_targets(&batch)?;
            let loss = self.loss_function.compute(&predictions, &targets)?;

            // Backward pass
            let gradients = self.loss_function.gradient(&predictions, &targets)?;
            self.backward_pass(&input, &gradients).await?;

            // Update metrics
            epoch_metrics.add_loss(loss);
            epoch_metrics.add_accuracy(self.compute_accuracy(&predictions, &targets)?);

            batch_count += 1;
            if batch_count % self.config.logging.log_interval == 0 {
                info!(
                    "Epoch {}, Batch {}, Loss: {:.4}, Accuracy: {:.4}",
                    epoch + 1,
                    batch_count,
                    loss,
                    epoch_metrics.accuracy()
                );
            }
        }

        epoch_metrics.finalize();
        Ok(epoch_metrics)
    }

    /// Validate for one epoch
    async fn validate_epoch(&self) -> Result<TrainingMetrics> {
        let mut val_metrics = TrainingMetrics::new();
        let val_samples = self
            .data_loader
            .get_val_samples()
            .ok_or_else(|| anyhow::anyhow!("No validation samples available"))?;

        let batch_iterator = crate::training::data::BatchIterator::new(
            val_samples.to_vec(),
            self.config.data.val_batch_size,
        );

        for batch in batch_iterator {
            let input = self.batch_to_tensor(&batch)?;

            // Forward pass (no gradients)
            let predictions = self.forward_pass_train(&input).await?;

            // Compute loss
            let targets = self.prepare_targets(&batch)?;
            let loss = self.loss_function.compute(&predictions, &targets)?;

            // Update metrics
            val_metrics.add_loss(loss);
            val_metrics.add_accuracy(self.compute_accuracy(&predictions, &targets)?);
        }

        val_metrics.finalize();
        Ok(val_metrics)
    }

    /// Evaluate on test set
    async fn evaluate(&self) -> Result<TrainingMetrics> {
        let mut test_metrics = TrainingMetrics::new();
        let test_samples = self
            .data_loader
            .get_test_samples()
            .ok_or_else(|| anyhow::anyhow!("No test samples available"))?;

        let batch_iterator = crate::training::data::BatchIterator::new(
            test_samples.to_vec(),
            self.config.data.val_batch_size,
        );

        for batch in batch_iterator {
            let input = self.batch_to_tensor(&batch)?;

            // Forward pass
            let predictions = self.forward_pass_train(&input).await?;

            // Compute loss
            let targets = self.prepare_targets(&batch)?;
            let loss = self.loss_function.compute(&predictions, &targets)?;

            // Update metrics
            test_metrics.add_loss(loss);
            test_metrics.add_accuracy(self.compute_accuracy(&predictions, &targets)?);
        }

        test_metrics.finalize();
        Ok(test_metrics)
    }

    /// Forward pass through the model (training mode)
    async fn forward_pass_train(
        &self,
        input: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        let model = self.model.read().await;

        if let Some(trainable) = model.as_trainable_ref() {
            Ok(trainable.forward_train(input)?)
        } else {
            Err(anyhow::anyhow!("Model is not trainable"))
        }
    }

    /// Backward pass (gradient computation and parameter update)
    async fn backward_pass(
        &mut self,
        input: &ndarray::Array2<f32>,
        gradients: &ndarray::Array2<f32>,
    ) -> Result<()> {
        // Apply gradient clipping if configured
        let clipped_gradients = if let Some(clip_norm) = self.config.training.gradient_clip_norm {
            self.clip_gradients(gradients, clip_norm)?
        } else {
            gradients.clone()
        };

        let mut model = self.model.write().await;

        if let Some(trainable) = model.as_trainable() {
            trainable.backward_train(input, &clipped_gradients)?;

            for (param, grad) in trainable.get_params_and_grads() {
                self.optimizer.update(param, grad)?;
            }
        } else {
            warn!("Model is not trainable, skipping backward pass");
        }

        Ok(())
    }

    /// Convert batch to tensor
    fn batch_to_tensor(&self, batch: &TrainingBatch) -> Result<ndarray::Array2<f32>> {
        let mut data = Vec::new();
        let mut dim = 0;

        for image in &batch.images {
            let vec = self.preprocess_image_float(image)?;
            dim = vec.len();
            data.extend(vec);
        }

        let batch_size = batch.images.len();
        if batch_size == 0 {
            return Ok(ndarray::Array2::zeros((0, 0)));
        }

        // Handle case where images might be different sizes (should be handled by loader)
        // Here we assume same size for simplicity or we should resize
        let array = ndarray::Array2::from_shape_vec((batch_size, dim), data)?;
        Ok(array)
    }

    /// Preprocess image for model input (float)
    fn preprocess_image_float(&self, image: &OcrImage) -> Result<Vec<f32>> {
        let (width, height) = (image.width, image.height);
        let mut floats = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y)?;
                floats.push(pixel.r as f32 / 255.0);
                floats.push(pixel.g as f32 / 255.0);
                floats.push(pixel.b as f32 / 255.0);
            }
        }

        Ok(floats)
    }

    /// Prepare targets for loss computation
    fn prepare_targets(&self, batch: &TrainingBatch) -> Result<ndarray::Array2<f32>> {
        let batch_size = batch.texts.len();
        let vocab_size = self.config.model.vocab_size;

        let mut targets = ndarray::Array2::<f32>::zeros((batch_size, vocab_size));

        for (i, text) in batch.texts.iter().enumerate() {
            // Convert text to one-hot encoding
            for ch in text.chars() {
                let char_idx = (ch as usize) % vocab_size;
                targets[[i, char_idx]] = 1.0;
            }
        }

        Ok(targets)
    }

    /// Compute accuracy
    fn compute_accuracy(
        &self,
        predictions: &ndarray::Array2<f32>,
        targets: &ndarray::Array2<f32>,
    ) -> Result<f32> {
        let mut correct = 0;
        let total = predictions.shape()[0];

        for i in 0..total {
            let pred_max = predictions
                .slice(s![i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let target_max = targets
                .slice(s![i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_max == target_max {
                correct += 1;
            }
        }

        Ok(correct as f32 / total as f32)
    }

    /// Preprocess image for model input
    fn preprocess_image(&self, image: &OcrImage) -> Result<Vec<u8>> {
        // Convert image to raw bytes
        // This is a simplified implementation
        let (width, height) = (image.width, image.height);
        let mut bytes = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y)?;
                bytes.push(pixel.r);
                bytes.push(pixel.g);
                bytes.push(pixel.b);
            }
        }

        Ok(bytes)
    }

    /// Convert model result to prediction vector
    fn result_to_vector(
        &self,
        result: &crate::core::recognition::RecognitionResult,
    ) -> Result<Vec<f32>> {
        // Convert recognition result to prediction vector
        // This is a simplified implementation
        let mut vector = vec![0.0; self.config.model.vocab_size];

        for (i, ch) in result.text.chars().enumerate() {
            if i < self.config.model.vocab_size {
                vector[i] = ch as u32 as f32 / 255.0; // Normalize
            }
        }

        Ok(vector)
    }

    /// Clip gradients to prevent exploding gradients
    fn clip_gradients(
        &self,
        gradients: &ndarray::Array2<f32>,
        max_norm: f32,
    ) -> Result<ndarray::Array2<f32>> {
        let norm = gradients.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm > max_norm {
            let scale = max_norm / norm;
            Ok(gradients * scale)
        } else {
            Ok(gradients.clone())
        }
    }

    /// Update learning rate
    async fn update_learning_rate(&mut self, epoch: usize) -> Result<()> {
        let new_lr = self.config.get_learning_rate(epoch, 0);

        // Update optimizer learning rate
        match self.optimizer.name() {
            "SGD" => {
                if let Some(sgd) = self.optimizer.as_any().downcast_ref::<SGD>() {
                    // Update learning rate
                    // Note: This would need to be implemented in the optimizer trait
                }
            }
            "Adam" => {
                if let Some(adam) = self.optimizer.as_any().downcast_ref::<Adam>() {
                    // Update learning rate
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Check if early stopping should be triggered
    async fn should_early_stop(&self, val_metrics: &TrainingMetrics) -> Result<bool> {
        // Simple early stopping based on validation loss
        // In practice, you would track the best validation loss over time
        Ok(false) // Placeholder
    }

    /// Save checkpoint
    async fn save_checkpoint(&mut self, epoch: usize) -> Result<()> {
        let checkpoint = Checkpoint {
            epoch,
            model_state: HashMap::new(), // Placeholder
            optimizer_state: self.optimizer.get_state(),
            metrics: self.metrics.get_current_metrics().cloned(),
            metadata: HashMap::new(),
        };

        self.checkpoint_manager.save_checkpoint(&checkpoint).await?;
        Ok(())
    }

    /// Create optimizer from config
    fn create_optimizer(config: &TrainingConfig) -> Result<Box<dyn Optimizer + Send + Sync>> {
        match config.optimizer.optimizer_type.as_str() {
            "SGD" => {
                let mut sgd = SGD::new(config.optimizer.learning_rate);
                if let Some(momentum) = config.optimizer.parameters.get("momentum") {
                    if let Some(momentum_val) = momentum.as_f64() {
                        sgd = sgd.with_momentum(momentum_val as f32);
                    }
                }
                if let Some(weight_decay) = config.optimizer.parameters.get("weight_decay") {
                    if let Some(weight_decay_val) = weight_decay.as_f64() {
                        sgd = sgd.with_weight_decay(weight_decay_val as f32);
                    }
                }
                Ok(Box::new(sgd))
            }
            "Adam" => {
                let mut adam = Adam::new(config.optimizer.learning_rate);
                if let Some(beta1) = config.optimizer.parameters.get("beta1") {
                    if let Some(beta1_val) = beta1.as_f64() {
                        adam = adam.with_betas(beta1_val as f32, 0.999);
                    }
                }
                if let Some(weight_decay) = config.optimizer.parameters.get("weight_decay") {
                    if let Some(weight_decay_val) = weight_decay.as_f64() {
                        adam = adam.with_weight_decay(weight_decay_val as f32);
                    }
                }
                Ok(Box::new(adam))
            }
            "RMSprop" => {
                let mut rmsprop = RMSprop::new(config.optimizer.learning_rate);
                if let Some(alpha) = config.optimizer.parameters.get("alpha") {
                    if let Some(alpha_val) = alpha.as_f64() {
                        rmsprop = rmsprop.with_alpha(alpha_val as f32);
                    }
                }
                if let Some(weight_decay) = config.optimizer.parameters.get("weight_decay") {
                    if let Some(weight_decay_val) = weight_decay.as_f64() {
                        rmsprop = rmsprop.with_weight_decay(weight_decay_val as f32);
                    }
                }
                Ok(Box::new(rmsprop))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown optimizer type: {}",
                config.optimizer.optimizer_type
            )),
        }
    }

    /// Create loss function from config
    fn create_loss_function(
        config: &TrainingConfig,
    ) -> Result<Box<dyn LossFunction + Send + Sync>> {
        // For now, use Cross-Entropy loss
        // In practice, you would choose based on the model type and task
        Ok(Box::new(CrossEntropyLoss::new()))
    }
}

/// Extension trait for optimizer downcasting
trait OptimizerAny {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: 'static> OptimizerAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
