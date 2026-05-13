//! Loss functions for OCR training

use anyhow::Result;
use ndarray::{Array1, Array2, s};

/// Trait for loss functions
pub trait LossFunction {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32>;
    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>>;
    fn name(&self) -> &'static str;
}

/// Connectionist Temporal Classification (CTC) Loss
///
/// CTC is commonly used for sequence-to-sequence tasks like OCR where
/// the alignment between input and output sequences is unknown.
pub struct CTCLoss {
    blank_token: usize,
    reduction: ReductionType,
}

#[derive(Debug, Clone)]
pub enum ReductionType {
    Mean,
    Sum,
    None,
}

impl CTCLoss {
    pub fn new(blank_token: usize) -> Self {
        Self {
            blank_token,
            reduction: ReductionType::Mean,
        }
    }

    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }

    /// Compute CTC loss for a batch of sequences
    fn compute_ctc_loss(
        &self,
        log_probs: &Array2<f32>,
        targets: &Array1<usize>,
        input_lengths: &Array1<usize>,
        target_lengths: &Array1<usize>,
    ) -> Result<f32> {
        let batch_size = log_probs.shape()[0];
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let input_len = input_lengths[i];
            let target_len = target_lengths[i];

            let sequence_log_probs = log_probs
                .slice_axis(ndarray::Axis(0), ndarray::Slice::from(i..i + 1))
                .slice_axis(ndarray::Axis(1), ndarray::Slice::from(0..input_len))
                .slice_axis(ndarray::Axis(2), ndarray::Slice::from(..))
                .to_owned();
            let target_sequence = targets
                .slice_axis(ndarray::Axis(0), ndarray::Slice::from(i..i + 1))
                .slice_axis(ndarray::Axis(1), ndarray::Slice::from(0..target_len))
                .to_owned();

            let loss = self.forward_algorithm(&sequence_log_probs, &target_sequence)?;
            total_loss += loss;
        }

        match self.reduction {
            ReductionType::Mean => Ok(total_loss / batch_size as f32),
            ReductionType::Sum => Ok(total_loss),
            ReductionType::None => Ok(total_loss),
        }
    }

    /// Forward algorithm for CTC
    fn forward_algorithm(&self, log_probs: &Array2<f32>, target: &Array1<usize>) -> Result<f32> {
        let t = log_probs.shape()[0];
        let s = target.len() * 2 + 1; // Extended target with blanks

        // Create extended target with blanks
        let mut extended_target = vec![self.blank_token; s];
        for (i, &token) in target.iter().enumerate() {
            extended_target[i * 2 + 1] = token;
        }

        // Initialize alpha matrix
        let mut alpha = Array2::<f32>::zeros((t, s));

        // Initialize first column
        alpha[[0, 0]] = log_probs[[0, self.blank_token]];
        if s > 1 {
            alpha[[0, 1]] = log_probs[[0, extended_target[1]]];
        }

        // Forward pass
        for t_idx in 1..t {
            for s_idx in 0..s {
                let current_token = extended_target[s_idx];
                let mut sum = alpha[[t_idx - 1, s_idx]] + log_probs[[t_idx, current_token]];

                if s_idx > 0 {
                    sum =
                        sum.max(alpha[[t_idx - 1, s_idx - 1]] + log_probs[[t_idx, current_token]]);
                }

                if s_idx > 1
                    && extended_target[s_idx] != self.blank_token
                    && extended_target[s_idx] != extended_target[s_idx - 2]
                {
                    sum =
                        sum.max(alpha[[t_idx - 1, s_idx - 2]] + log_probs[[t_idx, current_token]]);
                }

                alpha[[t_idx, s_idx]] = sum;
            }
        }

        // Compute final loss
        let final_loss = if s > 0 {
            alpha[[t - 1, s - 1]].max(alpha[[t - 1, s - 2]])
        } else {
            alpha[[t - 1, 0]]
        };

        Ok(-final_loss)
    }
}

impl LossFunction for CTCLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        // Convert targets to indices and lengths
        let batch_size = predictions.shape()[0];
        let mut target_indices = Vec::new();
        let mut target_lengths = Vec::new();
        let mut input_lengths = Vec::new();

        for i in 0..batch_size {
            // Find non-zero elements in target
            let target_row = targets.slice(s![i, ..]);
            let non_zero_indices: Vec<usize> = target_row
                .iter()
                .enumerate()
                .filter(|&(_, &val)| val != 0.0)
                .map(|(idx, _)| idx)
                .collect();

            target_indices.extend(non_zero_indices.clone());
            target_lengths.push(non_zero_indices.len());
            input_lengths.push(predictions.shape()[1]);
        }

        let target_array = Array1::from(target_indices);
        let target_lengths_array = Array1::from(target_lengths);
        let input_lengths_array = Array1::from(input_lengths);

        self.compute_ctc_loss(
            predictions,
            &target_array,
            &input_lengths_array,
            &target_lengths_array,
        )
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        // Simplified gradient computation
        // In practice, this would use the backward algorithm
        let mut gradients = predictions.clone();
        gradients -= targets;
        Ok(gradients)
    }

    fn name(&self) -> &'static str {
        "CTC Loss"
    }
}

/// Cross-Entropy Loss for classification tasks
pub struct CrossEntropyLoss {
    reduction: ReductionType,
    label_smoothing: f32,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            reduction: ReductionType::Mean,
            label_smoothing: 0.0,
        }
    }

    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pred_row = predictions.slice(s![i, ..]);
            let target_row = targets.slice(s![i, ..]);

            // Apply softmax to predictions
            let max_val = pred_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_pred: Array1<f32> = pred_row.mapv(|x| (x - max_val).exp());
            let sum_exp: f32 = exp_pred.sum();
            let softmax_pred = exp_pred.mapv(|x| x / sum_exp);

            // Compute cross-entropy loss
            let mut loss = 0.0;
            for j in 0..num_classes {
                if target_row[j] > 0.0 {
                    loss += -target_row[j] * softmax_pred[j].ln();
                }
            }

            total_loss += loss;
        }

        match self.reduction {
            ReductionType::Mean => Ok(total_loss / batch_size as f32),
            ReductionType::Sum => Ok(total_loss),
            ReductionType::None => Ok(total_loss),
        }
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];
        let mut gradients = Array2::<f32>::zeros((batch_size, num_classes));

        for i in 0..batch_size {
            let pred_row = predictions.slice(s![i, ..]);
            let target_row = targets.slice(s![i, ..]);

            // Apply softmax to predictions
            let max_val = pred_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_pred: Array1<f32> = pred_row.mapv(|x| (x - max_val).exp());
            let sum_exp: f32 = exp_pred.sum();
            let softmax_pred = exp_pred.mapv(|x| x / sum_exp);

            // Compute gradients
            for j in 0..num_classes {
                gradients[[i, j]] = softmax_pred[j] - target_row[j];
            }
        }

        Ok(gradients)
    }

    fn name(&self) -> &'static str {
        "Cross-Entropy Loss"
    }
}

/// Focal Loss for handling class imbalance
pub struct FocalLoss {
    alpha: f32,
    gamma: f32,
    reduction: ReductionType,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        Self {
            alpha,
            gamma,
            reduction: ReductionType::Mean,
        }
    }

    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
}

impl LossFunction for FocalLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pred_row = predictions.slice(s![i, ..]);
            let target_row = targets.slice(s![i, ..]);

            // Apply softmax to predictions
            let max_val = pred_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_pred: Array1<f32> = pred_row.mapv(|x| (x - max_val).exp());
            let sum_exp: f32 = exp_pred.sum();
            let softmax_pred = exp_pred.mapv(|x| x / sum_exp);

            // Compute focal loss
            let mut loss = 0.0;
            for j in 0..num_classes {
                if target_row[j] > 0.0 {
                    let pt = softmax_pred[j];
                    let focal_weight = self.alpha * (1.0 - pt).powf(self.gamma);
                    loss += -target_row[j] * focal_weight * pt.ln();
                }
            }

            total_loss += loss;
        }

        match self.reduction {
            ReductionType::Mean => Ok(total_loss / batch_size as f32),
            ReductionType::Sum => Ok(total_loss),
            ReductionType::None => Ok(total_loss),
        }
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        // Simplified gradient computation for focal loss
        let mut gradients = predictions.clone();
        gradients -= targets;
        Ok(gradients)
    }

    fn name(&self) -> &'static str {
        "Focal Loss"
    }
}

/// Attention-based loss for sequence-to-sequence models
pub struct AttentionLoss {
    reduction: ReductionType,
}

impl AttentionLoss {
    pub fn new() -> Self {
        Self {
            reduction: ReductionType::Mean,
        }
    }

    pub fn with_reduction(mut self, reduction: ReductionType) -> Self {
        self.reduction = reduction;
        self
    }
}

impl LossFunction for AttentionLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        // Attention loss combines cross-entropy with attention alignment
        let ce_loss = CrossEntropyLoss::new().with_reduction(ReductionType::None);
        let ce_value = ce_loss.compute(predictions, targets)?;

        // Add attention alignment penalty (simplified)
        let attention_penalty = 0.1 * predictions.sum();

        let total_loss = ce_value + attention_penalty;

        match self.reduction {
            ReductionType::Mean => Ok(total_loss / predictions.shape()[0] as f32),
            ReductionType::Sum => Ok(total_loss),
            ReductionType::None => Ok(total_loss),
        }
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        let ce_loss = CrossEntropyLoss::new();
        let mut gradients = ce_loss.gradient(predictions, targets)?;

        // Add attention gradient
        gradients += 0.1;

        Ok(gradients)
    }

    fn name(&self) -> &'static str {
        "Attention Loss"
    }
}

/// Combined loss function that can use multiple loss functions
pub struct CombinedLoss {
    losses: Vec<Box<dyn LossFunction + Send + Sync>>,
    weights: Vec<f32>,
}

impl CombinedLoss {
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn add_loss(mut self, loss: Box<dyn LossFunction + Send + Sync>, weight: f32) -> Self {
        self.losses.push(loss);
        self.weights.push(weight);
        self
    }
}

impl LossFunction for CombinedLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        let mut total_loss = 0.0;

        for (loss, &weight) in self.losses.iter().zip(self.weights.iter()) {
            let loss_value = loss.compute(predictions, targets)?;
            total_loss += weight * loss_value;
        }

        Ok(total_loss)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        let mut total_gradient = Array2::<f32>::zeros(predictions.raw_dim());

        for (loss, &weight) in self.losses.iter().zip(self.weights.iter()) {
            let gradient = loss.gradient(predictions, targets)?;
            total_gradient = &total_gradient + &(weight * &gradient);
        }

        Ok(total_gradient)
    }

    fn name(&self) -> &'static str {
        "Combined Loss"
    }
}
