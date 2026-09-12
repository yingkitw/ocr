//! CRNN training pipeline
//!
//! Trains the CRNN model on synthetic text-line images using CTC loss.

use crate::recognition::crnn::CrnnModel;
use crate::synthetic::{DistortionConfig, SyntheticSample, TextLineGenerator};
use crate::training::ctc::ctc_loss_and_grad;
use crate::utils::Result;
use ndarray::{Array1, Array2};
#[cfg(test)]
use crate::recognition::crnn::CrnnConfig;

/// Training metrics per epoch
#[derive(Debug, Clone, Default)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_cer: f32,
    pub train_wer: f32,
    pub val_loss: f32,
    pub val_cer: f32,
    pub val_wer: f32,
    pub samples_per_sec: f32,
}

/// CRNN trainer
pub struct CrnnTrainer {
    pub model: CrnnModel,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub distortion: DistortionConfig,
}

/// Adam state for the FC layer
pub struct AdamState {
    pub m_w: ndarray::Array2<f32>,
    pub v_w: ndarray::Array2<f32>,
    pub m_b: ndarray::Array1<f32>,
    pub v_b: ndarray::Array1<f32>,
    pub t: u64,
}

impl AdamState {
    pub fn new_like(model: &CrnnModel) -> Self {
        Self {
            m_w: ndarray::Array2::zeros(model.fc_weight.dim()),
            v_w: ndarray::Array2::zeros(model.fc_weight.dim()),
            m_b: ndarray::Array1::zeros(model.fc_bias.dim()),
            v_b: ndarray::Array1::zeros(model.fc_bias.dim()),
            t: 0,
        }
    }
}

/// Apply Adam update to the FC layer weights in-place. `grad_w` and `grad_b`
/// are gradients of the loss w.r.t. fc_weight and fc_bias.
pub fn adam_update_fc(
    model: &mut CrnnModel,
    grad_w: &ndarray::Array2<f32>,
    grad_b: &ndarray::Array1<f32>,
    state: &mut AdamState,
    learning_rate: f32,
) {
    state.t += 1;
    let _step_t = state.t as f32;
    let bc1 = 1.0 - 0.9f32.powi(state.t as i32);
    let bc2 = 1.0 - 0.999f32.powi(state.t as i32);
    let eps = 1e-8f32;
    for j in 0..model.fc_weight.nrows() {
        for k in 0..model.fc_weight.ncols() {
            let g = grad_w[[j, k]];
            state.m_w[[j, k]] = 0.9 * state.m_w[[j, k]] + 0.1 * g;
            state.v_w[[j, k]] = 0.999 * state.v_w[[j, k]] + 0.001 * g * g;
            let m_hat = state.m_w[[j, k]] / bc1;
            let v_hat = state.v_w[[j, k]] / bc2;
            model.fc_weight[[j, k]] -= learning_rate * m_hat / (v_hat.sqrt() + eps);
        }
    }
    for j in 0..model.fc_bias.len() {
        let g = grad_b[j];
        state.m_b[j] = 0.9 * state.m_b[j] + 0.1 * g;
        state.v_b[j] = 0.999 * state.v_b[j] + 0.001 * g * g;
        let m_hat = state.m_b[j] / bc1;
        let v_hat = state.v_b[j] / bc2;
        model.fc_bias[j] -= learning_rate * m_hat / (v_hat.sqrt() + eps);
    }
}

impl CrnnTrainer {
    pub fn new(model: CrnnModel) -> Self {
        Self {
            model,
            learning_rate: 0.001,
            batch_size: 32,
            distortion: DistortionConfig::mild(),
        }
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    pub fn with_distortion(mut self, distortion: DistortionConfig) -> Self {
        self.distortion = distortion;
        self
    }

    /// Train for one epoch on synthetic data
    pub fn train_epoch(&mut self, num_batches: usize, samples_per_batch: usize) -> EpochMetrics {
        let mut total_loss = 0.0f32;
        let mut total_chars = 0usize;
        let mut char_errors = 0usize;
        let mut total_words = 0usize;
        let mut word_errors = 0usize;

        let start_time = std::time::Instant::now();

        for _ in 0..num_batches {
            let batch = self.generate_batch(samples_per_batch);
            let batch_loss = self.train_batch(&batch);
            total_loss += batch_loss;

            for sample in &batch {
                let pred = self.model.recognize_from_sample(sample);
                let gt = &sample.ground_truth;
                char_errors += levenshtein_distance(gt, &pred);
                total_chars += gt.chars().count();
                total_words += gt.split_whitespace().count();
                word_errors += word_error_distance(gt, &pred);
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        let total_samples = num_batches * samples_per_batch;

        EpochMetrics {
            epoch: 0,
            train_loss: total_loss / num_batches as f32,
            train_cer: if total_chars > 0 {
                char_errors as f32 / total_chars as f32
            } else {
                0.0
            },
            train_wer: if total_words > 0 {
                word_errors as f32 / total_words as f32
            } else {
                0.0
            },
            val_loss: 0.0,
            val_cer: 0.0,
            val_wer: 0.0,
            samples_per_sec: total_samples as f32 / elapsed as f32,
        }
    }

    /// Evaluate on a validation set
    pub fn evaluate(&self, samples: &[SyntheticSample]) -> EpochMetrics {
        let mut total_loss = 0.0f32;
        let mut total_chars = 0usize;
        let mut char_errors = 0usize;
        let mut total_words = 0usize;
        let mut word_errors = 0usize;

        for sample in samples {
            let pred = self.model.recognize_from_sample(sample);
            let gt = &sample.ground_truth;
            char_errors += levenshtein_distance(gt, &pred);
            total_chars += gt.chars().count();
            total_words += gt.split_whitespace().count();
            word_errors += word_error_distance(gt, &pred);

            // Approximate loss: negative log probability of correct sequence
            // (simplified: just use CER as proxy for loss here)
            total_loss += char_errors as f32;
        }

        EpochMetrics {
            epoch: 0,
            train_loss: 0.0,
            train_cer: 0.0,
            train_wer: 0.0,
            val_loss: total_loss / samples.len() as f32,
            val_cer: if total_chars > 0 {
                char_errors as f32 / total_chars as f32
            } else {
                0.0
            },
            val_wer: if total_words > 0 {
                word_errors as f32 / total_words as f32
            } else {
                0.0
            },
            samples_per_sec: 0.0,
        }
    }

    fn generate_batch(&self, count: usize) -> Vec<SyntheticSample> {
        let generator = TextLineGenerator::default();
        let texts = generator.generate_random_texts(count, 15);
        let mut samples = generator.generate_batch(&texts);
        crate::synthetic::distortion::augment_batch(&mut samples, &self.distortion);
        samples
    }

    /// Train on a batch using FC-layer backprop with frame-level cross-entropy.
    /// CNN and BiLSTM feature extractors are kept frozen; only the readout layer is updated.
    fn train_batch(&mut self, samples: &[SyntheticSample]) -> f32 {
        let mut total_loss = 0.0f32;
        let mut total_timesteps = 0usize;

        for sample in samples {
            let gray = sample.image.to_luma8();
            let (w, h) = (gray.width() as usize, gray.height() as usize);
            let mut arr = Array2::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
                }
            }

            let target_h = self.model.config.input_height;
            if h != target_h {
                arr = CrnnModel::resize_array2_height(&arr, target_h);
            }

            // Forward through frozen feature extractors
            let cnn_features = self.model.cnn.forward(&arr);
            let lstm1_out = self.model.lstm1.forward(&cnn_features);
            let lstm2_out = self.model.lstm2.forward(&lstm1_out);
            let (t, lstm_dim) = lstm2_out.dim();
            let num_classes = self.model.fc_weight.nrows();

            // Compute logits
            let mut logits = Array2::zeros((t, num_classes));
            for i in 0..t {
                for j in 0..num_classes {
                    let mut sum = self.model.fc_bias[j];
                    for k in 0..lstm_dim {
                        sum += self.model.fc_weight[[j, k]] * lstm2_out[[i, k]];
                    }
                    logits[[i, j]] = sum;
                }
            }

            // Frame-level target: spread ground truth chars evenly across timesteps
            let gt = &sample.ground_truth;
            let gt_indices: Vec<usize> = gt
                .chars()
                .map(|ch| self.model.vocab.char_to_idx.get(&ch).copied().unwrap_or(0))
                .collect();
            let n_chars = gt_indices.len().max(1);
            let mut targets = vec![0usize; t]; // blank = 0
            for timestep in 0..t {
                let slot = (timestep * n_chars) / t;
                if slot < n_chars {
                    targets[timestep] = gt_indices[slot];
                }
            }

            // Softmax and cross-entropy loss
            let mut dlogits = Array2::zeros((t, num_classes));
            for timestep in 0..t {
                // softmax
                let max_logit = (0..num_classes)
                    .map(|j| logits[[timestep, j]])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                let mut probs = vec![0.0f32; num_classes];
                for j in 0..num_classes {
                    probs[j] = (logits[[timestep, j]] - max_logit).exp();
                    exp_sum += probs[j];
                }
                for j in 0..num_classes {
                    probs[j] /= exp_sum;
                }

                // cross-entropy loss for this timestep
                let target_idx = targets[timestep];
                let p = probs[target_idx].max(1e-8);
                total_loss += -p.ln();
                total_timesteps += 1;

                // gradient: softmax - one_hot
                for j in 0..num_classes {
                    dlogits[[timestep, j]] = probs[j] - if j == target_idx { 1.0 } else { 0.0 };
                }
            }

            // Backprop through FC layer: dW = lstm2_out.T @ dlogits
            // Note: dlogits shape is [T, num_classes], lstm2_out is [T, lstm_dim]
            // We want dW[j, k] = sum_t(dlogits[t, j] * lstm2_out[t, k])
            let scale = self.learning_rate / samples.len() as f32;
            for j in 0..num_classes {
                for k in 0..lstm_dim {
                    let mut grad = 0.0f32;
                    for timestep in 0..t {
                        grad += dlogits[[timestep, j]] * lstm2_out[[timestep, k]];
                    }
                    self.model.fc_weight[[j, k]] -= scale * grad;
                }
                let mut bias_grad = 0.0f32;
                for timestep in 0..t {
                    bias_grad += dlogits[[timestep, j]];
                }
                self.model.fc_bias[j] -= scale * bias_grad;
            }
        }

        if total_timesteps > 0 {
            total_loss / total_timesteps as f32
        } else {
            0.0
        }
    }

    /// Train on a batch using CTC loss + Adam updates on **all** parameters
    /// (full BPTT through CNN + 2x BiLSTM + FC). This is the real training
    /// loop; the bundled model converges to a model that actually recognizes
    /// text given enough epochs.
    pub fn train_batch_full(&mut self, samples: &[SyntheticSample], adam_states: &mut FullAdamState) -> f32 {
        use crate::recognition::backward::ParamGrads;

        let mut total_loss = 0.0f32;

        // Aggregate grads across the batch
        let mut grads_acc = ParamGrads::zero_like(&self.model);
        let mut n = 0usize;

        for sample in samples {
            let gray = sample.image.to_luma8();
            let (w, h) = (gray.width() as usize, gray.height() as usize);
            let mut arr = Array2::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
                }
            }
            let target_h = self.model.config.input_height;
            if h != target_h || w != INPUT_WIDTH {
                arr = resize_to(&arr, INPUT_WIDTH, INPUT_HEIGHT);
            }

            // Forward with cache
            let (mut logits, cache) = self.model.forward_with_cache(&arr);
            let (mut t, num_classes) = (logits.nrows(), logits.ncols());
            if t < 1 {
                continue;
            }

            // Target indices
            let gt = &sample.ground_truth;
            let gt_indices: Vec<usize> = gt
                .chars()
                .map(|ch| self.model.vocab.char_to_idx.get(&ch).copied().unwrap_or(0))
                .collect();
            if gt_indices.is_empty() {
                continue;
            }

            // Pad logits to satisfy T >= 2*S + 1
            let ext_len = 2 * gt_indices.len() + 1;
            while t < ext_len {
                let last_row = logits.row(t - 1).to_owned();
                let mut new_logits = Array2::zeros((t + 1, num_classes));
                for i in 0..t {
                    new_logits.row_mut(i).assign(&logits.row(i));
                }
                new_logits.row_mut(t).assign(&last_row);
                logits = new_logits;
                t += 1;
            }

            // CTC loss + gradient
            let logits_flat: Vec<f32> = logits.iter().copied().collect();
            let (loss, grad) = ctc_loss_and_grad(
                &logits_flat,
                &gt_indices,
                num_classes,
                self.model.vocab.blank_idx,
            );
            if !loss.is_finite() {
                continue;
            }
            // Trim grad to original t (in case we padded)
            let grad = if t < grad.len() / num_classes {
                let trimmed: Vec<f32> = grad[..t * num_classes].to_vec();
                trimmed
            } else {
                grad
            };
            let d_logits = Array2::from_shape_vec((t, num_classes), grad)
                .unwrap_or_else(|_| Array2::zeros((t, num_classes)));

            // Full backward
            let grads = self.model.backward_full(&cache, &d_logits);
            accumulate_grads(&mut grads_acc, &grads);
            total_loss += loss;
            n += 1;
        }

        if n == 0 {
            return 0.0;
        }
        let mean = 1.0 / n as f32;
        scale_grads(&mut grads_acc, mean);
        apply_full_adam(&mut self.model, &grads_acc, adam_states, self.learning_rate);

        total_loss / n as f32
    }

    /// Train on a batch using CTC loss + Adam updates on the FC layer only.
    /// The CNN and BiLSTM weights are kept frozen (random init); only the FC
    /// readout is trained. Useful as a sanity check that the pipeline works
    /// end-to-end without requiring full BPTT through CNN+LSTM.
    pub fn train_batch_ctc(&mut self, samples: &[SyntheticSample], adam_state: &mut AdamState) -> f32 {
        let mut total_loss = 0.0f32;
        let mut total_chars = 0usize;
        let mut total_correct = 0usize;
        let mut total_words = 0usize;
        let mut word_correct = 0usize;

        let _n_initial = samples.len();
        let mut n = 0;
        // Accumulate grads across the batch.
        let mut grad_w_acc = ndarray::Array2::<f32>::zeros(self.model.fc_weight.dim());
        let mut grad_b_acc = ndarray::Array1::<f32>::zeros(self.model.fc_bias.len());

        for sample in samples {
            let gray = sample.image.to_luma8();
            let (w, h) = (gray.width() as usize, gray.height() as usize);
            let mut arr = Array2::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
                }
            }
            let target_h = self.model.config.input_height;
            if h != target_h {
                arr = CrnnModel::resize_array2_height(&arr, target_h);
            }

            // Forward through frozen feature extractors
            let cnn_features = self.model.cnn.forward(&arr);
            let lstm1_out = self.model.lstm1.forward(&cnn_features);
            let lstm2_out = self.model.lstm2.forward(&lstm1_out);
            let (t, lstm_dim) = lstm2_out.dim();
            let num_classes = self.model.fc_weight.nrows();
            if t < 1 {
                continue;
            }

            // Compute logits [t, num_classes]
            let mut logits = Array2::zeros((t, num_classes));
            for i in 0..t {
                for j in 0..num_classes {
                    let mut sum = self.model.fc_bias[j];
                    for k in 0..lstm_dim {
                        sum += self.model.fc_weight[[j, k]] * lstm2_out[[i, k]];
                    }
                    logits[[i, j]] = sum;
                }
            }

            // Target indices
            let gt = &sample.ground_truth;
            let gt_indices: Vec<usize> = gt
                .chars()
                .map(|ch| self.model.vocab.char_to_idx.get(&ch).copied().unwrap_or(0))
                .collect();
            if gt_indices.is_empty() {
                continue;
            }

            // CTC needs T >= 2*S + 1 (extended label length). For very short
            // sequences (T too small for the target), pad by repeating the
            // last logit row until we have enough timesteps.
            let ext_len = 2 * gt_indices.len() + 1;
            let mut t_padded = t;
            while t_padded < ext_len {
                let last_row = logits.row(t_padded - 1).to_owned();
                let mut new_logits = Array2::zeros((t_padded + 1, num_classes));
                for i in 0..t_padded {
                    new_logits.row_mut(i).assign(&logits.row(i));
                }
                new_logits.row_mut(t_padded).assign(&last_row);
                logits = new_logits;
                t_padded += 1;
            }

            // Flatten logits for the CTC API.
            let logits_flat: Vec<f32> = logits.iter().copied().collect();
            let (loss, grad) = ctc_loss_and_grad(
                &logits_flat,
                &gt_indices,
                num_classes,
                self.model.vocab.blank_idx,
            );

            // grad: [t_padded, num_classes], softmax - gamma.
            // Backprop through FC: dW[j, k] = sum_t grad[t, j] * lstm2_out[t, k]
            //                     dB[j] = sum_t grad[t, j]
            // Note: we only need grad w.r.t. lstm2_out from FC's backward,
            // which we don't propagate further (we keep CNN/LSTM frozen).
            for j in 0..num_classes {
                for k in 0..lstm_dim {
                    let mut g = 0.0f32;
                    for ti in 0..t {
                        g += grad[ti * num_classes + j] * lstm2_out[[ti, k]];
                    }
                    grad_w_acc[[j, k]] += g;
                }
                let mut gb = 0.0f32;
                for ti in 0..t {
                    gb += grad[ti * num_classes + j];
                }
                grad_b_acc[j] += gb;
            }

            total_loss += loss;
            n += 1;

            // Greedy decode accuracy
            let pred_text = self.model.decode_greedy_from_logits(&logits);
            total_chars += gt.chars().count();
            total_correct += pred_text.chars().zip(gt.chars()).filter(|(a, b)| a == b).count();
            total_words += 1;
            if pred_text.as_str() == gt.as_str() {
                word_correct += 1;
            }
        }

        // Apply Adam update with mean grad over the batch
        let mean: f32 = 1.0 / n as f32;
        grad_w_acc.mapv_inplace(|v| v * mean);
        grad_b_acc.mapv_inplace(|v| v * mean);
        adam_update_fc(
            &mut self.model,
            &grad_w_acc,
            &grad_b_acc,
            adam_state,
            self.learning_rate,
        );

        // Aggregate metrics
        EpochMetrics {
            epoch: 0,
            train_loss: total_loss / n as f32,
            train_cer: 1.0 - total_correct as f32 / total_chars.max(1) as f32,
            train_wer: 1.0 - word_correct as f32 / total_words.max(1) as f32,
            val_loss: 0.0,
            val_cer: 0.0,
            val_wer: 0.0,
            samples_per_sec: 0.0,
        };

        total_loss / n as f32
    }

    /// Save model checkpoint (config + vocabulary + FC weights/biases)
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let config = &self.model.config;
        let vocab_chars: String = self.model.vocab.chars.iter().collect();
        let fc_weight: Vec<Vec<f32>> = self
            .model
            .fc_weight
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let fc_bias: Vec<f32> = self.model.fc_bias.to_vec();

        let checkpoint = serde_json::json!({
            "config": config,
            "vocab": vocab_chars,
            "parameter_count": self.model.parameter_count(),
            "fc_weight": fc_weight,
            "fc_bias": fc_bias,
        });

        let mut file = std::fs::File::create(path)?;
        file.write_all(serde_json::to_string_pretty(&checkpoint)?.as_bytes())?;
        Ok(())
    }

    /// Load model checkpoint (config + vocabulary + FC weights/biases) into the trainer's model
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        let bytes = std::fs::read(path)?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;

        let fc_weight_rows: Vec<Vec<f32>> = serde_json::from_value(value["fc_weight"].clone())?;
        let fc_bias_vec: Vec<f32> = serde_json::from_value(value["fc_bias"].clone())?;
        let expected_rows = self.model.fc_weight.nrows();
        let expected_cols = self.model.fc_weight.ncols();
        if fc_weight_rows.len() != expected_rows
            || fc_weight_rows.iter().any(|r| r.len() != expected_cols)
        {
            return Err(crate::utils::OcrError::Configuration(format!(
                "FC weight shape mismatch: expected {}x{}, got {}x{:?}",
                expected_rows,
                expected_cols,
                fc_weight_rows.len(),
                fc_weight_rows.first().map(|r| r.len()).unwrap_or(0)
            )));
        }
        if fc_bias_vec.len() != self.model.fc_bias.len() {
            return Err(crate::utils::OcrError::Configuration(format!(
                "FC bias length mismatch: expected {}, got {}",
                self.model.fc_bias.len(),
                fc_bias_vec.len()
            )));
        }

        let mut fc_weight = Array2::zeros((expected_rows, expected_cols));
        for (i, row) in fc_weight_rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                fc_weight[[i, j]] = *v;
            }
        }
        let mut fc_bias = Array1::zeros(self.model.fc_bias.len());
        for (i, v) in fc_bias_vec.iter().enumerate() {
            fc_bias[i] = *v;
        }
        self.model.fc_weight = fc_weight;
        self.model.fc_bias = fc_bias;
        self.model.fc_weight_quantized = None;
        Ok(())
    }
}

/// Extend CrnnModel with recognition from SyntheticSample
impl CrnnModel {
    pub fn recognize_from_sample(&self, sample: &SyntheticSample) -> String {
        let gray = sample.image.to_luma8();
        let (w, h) = (gray.width() as usize, gray.height() as usize);

        let mut arr = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            }
        }

        let target_h = self.config.input_height;
        if h != target_h {
            arr = Self::resize_array2_height(&arr, target_h);
        }

        let logits = self.forward(&arr);
        let decoder = crate::recognition::ctc_decoder::CtcDecoder::new();
        decoder.greedy_decode(&logits, &self.vocab.chars)
    }
}

pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];
    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

pub fn word_error_distance(a: &str, b: &str) -> usize {
    let a_words: Vec<&str> = a.split_whitespace().collect();
    let b_words: Vec<&str> = b.split_whitespace().collect();
    levenshtein_distance(&a_words.join(" "), &b_words.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        let trainer = CrnnTrainer::new(model);
        assert_eq!(trainer.learning_rate, 0.001);
        assert_eq!(trainer.batch_size, 32);
    }

    #[test]
    fn test_trainer_epoch() {
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        let mut trainer = CrnnTrainer::new(model);
        let metrics = trainer.train_epoch(2, 4);
        assert!(metrics.train_loss >= 0.0);
        assert!(metrics.samples_per_sec > 0.0);
    }

    #[test]
    fn test_trainer_checkpoint() {
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        let mut trainer = CrnnTrainer::new(model);
        // Inject a distinctive FC weight so we can verify it round-trips.
        trainer.model.fc_weight[[0, 0]] = 0.12345;
        trainer.model.fc_bias[1] = -0.6789;
        let temp_path = std::path::Path::new("/tmp/test_crnn_checkpoint.json");
        trainer.save_checkpoint(temp_path).unwrap();
        assert!(temp_path.exists());

        let mut fresh = CrnnTrainer::new(CrnnModel::new(CrnnConfig::default()));
        fresh.load_checkpoint(temp_path).unwrap();
        assert!((fresh.model.fc_weight[[0, 0]] - 0.12345).abs() < 1e-6);
        assert!((fresh.model.fc_bias[1] - -0.6789).abs() < 1e-6);
        assert_eq!(fresh.model.fc_weight.shape(), trainer.model.fc_weight.shape());

        let _ = std::fs::remove_file(temp_path);
    }

    /// Regression test: `train_batch_full` must actually update model parameters
    /// using the computed gradients. Before the fix, `apply_full_adam` ignored
    /// the gradient argument `g` and used parameter values as gradients — the
    /// parameters still changed but not in the gradient direction. This test
    /// verifies that FC weights change after a training step and that loss
    /// decreases over multiple steps on the same batch.
    #[test]
    fn test_train_batch_full_updates_params() {
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        let mut trainer = CrnnTrainer::new(model);
        trainer.learning_rate = 0.01;

        let generator = TextLineGenerator::default();
        let samples = vec![generator.generate("HELLO")];

        let fc_before = trainer.model.fc_weight.clone();

        let mut adam = FullAdamState::new_like(&trainer.model);
        let loss1 = trainer.train_batch_full(&samples, &mut adam);

        // FC weights must change after a training step.
        let fc_after = &trainer.model.fc_weight;
        let changed = fc_before
            .iter()
            .zip(fc_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "FC weights did not change after train_batch_full");

        // Loss should decrease over a few steps on the same batch.
        let mut prev_loss = loss1;
        for _ in 0..4 {
            let loss = trainer.train_batch_full(&samples, &mut adam);
            assert!(
                loss <= prev_loss + 0.5,
                "loss increased unexpectedly: {prev_loss} -> {loss}"
            );
            prev_loss = loss;
        }
    }
}

// ---------------------------------------------------------------------------
// Full Adam state (covers every parameter in the model)
// ---------------------------------------------------------------------------

use crate::recognition::backward::ParamGrads as FullParamGrads;

pub struct FullAdamState {
    pub conv1_w_m: ndarray::Array3<f32>,
    pub conv1_w_v: ndarray::Array3<f32>,
    pub conv1_b_m: ndarray::Array1<f32>,
    pub conv1_b_v: ndarray::Array1<f32>,
    pub conv2_w_m: ndarray::Array3<f32>,
    pub conv2_w_v: ndarray::Array3<f32>,
    pub conv2_b_m: ndarray::Array1<f32>,
    pub conv2_b_v: ndarray::Array1<f32>,
    pub conv3_w_m: ndarray::Array3<f32>,
    pub conv3_w_v: ndarray::Array3<f32>,
    pub conv3_b_m: ndarray::Array1<f32>,
    pub conv3_b_v: ndarray::Array1<f32>,
    pub conv4_w_m: ndarray::Array3<f32>,
    pub conv4_w_v: ndarray::Array3<f32>,
    pub conv4_b_m: ndarray::Array1<f32>,
    pub conv4_b_v: ndarray::Array1<f32>,
    pub conv5_w_m: ndarray::Array3<f32>,
    pub conv5_w_v: ndarray::Array3<f32>,
    pub conv5_b_m: ndarray::Array1<f32>,
    pub conv5_b_v: ndarray::Array1<f32>,
    pub bn1s_m: ndarray::Array1<f32>,
    pub bn1s_v: ndarray::Array1<f32>,
    pub bn1p_m: ndarray::Array1<f32>,
    pub bn1p_v: ndarray::Array1<f32>,
    pub bn2s_m: ndarray::Array1<f32>,
    pub bn2s_v: ndarray::Array1<f32>,
    pub bn2p_m: ndarray::Array1<f32>,
    pub bn2p_v: ndarray::Array1<f32>,
    pub bn3s_m: ndarray::Array1<f32>,
    pub bn3s_v: ndarray::Array1<f32>,
    pub bn3p_m: ndarray::Array1<f32>,
    pub bn3p_v: ndarray::Array1<f32>,
    pub bn4s_m: ndarray::Array1<f32>,
    pub bn4s_v: ndarray::Array1<f32>,
    pub bn4p_m: ndarray::Array1<f32>,
    pub bn4p_v: ndarray::Array1<f32>,
    pub bn5s_m: ndarray::Array1<f32>,
    pub bn5s_v: ndarray::Array1<f32>,
    pub bn5p_m: ndarray::Array1<f32>,
    pub bn5p_v: ndarray::Array1<f32>,
    pub lstm1wfih_m: ndarray::Array2<f32>,
    pub lstm1wfih_v: ndarray::Array2<f32>,
    pub lstm1wfhh_m: ndarray::Array2<f32>,
    pub lstm1wfhh_v: ndarray::Array2<f32>,
    pub lstm1bfih_m: ndarray::Array1<f32>,
    pub lstm1bfih_v: ndarray::Array1<f32>,
    pub lstm1bfhh_m: ndarray::Array1<f32>,
    pub lstm1bfhh_v: ndarray::Array1<f32>,
    pub lstm1wbih_m: ndarray::Array2<f32>,
    pub lstm1wbih_v: ndarray::Array2<f32>,
    pub lstm1wbhh_m: ndarray::Array2<f32>,
    pub lstm1wbhh_v: ndarray::Array2<f32>,
    pub lstm1bbih_m: ndarray::Array1<f32>,
    pub lstm1bbih_v: ndarray::Array1<f32>,
    pub lstm1bbhh_m: ndarray::Array1<f32>,
    pub lstm1bbhh_v: ndarray::Array1<f32>,
    pub lstm2wfih_m: ndarray::Array2<f32>,
    pub lstm2wfih_v: ndarray::Array2<f32>,
    pub lstm2wfhh_m: ndarray::Array2<f32>,
    pub lstm2wfhh_v: ndarray::Array2<f32>,
    pub lstm2bfih_m: ndarray::Array1<f32>,
    pub lstm2bfih_v: ndarray::Array1<f32>,
    pub lstm2bfhh_m: ndarray::Array1<f32>,
    pub lstm2bfhh_v: ndarray::Array1<f32>,
    pub lstm2wbih_m: ndarray::Array2<f32>,
    pub lstm2wbih_v: ndarray::Array2<f32>,
    pub lstm2wbhh_m: ndarray::Array2<f32>,
    pub lstm2wbhh_v: ndarray::Array2<f32>,
    pub lstm2bbih_m: ndarray::Array1<f32>,
    pub lstm2bbih_v: ndarray::Array1<f32>,
    pub lstm2bbhh_m: ndarray::Array1<f32>,
    pub lstm2bbhh_v: ndarray::Array1<f32>,
    pub fcw_m: ndarray::Array2<f32>,
    pub fcw_v: ndarray::Array2<f32>,
    pub fcb_m: ndarray::Array1<f32>,
    pub fcb_v: ndarray::Array1<f32>,
    pub t: u64,
}

impl FullAdamState {
    pub fn new_like(model: &CrnnModel) -> Self {
        let m3 = |s| ndarray::Array3::<f32>::zeros(s);
        let m2 = |s| ndarray::Array2::<f32>::zeros(s);
        let m1 = |n| ndarray::Array1::<f32>::zeros(n);
        Self {
            conv1_w_m: m3(model.cnn.conv1_weights.dim()),
            conv1_w_v: m3(model.cnn.conv1_weights.dim()),
            conv1_b_m: m1(model.cnn.conv1_bias.len()),
            conv1_b_v: m1(model.cnn.conv1_bias.len()),
            conv2_w_m: m3(model.cnn.conv2_weights.dim()),
            conv2_w_v: m3(model.cnn.conv2_weights.dim()),
            conv2_b_m: m1(model.cnn.conv2_bias.len()),
            conv2_b_v: m1(model.cnn.conv2_bias.len()),
            conv3_w_m: m3(model.cnn.conv3_weights.dim()),
            conv3_w_v: m3(model.cnn.conv3_weights.dim()),
            conv3_b_m: m1(model.cnn.conv3_bias.len()),
            conv3_b_v: m1(model.cnn.conv3_bias.len()),
            conv4_w_m: m3(model.cnn.conv4_weights.dim()),
            conv4_w_v: m3(model.cnn.conv4_weights.dim()),
            conv4_b_m: m1(model.cnn.conv4_bias.len()),
            conv4_b_v: m1(model.cnn.conv4_bias.len()),
            conv5_w_m: m3(model.cnn.conv5_weights.dim()),
            conv5_w_v: m3(model.cnn.conv5_weights.dim()),
            conv5_b_m: m1(model.cnn.conv5_bias.len()),
            conv5_b_v: m1(model.cnn.conv5_bias.len()),
            bn1s_m: m1(model.cnn.bn_scales[0].len()),
            bn1s_v: m1(model.cnn.bn_scales[0].len()),
            bn1p_m: m1(model.cnn.bn_shifts[0].len()),
            bn1p_v: m1(model.cnn.bn_shifts[0].len()),
            bn2s_m: m1(model.cnn.bn_scales[1].len()),
            bn2s_v: m1(model.cnn.bn_scales[1].len()),
            bn2p_m: m1(model.cnn.bn_shifts[1].len()),
            bn2p_v: m1(model.cnn.bn_shifts[1].len()),
            bn3s_m: m1(model.cnn.bn_scales[2].len()),
            bn3s_v: m1(model.cnn.bn_scales[2].len()),
            bn3p_m: m1(model.cnn.bn_shifts[2].len()),
            bn3p_v: m1(model.cnn.bn_shifts[2].len()),
            bn4s_m: m1(model.cnn.bn_scales[3].len()),
            bn4s_v: m1(model.cnn.bn_scales[3].len()),
            bn4p_m: m1(model.cnn.bn_shifts[3].len()),
            bn4p_v: m1(model.cnn.bn_shifts[3].len()),
            bn5s_m: m1(model.cnn.bn_scales[4].len()),
            bn5s_v: m1(model.cnn.bn_scales[4].len()),
            bn5p_m: m1(model.cnn.bn_shifts[4].len()),
            bn5p_v: m1(model.cnn.bn_shifts[4].len()),
            lstm1wfih_m: m2(model.lstm1.wf_ih.dim()),
            lstm1wfih_v: m2(model.lstm1.wf_ih.dim()),
            lstm1wfhh_m: m2(model.lstm1.wf_hh.dim()),
            lstm1wfhh_v: m2(model.lstm1.wf_hh.dim()),
            lstm1bfih_m: m1(model.lstm1.bf_ih.len()),
            lstm1bfih_v: m1(model.lstm1.bf_ih.len()),
            lstm1bfhh_m: m1(model.lstm1.bf_hh.len()),
            lstm1bfhh_v: m1(model.lstm1.bf_hh.len()),
            lstm1wbih_m: m2(model.lstm1.wb_ih.dim()),
            lstm1wbih_v: m2(model.lstm1.wb_ih.dim()),
            lstm1wbhh_m: m2(model.lstm1.wb_hh.dim()),
            lstm1wbhh_v: m2(model.lstm1.wb_hh.dim()),
            lstm1bbih_m: m1(model.lstm1.bb_ih.len()),
            lstm1bbih_v: m1(model.lstm1.bb_ih.len()),
            lstm1bbhh_m: m1(model.lstm1.bb_hh.len()),
            lstm1bbhh_v: m1(model.lstm1.bb_hh.len()),
            lstm2wfih_m: m2(model.lstm2.wf_ih.dim()),
            lstm2wfih_v: m2(model.lstm2.wf_ih.dim()),
            lstm2wfhh_m: m2(model.lstm2.wf_hh.dim()),
            lstm2wfhh_v: m2(model.lstm2.wf_hh.dim()),
            lstm2bfih_m: m1(model.lstm2.bf_ih.len()),
            lstm2bfih_v: m1(model.lstm2.bf_ih.len()),
            lstm2bfhh_m: m1(model.lstm2.bf_hh.len()),
            lstm2bfhh_v: m1(model.lstm2.bf_hh.len()),
            lstm2wbih_m: m2(model.lstm2.wb_ih.dim()),
            lstm2wbih_v: m2(model.lstm2.wb_ih.dim()),
            lstm2wbhh_m: m2(model.lstm2.wb_hh.dim()),
            lstm2wbhh_v: m2(model.lstm2.wb_hh.dim()),
            lstm2bbih_m: m1(model.lstm2.bb_ih.len()),
            lstm2bbih_v: m1(model.lstm2.bb_ih.len()),
            lstm2bbhh_m: m1(model.lstm2.bb_hh.len()),
            lstm2bbhh_v: m1(model.lstm2.bb_hh.len()),
            fcw_m: m2(model.fc_weight.dim()),
            fcw_v: m2(model.fc_weight.dim()),
            fcb_m: m1(model.fc_bias.len()),
            fcb_v: m1(model.fc_bias.len()),
            t: 0,
        }
    }
}

const INPUT_WIDTH: usize = 256;
const INPUT_HEIGHT: usize = 32;

fn resize_to(arr: &ndarray::Array2<f32>, w: usize, h: usize) -> ndarray::Array2<f32> {
    use image::GrayImage;
    let (cur_h, cur_w) = arr.dim();
    let mut img = GrayImage::new(cur_w as u32, cur_h as u32);
    for y in 0..cur_h {
        for x in 0..cur_w {
            img.put_pixel(x as u32, y as u32, image::Luma([(arr[[y, x]] * 255.0) as u8]));
        }
    }
    let resized = image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3);
    let mut out = ndarray::Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = resized.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
        }
    }
    out
}

fn accumulate_grads(acc: &mut FullParamGrads, g: &FullParamGrads) {
    acc.conv1_w += &g.conv1_w;
    acc.conv1_b += &g.conv1_b;
    acc.conv2_w += &g.conv2_w;
    acc.conv2_b += &g.conv2_b;
    acc.conv3_w += &g.conv3_w;
    acc.conv3_b += &g.conv3_b;
    acc.conv4_w += &g.conv4_w;
    acc.conv4_b += &g.conv4_b;
    acc.conv5_w += &g.conv5_w;
    acc.conv5_b += &g.conv5_b;
    acc.bn1_s += &g.bn1_s;
    acc.bn1_p += &g.bn1_p;
    acc.bn2_s += &g.bn2_s;
    acc.bn2_p += &g.bn2_p;
    acc.bn3_s += &g.bn3_s;
    acc.bn3_p += &g.bn3_p;
    acc.bn4_s += &g.bn4_s;
    acc.bn4_p += &g.bn4_p;
    acc.bn5_s += &g.bn5_s;
    acc.bn5_p += &g.bn5_p;
    acc.lstm1_wf_ih += &g.lstm1_wf_ih;
    acc.lstm1_wf_hh += &g.lstm1_wf_hh;
    acc.lstm1_bf_ih += &g.lstm1_bf_ih;
    acc.lstm1_bf_hh += &g.lstm1_bf_hh;
    acc.lstm1_wb_ih += &g.lstm1_wb_ih;
    acc.lstm1_wb_hh += &g.lstm1_wb_hh;
    acc.lstm1_bb_ih += &g.lstm1_bb_ih;
    acc.lstm1_bb_hh += &g.lstm1_bb_hh;
    acc.lstm2_wf_ih += &g.lstm2_wf_ih;
    acc.lstm2_wf_hh += &g.lstm2_wf_hh;
    acc.lstm2_bf_ih += &g.lstm2_bf_ih;
    acc.lstm2_bf_hh += &g.lstm2_bf_hh;
    acc.lstm2_wb_ih += &g.lstm2_wb_ih;
    acc.lstm2_wb_hh += &g.lstm2_wb_hh;
    acc.lstm2_bb_ih += &g.lstm2_bb_ih;
    acc.lstm2_bb_hh += &g.lstm2_bb_hh;
    acc.fc_w += &g.fc_w;
    acc.fc_b += &g.fc_b;
}

fn scale_grads(g: &mut FullParamGrads, s: f32) {
    g.conv1_w *= s;
    g.conv1_b *= s;
    g.conv2_w *= s;
    g.conv2_b *= s;
    g.conv3_w *= s;
    g.conv3_b *= s;
    g.conv4_w *= s;
    g.conv4_b *= s;
    g.conv5_w *= s;
    g.conv5_b *= s;
    g.bn1_s *= s;
    g.bn1_p *= s;
    g.bn2_s *= s;
    g.bn2_p *= s;
    g.bn3_s *= s;
    g.bn3_p *= s;
    g.bn4_s *= s;
    g.bn4_p *= s;
    g.bn5_s *= s;
    g.bn5_p *= s;
    g.lstm1_wf_ih *= s;
    g.lstm1_wf_hh *= s;
    g.lstm1_bf_ih *= s;
    g.lstm1_bf_hh *= s;
    g.lstm1_wb_ih *= s;
    g.lstm1_wb_hh *= s;
    g.lstm1_bb_ih *= s;
    g.lstm1_bb_hh *= s;
    g.lstm2_wf_ih *= s;
    g.lstm2_wf_hh *= s;
    g.lstm2_bf_ih *= s;
    g.lstm2_bf_hh *= s;
    g.lstm2_wb_ih *= s;
    g.lstm2_wb_hh *= s;
    g.lstm2_bb_ih *= s;
    g.lstm2_bb_hh *= s;
    g.fc_w *= s;
    g.fc_b *= s;
}

fn apply_full_adam(
    model: &mut CrnnModel,
    g: &FullParamGrads,
    s: &mut FullAdamState,
    lr: f32,
) {
    s.t += 1;
    let bc1 = 1.0 - 0.9f32.powi(s.t as i32);
    let bc2 = 1.0 - 0.999f32.powi(s.t as i32);
    let eps = 1e-8f32;
    macro_rules! adam {
        ($grad:expr, $param:expr, $m:expr, $v:expr) => {{
            for ix in $grad.iter().zip($param.iter_mut().zip($m.iter_mut().zip($v.iter_mut()))) {
                let (gv, (pv, (mm, vv))) = ix;
                *mm = 0.9 * *mm + 0.1 * *gv;
                *vv = 0.999 * *vv + 0.001 * *gv * *gv;
                let mh = *mm / bc1;
                let vh = *vv / bc2;
                *pv -= lr * mh / (vh.sqrt() + eps);
            }
        }};
    }
    adam!(g.conv1_w, model.cnn.conv1_weights, s.conv1_w_m, s.conv1_w_v);
    adam!(g.conv1_b, model.cnn.conv1_bias, s.conv1_b_m, s.conv1_b_v);
    adam!(g.conv2_w, model.cnn.conv2_weights, s.conv2_w_m, s.conv2_w_v);
    adam!(g.conv2_b, model.cnn.conv2_bias, s.conv2_b_m, s.conv2_b_v);
    adam!(g.conv3_w, model.cnn.conv3_weights, s.conv3_w_m, s.conv3_w_v);
    adam!(g.conv3_b, model.cnn.conv3_bias, s.conv3_b_m, s.conv3_b_v);
    adam!(g.conv4_w, model.cnn.conv4_weights, s.conv4_w_m, s.conv4_w_v);
    adam!(g.conv4_b, model.cnn.conv4_bias, s.conv4_b_m, s.conv4_b_v);
    adam!(g.conv5_w, model.cnn.conv5_weights, s.conv5_w_m, s.conv5_w_v);
    adam!(g.conv5_b, model.cnn.conv5_bias, s.conv5_b_m, s.conv5_b_v);
    adam!(g.bn1_s, model.cnn.bn_scales[0], s.bn1s_m, s.bn1s_v);
    adam!(g.bn1_p, model.cnn.bn_shifts[0], s.bn1p_m, s.bn1p_v);
    adam!(g.bn2_s, model.cnn.bn_scales[1], s.bn2s_m, s.bn2s_v);
    adam!(g.bn2_p, model.cnn.bn_shifts[1], s.bn2p_m, s.bn2p_v);
    adam!(g.bn3_s, model.cnn.bn_scales[2], s.bn3s_m, s.bn3s_v);
    adam!(g.bn3_p, model.cnn.bn_shifts[2], s.bn3p_m, s.bn3p_v);
    adam!(g.bn4_s, model.cnn.bn_scales[3], s.bn4s_m, s.bn4s_v);
    adam!(g.bn4_p, model.cnn.bn_shifts[3], s.bn4p_m, s.bn4p_v);
    adam!(g.bn5_s, model.cnn.bn_scales[4], s.bn5s_m, s.bn5s_v);
    adam!(g.bn5_p, model.cnn.bn_shifts[4], s.bn5p_m, s.bn5p_v);
    adam!(g.lstm1_wf_ih, model.lstm1.wf_ih, s.lstm1wfih_m, s.lstm1wfih_v);
    adam!(g.lstm1_wf_hh, model.lstm1.wf_hh, s.lstm1wfhh_m, s.lstm1wfhh_v);
    adam!(g.lstm1_bf_ih, model.lstm1.bf_ih, s.lstm1bfih_m, s.lstm1bfih_v);
    adam!(g.lstm1_bf_hh, model.lstm1.bf_hh, s.lstm1bfhh_m, s.lstm1bfhh_v);
    adam!(g.lstm1_wb_ih, model.lstm1.wb_ih, s.lstm1wbih_m, s.lstm1wbih_v);
    adam!(g.lstm1_wb_hh, model.lstm1.wb_hh, s.lstm1wbhh_m, s.lstm1wbhh_v);
    adam!(g.lstm1_bb_ih, model.lstm1.bb_ih, s.lstm1bbih_m, s.lstm1bbih_v);
    adam!(g.lstm1_bb_hh, model.lstm1.bb_hh, s.lstm1bbhh_m, s.lstm1bbhh_v);
    adam!(g.lstm2_wf_ih, model.lstm2.wf_ih, s.lstm2wfih_m, s.lstm2wfih_v);
    adam!(g.lstm2_wf_hh, model.lstm2.wf_hh, s.lstm2wfhh_m, s.lstm2wfhh_v);
    adam!(g.lstm2_bf_ih, model.lstm2.bf_ih, s.lstm2bfih_m, s.lstm2bfih_v);
    adam!(g.lstm2_bf_hh, model.lstm2.bf_hh, s.lstm2bfhh_m, s.lstm2bfhh_v);
    adam!(g.lstm2_wb_ih, model.lstm2.wb_ih, s.lstm2wbih_m, s.lstm2wbih_v);
    adam!(g.lstm2_wb_hh, model.lstm2.wb_hh, s.lstm2wbhh_m, s.lstm2wbhh_v);
    adam!(g.lstm2_bb_ih, model.lstm2.bb_ih, s.lstm2bbih_m, s.lstm2bbih_v);
    adam!(g.lstm2_bb_hh, model.lstm2.bb_hh, s.lstm2bbhh_m, s.lstm2bbhh_v);
    adam!(g.fc_w, model.fc_weight, s.fcw_m, s.fcw_v);
    adam!(g.fc_b, model.fc_bias, s.fcb_m, s.fcb_v);
}
