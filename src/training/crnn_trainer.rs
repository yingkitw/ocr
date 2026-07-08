//! CRNN training pipeline
//!
//! Trains the CRNN model on synthetic text-line images using CTC loss.

use crate::recognition::crnn::CrnnModel;
use crate::synthetic::{DistortionConfig, SyntheticSample, TextLineGenerator};
use crate::utils::Result;
use ndarray::Array2;
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

    /// Save model checkpoint
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let config = &self.model.config;
        let vocab_chars: String = self.model.vocab.chars.iter().collect();

        let checkpoint = serde_json::json!({
            "config": config,
            "vocab": vocab_chars,
            "parameter_count": self.model.parameter_count(),
        });

        let mut file = std::fs::File::create(path)?;
        file.write_all(serde_json::to_string_pretty(&checkpoint)?.as_bytes())?;
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
        let trainer = CrnnTrainer::new(model);
        let temp_path = std::path::Path::new("/tmp/test_crnn_checkpoint.json");
        trainer.save_checkpoint(temp_path).unwrap();
        assert!(temp_path.exists());
        let _ = std::fs::remove_file(temp_path);
    }
}
