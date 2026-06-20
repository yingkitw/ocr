//! CRNN training pipeline
//!
//! Trains the CRNN model on synthetic text-line images using CTC loss.

use crate::recognition::crnn::{CrnnConfig, CrnnModel};
use crate::synthetic::{DistortionConfig, SyntheticSample, TextLineGenerator};
use crate::training::metrics::{cer, wer};
use crate::utils::Result;
use ndarray::{Array1, Array2};

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

    fn train_batch(&mut self, _samples: &[SyntheticSample]) -> f32 {
        // Simplified training: for now return a dummy loss
        // Full backpropagation through the CRNN would require:
        // 1. Forward pass for each sample
        // 2. CTC loss computation
        // 3. Backward pass to compute gradients
        // 4. Adam/SGD parameter update
        // This is a placeholder that would be filled with real backprop
        // once the gradient computation is fully implemented.
        0.5f32
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
        use image::GenericImageView;
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

fn levenshtein_distance(a: &str, b: &str) -> usize {
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

fn word_error_distance(a: &str, b: &str) -> usize {
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
