//! Benchmark harness for OCR engines on synthetic data
//!
//! Measures Character Error Rate (CER) and Word Error Rate (WER)
//! to track improvement across phases.

use crate::synthetic::{DistortionConfig, SyntheticSample, TextLineGenerator};
use crate::utils::Result;
use std::path::Path;

/// Benchmark metrics for OCR evaluation
#[derive(Debug, Clone, Default)]
pub struct BenchmarkMetrics {
    /// Total samples evaluated
    pub total_samples: usize,
    /// Samples with perfect recognition (CER == 0)
    pub perfect_recognitions: usize,
    /// Total characters in ground truth
    pub total_chars: usize,
    /// Character errors (insertions + deletions + substitutions)
    pub char_errors: usize,
    /// Total words in ground truth
    pub total_words: usize,
    /// Word errors
    pub word_errors: usize,
    /// Average inference time per sample in milliseconds
    pub avg_time_ms: f64,
}

impl BenchmarkMetrics {
    /// Character Error Rate (0.0 = perfect, 1.0 = all wrong)
    pub fn cer(&self) -> f64 {
        if self.total_chars == 0 {
            0.0
        } else {
            self.char_errors as f64 / self.total_chars as f64
        }
    }

    /// Word Error Rate
    pub fn wer(&self) -> f64 {
        if self.total_words == 0 {
            0.0
        } else {
            self.word_errors as f64 / self.total_words as f64
        }
    }

    /// Accuracy percentage (perfect recognitions)
    pub fn accuracy(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.perfect_recognitions as f64 / self.total_samples as f64 * 100.0
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Samples: {} | Perfect: {} ({:.1}%) | CER: {:.2}% | WER: {:.2}% | Avg time: {:.2}ms",
            self.total_samples,
            self.perfect_recognitions,
            self.accuracy(),
            self.cer() * 100.0,
            self.wer() * 100.0,
            self.avg_time_ms
        )
    }
}

/// A recognizer function that can be benchmarked
pub trait BenchmarkRecognizer {
    fn recognize(&self, image: &image::DynamicImage) -> Result<String>;
}

/// Run a benchmark suite against a recognizer
pub fn run_benchmark(
    recognizer: &dyn BenchmarkRecognizer,
    samples: &[SyntheticSample],
) -> BenchmarkMetrics {
    let mut metrics = BenchmarkMetrics::default();
    let mut total_time_ms = 0.0;

    for sample in samples {
        let start = std::time::Instant::now();
        let predicted = recognizer.recognize(&sample.image).unwrap_or_default();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        total_time_ms += elapsed;

        metrics.total_samples += 1;
        metrics.total_chars += sample.ground_truth.chars().count();
        metrics.total_words += sample.ground_truth.split_whitespace().count();

        let char_errs = levenshtein_distance(&sample.ground_truth, &predicted);
        metrics.char_errors += char_errs;

        if predicted == sample.ground_truth {
            metrics.perfect_recognitions += 1;
        }

        let word_errs = word_error_distance(&sample.ground_truth, &predicted);
        metrics.word_errors += word_errs;
    }

    if !samples.is_empty() {
        metrics.avg_time_ms = total_time_ms / samples.len() as f64;
    }

    metrics
}

/// Generate a synthetic benchmark dataset
pub struct BenchmarkDataset;

impl BenchmarkDataset {
    /// Generate a standard test set for Latin text
    pub fn generate_latin_test_set(count: usize) -> Vec<SyntheticSample> {
        let generator = TextLineGenerator::default();
        let texts = generator.generate_random_texts(count, 15);
        generator.generate_batch(&texts)
    }

    /// Generate a clean test set (no distortions)
    pub fn generate_clean(count: usize) -> Vec<SyntheticSample> {
        Self::generate_latin_test_set(count)
    }

    /// Generate a mildly distorted test set
    pub fn generate_mild(count: usize) -> Vec<SyntheticSample> {
        let generator = TextLineGenerator::default();
        let texts = generator.generate_random_texts(count, 15);
        let mut samples = generator.generate_batch(&texts);
        crate::synthetic::distortion::augment_batch(&mut samples, &DistortionConfig::mild());
        samples
    }

    /// Generate a heavily distorted test set
    pub fn generate_heavy(count: usize) -> Vec<SyntheticSample> {
        let generator = TextLineGenerator::default();
        let texts = generator.generate_random_texts(count, 15);
        let mut samples = generator.generate_batch(&texts);
        crate::synthetic::distortion::augment_batch(&mut samples, &DistortionConfig::heavy());
        samples
    }
}

/// Compute Levenshtein edit distance between two strings
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

/// Compute word-level edit distance
fn word_error_distance(a: &str, b: &str) -> usize {
    let a_words: Vec<&str> = a.split_whitespace().collect();
    let b_words: Vec<&str> = b.split_whitespace().collect();
    levenshtein_distance(&a_words.join(" "), &b_words.join(" "))
}

/// Save benchmark results to a JSON file for tracking over time
pub fn save_results(metrics: &BenchmarkMetrics, path: &Path) -> Result<()> {
    let json = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "total_samples": metrics.total_samples,
        "perfect_recognitions": metrics.perfect_recognitions,
        "accuracy_percent": metrics.accuracy(),
        "cer": metrics.cer(),
        "wer": metrics.wer(),
        "avg_time_ms": metrics.avg_time_ms,
    });

    let content = serde_json::to_string_pretty(&json)?;
    std::fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        assert_eq!(levenshtein_distance("", "hello"), 5);
    }

    #[test]
    fn test_benchmark_metrics() {
        let mut metrics = BenchmarkMetrics::default();
        metrics.total_samples = 100;
        metrics.perfect_recognitions = 85;
        metrics.total_chars = 1000;
        metrics.char_errors = 50;
        metrics.total_words = 200;
        metrics.word_errors = 20;

        assert_eq!(metrics.accuracy(), 85.0);
        assert_eq!(metrics.cer(), 0.05);
        assert_eq!(metrics.wer(), 0.10);
    }

    #[test]
    fn test_generate_benchmark_dataset() {
        let samples = BenchmarkDataset::generate_clean(10);
        assert_eq!(samples.len(), 10);
        for sample in &samples {
            assert!(!sample.ground_truth.is_empty());
        }
    }

    #[test]
    fn test_generate_distorted_dataset() {
        let samples = BenchmarkDataset::generate_mild(5);
        assert_eq!(samples.len(), 5);
    }
}
