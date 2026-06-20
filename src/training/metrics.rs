//! Training metrics and monitoring

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Character Error Rate (CER) between two strings
pub fn cer(predicted: &str, ground_truth: &str) -> f32 {
    if ground_truth.is_empty() {
        return if predicted.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein_distance(predicted, ground_truth);
    dist as f32 / ground_truth.chars().count() as f32
}

/// Word Error Rate (WER) between two strings
pub fn wer(predicted: &str, ground_truth: &str) -> f32 {
    let pred_words: Vec<&str> = predicted.split_whitespace().collect();
    let gt_words: Vec<&str> = ground_truth.split_whitespace().collect();
    if gt_words.is_empty() {
        return if pred_words.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein_distance(&pred_words.join(" "), &gt_words.join(" "));
    dist as f32 / gt_words.len() as f32
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

/// Training metrics for a single epoch or batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub learning_rate: f32,
    pub processing_time: Duration,
    pub samples_processed: usize,
    pub additional_metrics: HashMap<String, f32>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            loss: 0.0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            learning_rate: 0.0,
            processing_time: Duration::from_secs(0),
            samples_processed: 0,
            additional_metrics: HashMap::new(),
        }
    }

    pub fn add_loss(&mut self, loss: f32) {
        self.loss += loss;
    }

    pub fn add_accuracy(&mut self, accuracy: f32) {
        self.accuracy += accuracy;
    }

    pub fn add_precision(&mut self, precision: f32) {
        self.precision += precision;
    }

    pub fn add_recall(&mut self, recall: f32) {
        self.recall += recall;
    }

    pub fn add_f1_score(&mut self, f1_score: f32) {
        self.f1_score += f1_score;
    }

    pub fn add_metric(&mut self, name: String, value: f32) {
        self.additional_metrics.insert(name, value);
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    pub fn set_processing_time(&mut self, time: Duration) {
        self.processing_time = time;
    }

    pub fn set_samples_processed(&mut self, count: usize) {
        self.samples_processed = count;
    }

    pub fn finalize(&mut self) {
        if self.samples_processed > 0 {
            self.loss /= self.samples_processed as f32;
            self.accuracy /= self.samples_processed as f32;
            self.precision /= self.samples_processed as f32;
            self.recall /= self.samples_processed as f32;
            self.f1_score /= self.samples_processed as f32;
        }
    }

    pub fn loss(&self) -> f32 {
        self.loss
    }

    pub fn accuracy(&self) -> f32 {
        self.accuracy
    }

    pub fn precision(&self) -> f32 {
        self.precision
    }

    pub fn recall(&self) -> f32 {
        self.recall
    }

    pub fn f1_score(&self) -> f32 {
        self.f1_score
    }
}

/// Metric tracker for monitoring training progress
pub struct MetricTracker {
    metrics_history: Vec<TrainingMetrics>,
    current_metrics: Option<TrainingMetrics>,
    best_metrics: Option<TrainingMetrics>,
    start_time: Option<Instant>,
}

impl MetricTracker {
    pub fn new() -> Self {
        Self {
            metrics_history: Vec::new(),
            current_metrics: None,
            best_metrics: None,
            start_time: None,
        }
    }

    pub fn start_epoch(&mut self) {
        self.start_time = Some(Instant::now());
        self.current_metrics = Some(TrainingMetrics::new());
    }

    pub fn end_epoch(&mut self) -> Option<TrainingMetrics> {
        if let Some(mut metrics) = self.current_metrics.take() {
            if let Some(start_time) = self.start_time {
                metrics.set_processing_time(start_time.elapsed());
            }
            metrics.finalize();

            // Update best metrics
            if self.best_metrics.is_none()
                || metrics.accuracy() > self.best_metrics.as_ref().unwrap().accuracy()
            {
                self.best_metrics = Some(metrics.clone());
            }

            self.metrics_history.push(metrics.clone());
            Some(metrics)
        } else {
            None
        }
    }

    pub fn add_batch_metrics(&mut self, loss: f32, accuracy: f32, samples: usize) {
        if let Some(ref mut metrics) = self.current_metrics {
            metrics.add_loss(loss);
            metrics.add_accuracy(accuracy);
            metrics.set_samples_processed(metrics.samples_processed + samples);
        }
    }

    pub fn add_custom_metric(&mut self, name: String, value: f32) {
        if let Some(ref mut metrics) = self.current_metrics {
            metrics.add_metric(name, value);
        }
    }

    pub fn get_current_metrics(&self) -> Option<&TrainingMetrics> {
        self.current_metrics.as_ref()
    }

    pub fn get_best_metrics(&self) -> Option<&TrainingMetrics> {
        self.best_metrics.as_ref()
    }

    pub fn get_metrics_history(&self) -> &[TrainingMetrics] {
        &self.metrics_history
    }

    pub fn reset(&mut self) {
        self.metrics_history.clear();
        self.current_metrics = None;
        self.best_metrics = None;
        self.start_time = None;
    }

    pub fn get_training_summary(&self) -> TrainingSummary {
        TrainingSummary {
            total_epochs: self.metrics_history.len(),
            best_accuracy: self
                .best_metrics
                .as_ref()
                .map(|m| m.accuracy())
                .unwrap_or(0.0),
            best_loss: self
                .best_metrics
                .as_ref()
                .map(|m| m.loss())
                .unwrap_or(f32::INFINITY),
            average_accuracy: self.calculate_average_accuracy(),
            average_loss: self.calculate_average_loss(),
            total_training_time: self.calculate_total_training_time(),
        }
    }

    fn calculate_average_accuracy(&self) -> f32 {
        if self.metrics_history.is_empty() {
            return 0.0;
        }
        self.metrics_history
            .iter()
            .map(|m| m.accuracy())
            .sum::<f32>()
            / self.metrics_history.len() as f32
    }

    fn calculate_average_loss(&self) -> f32 {
        if self.metrics_history.is_empty() {
            return 0.0;
        }
        self.metrics_history.iter().map(|m| m.loss()).sum::<f32>()
            / self.metrics_history.len() as f32
    }

    fn calculate_total_training_time(&self) -> Duration {
        self.metrics_history
            .iter()
            .map(|m| m.processing_time)
            .fold(Duration::from_secs(0), |acc, time| acc + time)
    }
}

/// Training summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub total_epochs: usize,
    pub best_accuracy: f32,
    pub best_loss: f32,
    pub average_accuracy: f32,
    pub average_loss: f32,
    pub total_training_time: Duration,
}

/// Real-time metrics monitor
pub struct MetricsMonitor {
    metrics: MetricTracker,
    log_interval: Duration,
    last_log_time: Instant,
}

impl MetricsMonitor {
    pub fn new(log_interval: Duration) -> Self {
        Self {
            metrics: MetricTracker::new(),
            log_interval,
            last_log_time: Instant::now(),
        }
    }

    pub fn start_epoch(&mut self) {
        self.metrics.start_epoch();
    }

    pub fn end_epoch(&mut self) -> Option<TrainingMetrics> {
        self.metrics.end_epoch()
    }

    pub fn log_batch_metrics(&mut self, loss: f32, accuracy: f32, samples: usize) {
        self.metrics.add_batch_metrics(loss, accuracy, samples);

        // Log if enough time has passed
        if self.last_log_time.elapsed() >= self.log_interval {
            if let Some(metrics) = self.metrics.get_current_metrics() {
                tracing::info!(
                    "Batch metrics - Loss: {:.4}, Accuracy: {:.4}, Samples: {}",
                    metrics.loss(),
                    metrics.accuracy(),
                    metrics.samples_processed
                );
            }
            self.last_log_time = Instant::now();
        }
    }

    pub fn log_custom_metric(&mut self, name: String, value: f32) {
        self.metrics.add_custom_metric(name, value);
    }

    pub fn get_summary(&self) -> TrainingSummary {
        self.metrics.get_training_summary()
    }
}

/// Metrics exporter for external monitoring systems
pub trait MetricsExporter {
    fn export_metrics(&self, metrics: &TrainingMetrics) -> Result<()>;
    fn export_summary(&self, summary: &TrainingSummary) -> Result<()>;
}

/// Console metrics exporter
pub struct ConsoleMetricsExporter;

impl MetricsExporter for ConsoleMetricsExporter {
    fn export_metrics(&self, metrics: &TrainingMetrics) -> Result<()> {
        println!(
            "Metrics: Loss={:.4}, Accuracy={:.4}, Precision={:.4}, Recall={:.4}, F1={:.4}",
            metrics.loss(),
            metrics.accuracy(),
            metrics.precision(),
            metrics.recall(),
            metrics.f1_score()
        );
        Ok(())
    }

    fn export_summary(&self, summary: &TrainingSummary) -> Result<()> {
        println!("Training Summary:");
        println!("  Total Epochs: {}", summary.total_epochs);
        println!("  Best Accuracy: {:.4}", summary.best_accuracy);
        println!("  Best Loss: {:.4}", summary.best_loss);
        println!("  Average Accuracy: {:.4}", summary.average_accuracy);
        println!("  Average Loss: {:.4}", summary.average_loss);
        println!("  Total Training Time: {:?}", summary.total_training_time);
        Ok(())
    }
}

/// JSON metrics exporter
pub struct JsonMetricsExporter {
    output_dir: String,
}

impl JsonMetricsExporter {
    pub fn new(output_dir: String) -> Self {
        Self { output_dir }
    }
}

impl MetricsExporter for JsonMetricsExporter {
    fn export_metrics(&self, metrics: &TrainingMetrics) -> Result<()> {
        let json = serde_json::to_string_pretty(metrics)?;
        let filename = format!(
            "{}/metrics_{}.json",
            self.output_dir,
            chrono::Utc::now().timestamp()
        );
        std::fs::write(filename, json)?;
        Ok(())
    }

    fn export_summary(&self, summary: &TrainingSummary) -> Result<()> {
        let json = serde_json::to_string_pretty(summary)?;
        let filename = format!("{}/summary.json", self.output_dir);
        std::fs::write(filename, json)?;
        Ok(())
    }
}

/// CSV metrics exporter
pub struct CsvMetricsExporter {
    output_file: String,
    headers_written: bool,
    metrics_history: Vec<TrainingMetrics>,
}

impl CsvMetricsExporter {
    pub fn new(output_file: String) -> Self {
        Self {
            output_file,
            headers_written: false,
            metrics_history: Vec::new(),
        }
    }

    pub fn add_metrics(&mut self, metrics: TrainingMetrics) {
        self.metrics_history.push(metrics);
    }
}

impl MetricsExporter for CsvMetricsExporter {
    fn export_metrics(&self, metrics: &TrainingMetrics) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_file)?;

        if !self.headers_written {
            writeln!(
                file,
                "epoch,loss,accuracy,precision,recall,f1_score,learning_rate,processing_time_ms,samples_processed"
            )?;
        }

        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            self.metrics_history.len(),
            metrics.loss(),
            metrics.accuracy(),
            metrics.precision(),
            metrics.recall(),
            metrics.f1_score(),
            metrics.learning_rate,
            metrics.processing_time.as_millis(),
            metrics.samples_processed
        )?;

        Ok(())
    }

    fn export_summary(&self, _summary: &TrainingSummary) -> Result<()> {
        // CSV summary export not implemented
        Ok(())
    }
}
