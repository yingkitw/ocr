//! Time utilities for OCR

use chrono::{DateTime, Utc};
use std::time::{Duration, Instant, SystemTime};

/// A timer for measuring execution time
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Create a new timer
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since creation
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get the elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }

    /// Get the elapsed time in microseconds
    pub fn elapsed_micros(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }

    /// Get the elapsed time in nanoseconds
    pub fn elapsed_nanos(&self) -> u64 {
        self.elapsed().as_nanos() as u64
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// A performance profiler for measuring multiple operations
#[derive(Debug, Clone)]
pub struct Profiler {
    operations: std::collections::HashMap<String, Vec<Duration>>,
    current_operation: Option<(String, Instant)>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            operations: std::collections::HashMap::new(),
            current_operation: None,
        }
    }

    /// Start timing an operation
    pub fn start_operation(&mut self, name: &str) {
        if let Some((prev_name, start)) = self.current_operation.take() {
            let duration = start.elapsed();
            self.operations
                .entry(prev_name)
                .or_insert_with(Vec::new)
                .push(duration);
        }

        self.current_operation = Some((name.to_string(), Instant::now()));
    }

    /// Stop the current operation
    pub fn stop_operation(&mut self) {
        if let Some((name, start)) = self.current_operation.take() {
            let duration = start.elapsed();
            self.operations
                .entry(name)
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Get statistics for an operation
    pub fn get_stats(&self, operation: &str) -> Option<OperationStats> {
        self.operations.get(operation).map(|durations| {
            let mut sorted = durations.clone();
            sorted.sort();

            let count = sorted.len();
            let total: Duration = sorted.iter().sum();
            let average = if count > 0 {
                Duration::from_nanos(total.as_nanos() as u64 / count as u64)
            } else {
                Duration::ZERO
            };

            let min = sorted.first().copied().unwrap_or_default();
            let max = sorted.last().copied().unwrap_or_default();
            let median = if count % 2 == 0 {
                let mid = count / 2;
                Duration::from_nanos(
                    ((sorted[mid - 1].as_nanos() + sorted[mid].as_nanos()) / 2) as u64,
                )
            } else {
                sorted[count / 2]
            };

            OperationStats {
                count,
                total,
                average,
                min,
                max,
                median,
            }
        })
    }

    /// Get all operation names
    pub fn operation_names(&self) -> Vec<String> {
        self.operations.keys().cloned().collect()
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.operations.clear();
        self.current_operation = None;
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for an operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub count: usize,
    pub total: Duration,
    pub average: Duration,
    pub min: Duration,
    pub max: Duration,
    pub median: Duration,
}

impl OperationStats {
    /// Get the total time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }

    /// Get the average time in milliseconds
    pub fn average_ms(&self) -> f64 {
        self.average.as_secs_f64() * 1000.0
    }

    /// Get the min time in milliseconds
    pub fn min_ms(&self) -> f64 {
        self.min.as_secs_f64() * 1000.0
    }

    /// Get the max time in milliseconds
    pub fn max_ms(&self) -> f64 {
        self.max.as_secs_f64() * 1000.0
    }

    /// Get the median time in milliseconds
    pub fn median_ms(&self) -> f64 {
        self.median.as_secs_f64() * 1000.0
    }
}

/// Get the current timestamp as a string
pub fn current_timestamp() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.format("%Y-%m-%d %H:%M:%S%.3f UTC").to_string()
}

/// Get the current timestamp as a Unix timestamp
pub fn current_unix_timestamp() -> i64 {
    Utc::now().timestamp()
}

/// Convert a SystemTime to a `DateTime<Utc>`
pub fn system_time_to_datetime(time: SystemTime) -> DateTime<Utc> {
    DateTime::from(time)
}

/// Create a duration from milliseconds
pub fn duration_from_ms(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

/// Create a duration from microseconds
pub fn duration_from_micros(micros: u64) -> Duration {
    Duration::from_micros(micros)
}

/// Create a duration from nanoseconds
pub fn duration_from_nanos(nanos: u64) -> Duration {
    Duration::from_nanos(nanos)
}
