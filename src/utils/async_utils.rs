//! Async utilities for OCR

use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tracing::debug;

use crate::utils::error::*;

/// Async semaphore for controlling concurrency
pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
}

impl ConcurrencyLimiter {
    /// Create a new concurrency limiter with the given limit
    pub fn new(limit: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(limit)),
        }
    }

    /// Acquire a permit from the semaphore
    pub async fn acquire(&self) -> Result<tokio::sync::SemaphorePermit<'_>> {
        self.semaphore
            .acquire()
            .await
            .map_err(|_| OcrError::Internal("Failed to acquire semaphore permit".to_string()))
    }
}

/// Async cache with TTL support
pub struct AsyncCache<K, V> {
    cache: Arc<Mutex<std::collections::HashMap<K, (V, std::time::Instant)>>>,
    ttl: std::time::Duration,
}

impl<K, V> AsyncCache<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + 'static,
    V: Clone + Send + 'static,
{
    /// Create a new async cache with the given TTL
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            ttl,
        }
    }

    /// Get a value from the cache
    pub async fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().await;
        let now = std::time::Instant::now();

        if let Some((value, timestamp)) = cache.get(key) {
            if now.duration_since(*timestamp) < self.ttl {
                Some(value.clone())
            } else {
                cache.remove(key);
                None
            }
        } else {
            None
        }
    }

    /// Insert a value into the cache
    pub async fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.lock().await;
        cache.insert(key, (value, std::time::Instant::now()));
    }

    /// Clear expired entries from the cache
    pub async fn cleanup(&self) {
        let mut cache = self.cache.lock().await;
        let now = std::time::Instant::now();

        cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.ttl);
    }
}

/// Async progress tracker
pub struct AsyncProgressTracker {
    total: Arc<Mutex<usize>>,
    completed: Arc<Mutex<usize>>,
    callback: Arc<Mutex<Option<Box<dyn Fn(f32) + Send + Sync>>>>,
}

impl AsyncProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        Self {
            total: Arc::new(Mutex::new(total)),
            completed: Arc::new(Mutex::new(0)),
            callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Set a progress callback
    pub async fn set_callback<F>(&self, callback: F)
    where
        F: Fn(f32) + Send + Sync + 'static,
    {
        let mut cb = self.callback.lock().await;
        *cb = Some(Box::new(callback));
    }

    /// Update progress
    pub async fn update(&self, increment: usize) -> Result<()> {
        let mut completed = self.completed.lock().await;
        *completed += increment;

        let total = *self.total.lock().await;
        let progress = if total > 0 {
            *completed as f32 / total as f32
        } else {
            0.0
        };

        if let Some(ref callback) = *self.callback.lock().await {
            callback(progress);
        }

        debug!("Progress: {:.1}% ({}/{})", progress * 100.0, *completed, total);
        Ok(())
    }
}
