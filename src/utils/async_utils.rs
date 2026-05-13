//! Async utilities for MiniOCR

use futures::future::BoxFuture;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tracing::debug;

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
            .map_err(|_| MiniOcrError::Internal("Failed to acquire semaphore permit".to_string()))
    }
}

/// Async batch processor for processing items in parallel
pub struct AsyncBatchProcessor<T, R> {
    concurrency_limiter: ConcurrencyLimiter,
    processor: Arc<dyn Fn(T) -> BoxFuture<'static, Result<R>> + Send + Sync>,
}

impl<T, R> AsyncBatchProcessor<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    /// Create a new async batch processor
    pub fn new<F>(concurrency_limit: usize, processor: F) -> Self
    where
        F: Fn(T) -> BoxFuture<'static, Result<R>> + Send + Sync + 'static,
    {
        Self {
            concurrency_limiter: ConcurrencyLimiter::new(concurrency_limit),
            processor: Arc::new(processor),
        }
    }

    /// Process a batch of items
    pub async fn process_batch(&self, items: Vec<T>) -> Result<Vec<R>> {
        let mut handles = Vec::new();

        for item in items {
            let processor = Arc::clone(&self.processor);
            let limiter = Arc::clone(&self.concurrency_limiter.semaphore);

            let handle = tokio::spawn(async move {
                let _permit = limiter.acquire().await?;
                processor(item).await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result?),
                Err(e) => return Err(MiniOcrError::Internal(format!("Task failed: {}", e))),
            }
        }

        Ok(results)
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

        let progress = {
            let total = *self.total.lock().await;
            if total > 0 {
                *completed as f32 / total as f32
            } else {
                0.0
            }
        };

        if let Some(callback) = self.callback.lock().await.as_ref() {
            callback(progress);
        }

        debug!("Progress updated: {:.2}%", progress * 100.0);
        Ok(())
    }

    /// Get current progress
    pub async fn progress(&self) -> f32 {
        let completed = *self.completed.lock().await;
        let total = *self.total.lock().await;

        if total > 0 {
            completed as f32 / total as f32
        } else {
            0.0
        }
    }
}

use crate::utils::error::*;
