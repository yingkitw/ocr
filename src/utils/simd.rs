//! SIMD utilities for OCR
//!
//! This module provides optimized vector operations for performance-critical code.
//! Currently implements scalar fallbacks, but can be extended with actual SIMD
//! instructions when needed.

/// SIMD-enabled vector operations for performance-critical code
pub struct SimdOps;

impl SimdOps {
    /// Add two vectors (scalar fallback)
    pub fn add_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = a[i] + b[i];
        }
        result
    }

    /// Multiply two vectors (scalar fallback)
    pub fn mul_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = a[i] * b[i];
        }
        result
    }

    /// Calculate the sum of a vector (scalar fallback)
    pub fn sum_f32x8(v: [f32; 8]) -> f32 {
        v.iter().sum()
    }

    /// Calculate the maximum value in a vector (scalar fallback)
    pub fn max_f32x8(v: [f32; 8]) -> f32 {
        v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Calculate the minimum value in a vector (scalar fallback)
    pub fn min_f32x8(v: [f32; 8]) -> f32 {
        v.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    /// Clamp values in a vector (scalar fallback)
    pub fn clamp_f32x8(v: [f32; 8], min: [f32; 8], max: [f32; 8]) -> [f32; 8] {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = v[i].max(min[i]).min(max[i]);
        }
        result
    }

    /// Calculate the absolute value of a vector (scalar fallback)
    pub fn abs_f32x8(v: [f32; 8]) -> [f32; 8] {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = v[i].abs();
        }
        result
    }

    /// Calculate the square root of a vector (scalar fallback)
    pub fn sqrt_f32x8(v: [f32; 8]) -> [f32; 8] {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = v[i].sqrt();
        }
        result
    }

    /// Calculate the dot product of two vectors (scalar fallback)
    pub fn dot_f32x8(a: [f32; 8], b: [f32; 8]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Convert a slice to f32x8, padding with zeros if necessary
    pub fn from_slice_f32x8(slice: &[f32]) -> [f32; 8] {
        let mut data = [0.0; 8];
        let len = slice.len().min(8);
        data[..len].copy_from_slice(&slice[..len]);
        data
    }

    /// Convert f32x8 to array (no-op for scalar implementation)
    pub fn to_array_f32x8(v: [f32; 8]) -> [f32; 8] {
        v
    }
}

/// SIMD-enabled image processing operations
pub struct SimdImageOps;

impl SimdImageOps {
    /// Apply a kernel to an image using optimized operations
    pub fn apply_kernel_simd(
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        kernel: &[f32],
        kernel_size: usize,
    ) {
        let half_kernel = kernel_size / 2;

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut sum = 0.0;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let pixel_y = y + ky - half_kernel;
                        let pixel_x = x + kx - half_kernel;
                        let pixel_idx = pixel_y * width + pixel_x;
                        let kernel_idx = ky * kernel_size + kx;

                        sum += input[pixel_idx] * kernel[kernel_idx];
                    }
                }

                let output_idx = y * width + x;
                output[output_idx] = sum;
            }
        }
    }

    /// Apply a 3x3 kernel using optimized operations
    pub fn apply_3x3_kernel_simd(
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        kernel: &[f32; 9],
    ) {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0.0;

                for ky in 0..3 {
                    let pixel_y = y + ky - 1;
                    let row_start = pixel_y * width;

                    for kx in 0..3 {
                        let pixel_x = x + kx - 1;
                        let pixel_idx = row_start + pixel_x;
                        let kernel_idx = ky * 3 + kx;

                        sum += input[pixel_idx] * kernel[kernel_idx];
                    }
                }

                let output_idx = y * width + x;
                output[output_idx] = sum;
            }
        }
    }
}
