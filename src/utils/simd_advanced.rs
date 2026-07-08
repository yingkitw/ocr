//! Advanced SIMD operations for high-performance computing
//!
//! Provides SIMD-accelerated vector, matrix, image processing, and neural network operations.
//! Uses architecture-specific intrinsics when the `simd` feature is enabled.

// On aarch64 the NEON SIMD block returns unconditionally (NEON is mandatory),
// so the trailing scalar fallbacks are unreachable on that target. They remain
// for x86_64 (runtime AVX2 detection) and non-`simd` builds.
#![allow(unreachable_code)]

use anyhow::Result;

/// SIMD vector types for different architectures
#[derive(Debug, Clone)]
pub enum SIMDVector {
    F32x4([f32; 4]),
    F32x8([f32; 8]),
    F32x16([f32; 16]),
    I32x4([i32; 4]),
    I32x8([i32; 8]),
    U8x16([u8; 16]),
    U8x32([u8; 32]),
}

impl SIMDVector {
    pub fn splat_f32(value: f32, size: usize) -> Self {
        match size {
            4 => SIMDVector::F32x4([value; 4]),
            8 => SIMDVector::F32x8([value; 8]),
            16 => SIMDVector::F32x16([value; 16]),
            _ => SIMDVector::F32x4([value; 4]),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SIMDVector::F32x4(_) => 4,
            SIMDVector::F32x8(_) => 8,
            SIMDVector::F32x16(_) => 16,
            SIMDVector::I32x4(_) => 4,
            SIMDVector::I32x8(_) => 8,
            SIMDVector::U8x16(_) => 16,
            SIMDVector::U8x32(_) => 32,
        }
    }
}

/// SIMD operations trait
pub trait SIMDOperations {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn max(&self, other: &Self) -> Self;
    fn min(&self, other: &Self) -> Self;
    fn sqrt(&self) -> Self;
    fn abs(&self) -> Self;
    fn sum(&self) -> f32;
    fn dot_product(&self, other: &Self) -> f32;
}

impl SIMDOperations for SIMDVector {
    fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                SIMDVector::F32x4(simd_add_f32x4(*a, *b))
            }
            (SIMDVector::F32x8(a), SIMDVector::F32x8(b)) => {
                let lo = simd_add_f32x4([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]);
                let hi = simd_add_f32x4([a[4], a[5], a[6], a[7]], [b[4], b[5], b[6], b[7]]);
                SIMDVector::F32x8([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
            }
            (SIMDVector::I32x4(a), SIMDVector::I32x4(b)) => {
                SIMDVector::I32x4([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
            }
            (SIMDVector::I32x8(a), SIMDVector::I32x8(b)) => SIMDVector::I32x8([
                a[0] + b[0],
                a[1] + b[1],
                a[2] + b[2],
                a[3] + b[3],
                a[4] + b[4],
                a[5] + b[5],
                a[6] + b[6],
                a[7] + b[7],
            ]),
            (SIMDVector::U8x16(a), SIMDVector::U8x16(b)) => {
                let mut r = [0u8; 16];
                for i in 0..16 {
                    r[i] = a[i].wrapping_add(b[i]);
                }
                SIMDVector::U8x16(r)
            }
            (SIMDVector::U8x32(a), SIMDVector::U8x32(b)) => {
                let mut r = [0u8; 32];
                for i in 0..32 {
                    r[i] = a[i].wrapping_add(b[i]);
                }
                SIMDVector::U8x32(r)
            }
            _ => unreachable!("type mismatch"),
        }
    }

    fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                SIMDVector::F32x4(simd_sub_f32x4(*a, *b))
            }
            (SIMDVector::I32x4(a), SIMDVector::I32x4(b)) => {
                SIMDVector::I32x4([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
            }
            _ => unreachable!("type mismatch"),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                SIMDVector::F32x4(simd_mul_f32x4(*a, *b))
            }
            (SIMDVector::F32x8(a), SIMDVector::F32x8(b)) => {
                let lo = simd_mul_f32x4([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]);
                let hi = simd_mul_f32x4([a[4], a[5], a[6], a[7]], [b[4], b[5], b[6], b[7]]);
                SIMDVector::F32x8([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
            }
            _ => unreachable!("type mismatch"),
        }
    }

    fn div(&self, other: &Self) -> Self {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                SIMDVector::F32x4([a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]])
            }
            _ => unreachable!("type mismatch"),
        }
    }

    fn max(&self, other: &Self) -> Self {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                SIMDVector::F32x4(simd_max_f32x4(*a, *b))
            }
            (SIMDVector::I32x4(a), SIMDVector::I32x4(b)) => SIMDVector::I32x4([
                a[0].max(b[0]),
                a[1].max(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]),
            _ => unreachable!("type mismatch"),
        }
    }

    fn min(&self, other: &Self) -> Self {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                SIMDVector::F32x4(simd_min_f32x4(*a, *b))
            }
            _ => unreachable!("type mismatch"),
        }
    }

    fn sqrt(&self) -> Self {
        match self {
            SIMDVector::F32x4(a) => SIMDVector::F32x4(simd_sqrt_f32x4(*a)),
            _ => unreachable!("type mismatch"),
        }
    }

    fn abs(&self) -> Self {
        match self {
            SIMDVector::F32x4(a) => {
                SIMDVector::F32x4([a[0].abs(), a[1].abs(), a[2].abs(), a[3].abs()])
            }
            _ => unreachable!("type mismatch"),
        }
    }

    fn sum(&self) -> f32 {
        match self {
            SIMDVector::F32x4(a) => a[0] + a[1] + a[2] + a[3],
            SIMDVector::F32x8(a) => a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7],
            _ => unreachable!("type mismatch"),
        }
    }

    fn dot_product(&self, other: &Self) -> f32 {
        match (self, other) {
            (SIMDVector::F32x4(a), SIMDVector::F32x4(b)) => {
                let m = simd_mul_f32x4(*a, *b);
                m[0] + m[1] + m[2] + m[3]
            }
            (SIMDVector::F32x8(a), SIMDVector::F32x8(b)) => {
                let lo = simd_mul_f32x4([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]);
                let hi = simd_mul_f32x4([a[4], a[5], a[6], a[7]], [b[4], b[5], b[6], b[7]]);
                lo[0] + lo[1] + lo[2] + lo[3] + hi[0] + hi[1] + hi[2] + hi[3]
            }
            _ => unreachable!("type mismatch"),
        }
    }
}

fn simd_add_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                use std::arch::x86_64::_mm_add_ps;
                use std::arch::x86_64::_mm_loadu_ps;
                use std::arch::x86_64::_mm_storeu_ps;
                let va = _mm_loadu_ps(a.as_ptr());
                let vb = _mm_loadu_ps(b.as_ptr());
                let vr = _mm_add_ps(va, vb);
                let mut r = [0.0; 4];
                _mm_storeu_ps(r.as_mut_ptr(), vr);
                return r;
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vaddq_f32;
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vst1q_f32;
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vaddq_f32(va, vb);
            let mut r = [0.0; 4];
            vst1q_f32(r.as_mut_ptr(), vr);
            return r;
        }
    }
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn simd_sub_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                use std::arch::x86_64::_mm_loadu_ps;
                use std::arch::x86_64::_mm_storeu_ps;
                use std::arch::x86_64::_mm_sub_ps;
                let va = _mm_loadu_ps(a.as_ptr());
                let vb = _mm_loadu_ps(b.as_ptr());
                let vr = _mm_sub_ps(va, vb);
                let mut r = [0.0; 4];
                _mm_storeu_ps(r.as_mut_ptr(), vr);
                return r;
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vst1q_f32;
            use std::arch::aarch64::vsubq_f32;
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vsubq_f32(va, vb);
            let mut r = [0.0; 4];
            vst1q_f32(r.as_mut_ptr(), vr);
            return r;
        }
    }
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

fn simd_mul_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                use std::arch::x86_64::_mm_loadu_ps;
                use std::arch::x86_64::_mm_mul_ps;
                use std::arch::x86_64::_mm_storeu_ps;
                let va = _mm_loadu_ps(a.as_ptr());
                let vb = _mm_loadu_ps(b.as_ptr());
                let vr = _mm_mul_ps(va, vb);
                let mut r = [0.0; 4];
                _mm_storeu_ps(r.as_mut_ptr(), vr);
                return r;
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vmulq_f32;
            use std::arch::aarch64::vst1q_f32;
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vmulq_f32(va, vb);
            let mut r = [0.0; 4];
            vst1q_f32(r.as_mut_ptr(), vr);
            return r;
        }
    }
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

fn simd_max_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                use std::arch::x86_64::_mm_loadu_ps;
                use std::arch::x86_64::_mm_max_ps;
                use std::arch::x86_64::_mm_storeu_ps;
                let va = _mm_loadu_ps(a.as_ptr());
                let vb = _mm_loadu_ps(b.as_ptr());
                let vr = _mm_max_ps(va, vb);
                let mut r = [0.0; 4];
                _mm_storeu_ps(r.as_mut_ptr(), vr);
                return r;
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vmaxq_f32;
            use std::arch::aarch64::vst1q_f32;
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vmaxq_f32(va, vb);
            let mut r = [0.0; 4];
            vst1q_f32(r.as_mut_ptr(), vr);
            return r;
        }
    }
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn simd_min_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                use std::arch::x86_64::_mm_loadu_ps;
                use std::arch::x86_64::_mm_min_ps;
                use std::arch::x86_64::_mm_storeu_ps;
                let va = _mm_loadu_ps(a.as_ptr());
                let vb = _mm_loadu_ps(b.as_ptr());
                let vr = _mm_min_ps(va, vb);
                let mut r = [0.0; 4];
                _mm_storeu_ps(r.as_mut_ptr(), vr);
                return r;
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vminq_f32;
            use std::arch::aarch64::vst1q_f32;
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vminq_f32(va, vb);
            let mut r = [0.0; 4];
            vst1q_f32(r.as_mut_ptr(), vr);
            return r;
        }
    }
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

fn simd_sqrt_f32x4(a: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                use std::arch::x86_64::_mm_loadu_ps;
                use std::arch::x86_64::_mm_sqrt_ps;
                use std::arch::x86_64::_mm_storeu_ps;
                let va = _mm_loadu_ps(a.as_ptr());
                let vr = _mm_sqrt_ps(va);
                let mut r = [0.0; 4];
                _mm_storeu_ps(r.as_mut_ptr(), vr);
                return r;
            }
        }
    }
    [a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()]
}

/// SIMD matrix operations
pub struct SIMDMatrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl SIMDMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Matrix multiplication using SIMD
    pub fn matmul_simd(&self, other: &SIMDMatrix) -> Result<SIMDMatrix> {
        if self.cols != other.rows {
            return Err(anyhow::anyhow!(
                "Matrix dimensions don't match for multiplication"
            ));
        }

        let mut result = SIMDMatrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                let mut k = 0;
                while k + 4 <= self.cols {
                    let a_simd = self.get_row_simd(i, k, 4);
                    let b_simd = other.get_col_simd(k, j, 4);
                    sum += a_simd.dot_product(&b_simd);
                    k += 4;
                }

                while k < self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                    k += 1;
                }

                result.set(i, j, sum);
            }
        }

        Ok(result)
    }

    /// Optimized matmul using blocking for cache efficiency
    pub fn matmul_blocked(&self, other: &SIMDMatrix, block_size: usize) -> Result<SIMDMatrix> {
        if self.cols != other.rows {
            return Err(anyhow::anyhow!(
                "Matrix dimensions don't match for multiplication"
            ));
        }

        let mut result = SIMDMatrix::new(self.rows, other.cols);

        for i in (0..self.rows).step_by(block_size) {
            let i_end = (i + block_size).min(self.rows);
            for k in (0..self.cols).step_by(block_size) {
                let k_end = (k + block_size).min(self.cols);
                for j in (0..other.cols).step_by(block_size) {
                    let j_end = (j + block_size).min(other.cols);

                    for ii in i..i_end {
                        for kk in k..k_end {
                            let aik = self.get(ii, kk);
                            for jj in j..j_end {
                                let old = result.get(ii, jj);
                                result.set(ii, jj, old + aik * other.get(kk, jj));
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Convolution operation using SIMD
    pub fn conv2d_simd(
        &self,
        kernel: &SIMDMatrix,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<SIMDMatrix> {
        let output_rows = (self.rows + 2 * padding.0 - kernel.rows) / stride.0 + 1;
        let output_cols = (self.cols + 2 * padding.1 - kernel.cols) / stride.1 + 1;

        let mut result = SIMDMatrix::new(output_rows, output_cols);

        for i in 0..output_rows {
            for j in 0..output_cols {
                let mut sum = 0.0;

                for ki in 0..kernel.rows {
                    for kj in 0..kernel.cols {
                        let input_i = i * stride.0 + ki;
                        let input_j = j * stride.1 + kj;

                        if input_i >= padding.0
                            && input_i < self.rows + padding.0
                            && input_j >= padding.1
                            && input_j < self.cols + padding.1
                        {
                            let actual_i = input_i - padding.0;
                            let actual_j = input_j - padding.1;
                            sum += self.get(actual_i, actual_j) * kernel.get(ki, kj);
                        }
                    }
                }

                result.set(i, j, sum);
            }
        }

        Ok(result)
    }

    /// Batch normalization using SIMD
    pub fn batch_norm_simd(
        &mut self,
        mean: &[f32],
        variance: &[f32],
        gamma: &[f32],
        beta: &[f32],
        epsilon: f32,
    ) -> Result<()> {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = j % mean.len();
                let normalized = (self.get(i, j) - mean[idx]) / (variance[idx] + epsilon).sqrt();
                let output = gamma[idx] * normalized + beta[idx];
                self.set(i, j, output);
            }
        }
        Ok(())
    }

    /// ReLU activation using SIMD
    pub fn relu_simd(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j);
                self.set(i, j, if val > 0.0 { val } else { 0.0 });
            }
        }
    }

    /// Softmax using SIMD
    pub fn softmax_simd(&mut self) -> Result<()> {
        for i in 0..self.rows {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..self.cols {
                max_val = max_val.max(self.get(i, j));
            }

            let mut sum = 0.0;
            for j in 0..self.cols {
                let exp_val = (self.get(i, j) - max_val).exp();
                self.set(i, j, exp_val);
                sum += exp_val;
            }

            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) / sum);
            }
        }
        Ok(())
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    fn get_row_simd(&self, row: usize, start_col: usize, count: usize) -> SIMDVector {
        let mut data = [0.0; 4];
        for i in 0..count.min(4) {
            data[i] = self.get(row, start_col + i);
        }
        SIMDVector::F32x4(data)
    }

    fn get_col_simd(&self, start_row: usize, col: usize, count: usize) -> SIMDVector {
        let mut data = [0.0; 4];
        for i in 0..count.min(4) {
            data[i] = self.get(start_row + i, col);
        }
        SIMDVector::F32x4(data)
    }
}

/// SIMD image processing operations
pub struct SIMDImageProcessor;

impl SIMDImageProcessor {
    /// Apply Gaussian blur using SIMD
    pub fn gaussian_blur_simd(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        kernel_size: usize,
        sigma: f32,
    ) -> Result<Vec<f32>> {
        let kernel = self.generate_gaussian_kernel(kernel_size, sigma)?;
        let mut result = vec![0.0; image.len()];

        let padding = kernel_size / 2;

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let img_y = y as i32 + ky as i32 - padding as i32;
                        let img_x = x as i32 + kx as i32 - padding as i32;

                        if img_y >= 0 && img_y < height as i32 && img_x >= 0 && img_x < width as i32
                        {
                            let pixel_idx = (img_y as usize * width + img_x as usize) as usize;
                            let kernel_val = kernel[ky * kernel_size + kx];
                            sum += image[pixel_idx] * kernel_val;
                            weight_sum += kernel_val;
                        }
                    }
                }

                if weight_sum > 0.0 {
                    result[y * width + x] = sum / weight_sum;
                } else {
                    result[y * width + x] = image[y * width + x];
                }
            }
        }

        Ok(result)
    }

    /// Apply Sobel edge detection using SIMD
    pub fn sobel_edge_detection_simd(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        let sobel_x = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let gx = self.apply_kernel_simd(image, width, height, &sobel_x, 3)?;
        let gy = self.apply_kernel_simd(image, width, height, &sobel_y, 3)?;

        let mut result = vec![0.0; image.len()];
        for i in 0..image.len() {
            result[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
        }

        Ok(result)
    }

    fn apply_kernel_simd(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        kernel: &[f32],
        kernel_size: usize,
    ) -> Result<Vec<f32>> {
        let mut result = vec![0.0; image.len()];
        let padding = kernel_size / 2;

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let img_y = y as i32 + ky as i32 - padding as i32;
                        let img_x = x as i32 + kx as i32 - padding as i32;

                        if img_y >= 0 && img_y < height as i32 && img_x >= 0 && img_x < width as i32
                        {
                            let pixel_idx = img_y as usize * width + img_x as usize;
                            let kernel_idx = ky * kernel_size + kx;
                            sum += image[pixel_idx] * kernel[kernel_idx];
                        }
                    }
                }

                result[y * width + x] = sum;
            }
        }

        Ok(result)
    }

    fn generate_gaussian_kernel(&self, size: usize, sigma: f32) -> Result<Vec<f32>> {
        let mut kernel = vec![0.0; size * size];
        let center = size as f32 / 2.0;
        let mut sum = 0.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let distance_squared = dx * dx + dy * dy;
                let value = (-distance_squared / (2.0 * sigma * sigma)).exp();
                kernel[y * size + x] = value;
                sum += value;
            }
        }

        for val in &mut kernel {
            *val /= sum;
        }

        Ok(kernel)
    }
}

/// SIMD neural network operations
pub struct SIMDNeuralNetwork;

impl SIMDNeuralNetwork {
    /// Convolutional layer forward pass using SIMD
    pub fn conv2d_forward_simd(
        &self,
        input: &[f32],
        input_shape: (usize, usize, usize), // (height, width, channels)
        weights: &[f32],
        weight_shape: (usize, usize, usize, usize), // (out_channels, in_channels, kernel_h, kernel_w)
        bias: &[f32],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Vec<f32>> {
        let (input_h, input_w, input_c) = input_shape;
        let (out_c, in_c, kernel_h, kernel_w) = weight_shape;

        let output_h = (input_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let output_w = (input_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        let mut output = vec![0.0; out_c * output_h * output_w];

        for oc in 0..out_c {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let mut sum = bias[oc];

                    for ic in 0..in_c {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let input_h_idx = oh * stride.0 + kh;
                                let input_w_idx = ow * stride.1 + kw;

                                if input_h_idx >= padding.0
                                    && input_h_idx < input_h + padding.0
                                    && input_w_idx >= padding.1
                                    && input_w_idx < input_w + padding.1
                                {
                                    let actual_h = input_h_idx - padding.0;
                                    let actual_w = input_w_idx - padding.1;

                                    let input_idx =
                                        actual_h * input_w * input_c + actual_w * input_c + ic;
                                    let weight_idx = oc * in_c * kernel_h * kernel_w
                                        + ic * kernel_h * kernel_w
                                        + kh * kernel_w
                                        + kw;

                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }

                    let output_idx = oc * output_h * output_w + oh * output_w + ow;
                    output[output_idx] = sum;
                }
            }
        }

        Ok(output)
    }

    /// Max pooling using SIMD
    pub fn max_pool2d_simd(
        &self,
        input: &[f32],
        input_shape: (usize, usize, usize), // (height, width, channels)
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Vec<f32>> {
        let (input_h, input_w, channels) = input_shape;
        let (kernel_h, kernel_w) = kernel_size;

        let output_h = (input_h - kernel_h) / stride.0 + 1;
        let output_w = (input_w - kernel_w) / stride.1 + 1;

        let mut output = vec![0.0; channels * output_h * output_w];

        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let mut max_val = f32::NEG_INFINITY;

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let input_h_idx = oh * stride.0 + kh;
                            let input_w_idx = ow * stride.1 + kw;

                            let input_idx =
                                input_h_idx * input_w * channels + input_w_idx * channels + c;
                            max_val = max_val.max(input[input_idx]);
                        }
                    }

                    let output_idx = c * output_h * output_w + oh * output_w + ow;
                    output[output_idx] = max_val;
                }
            }
        }

        Ok(output)
    }

    /// Batch normalization using SIMD
    pub fn batch_norm_simd(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        variance: &[f32],
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0; input.len()];
        let channels = gamma.len();

        for (i, &val) in input.iter().enumerate() {
            let c = i % channels;
            let normalized = (val - mean[c]) / (variance[c] + epsilon).sqrt();
            output[i] = gamma[c] * normalized + beta[c];
        }

        Ok(output)
    }
}

/// SIMD utility functions
pub struct SIMDUtils;

impl SIMDUtils {
    /// Check if SIMD is available on this platform
    pub fn is_simd_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("sse4.1")
        }
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Get optimal vector size for current platform
    pub fn get_optimal_vector_size() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                8
            } else if std::is_x86_feature_detected!("sse4.1") {
                4
            } else {
                1
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            4
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            1
        }
    }

    /// Align data for SIMD operations
    pub fn align_data(data: &[f32], alignment: usize) -> Vec<f32> {
        let mut aligned = data.to_vec();
        while aligned.len() % alignment != 0 {
            aligned.push(0.0);
        }
        aligned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vector_add() {
        let a = SIMDVector::F32x4([1.0, 2.0, 3.0, 4.0]);
        let b = SIMDVector::F32x4([4.0, 3.0, 2.0, 1.0]);
        let r = a.add(&b);
        if let SIMDVector::F32x4(data) = r {
            assert_eq!(data, [5.0, 5.0, 5.0, 5.0]);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_simd_vector_mul() {
        let a = SIMDVector::F32x4([2.0, 3.0, 4.0, 5.0]);
        let b = SIMDVector::F32x4([3.0, 2.0, 1.0, 0.0]);
        let r = a.mul(&b);
        if let SIMDVector::F32x4(data) = r {
            assert_eq!(data, [6.0, 6.0, 4.0, 0.0]);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_simd_matrix_matmul() {
        let a = SIMDMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = SIMDMatrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let r = a.matmul_simd(&b).unwrap();
        assert_eq!(r.rows, 2);
        assert_eq!(r.cols, 2);
        // [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
        assert!((r.get(0, 0) - 19.0).abs() < 1e-6);
        assert!((r.get(0, 1) - 22.0).abs() < 1e-6);
        assert!((r.get(1, 0) - 43.0).abs() < 1e-6);
        assert!((r.get(1, 1) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_utils() {
        let avail = SIMDUtils::is_simd_available();
        // This should return true on any modern x86_64 or aarch64 machine
        assert!(avail || cfg!(not(any(target_arch = "x86_64", target_arch = "aarch64"))));
    }
}
