//! SIMD utilities for OCR
//!
//! Provides optimized vector operations using SIMD intrinsics when the
//! `simd` feature is enabled, with scalar fallbacks otherwise.

/// SIMD-enabled vector operations for performance-critical code
pub struct SimdOps;

impl SimdOps {
    /// Add two vectors
    pub fn add_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        simd_add_f32x8(a, b)
    }

    /// Multiply two vectors
    pub fn mul_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        simd_mul_f32x8(a, b)
    }

    /// Calculate the sum of a vector
    pub fn sum_f32x8(v: [f32; 8]) -> f32 {
        v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]
    }

    /// Calculate the maximum value in a vector
    pub fn max_f32x8(v: [f32; 8]) -> f32 {
        simd_max_f32x8(v)
    }

    /// Calculate the minimum value in a vector
    pub fn min_f32x8(v: [f32; 8]) -> f32 {
        simd_min_f32x8(v)
    }

    /// Clamp values in a vector
    pub fn clamp_f32x8(v: [f32; 8], min: [f32; 8], max: [f32; 8]) -> [f32; 8] {
        simd_clamp_f32x8(v, min, max)
    }

    /// Calculate the absolute value of a vector
    pub fn abs_f32x8(v: [f32; 8]) -> [f32; 8] {
        simd_abs_f32x8(v)
    }

    /// Calculate the square root of a vector
    pub fn sqrt_f32x8(v: [f32; 8]) -> [f32; 8] {
        simd_sqrt_f32x8(v)
    }

    /// Calculate the dot product of two vectors
    pub fn dot_f32x8(a: [f32; 8], b: [f32; 8]) -> f32 {
        let m = simd_mul_f32x8(a, b);
        m[0] + m[1] + m[2] + m[3] + m[4] + m[5] + m[6] + m[7]
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

    /// Process a slice of f32 values with a SIMD operation
    ///
    /// Processes 8 elements at a time using SIMD, then handles remaining elements with scalar
    pub fn process_slice<F>(input: &[f32], output: &mut [f32], mut f: F)
    where
        F: FnMut(&[f32]) -> [f32; 8],
    {
        let len = input.len().min(output.len());
        let mut i = 0;
        while i + 8 <= len {
            let mut chunk = [0.0; 8];
            chunk.copy_from_slice(&input[i..i + 8]);
            let result = f(&chunk);
            output[i..i + 8].copy_from_slice(&result);
            i += 8;
        }
        if i < len {
            let mut chunk = [0.0; 8];
            let remaining = len - i;
            chunk[..remaining].copy_from_slice(&input[i..len]);
            let result = f(&chunk);
            output[i..len].copy_from_slice(&result[..remaining]);
        }
    }
}

/// SIMD-enabled image processing operations
pub struct SimdImageOps;

impl SimdImageOps {
    /// Adjust contrast of a grayscale image using SIMD
    ///
    /// For each pixel: new = (pixel - 128) * factor + 128, clamped to [0, 255]
    pub fn contrast_adjust(pixels: &[u8], factor: f32) -> Vec<u8> {
        let mut result = vec![0u8; pixels.len()];
        let len = pixels.len();
        let mut i = 0;

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            unsafe {
                use std::arch::aarch64::*;
                let factor_v = vdupq_n_f32(factor);
                let offset = vdupq_n_f32(128.0);
                let zero = vdupq_n_f32(0.0);
                let max_val = vdupq_n_f32(255.0);

                while i + 16 <= len {
                    let chunk = vld1q_u8(pixels.as_ptr().add(i));
                    let low_u16 = vmovl_u8(vget_low_u8(chunk));
                    let high_u16 = vmovl_u8(vget_high_u8(chunk));
                    let low_u32_lo = vmovl_u16(vget_low_u16(low_u16));
                    let low_u32_hi = vmovl_u16(vget_high_u16(low_u16));
                    let high_u32_lo = vmovl_u16(vget_low_u16(high_u16));
                    let high_u32_hi = vmovl_u16(vget_high_u16(high_u16));

                    let f0 = vcvtq_f32_u32(low_u32_lo);
                    let f1 = vcvtq_f32_u32(low_u32_hi);
                    let f2 = vcvtq_f32_u32(high_u32_lo);
                    let f3 = vcvtq_f32_u32(high_u32_hi);

                    let adj0 = vmlaq_f32(offset, vsubq_f32(f0, offset), factor_v);
                    let adj1 = vmlaq_f32(offset, vsubq_f32(f1, offset), factor_v);
                    let adj2 = vmlaq_f32(offset, vsubq_f32(f2, offset), factor_v);
                    let adj3 = vmlaq_f32(offset, vsubq_f32(f3, offset), factor_v);

                    let c0 = vminq_f32(vmaxq_f32(adj0, zero), max_val);
                    let c1 = vminq_f32(vmaxq_f32(adj1, zero), max_val);
                    let c2 = vminq_f32(vmaxq_f32(adj2, zero), max_val);
                    let c3 = vminq_f32(vmaxq_f32(adj3, zero), max_val);

                    let u0 = vcvtq_u32_f32(c0);
                    let u1 = vcvtq_u32_f32(c1);
                    let u2 = vcvtq_u32_f32(c2);
                    let u3 = vcvtq_u32_f32(c3);

                    let n0 = vqmovn_u32(u0);
                    let n1 = vqmovn_u32(u1);
                    let n2 = vqmovn_u32(u2);
                    let n3 = vqmovn_u32(u3);
                    let lo = vqmovn_u16(vcombine_u16(n0, n1));
                    let hi = vqmovn_u16(vcombine_u16(n2, n3));
                    let out = vcombine_u8(lo, hi);

                    vst1q_u8(result.as_mut_ptr().add(i), out);
                    i += 16;
                }
            }
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                unsafe { i = sse_contrast_adjust(pixels, &mut result, factor); }
            }
        }

        // Scalar fallback for remaining pixels
        for j in i..len {
            let p = pixels[j] as f32;
            let adjusted = (p - 128.0) * factor + 128.0;
            result[j] = (adjusted.clamp(0.0, 255.0)) as u8;
        }

        result
    }

    /// Fast binary threshold using SIMD
    ///
    /// Pixels >= threshold become 255, others become 0
    pub fn threshold(pixels: &[u8], threshold: u8) -> Vec<u8> {
        let mut result = vec![0u8; pixels.len()];
        let len = pixels.len();
        let mut i = 0;

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            unsafe {
                use std::arch::aarch64::*;
                let thresh = vdupq_n_u8(threshold);
                let white = vdupq_n_u8(255u8);
                let zero = vdupq_n_u8(0u8);

                while i + 16 <= len {
                    let chunk = vld1q_u8(pixels.as_ptr().add(i));
                    let mask = vcgeq_u8(chunk, thresh);
                    let out = vbslq_u8(mask, white, zero);
                    vst1q_u8(result.as_mut_ptr().add(i), out);
                    i += 16;
                }
            }
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse2") {
                unsafe { i = sse_threshold(pixels, &mut result, threshold); }
            }
        }

        // Scalar fallback
        for j in i..len {
            result[j] = if pixels[j] >= threshold { 255 } else { 0 };
        }

        result
    }

    /// Compute horizontal and vertical projections using SIMD
    ///
    /// Returns (horizontal_projection, vertical_projection) where each element
    /// counts dark pixels (value < 128) in that row/column.
    pub fn compute_projections(data: &[u8], width: usize, height: usize) -> (Vec<u32>, Vec<u32>) {
        let mut h_proj = vec![0u32; height];
        let mut v_proj = vec![0u32; width];

        for y in 0..height {
            let row_start = y * width;
            let mut count = 0u32;
            let mut x = 0;

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    use std::arch::aarch64::*;
                    let threshold = vdupq_n_u8(127u8);
                    while x + 16 <= width {
                        let chunk = vld1q_u8(data.as_ptr().add(row_start + x));
                        let mask = vcltq_u8(chunk, threshold);
                        let bits = vcntq_u8(mask);
                        let sum = vaddlvq_u8(bits);
                        count += sum as u32;
                        x += 16;
                    }
                }
            }

            for xi in x..width {
                if data[row_start + xi] < 128 {
                    count += 1;
                }
            }
            h_proj[y] = count;
        }

        for x in 0..width {
            let mut count = 0u32;
            for y in 0..height {
                if data[y * width + x] < 128 {
                    count += 1;
                }
            }
            v_proj[x] = count;
        }

        (h_proj, v_proj)
    }
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

    /// Fast box blur using summed-area table
    #[cfg(feature = "simd")]
    pub fn box_blur(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        let _size = (2 * radius + 1) as f32;

        // Horizontal pass
        let mut temp = vec![0.0; input.len()];
        scalar_horizontal_blur(input, &mut temp, width, height, radius);

        // Vertical pass
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let y_start = if y >= radius { y - radius } else { 0 };
                let y_end = (y + radius + 1).min(height);
                let count = (y_end - y_start) as f32;
                for yi in y_start..y_end {
                    sum += temp[yi * width + x];
                }
                output[y * width + x] = sum / count;
            }
        }

        output
    }

    /// Fast box blur (scalar fallback)
    #[cfg(not(feature = "simd"))]
    pub fn box_blur(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        let mut temp = vec![0.0; input.len()];
        scalar_horizontal_blur(input, &mut temp, width, height, radius);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let y_start = if y >= radius { y - radius } else { 0 };
                let y_end = (y + radius + 1).min(height);
                let count = (y_end - y_start) as f32;
                for yi in y_start..y_end {
                    sum += temp[yi * width + x];
                }
                output[y * width + x] = sum / count;
            }
        }
        output
    }
}

fn scalar_horizontal_blur(input: &[f32], output: &mut [f32], width: usize, height: usize, radius: usize) {
    for y in 0..height {
        let row_start = y * width;
        for x in 0..width {
            let mut sum = 0.0;
            let x_start = if x >= radius { x - radius } else { 0 };
            let x_end = (x + radius + 1).min(width);
            let count = (x_end - x_start) as f32;
            for xi in x_start..x_end {
                sum += input[row_start + xi];
            }
            output[row_start + x] = sum / count;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_contrast_adjust(pixels: &[u8], result: &mut [u8], factor: f32) -> usize {
    use std::arch::x86_64::*;
    let len = pixels.len();
    let mut i = 0;
    let factor_v = _mm_set1_ps(factor);
    let offset = _mm_set1_ps(128.0);
    let zero = _mm_set1_ps(0.0);
    let max_val = _mm_set1_ps(255.0);

    while i + 16 <= len {
        let chunk = _mm_loadu_si128(pixels.as_ptr().add(i) as *const __m128i);
        let zero_i = _mm_setzero_si128();
        let low16 = _mm_unpacklo_epi8(chunk, zero_i);
        let high16 = _mm_unpackhi_epi8(chunk, zero_i);
        let low32 = _mm_unpacklo_epi16(low16, zero_i);
        let high32 = _mm_unpackhi_epi16(low16, zero_i);
        let low32_2 = _mm_unpacklo_epi16(high16, zero_i);
        let high32_2 = _mm_unpackhi_epi16(high16, zero_i);

        let f1 = _mm_cvtepi32_ps(low32);
        let f2 = _mm_cvtepi32_ps(high32);
        let f3 = _mm_cvtepi32_ps(low32_2);
        let f4 = _mm_cvtepi32_ps(high32_2);

        let adj1 = _mm_add_ps(offset, _mm_mul_ps(_mm_sub_ps(f1, offset), factor_v));
        let adj2 = _mm_add_ps(offset, _mm_mul_ps(_mm_sub_ps(f2, offset), factor_v));
        let adj3 = _mm_add_ps(offset, _mm_mul_ps(_mm_sub_ps(f3, offset), factor_v));
        let adj4 = _mm_add_ps(offset, _mm_mul_ps(_mm_sub_ps(f4, offset), factor_v));

        let clamp1 = _mm_min_ps(_mm_max_ps(adj1, zero), max_val);
        let clamp2 = _mm_min_ps(_mm_max_ps(adj2, zero), max_val);
        let clamp3 = _mm_min_ps(_mm_max_ps(adj3, zero), max_val);
        let clamp4 = _mm_min_ps(_mm_max_ps(adj4, zero), max_val);

        let i1 = _mm_cvtps_epi32(clamp1);
        let i2 = _mm_cvtps_epi32(clamp2);
        let i3 = _mm_cvtps_epi32(clamp3);
        let i4 = _mm_cvtps_epi32(clamp4);

        let p1 = _mm_packs_epi32(i1, i2);
        let p2 = _mm_packs_epi32(i3, i4);
        let out = _mm_packus_epi16(p1, p2);

        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, out);
        i += 16;
    }
    i
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sse_threshold(pixels: &[u8], result: &mut [u8], threshold: u8) -> usize {
    use std::arch::x86_64::*;
    let len = pixels.len();
    let mut i = 0;
    let thresh = _mm_set1_epi8(threshold as i8);
    let white = _mm_set1_epi8(-1i8);
    let zero = _mm_setzero_si128();

    while i + 16 <= len {
        let chunk = _mm_loadu_si128(pixels.as_ptr().add(i) as *const __m128i);
        let mask = _mm_cmpgt_epi8(chunk, thresh);
        let out = _mm_or_si128(_mm_and_si128(mask, white), _mm_andnot_si128(mask, zero));
        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, out);
        i += 16;
    }
    i
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_add_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    use std::arch::x86_64::_mm_add_ps;
    use std::arch::x86_64::_mm_loadu_ps;
    use std::arch::x86_64::_mm_storeu_ps;
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let vr = _mm_add_ps(va, vb);
    let mut r = [0.0; 4];
    _mm_storeu_ps(r.as_mut_ptr(), vr);
    r
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_mul_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    use std::arch::x86_64::_mm_loadu_ps;
    use std::arch::x86_64::_mm_mul_ps;
    use std::arch::x86_64::_mm_storeu_ps;
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let vr = _mm_mul_ps(va, vb);
    let mut r = [0.0; 4];
    _mm_storeu_ps(r.as_mut_ptr(), vr);
    r
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_max_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    use std::arch::x86_64::_mm_loadu_ps;
    use std::arch::x86_64::_mm_max_ps;
    use std::arch::x86_64::_mm_storeu_ps;
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let vr = _mm_max_ps(va, vb);
    let mut r = [0.0; 4];
    _mm_storeu_ps(r.as_mut_ptr(), vr);
    r
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_min_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    use std::arch::x86_64::_mm_loadu_ps;
    use std::arch::x86_64::_mm_min_ps;
    use std::arch::x86_64::_mm_storeu_ps;
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let vr = _mm_min_ps(va, vb);
    let mut r = [0.0; 4];
    _mm_storeu_ps(r.as_mut_ptr(), vr);
    r
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_add_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    use std::arch::x86_64::_mm256_add_ps;
    use std::arch::x86_64::_mm256_loadu_ps;
    use std::arch::x86_64::_mm256_storeu_ps;
    let va = _mm256_loadu_ps(a.as_ptr());
    let vb = _mm256_loadu_ps(b.as_ptr());
    let vr = _mm256_add_ps(va, vb);
    let mut r = [0.0; 8];
    _mm256_storeu_ps(r.as_mut_ptr(), vr);
    r
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_mul_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    use std::arch::x86_64::_mm256_loadu_ps;
    use std::arch::x86_64::_mm256_mul_ps;
    use std::arch::x86_64::_mm256_storeu_ps;
    let va = _mm256_loadu_ps(a.as_ptr());
    let vb = _mm256_loadu_ps(b.as_ptr());
    let vr = _mm256_mul_ps(va, vb);
    let mut r = [0.0; 8];
    _mm256_storeu_ps(r.as_mut_ptr(), vr);
    r
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_max_f32x8(v: [f32; 8]) -> f32 {
    use std::arch::x86_64::_mm256_loadu_ps;
    use std::arch::x86_64::_mm256_max_ps;
    use std::arch::x86_64::_mm256_permutevar8x32_ps;
    use std::arch::x86_64::_mm256_castps256_ps128;
    use std::arch::x86_64::_mm_max_ps;
    use std::arch::x86_64::_mm_storeu_ps;
    use std::arch::x86_64::_mm256_extractf128_ps;
    let vv = _mm256_loadu_ps(v.as_ptr());
    let perm = _mm256_permutevar8x32_ps(vv, _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
    let max_v = _mm256_max_ps(vv, perm);
    let low = _mm256_castps256_ps128(max_v);
    let high = _mm256_extractf128_ps(max_v, 1);
    let max128 = _mm_max_ps(low, high);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), max128);
    r[0].max(r[1]).max(r[2]).max(r[3])
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_min_f32x8(v: [f32; 8]) -> f32 {
    use std::arch::x86_64::_mm256_loadu_ps;
    use std::arch::x86_64::_mm256_min_ps;
    use std::arch::x86_64::_mm256_castps256_ps128;
    use std::arch::x86_64::_mm_min_ps;
    use std::arch::x86_64::_mm_storeu_ps;
    use std::arch::x86_64::_mm256_extractf128_ps;
    let vv = _mm256_loadu_ps(v.as_ptr());
    let low = _mm256_castps256_ps128(vv);
    let high = _mm256_extractf128_ps(vv, 1);
    let min128 = _mm_min_ps(low, high);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), min128);
    r[0].min(r[1]).min(r[2]).min(r[3])
}

fn simd_add_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return avx2_add_f32x8(a, b); }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vaddq_f32;
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vst1q_f32;
            let va = vld1q_f32(a[..4].as_ptr());
            let vb = vld1q_f32(b[..4].as_ptr());
            let vr_low = vaddq_f32(va, vb);
            let va_high = vld1q_f32(a[4..].as_ptr());
            let vb_high = vld1q_f32(b[4..].as_ptr());
            let vr_high = vaddq_f32(va_high, vb_high);
            let mut r = [0.0; 8];
            vst1q_f32(r[..4].as_mut_ptr(), vr_low);
            vst1q_f32(r[4..].as_mut_ptr(), vr_high);
            return r;
        }
    }
    // Scalar fallback
    [
        a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3],
        a[4] + b[4], a[5] + b[5], a[6] + b[6], a[7] + b[7],
    ]
}

fn simd_mul_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return avx2_mul_f32x8(a, b); }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::vld1q_f32;
            use std::arch::aarch64::vmulq_f32;
            use std::arch::aarch64::vst1q_f32;
            let va = vld1q_f32(a[..4].as_ptr());
            let vb = vld1q_f32(b[..4].as_ptr());
            let vr_low = vmulq_f32(va, vb);
            let va_high = vld1q_f32(a[4..].as_ptr());
            let vb_high = vld1q_f32(b[4..].as_ptr());
            let vr_high = vmulq_f32(va_high, vb_high);
            let mut r = [0.0; 8];
            vst1q_f32(r[..4].as_mut_ptr(), vr_low);
            vst1q_f32(r[4..].as_mut_ptr(), vr_high);
            return r;
        }
    }
    [
        a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3],
        a[4] * b[4], a[5] * b[5], a[6] * b[6], a[7] * b[7],
    ]
}

fn simd_max_f32x8(v: [f32; 8]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return avx2_max_f32x8(v); }
        }
    }
    v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
}

fn simd_min_f32x8(v: [f32; 8]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return avx2_min_f32x8(v); }
        }
    }
    v.iter().fold(f32::INFINITY, |a, &b| a.min(b))
}

fn simd_clamp_f32x8(v: [f32; 8], min: [f32; 8], max: [f32; 8]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = v[i].max(min[i]).min(max[i]);
    }
    result
}

fn simd_abs_f32x8(v: [f32; 8]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = v[i].abs();
    }
    result
}

fn simd_sqrt_f32x8(v: [f32; 8]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = v[i].sqrt();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32x8() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let r = SimdOps::add_f32x8(a, b);
        assert_eq!(r, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn test_mul_f32x8() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0; 8];
        let r = SimdOps::mul_f32x8(a, b);
        assert_eq!(r, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_f32x8() {
        let v = [1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0];
        let m = SimdOps::max_f32x8(v);
        assert!((m - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_f32x8() {
        let v = [1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0];
        let m = SimdOps::min_f32x8(v);
        assert!((m - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_f32x8() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let s = SimdOps::sum_f32x8(v);
        assert!((s - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_slice() {
        let slice = &[1.0, 2.0, 3.0];
        let r = SimdOps::from_slice_f32x8(slice);
        assert_eq!(r[..3], [1.0, 2.0, 3.0]);
        assert_eq!(r[3..], [0.0; 5]);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0; 8];
        let d = SimdOps::dot_f32x8(a, b);
        assert!((d - 36.0).abs() < 1e-6);
    }
}
