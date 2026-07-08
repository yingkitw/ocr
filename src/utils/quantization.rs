//! INT8 quantization for edge deployment
//!
//! Provides symmetric per-tensor quantization:
//!   quantized = round(clamp(value / scale, -127, 127))
//!   dequantized = quantized * scale
//!
//! Benefits: 4x memory reduction, faster cache-friendly inference on CPU.

use ndarray::{Array1, Array2};

/// Quantized tensor with symmetric INT8 weights
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub weights: Vec<i8>,
    pub scale: f32,
    pub shape: (usize, usize), // rows, cols
}

impl QuantizedTensor {
    /// Create from a flattened INT8 buffer with known shape
    pub fn from_vec(weights: Vec<i8>, scale: f32, rows: usize, cols: usize) -> Self {
        Self {
            weights,
            scale,
            shape: (rows, cols),
        }
    }

    /// Convert back to f32 Array2
    pub fn to_array2(&self) -> Array2<f32> {
        let (rows, cols) = self.shape;
        let mut arr = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                arr[[i, j]] = self.weights[i * cols + j] as f32 * self.scale;
            }
        }
        arr
    }

    /// Number of parameters
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weights.len() * std::mem::size_of::<i8>()
    }
}

/// Quantize an f32 Array2 to symmetric INT8
pub fn quantize_array2(arr: &Array2<f32>) -> QuantizedTensor {
    let (rows, cols) = (arr.nrows(), arr.ncols());

    // Find max absolute value for scale
    let max_abs = arr.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    if max_abs < 1e-8 {
        return QuantizedTensor {
            weights: vec![0i8; rows * cols],
            scale: 1.0,
            shape: (rows, cols),
        };
    }

    let scale = max_abs / 127.0;
    let mut weights = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            let q = (arr[[i, j]] / scale).round().clamp(-127.0, 127.0) as i8;
            weights.push(q);
        }
    }

    QuantizedTensor {
        weights,
        scale,
        shape: (rows, cols),
    }
}

/// Quantize an f32 Array1 to symmetric INT8
pub fn quantize_array1(arr: &Array1<f32>) -> (Vec<i8>, f32) {
    let max_abs = arr.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    if max_abs < 1e-8 {
        return (vec![0i8; arr.len()], 1.0);
    }

    let scale = max_abs / 127.0;
    let weights: Vec<i8> = arr
        .iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (weights, scale)
}

/// Dequantize INT8 weights back to f32 Array2
pub fn dequantize_to_array2(weights: &[i8], scale: f32, rows: usize, cols: usize) -> Array2<f32> {
    let mut arr = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = weights[i * cols + j] as f32 * scale;
        }
    }
    arr
}

/// Dequantize INT8 weights back to f32 Array1
pub fn dequantize_to_array1(weights: &[i8], scale: f32) -> Array1<f32> {
    Array1::from_iter(weights.iter().map(|&w| w as f32 * scale))
}

/// Matrix multiplication with quantized weights: output = input @ quantized_weight.T
/// Returns f32 output. Input stays f32; weights are dequantized on-the-fly per element.
/// This saves memory bandwidth (1 byte reads vs 4 byte reads) at the cost of
/// per-element dequantization.
pub fn quantized_matmul(input: &Array2<f32>, qw: &QuantizedTensor) -> Array2<f32> {
    let (batch, in_features) = input.dim();
    let (out_features, _) = qw.shape;
    assert_eq!(in_features, qw.shape.1, "Input features must match weight cols");

    let mut output = Array2::zeros((batch, out_features));
    let scale = qw.scale;

    for b in 0..batch {
        for o in 0..out_features {
            let mut sum = 0.0f32;
            for i in 0..in_features {
                let w = qw.weights[o * in_features + i] as f32 * scale;
                sum += input[[b, i]] * w;
            }
            output[[b, o]] = sum;
        }
    }

    output
}

/// Compute compression ratio: original f32 size / quantized size
pub fn compression_ratio(element_count: usize) -> f32 {
    (element_count * std::mem::size_of::<f32>()) as f32
            / (element_count * std::mem::size_of::<i8>()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let mut arr = Array2::zeros((3, 4));
        arr[[0, 0]] = 1.0;
        arr[[0, 1]] = -0.5;
        arr[[1, 2]] = 2.0;
        arr[[2, 3]] = -1.5;

        let qt = quantize_array2(&arr);
        let recovered = qt.to_array2();

        // Max error should be <= scale (quantization step)
        let max_err = arr
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err <= qt.scale + 1e-6, "Max error {} > scale {}", max_err, qt.scale);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let arr = Array2::zeros((5, 5));
        let qt = quantize_array2(&arr);
        assert!(qt.weights.iter().all(|&w| w == 0));
        assert_eq!(qt.scale, 1.0);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(1000);
        assert!((ratio - 4.0).abs() < 0.01, "Expected ~4x compression, got {}", ratio);
    }

    #[test]
    fn test_quantized_matmul() {
        let mut w = Array2::zeros((2, 3));
        w[[0, 0]] = 1.0;
        w[[0, 1]] = 2.0;
        w[[0, 2]] = 3.0;
        w[[1, 0]] = -1.0;
        w[[1, 1]] = 0.0;
        w[[1, 2]] = 1.0;

        let qw = quantize_array2(&w);
        let input = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let output = quantized_matmul(&input, &qw);

        // Expected: [6.0, 0.0] (with some quantization error)
        assert!((output[[0, 0]] - 6.0).abs() < 0.1, "Expected ~6.0, got {}", output[[0, 0]]);
        assert!(output[[0, 1]].abs() < 0.1, "Expected ~0.0, got {}", output[[0, 1]]);
    }

    #[test]
    fn test_quantized_tensor_size() {
        let arr = Array2::zeros((100, 100));
        let qt = quantize_array2(&arr);
        assert_eq!(qt.size_bytes(), 10000);
        assert_eq!(qt.len(), 10000);
    }

    #[test]
    fn test_quantize_array1() {
        let arr = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let (weights, scale) = quantize_array1(&arr);
        assert_eq!(weights.len(), 5);
        assert!(scale > 0.0);
        // Max should map to 127
        assert_eq!(weights[4], 127);
        assert_eq!(weights[0], -127);
    }

    #[test]
    fn test_dequantize_to_array1() {
        let weights = vec![-127i8, 0, 127];
        let scale = 0.1;
        let arr = dequantize_to_array1(&weights, scale);
        assert!((arr[0] - (-12.7)).abs() < 1e-5);
        assert!((arr[1] - 0.0).abs() < 1e-5);
        assert!((arr[2] - 12.7).abs() < 1e-5);
    }
}
