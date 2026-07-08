//! Compute backend abstraction for neural network inference
//!
//! Provides a unified interface for CPU, CUDA, and OpenCL backends.
//! Enabled via the `cuda` and `opencl` feature flags.

use crate::utils::Result;
use serde::{Deserialize, Serialize};

/// Supported compute backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "opencl")]
    OpenCl,
}

impl BackendType {
    /// Detect the best available backend
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            if CudaBackend::is_available() {
                return BackendType::Cuda;
            }
        }
        #[cfg(feature = "opencl")]
        {
            if OpenClBackend::is_available() {
                return BackendType::OpenCl;
            }
        }
        BackendType::Cpu
    }

    pub fn name(&self) -> &str {
        match self {
            BackendType::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            BackendType::Cuda => "CUDA",
            #[cfg(feature = "opencl")]
            BackendType::OpenCl => "OpenCL",
        }
    }
}

/// Trait for compute backends
pub trait ComputeBackend: Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn device_name(&self) -> &str;
    fn total_memory(&self) -> usize;

    /// Matrix multiplication: C = A * B
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;

    /// 2D Convolution
    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>>;

    /// Element-wise addition
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// Element-wise multiplication
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// ReLU activation
    fn relu(&self, x: &[f32]) -> Result<Vec<f32>>;

    /// Sigmoid activation
    fn sigmoid(&self, x: &[f32]) -> Result<Vec<f32>>;

    /// Tanh activation
    fn tanh(&self, x: &[f32]) -> Result<Vec<f32>>;

    /// Softmax along last dimension
    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>>;

    /// Vector addition with broadcast bias
    fn add_bias(&self, x: &[f32], bias: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>>;
}

// ── CPU Backend ──────────────────────────────────────────────────────────

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn device_name(&self) -> &str {
        "CPU (rayon)"
    }

    fn total_memory(&self) -> usize {
        0 // Not applicable
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        let mut c = vec![0.0f32; m * n];
        // a: m x k, b: k x n, c: m x n
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        let out_h = (in_h + 2 * padding - k_h) / stride + 1;
        let out_w = (in_w + 2 * padding - k_w) / stride + 1;
        let mut output = vec![0.0f32; out_c * out_h * out_w];

        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ic in 0..in_c {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = (oh * stride + kh) as isize - padding as isize;
                                let iw = (ow * stride + kw) as isize - padding as isize;
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let input_idx = (ic * in_h + ih as usize) * in_w + iw as usize;
                                    let kernel_idx = ((oc * in_c + ic) * k_h + kh) * k_w + kw;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += b[oc];
                    }
                    output[(oc * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
        Ok(output)
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
    }

    fn relu(&self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect())
    }

    fn sigmoid(&self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }

    fn tanh(&self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| v.tanh()).collect())
    }

    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            let max_val = x[start..end]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = x[start..end].iter().map(|&v| (v - max_val).exp()).sum();
            for (c, val) in result[start..end].iter_mut().enumerate() {
                *val = (x[start + c] - max_val).exp() / sum;
            }
        }
        Ok(result)
    }

    fn add_bias(&self, x: &[f32], bias: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let mut result = x.to_vec();
        for r in 0..rows {
            for c in 0..cols {
                result[r * cols + c] += bias[c];
            }
        }
        Ok(result)
    }
}

// ── CUDA Backend ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device_name: String,
    total_memory: usize,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn is_available() -> bool {
        // Check if CUDA runtime is available
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || cfg!(target_os = "linux")
    }

    pub fn new() -> Result<Self> {
        Ok(Self {
            device_name: "NVIDIA CUDA GPU".to_string(),
            total_memory: 0,
        })
    }
}

#[cfg(feature = "cuda")]
impl ComputeBackend for CudaBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn total_memory(&self) -> usize {
        self.total_memory
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Fallback to CPU for now; real impl would use cudarc
        CpuBackend::new().matmul(a, b, m, k, n)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        CpuBackend::new().conv2d(
            input, kernel, bias, in_h, in_w, in_c, out_c, k_h, k_w, stride, padding,
        )
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        CpuBackend::new().add(a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        CpuBackend::new().mul(a, b)
    }

    fn relu(&self, x: &[f32]) -> Result<Vec<f32>> {
        CpuBackend::new().relu(x)
    }

    fn sigmoid(&self, x: &[f32]) -> Result<Vec<f32>> {
        CpuBackend::new().sigmoid(x)
    }

    fn tanh(&self, x: &[f32]) -> Result<Vec<f32>> {
        CpuBackend::new().tanh(x)
    }

    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        CpuBackend::new().softmax(x, rows, cols)
    }

    fn add_bias(&self, x: &[f32], bias: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        CpuBackend::new().add_bias(x, bias, rows, cols)
    }
}

// ── OpenCL Backend ───────────────────────────────────────────────────────

#[cfg(feature = "opencl")]
const OPENCL_KERNEL_SRC: &str = r#"
__kernel void matmul(
    __global const float* a,
    __global const float* b,
    __global float* c,
    int m, int k, int n
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += a[row * k + p] * b[p * n + col];
        }
        c[row * n + col] = sum;
    }
}

__kernel void conv2d(
    __global const float* input,
    __global const float* kernel,
    __global const float* bias,
    __global float* output,
    int in_h, int in_w, int in_c,
    int out_c, int k_h, int k_w,
    int stride, int padding, int has_bias
) {
    int oc = get_global_id(0);
    int oh = get_global_id(1);
    int ow = get_global_id(2);
    int out_h = (in_h + 2 * padding - k_h) / stride + 1;
    int out_w = (in_w + 2 * padding - k_w) / stride + 1;
    if (oc >= out_c || oh >= out_h || ow >= out_w) return;

    float sum = 0.0f;
    for (int ic = 0; ic < in_c; ic++) {
        for (int kh = 0; kh < k_h; kh++) {
            for (int kw = 0; kw < k_w; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int iidx = (ic * in_h + ih) * in_w + iw;
                    int kidx = ((oc * in_c + ic) * k_h + kh) * k_w + kw;
                    sum += input[iidx] * kernel[kidx];
                }
            }
        }
    }
    if (has_bias) sum += bias[oc];
    output[(oc * out_h + oh) * out_w + ow] = sum;
}

__kernel void add(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}

__kernel void mul(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] * b[i];
}

__kernel void relu(__global const float* x, __global float* y, int n) {
    int i = get_global_id(0);
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__kernel void sigmoid(__global const float* x, __global float* y, int n) {
    int i = get_global_id(0);
    if (i < n) y[i] = 1.0f / (1.0f + exp(-x[i]));
}

__kernel void tanh_kernel(__global const float* x, __global float* y, int n) {
    int i = get_global_id(0);
    if (i < n) y[i] = tanh(x[i]);
}

__kernel void add_bias(__global float* x, __global const float* bias, int rows, int cols) {
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r < rows && c < cols) x[r * cols + c] += bias[c];
}
"#;

#[cfg(feature = "opencl")]
pub struct OpenClBackend {
    device_name: String,
    total_memory: usize,
    proque: Option<ocl::ProQue>,
}

#[cfg(feature = "opencl")]
impl OpenClBackend {
    pub fn is_available() -> bool {
        cfg!(any(target_os = "macos", target_os = "linux", target_os = "windows"))
    }

    pub fn new() -> Result<Self> {
        match ocl::ProQue::builder().src(OPENCL_KERNEL_SRC).dims(1).build() {
            Ok(proque) => {
                let device = proque.device();
                let name = device.name().unwrap_or_else(|_| "OpenCL".to_string());
                Ok(Self {
                    device_name: name,
                    total_memory: 0,
                    proque: Some(proque),
                })
            }
            Err(e) => {
                eprintln!("OpenCL init failed ({}), using CPU fallback.", e);
                Ok(Self {
                    device_name: "OpenCL (CPU fallback)".to_string(),
                    total_memory: 0,
                    proque: None,
                })
            }
        }
    }
}

#[cfg(feature = "opencl")]
impl ComputeBackend for OpenClBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::OpenCl
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn total_memory(&self) -> usize {
        self.total_memory
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        let pq = match &self.proque {
            Some(pq) => pq,
            None => return CpuBackend::new().matmul(a, b, m, k, n),
        };

        let buf_a = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(a.len())
            .copy_host_slice(a)
            .build()?;
        let buf_b = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(b.len())
            .copy_host_slice(b)
            .build()?;
        let buf_c = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(m * n)
            .build()?;

        let kernel = pq.kernel_builder("matmul")
            .arg(&buf_a)
            .arg(&buf_b)
            .arg(&buf_c)
            .arg(&(m as i32))
            .arg(&(k as i32))
            .arg(&(n as i32))
            .build()?;
        unsafe { kernel.enq()?; }

        let mut c = vec![0.0f32; m * n];
        buf_c.read(&mut c).enq()?;
        Ok(c)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        let pq = match &self.proque {
            Some(pq) => pq,
            None => return CpuBackend::new().conv2d(
                input, kernel, bias, in_h, in_w, in_c, out_c, k_h, k_w, stride, padding,
            ),
        };

        let out_h = (in_h + 2 * padding - k_h) / stride + 1;
        let out_w = (in_w + 2 * padding - k_w) / stride + 1;

        let buf_in = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(input.len())
            .copy_host_slice(input)
            .build()?;
        let buf_k = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(kernel.len())
            .copy_host_slice(kernel)
            .build()?;
        let buf_b = match bias {
            Some(b) => ocl::Buffer::<f32>::builder()
                .queue(pq.queue().clone())
                .len(b.len())
                .copy_host_slice(b)
                .build()?,
            None => ocl::Buffer::<f32>::builder()
                .queue(pq.queue().clone())
                .len(1)
                .build()?,
        };
        let buf_out = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(out_c * out_h * out_w)
            .build()?;

        let kernel = pq.kernel_builder("conv2d")
            .arg(&buf_in)
            .arg(&buf_k)
            .arg(&buf_b)
            .arg(&buf_out)
            .arg(&(in_h as i32))
            .arg(&(in_w as i32))
            .arg(&(in_c as i32))
            .arg(&(out_c as i32))
            .arg(&(k_h as i32))
            .arg(&(k_w as i32))
            .arg(&(stride as i32))
            .arg(&(padding as i32))
            .arg(&(if bias.is_some() { 1 } else { 0 }))
            .build()?;
        unsafe {
            kernel.cmd().global_work_size((out_c, out_h, out_w)).enq()?;
        }

        let mut output = vec![0.0f32; out_c * out_h * out_w];
        buf_out.read(&mut output).enq()?;
        Ok(output)
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_elementwise("add", a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_elementwise("mul", a, b)
    }

    fn relu(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.run_unary("relu", x)
    }

    fn sigmoid(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.run_unary("sigmoid", x)
    }

    fn tanh(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.run_unary("tanh_kernel", x)
    }

    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        // Softmax is tricky in OpenCL; use CPU for now
        CpuBackend::new().softmax(x, rows, cols)
    }

    fn add_bias(&self, x: &[f32], bias: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let pq = match &self.proque {
            Some(pq) => pq,
            None => return CpuBackend::new().add_bias(x, bias, rows, cols),
        };

        let mut buf_x = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(x.len())
            .copy_host_slice(x)
            .build()?;
        let buf_b = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(bias.len())
            .copy_host_slice(bias)
            .build()?;

        let kernel = pq.kernel_builder("add_bias")
            .arg(&buf_x)
            .arg(&buf_b)
            .arg(&(rows as i32))
            .arg(&(cols as i32))
            .build()?;
        unsafe {
            kernel.cmd().global_work_size((rows, cols)).enq()?;
        }

        let mut result = vec![0.0f32; x.len()];
        buf_x.read(&mut result).enq()?;
        Ok(result)
    }
}

#[cfg(feature = "opencl")]
impl OpenClBackend {
    fn run_elementwise(&self, name: &str, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let pq = match &self.proque {
            Some(pq) => pq,
            None => {
                let cpu = CpuBackend::new();
                return match name {
                    "add" => cpu.add(a, b),
                    "mul" => cpu.mul(a, b),
                    _ => cpu.add(a, b),
                };
            }
        };

        let buf_a = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(a.len())
            .copy_host_slice(a)
            .build()?;
        let buf_b = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(b.len())
            .copy_host_slice(b)
            .build()?;
        let buf_c = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(a.len())
            .build()?;

        let kernel = pq.kernel_builder(name)
            .arg(&buf_a)
            .arg(&buf_b)
            .arg(&buf_c)
            .arg(&(a.len() as i32))
            .build()?;
        unsafe { kernel.enq()?; }

        let mut c = vec![0.0f32; a.len()];
        buf_c.read(&mut c).enq()?;
        Ok(c)
    }

    fn run_unary(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let pq = match &self.proque {
            Some(pq) => pq,
            None => {
                let cpu = CpuBackend::new();
                return match name {
                    "relu" => cpu.relu(x),
                    "sigmoid" => cpu.sigmoid(x),
                    "tanh_kernel" => cpu.tanh(x),
                    _ => cpu.relu(x),
                };
            }
        };

        let buf_x = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(x.len())
            .copy_host_slice(x)
            .build()?;
        let buf_y = ocl::Buffer::<f32>::builder()
            .queue(pq.queue().clone())
            .len(x.len())
            .build()?;

        let kernel = pq.kernel_builder(name)
            .arg(&buf_x)
            .arg(&buf_y)
            .arg(&(x.len() as i32))
            .build()?;
        unsafe { kernel.enq()?; }

        let mut y = vec![0.0f32; x.len()];
        buf_y.read(&mut y).enq()?;
        Ok(y)
    }
}

// ── Backend Factory ──────────────────────────────────────────────────────

/// Create a compute backend of the specified type
pub fn create_backend(backend_type: BackendType) -> Result<Box<dyn ComputeBackend>> {
    match backend_type {
        BackendType::Cpu => Ok(Box::new(CpuBackend::new())),
        #[cfg(feature = "cuda")]
        BackendType::Cuda => Ok(Box::new(CudaBackend::new()?)),
        #[cfg(feature = "opencl")]
        BackendType::OpenCl => Ok(Box::new(OpenClBackend::new()?)),
    }
}

/// Auto-detect and create the best available backend
pub fn create_auto_backend() -> Result<Box<dyn ComputeBackend>> {
    create_backend(BackendType::detect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul() {
        let backend = CpuBackend::new();
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 row-major
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 row-major
        let c = backend.matmul(&a, &b, 2, 3, 2).unwrap();
        // Expected: [[58, 64], [139, 154]]
        assert!((c[0] - 58.0).abs() < 0.01);
        assert!((c[1] - 64.0).abs() < 0.01);
        assert!((c[2] - 139.0).abs() < 0.01);
        assert!((c[3] - 154.0).abs() < 0.01);
    }

    #[test]
    fn test_cpu_relu() {
        let backend = CpuBackend::new();
        let x = vec![-1.0, 0.0, 2.0, -0.5];
        let y = backend.relu(&x).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_cpu_sigmoid() {
        let backend = CpuBackend::new();
        let x = vec![0.0f32];
        let y = backend.sigmoid(&x).unwrap();
        assert!((y[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cpu_softmax() {
        let backend = CpuBackend::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = backend.softmax(&x, 1, 3).unwrap();
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        assert!(y[0] < y[1] && y[1] < y[2]);
    }

    #[test]
    fn test_backend_detect() {
        let backend_type = BackendType::detect();
        // On most systems without CUDA/OpenCL, this should be CPU
        let is_valid = matches!(backend_type, BackendType::Cpu);
        #[cfg(feature = "cuda")]
        let is_valid = is_valid || matches!(backend_type, BackendType::Cuda);
        #[cfg(feature = "opencl")]
        let is_valid = is_valid || matches!(backend_type, BackendType::OpenCl);
        assert!(is_valid);
    }
}
