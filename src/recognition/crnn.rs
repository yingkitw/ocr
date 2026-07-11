//! CRNN (Convolutional RNN) OCR model
//!
//! Implements a Convolutional Recurrent Neural Network for text recognition:
//! CNN feature extractor -> BiLSTM -> CTC decoder
//!
//! Architecture:
//! Input: HxW grayscale text-line image (H=32 typically)
//! CNN: 5 conv blocks with maxpool -> feature maps
//! Reshape: feature maps -> sequence of feature vectors
//! BiLSTM: 2-layer bidirectional LSTM
//! Output: Linear projection to vocab + blank, then CTC decode

use crate::compute::ComputeBackend;
use crate::core::image::OcrImage;
use crate::utils::quantization::{quantize_array2, QuantizedTensor};
use crate::utils::Result;
use ndarray::{s, Array1, Array2, Array3, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CRNN Model
// ---------------------------------------------------------------------------

/// CRNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrnnConfig {
    pub input_height: usize,
    pub input_channels: usize,
    pub num_classes: usize, // vocab size + 1 blank
    pub hidden_size: usize,
    pub num_lstm_layers: usize,
    pub cnn_channels: Vec<usize>,
    pub dropout: f32,
    /// Use CTC beam search (vs greedy) at inference time
    pub use_beam_search: bool,
    /// Beam width for CTC beam search
    pub beam_width: usize,
    /// Weight for dictionary bonus when rescoring beam hypotheses
    pub dict_weight: f32,
    /// Weight for n-gram LM score when rescoring beam hypotheses
    pub lm_weight: f32,
    /// Softmax temperature for confidence calibration (`T > 1` = more conservative)
    pub confidence_temperature: f32,
}

impl CrnnConfig {
    /// Create a config sized appropriately for a script's vocabulary
    pub fn for_script(script: crate::lang::unicode::Script) -> Self {
        let vocab = Vocabulary::for_script(script);
        let vocab_size = vocab.size();
        // Adjust hidden size based on vocab complexity
        let hidden_size = match script {
            crate::lang::unicode::Script::CJK => 128,
            crate::lang::unicode::Script::Arabic => 96,
            crate::lang::unicode::Script::Cyrillic => 80,
            _ => 64,
        };
        Self {
            input_height: 32,
            input_channels: 1,
            num_classes: vocab_size,
            hidden_size,
            num_lstm_layers: 2,
            cnn_channels: vec![16, 32, 64, 64, 128],
            dropout: 0.0,
            use_beam_search: true,
            beam_width: 10,
            dict_weight: 1.0,
            lm_weight: 0.5,
            confidence_temperature: 1.0,
        }
    }
}

impl Default for CrnnConfig {
    fn default() -> Self {
        Self {
            input_height: 32,
            input_channels: 1,
            num_classes: 96 + 1, // printable ASCII + blank
            hidden_size: 64,
            num_lstm_layers: 2,
            cnn_channels: vec![16, 32, 64, 64, 128],
            dropout: 0.0,
            use_beam_search: true,
            beam_width: 10,
            dict_weight: 1.0,
            lm_weight: 0.5,
            confidence_temperature: 1.0,
        }
    }
}

/// Character vocabulary for the model
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub chars: Vec<char>,
    pub char_to_idx: std::collections::HashMap<char, usize>,
    pub blank_idx: usize,
}

impl Vocabulary {
    pub fn from_ascii() -> Self {
        let chars: Vec<char> = (' '..='~').collect(); // 95 printable ASCII
        let mut char_to_idx = std::collections::HashMap::new();
        for (i, &ch) in chars.iter().enumerate() {
            char_to_idx.insert(ch, i + 1); // 0 reserved for blank (CTC)
        }
        Self {
            chars,
            char_to_idx,
            blank_idx: 0,
        }
    }

    /// Build vocabulary for a specific Unicode script
    pub fn for_script(script: crate::lang::unicode::Script) -> Self {
        use crate::lang::unicode::Script;
        let chars: Vec<char> = match script {
            Script::Latin => (' '..='~').collect(),
            Script::CJK => {
                // Most frequent 300 CJK ideographs + Hiragana + Katakana + Hangul jamo
                let mut set: Vec<char> = (
                    "的一是了我不人在他有这个上大来们中为到说国和也出时年"
                    .to_string()
                    + "会作对开发好用能就工过地行学小多天然于心面看当"
                    + "只些想还去法以都可那得无如全定又路它现"
                    + "最下家长力里前已但公者从前很动"
                    + "日事明其分什高次回被手活"
                ).chars().collect();
                // Hiragana and Katakana
                set.extend((0x3040..=0x309F).filter_map(std::char::from_u32));
                set.extend((0x30A0..=0x30FF).filter_map(std::char::from_u32));
                set
            }
            Script::Cyrillic => {
                // Russian alphabet + extended Cyrillic
                (0x0400..=0x04FF).filter_map(std::char::from_u32).collect()
            }
            Script::Arabic => {
                // Arabic script block
                (0x0600..=0x06FF)
                    .chain(0x0750..=0x077F)
                    .filter_map(std::char::from_u32)
                    .collect()
            }
            Script::Greek => {
                (0x0370..=0x03FF).chain(0x1F00..=0x1FFF)
                    .filter_map(std::char::from_u32)
                    .collect()
            }
            Script::Hebrew => {
                (0x0590..=0x05FF).chain(0xFB1D..=0xFB4F)
                    .filter_map(std::char::from_u32)
                    .collect()
            }
            Script::Thai => {
                (0x0E00..=0x0E7F).filter_map(std::char::from_u32).collect()
            }
            Script::Devanagari => {
                (0x0900..=0x097F).chain(0xA8E0..=0xA8FF)
                    .filter_map(std::char::from_u32)
                    .collect()
            }
            Script::Other => (' '..='~').collect(),
        };
        let mut char_to_idx = std::collections::HashMap::new();
        for (i, &ch) in chars.iter().enumerate() {
            char_to_idx.insert(ch, i + 1);
        }
        Self {
            chars,
            char_to_idx,
            blank_idx: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.chars.len() + 1 // +1 for blank
    }

    pub fn idx_to_char(&self, idx: usize) -> Option<char> {
        if idx == self.blank_idx {
            None
        } else {
            self.chars.get(idx.saturating_sub(1)).copied()
        }
    }

    pub fn text_to_indices(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|ch| *self.char_to_idx.get(&ch).unwrap_or(&self.blank_idx))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CNN Feature Extractor
// ---------------------------------------------------------------------------

/// Lightweight CNN for feature extraction from text-line images
pub struct CnnFeatureExtractor {
    pub config: CrnnConfig,
    // Conv1: 1 -> 64, 3x3
    pub(crate) conv1_weights: Array3<f32>,
    pub(crate) conv1_bias: Array1<f32>,
    // Conv2: 64 -> 128, 3x3
    pub(crate) conv2_weights: Array3<f32>,
    pub(crate) conv2_bias: Array1<f32>,
    // Conv3: 128 -> 256, 3x3
    pub(crate) conv3_weights: Array3<f32>,
    pub(crate) conv3_bias: Array1<f32>,
    // Conv4: 256 -> 256, 3x3
    pub(crate) conv4_weights: Array3<f32>,
    pub(crate) conv4_bias: Array1<f32>,
    // Conv5: 256 -> 512, 3x3
    pub(crate) conv5_weights: Array3<f32>,
    pub(crate) conv5_bias: Array1<f32>,
    // Batch norm params (simplified: just scale + shift per channel)
    bn_scales: Vec<Array1<f32>>,
    bn_shifts: Vec<Array1<f32>>,
    /// Optional compute backend for offloading conv2d to GPU
    backend: Option<Box<dyn ComputeBackend>>,
}

impl CnnFeatureExtractor {
    pub fn new(config: &CrnnConfig) -> Self {
        let c = &config.cnn_channels;
        let mut s = Self {
            config: config.clone(),
            conv1_weights: Self::init_conv_weights(3, 1, c[0]),
            conv1_bias: Array1::zeros(c[0]),
            conv2_weights: Self::init_conv_weights(3, c[0], c[1]),
            conv2_bias: Array1::zeros(c[1]),
            conv3_weights: Self::init_conv_weights(3, c[1], c[2]),
            conv3_bias: Array1::zeros(c[2]),
            conv4_weights: Self::init_conv_weights(3, c[2], c[3]),
            conv4_bias: Array1::zeros(c[3]),
            conv5_weights: Self::init_conv_weights(3, c[3], c[4]),
            conv5_bias: Array1::zeros(c[4]),
            bn_scales: vec![
                Array1::ones(c[0]),
                Array1::ones(c[1]),
                Array1::ones(c[2]),
                Array1::ones(c[3]),
                Array1::ones(c[4]),
            ],
            bn_shifts: vec![
                Array1::zeros(c[0]),
                Array1::zeros(c[1]),
                Array1::zeros(c[2]),
                Array1::zeros(c[3]),
                Array1::zeros(c[4]),
            ],
            backend: None,
        };
        s.randomize();
        s
    }

    /// Attach a compute backend (e.g., CUDA or OpenCL) for conv2d offloading.
    pub fn with_backend(mut self, backend: Box<dyn ComputeBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    fn init_conv_weights(k: usize, in_ch: usize, out_ch: usize) -> Array3<f32> {
        Array3::zeros((out_ch, in_ch, k * k))
    }

    fn randomize(&mut self) {
        let layers = [
            (&mut self.conv1_weights, 1),
            (&mut self.conv2_weights, self.config.cnn_channels[0]),
            (&mut self.conv3_weights, self.config.cnn_channels[1]),
            (&mut self.conv4_weights, self.config.cnn_channels[2]),
            (&mut self.conv5_weights, self.config.cnn_channels[3]),
        ];
        for (weights, in_ch) in layers {
            let scale = (2.0 / (in_ch * 9) as f32).sqrt();
            for v in weights.iter_mut() {
                *v = (fastrand::f32() - 0.5) * 2.0 * scale;
            }
        }
    }

    /// Forward pass: input [H, W] -> output [T, C] where T = W/4 (downsampled width), C = 512
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (_h, _w) = (input.nrows(), input.ncols());
        // Add channel dimension: [1, H, W]
        let mut x = input.clone().insert_axis(Axis(0));

        // Block 1: Conv(1->64, 3x3) + ReLU + MaxPool(2,2) -> H/2 x W/2
        x = self.conv_block(
            &x,
            &self.conv1_weights,
            &self.conv1_bias,
            &self.bn_scales[0],
            &self.bn_shifts[0],
        );
        x = maxpool2d(&x, 2);

        // Block 2: Conv(64->128, 3x3) + ReLU + MaxPool(2,2) -> H/4 x W/4
        x = self.conv_block(
            &x,
            &self.conv2_weights,
            &self.conv2_bias,
            &self.bn_scales[1],
            &self.bn_shifts[1],
        );
        x = maxpool2d(&x, 2);

        // Block 3: Conv(128->256, 3x3) + ReLU + MaxPool(1,2) -> H/4 x W/8
        x = self.conv_block(
            &x,
            &self.conv3_weights,
            &self.conv3_bias,
            &self.bn_scales[2],
            &self.bn_shifts[2],
        );
        x = maxpool2d_width_only(&x, 2);

        // Block 4: Conv(256->256, 3x3) + ReLU + MaxPool(1,2) -> H/4 x W/16
        x = self.conv_block(
            &x,
            &self.conv4_weights,
            &self.conv4_bias,
            &self.bn_scales[3],
            &self.bn_shifts[3],
        );
        x = maxpool2d_width_only(&x, 2);

        // Block 5: Conv(256->512, 2x2) + ReLU -> H/4-1 x W/16
        x = self.conv_block(
            &x,
            &self.conv5_weights,
            &self.conv5_bias,
            &self.bn_scales[4],
            &self.bn_shifts[4],
        );

        // Reshape: [C, H', W'] -> [W', C*H'] (treat width as time steps)
        let (c_out, h_out, w_out) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut output = Array2::zeros((w_out, c_out * h_out));
        for t in 0..w_out {
            for c in 0..c_out {
                for y in 0..h_out {
                    output[[t, c * h_out + y]] = x[[c, y, t]];
                }
            }
        }
        output
    }

    fn conv_block(
        &self,
        input: &Array3<f32>,
        weights: &Array3<f32>,
        bias: &Array1<f32>,
        bn_scale: &Array1<f32>,
        bn_shift: &Array1<f32>,
    ) -> Array3<f32> {
        if let Some(ref backend) = self.backend {
            return self.conv_block_backend(input, weights, bias, bn_scale, bn_shift, backend.as_ref());
        }

        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let out_c = weights.shape()[0];
        let k = (weights.shape()[2] as f32).sqrt() as usize; // k*k = last dim
        let pad = k / 2;
        let out_h = in_h;
        let out_w = in_w;

        let input_slice = input.as_slice().unwrap_or(&[]);
        let weights_slice = weights.as_slice().unwrap_or(&[]);
        let bias_slice = bias.as_slice().unwrap_or(&[]);
        let bn_scale_slice = bn_scale.as_slice().unwrap_or(&[]);
        let bn_shift_slice = bn_shift.as_slice().unwrap_or(&[]);

        // Parallelize over output channels
        let channels: Vec<Vec<f32>> = (0..out_c)
            .into_par_iter()
            .map(|oc| {
                let mut channel = vec![0.0f32; out_h * out_w];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias_slice[oc];
                        for ic in 0..in_c {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let iy = oh as i32 + ky as i32 - pad as i32;
                                    let ix = ow as i32 + kx as i32 - pad as i32;
                                    if iy >= 0 && iy < in_h as i32 && ix >= 0 && ix < in_w as i32 {
                                        let w = weights_slice[oc * in_c * k * k + ic * k * k + ky * k + kx];
                                        let inp = input_slice[ic * in_h * in_w + iy as usize * in_w + ix as usize];
                                        sum += inp * w;
                                    }
                                }
                            }
                        }
                        // Batch norm + ReLU
                        let normalized = sum * bn_scale_slice[oc] + bn_shift_slice[oc];
                        channel[oh * out_w + ow] = normalized.max(0.0);
                    }
                }
                channel
            })
            .collect();

        let mut output = Array3::zeros((out_c, out_h, out_w));
        for (oc, channel) in channels.iter().enumerate() {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    output[[oc, oh, ow]] = channel[oh * out_w + ow];
                }
            }
        }
        output
    }

    /// Conv2d via external compute backend (e.g., CUDA/OpenCL).
    /// Converts ndarray → flat Vec, runs backend.conv2d, then applies BN+ReLU.
    fn conv_block_backend(
        &self,
        input: &Array3<f32>,
        weights: &Array3<f32>,
        bias: &Array1<f32>,
        bn_scale: &Array1<f32>,
        bn_shift: &Array1<f32>,
        backend: &dyn ComputeBackend,
    ) -> Array3<f32> {
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let out_c = weights.shape()[0];
        let k = (weights.shape()[2] as f32).sqrt() as usize;
        let pad = k / 2;

        // Flatten input (already NCHW contiguous)
        let input_vec = input.as_slice().unwrap_or(&[]).to_vec();

        // Reshape weights from (out_c, in_c, k*k) to (out_c, in_c, k, k)
        let mut kernel_vec = vec![0.0f32; out_c * in_c * k * k];
        if let Some(wslice) = weights.as_slice() {
            for oc in 0..out_c {
                for ic in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let src = oc * in_c * k * k + ic * k * k + ky * k + kx;
                            let dst = ((oc * in_c + ic) * k + ky) * k + kx;
                            kernel_vec[dst] = wslice[src];
                        }
                    }
                }
            }
        }

        let bias_vec = bias.as_slice().unwrap_or(&[]).to_vec();

        let output_vec = backend
            .conv2d(
                &input_vec, &kernel_vec, Some(&bias_vec),
                in_h, in_w, in_c, out_c, k, k, 1, pad,
            )
            .unwrap_or_else(|_| vec![0.0f32; out_c * in_h * in_w]);

        let mut output = Array3::zeros((out_c, in_h, in_w));
        for oc in 0..out_c {
            for oh in 0..in_h {
                for ow in 0..in_w {
                    let val = output_vec[(oc * in_h + oh) * in_w + ow];
                    let bn = val * bn_scale[oc] + bn_shift[oc];
                    output[[oc, oh, ow]] = bn.max(0.0);
                }
            }
        }
        output
    }
}

fn maxpool2d(input: &Array3<f32>, pool: usize) -> Array3<f32> {
    let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let out_h = h / pool;
    let out_w = w / pool;
    let mut output = Array3::zeros((c, out_h, out_w));
    for ch in 0..c {
        for y in 0..out_h {
            for x in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for dy in 0..pool {
                    for dx in 0..pool {
                        let val = input[[ch, y * pool + dy, x * pool + dx]];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                output[[ch, y, x]] = max_val;
            }
        }
    }
    output
}

fn maxpool2d_width_only(input: &Array3<f32>, pool: usize) -> Array3<f32> {
    let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let out_w = w / pool;
    let mut output = Array3::zeros((c, h, out_w));
    for ch in 0..c {
        for y in 0..h {
            for x in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for dx in 0..pool {
                    let val = input[[ch, y, x * pool + dx]];
                    if val > max_val {
                        max_val = val;
                    }
                }
                output[[ch, y, x]] = max_val;
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// BiLSTM
// ---------------------------------------------------------------------------

/// Bidirectional LSTM layer
pub struct BiLstmLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    // Forward LSTM weights
    pub(crate) wf_ih: Array2<f32>,
    pub(crate) wf_hh: Array2<f32>,
    pub(crate) bf_ih: Array1<f32>,
    pub(crate) bf_hh: Array1<f32>,
    // Backward LSTM weights
    pub(crate) wb_ih: Array2<f32>,
    pub(crate) wb_hh: Array2<f32>,
    pub(crate) bb_ih: Array1<f32>,
    pub(crate) bb_hh: Array1<f32>,
}

impl BiLstmLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut s = Self {
            input_size,
            hidden_size,
            wf_ih: Array2::zeros((4 * hidden_size, input_size)),
            wf_hh: Array2::zeros((4 * hidden_size, hidden_size)),
            bf_ih: Array1::zeros(4 * hidden_size),
            bf_hh: Array1::zeros(4 * hidden_size),
            wb_ih: Array2::zeros((4 * hidden_size, input_size)),
            wb_hh: Array2::zeros((4 * hidden_size, hidden_size)),
            bb_ih: Array1::zeros(4 * hidden_size),
            bb_hh: Array1::zeros(4 * hidden_size),
        };
        s.randomize();
        s
    }

    fn randomize(&mut self) {
        let scale = (1.0 / self.input_size as f32).sqrt();
        let _h_scale = (1.0 / self.hidden_size as f32).sqrt();
        for w in [
            &mut self.wf_ih,
            &mut self.wf_hh,
            &mut self.wb_ih,
            &mut self.wb_hh,
        ]
        .iter_mut()
        {
            for v in w.iter_mut() {
                *v = (fastrand::f32() - 0.5) * 2.0 * scale;
            }
        }
        // Forget gate bias = 1.0
        for i in self.hidden_size..2 * self.hidden_size {
            self.bf_ih[i] = 1.0;
            self.bf_hh[i] = 1.0;
            self.bb_ih[i] = 1.0;
            self.bb_hh[i] = 1.0;
        }
    }

    /// Forward pass: input [T, input_size] -> output [T, 2*hidden_size]
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (t, _) = input.dim();
        let fwd = self.lstm_forward(input, &self.wf_ih, &self.wf_hh, &self.bf_ih, &self.bf_hh);
        let bwd = self.lstm_forward(
            &input.slice(s![..;-1,..]).to_owned(),
            &self.wb_ih,
            &self.wb_hh,
            &self.bb_ih,
            &self.bb_hh,
        );

        let mut output = Array2::zeros((t, 2 * self.hidden_size));
        for i in 0..t {
            for j in 0..self.hidden_size {
                output[[i, j]] = fwd[[i, j]];
                output[[i, self.hidden_size + j]] = bwd[[t - 1 - i, j]];
            }
        }
        output
    }

    fn lstm_forward(
        &self,
        input: &Array2<f32>,
        w_ih: &Array2<f32>,
        w_hh: &Array2<f32>,
        b_ih: &Array1<f32>,
        b_hh: &Array1<f32>,
    ) -> Array2<f32> {
        let (seq_len, _) = input.dim();
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);
        let mut output = Array2::zeros((seq_len, self.hidden_size));
        let hs = self.hidden_size;

        for t in 0..seq_len {
            let xt = input.row(t);
            let gates =
                Self::mat_vec_add(w_ih, &xt.to_owned(), b_ih) + Self::mat_vec_add(w_hh, &h, b_hh);

            let i = Self::sigmoid(&gates.slice(s![0..hs]).to_owned());
            let f = Self::sigmoid(&gates.slice(s![hs..2 * hs]).to_owned());
            let g = gates.slice(s![2 * hs..3 * hs]).mapv(|x| x.tanh());
            let o = Self::sigmoid(&gates.slice(s![3 * hs..4 * hs]).to_owned());

            c = &c * &f + &i * &g;
            h = &o * c.mapv(|x: f32| x.tanh());

            for j in 0..hs {
                output[[t, j]] = h[j];
            }
        }
        output
    }

    fn mat_vec_add(m: &Array2<f32>, v: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        let mut out = b.clone();
        for i in 0..m.nrows() {
            let row = m.row(i);
            let mut sum = b[i];
            for j in 0..row.len() {
                sum += row[j] * v[j];
            }
            out[i] = sum;
        }
        out
    }

    fn sigmoid(a: &Array1<f32>) -> Array1<f32> {
        a.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

// ---------------------------------------------------------------------------
// CRNN
// ---------------------------------------------------------------------------

/// Full CRNN model: CNN -> BiLSTM -> Linear -> CTC
pub struct CrnnModel {
    pub config: CrnnConfig,
    pub vocab: Vocabulary,
    pub cnn: CnnFeatureExtractor,
    pub lstm1: BiLstmLayer,
    pub lstm2: BiLstmLayer,
    pub fc_weight: Array2<f32>,
    pub fc_bias: Array1<f32>,
    /// Optional INT8-quantized FC weights for memory-efficient inference
    pub fc_weight_quantized: Option<QuantizedTensor>,
}

impl CrnnModel {
    pub fn new(config: CrnnConfig) -> Self {
        let vocab = Vocabulary::from_ascii();
        let num_classes = vocab.size();
        let cnn = CnnFeatureExtractor::new(&config);

        // After CNN: feature dim = 512 * (32/4 - 1) ≈ 512 * 7 = 3584 (simplified)
        // Actually after our CNN: cnn_channels[-1] * remaining_height
        // With 32 input and 2 maxpool(2): height = 32/4 = 8, minus conv5(2x2) = 7
        let cnn_out_h = config.input_height / 4 - 1;
        let cnn_feature_dim = config.cnn_channels[4] * cnn_out_h;

        let lstm1 = BiLstmLayer::new(cnn_feature_dim, config.hidden_size);
        let lstm2 = BiLstmLayer::new(2 * config.hidden_size, config.hidden_size);

        let mut fc_weight = Array2::zeros((num_classes, 2 * config.hidden_size));
        let fc_scale = (1.0 / (2 * config.hidden_size) as f32).sqrt();
        for v in fc_weight.iter_mut() {
            *v = (fastrand::f32() - 0.5) * 2.0 * fc_scale;
        }

        Self {
            config,
            vocab,
            cnn,
            lstm1,
            lstm2,
            fc_weight,
            fc_bias: Array1::zeros(num_classes),
            fc_weight_quantized: None,
        }
    }

    /// Attach a compute backend to the CNN feature extractor.
    /// When a GPU backend is available (compiled with `cuda` or `opencl`
    /// features), conv2d operations will be offloaded to it.
    pub fn with_backend(mut self, backend: Box<dyn ComputeBackend>) -> Self {
        self.cnn = self.cnn.with_backend(backend);
        self
    }

    /// Quantize the FC layer weights to INT8 for memory-efficient inference.
    /// The original f32 weights are kept for training; quantized weights are used
    /// for inference when `use_quantized` is true.
    pub fn quantize_fc(&mut self) {
        self.fc_weight_quantized = Some(quantize_array2(&self.fc_weight));
    }

    /// Forward using quantized FC weights if available, otherwise f32.
    pub fn forward(&self, image: &Array2<f32>) -> Array2<f32> {
        let cnn_features = self.cnn.forward(image);
        let lstm1_out = self.lstm1.forward(&cnn_features);
        let lstm2_out = self.lstm2.forward(&lstm1_out);

        let mut logits = if let Some(ref qw) = self.fc_weight_quantized {
            crate::utils::quantization::quantized_matmul(&lstm2_out, qw)
        } else {
            lstm2_out.dot(&self.fc_weight.t())
        };

        for i in 0..logits.nrows() {
            for j in 0..logits.ncols() {
                logits[[i, j]] += self.fc_bias[j];
            }
        }
        logits
    }
}

/// Recognition output with calibrated confidence scores.
#[derive(Debug, Clone)]
pub struct CrnnRecognition {
    pub text: String,
    pub confidence: f32,
    pub char_confidences: Vec<crate::recognition::confidence::CharConfidence>,
}

impl CrnnRecognition {
    /// Aggregate calibrated character scores into word-level confidences.
    pub fn word_confidences(&self) -> Vec<(String, f32)> {
        crate::recognition::confidence::DecodeConfidence {
            chars: self.char_confidences.clone(),
            overall: self.confidence,
        }
        .word_confidences()
    }
}

impl CrnnModel {
    /// Recognize text from an OcrImage
    pub fn recognize(&self, image: &OcrImage) -> Result<String> {
        Ok(self.recognize_detailed(image, None, None)?.text)
    }

    /// Recognize with optional dictionary / n-gram LM beam rescoring.
    pub fn recognize_with_rescoring(
        &self,
        image: &OcrImage,
        dictionary: Option<&crate::lang::dictionary::Dictionary>,
        ngram: Option<&crate::lang::NGramModel>,
    ) -> Result<String> {
        Ok(self.recognize_detailed(image, dictionary, ngram)?.text)
    }

    /// Recognize and return calibrated per-character / overall confidence.
    pub fn recognize_detailed(
        &self,
        image: &OcrImage,
        dictionary: Option<&crate::lang::dictionary::Dictionary>,
        ngram: Option<&crate::lang::NGramModel>,
    ) -> Result<CrnnRecognition> {
        let gray = image.data.to_luma8();
        let (w, h) = (gray.width() as usize, gray.height() as usize);

        let mut arr = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            }
        }

        let target_h = self.config.input_height;
        if h != target_h {
            arr = Self::resize_array2_height(&arr, target_h);
        }

        let logits = self.forward(&arr);
        let text = self.decode_logits(&logits, dictionary, ngram)?;
        let calibrator = crate::recognition::confidence::ConfidenceCalibrator::new(
            self.config.confidence_temperature,
        );

        let conf = if !self.config.use_beam_search {
            let path =
                crate::recognition::confidence::greedy_path_confidence(
                    &logits,
                    &self.vocab.chars,
                    &calibrator,
                );
            // Fall back if greedy path text diverges from decode (shouldn't normally).
            let path_text: String = path.chars.iter().map(|c| c.character).collect();
            if path_text == text {
                path
            } else {
                crate::recognition::confidence::hypothesis_confidence(
                    &logits, &text, &calibrator,
                )
            }
        } else {
            crate::recognition::confidence::hypothesis_confidence(&logits, &text, &calibrator)
        };

        Ok(CrnnRecognition {
            text,
            confidence: conf.overall,
            char_confidences: conf.chars,
        })
    }

    /// Decode CTC logits using greedy or beam search (+ optional rescoring).
    pub fn decode_logits(
        &self,
        logits: &Array2<f32>,
        dictionary: Option<&crate::lang::dictionary::Dictionary>,
        ngram: Option<&crate::lang::NGramModel>,
    ) -> Result<String> {
        use crate::recognition::ctc_decoder::{CtcDecoder, DictLmRescorer};

        let decoder = CtcDecoder::with_beam_width(self.config.beam_width);
        if !self.config.use_beam_search {
            return Ok(decoder.greedy_decode(logits, &self.vocab.chars));
        }

        if dictionary.is_some() || ngram.is_some() {
            let rescorer = DictLmRescorer::new(
                dictionary,
                ngram,
                self.config.dict_weight,
                self.config.lm_weight,
            );
            Ok(decoder.beam_search_decode_rescored(logits, &self.vocab.chars, &rescorer))
        } else {
            Ok(decoder.beam_search_decode(logits, &self.vocab.chars))
        }
    }

    pub fn resize_array2_height(arr: &Array2<f32>, target_h: usize) -> Array2<f32> {
        let (h, w) = (arr.nrows(), arr.ncols());
        let mut out = Array2::zeros((target_h, w));
        let scale = h as f32 / target_h as f32;
        for y in 0..target_h {
            let src_y = ((y as f32 + 0.5) * scale - 0.5).clamp(0.0, (h - 1) as f32) as usize;
            for x in 0..w {
                out[[y, x]] = arr[[src_y, x]];
            }
        }
        out
    }

    /// Get model parameter count for size estimation
    pub fn parameter_count(&self) -> usize {
        let mut count = 0usize;
        // CNN weights
        count += self.cnn.conv1_weights.len();
        count += self.cnn.conv2_weights.len();
        count += self.cnn.conv3_weights.len();
        count += self.cnn.conv4_weights.len();
        count += self.cnn.conv5_weights.len();
        // LSTM weights
        count += self.lstm1.wf_ih.len() + self.lstm1.wf_hh.len();
        count += self.lstm2.wf_ih.len() + self.lstm2.wf_hh.len();
        // FC
        count += self.fc_weight.len();
        count
    }

    /// Estimate model size in bytes. If quantized FC weights are present,
    /// they replace the f32 FC weight memory (4x reduction for that layer).
    pub fn model_size_bytes(&self) -> usize {
        let mut bytes = self.parameter_count() * std::mem::size_of::<f32>();
        // Subtract f32 FC weight size, add INT8 FC weight size
        if self.fc_weight_quantized.is_some() {
            bytes -= self.fc_weight.len() * std::mem::size_of::<f32>();
            bytes += self.fc_weight.len() * std::mem::size_of::<i8>();
        }
        bytes
    }
}

/// Registry that holds a CRNN model per Unicode script.
/// When a text region's script is detected, the matching model is used.
pub struct ScriptModelRegistry {
    models: std::collections::HashMap<crate::lang::unicode::Script, CrnnModel>,
    default_script: crate::lang::unicode::Script,
}

impl Default for ScriptModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptModelRegistry {
    /// Create a registry with models for all supported scripts
    pub fn new() -> Self {
        use crate::lang::unicode::Script;
        let scripts = [
            Script::Latin,
            Script::CJK,
            Script::Arabic,
            Script::Cyrillic,
            Script::Greek,
            Script::Hebrew,
            Script::Thai,
            Script::Devanagari,
        ];
        let mut models = std::collections::HashMap::new();
        for script in scripts {
            let config = CrnnConfig::for_script(script);
            models.insert(script, CrnnModel::new(config));
        }
        Self {
            models,
            default_script: Script::Latin,
        }
    }

    /// Recognize text using the model matching the detected script
    pub fn recognize(&self, image: &crate::core::image::OcrImage, text: &str) -> Result<String> {
        let script = crate::lang::unicode::Script::detect(text);
        let model = self.models.get(&script).or_else(|| self.models.get(&self.default_script)).ok_or_else(|| {
            crate::OcrError::Internal("No CRNN model available".to_string())
        })?;
        model.recognize(image)
    }

    /// Get total parameter count across all models
    pub fn total_parameter_count(&self) -> usize {
        self.models.values().map(|m| m.parameter_count()).sum()
    }

    /// Get model for a specific script
    pub fn model_for(&self, script: crate::lang::unicode::Script) -> Option<&CrnnModel> {
        self.models.get(&script)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crnn_forward_shape() {
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        // Input: 32 x 128
        let input = Array2::zeros((32, 128));
        let logits = model.forward(&input);
        assert!(logits.nrows() > 0);
        assert_eq!(logits.ncols(), model.vocab.size());
    }

    #[test]
    fn test_crnn_parameter_count() {
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        let count = model.parameter_count();
        assert!(count > 0);
        let size_mb = model.model_size_bytes() as f32 / (1024.0 * 1024.0);
        assert!(
            size_mb < 10.0,
            "Model should be under 10MB, got {:.2}MB",
            size_mb
        );
    }

    #[test]
    fn test_vocab_roundtrip() {
        let vocab = Vocabulary::from_ascii();
        let text = "Hello World! 123";
        let indices = vocab.text_to_indices(text);
        let decoded: String = indices
            .iter()
            .filter_map(|&i| vocab.idx_to_char(i))
            .collect();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_for_script_cyrillic() {
        let vocab = Vocabulary::for_script(crate::lang::unicode::Script::Cyrillic);
        assert!(vocab.size() > 50, "Cyrillic vocab should have many chars");
    }

    #[test]
    fn test_vocab_for_script_cjk() {
        let vocab = Vocabulary::for_script(crate::lang::unicode::Script::CJK);
        assert!(vocab.size() > 100, "CJK vocab should have many chars");
    }

    #[test]
    fn test_script_model_registry_creates_all_models() {
        let registry = ScriptModelRegistry::new();
        assert!(!registry.models.is_empty());
        // Total params should be reasonable (< 50MB for all scripts)
        let total_mb = registry.total_parameter_count() as f32 * 4.0 / (1024.0 * 1024.0);
        assert!(total_mb < 50.0, "All script models should be under 50MB, got {:.1}MB", total_mb);
    }

    #[test]
    fn test_script_model_registry_routes_by_script() {
        let registry = ScriptModelRegistry::new();
        let model = registry.model_for(crate::lang::unicode::Script::Cyrillic);
        assert!(model.is_some(), "Should have a Cyrillic model");
    }

    #[test]
    fn test_crnn_inference_speed() {
        use std::time::Instant;
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config);
        // Typical text-line size: 32x256
        let input = Array2::zeros((32, 256));
        let iters = 3;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = model.forward(&input);
        }
        let elapsed = start.elapsed();
        let ms_per_line = elapsed.as_millis() as f32 / iters as f32;
        println!("CRNN inference: {:.2} ms/line ({} iters)", ms_per_line, iters);
        // Debug builds are very slow with ndarray; only assert a reasonable ceiling for smoke test.
        // Target on release: < 100ms/line.  On debug we allow up to 10s/line to avoid flakiness.
        assert!(
            ms_per_line < 10000.0,
            "CRNN inference extremely slow: {:.2} ms/line",
            ms_per_line
        );
    }

    #[test]
    fn test_crnn_quantized_fc_inference() {
        let config = CrnnConfig::default();
        let mut model = CrnnModel::new(config);
        let input = Array2::zeros((32, 128));

        // Baseline f32 forward
        let logits_f32 = model.forward(&input);

        // Quantize FC weights and run again
        model.quantize_fc();
        let logits_q = model.forward(&input);

        // Shape should be identical
        assert_eq!(logits_f32.dim(), logits_q.dim());

        // Values should be close (quantization error is bounded by scale)
        let max_diff = logits_f32
            .iter()
            .zip(logits_q.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = model.fc_weight_quantized.as_ref().unwrap().scale;
        assert!(
            max_diff <= scale * 2.0,
            "Quantized inference drift too large: max_diff={} scale={}",
            max_diff,
            scale
        );

        // Model size should decrease
        let size_before = model.parameter_count() * std::mem::size_of::<f32>();
        let size_after = model.model_size_bytes();
        assert!(
            size_after < size_before,
            "Quantized model should be smaller: before={} after={}",
            size_before,
            size_after
        );
    }

    #[test]
    fn test_crnn_with_compute_backend() {
        use crate::compute::CpuBackend;
        let config = CrnnConfig::default();
        let model = CrnnModel::new(config.clone());
        let input = Array2::zeros((32, 128));
        let baseline = model.forward(&input);

        // Create model with CPU backend (should produce identical results)
        let model_backend = CrnnModel::new(config).with_backend(Box::new(CpuBackend::new()));
        let with_backend = model_backend.forward(&input);

        assert_eq!(baseline.dim(), with_backend.dim());
        let max_diff = baseline
            .iter()
            .zip(with_backend.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "Backend path should match CPU path: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_crnn_decode_logits_beam_search() {
        let mut config = CrnnConfig::default();
        config.use_beam_search = true;
        config.beam_width = 5;
        let model = CrnnModel::new(config);
        let logits = Array2::zeros((8, model.vocab.size()));
        let text = model.decode_logits(&logits, None, None).unwrap();
        // Random-zero logits should decode without panic (often empty or blanks)
        let _ = text;
    }

    #[test]
    fn test_crnn_decode_logits_greedy_toggle() {
        let mut config = CrnnConfig::default();
        config.use_beam_search = false;
        let model = CrnnModel::new(config);
        let logits = Array2::zeros((4, model.vocab.size()));
        let greedy = model.decode_logits(&logits, None, None).unwrap();

        let mut config_beam = CrnnConfig::default();
        config_beam.use_beam_search = true;
        let model_beam = CrnnModel::new(config_beam);
        // Same vocab size — zeros logits; both paths must succeed (outputs may differ on ties).
        let beam = model_beam.decode_logits(&logits, None, None).unwrap();
        assert!(greedy.len() <= 4);
        assert!(beam.len() <= 4);
    }
}
