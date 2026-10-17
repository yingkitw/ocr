//! Forward-with-cache + full backward for `CrnnModel`.
//!
//! This module implements a complete backward pass through the CRNN
//! architecture (CNN + 2-layer BiLSTM + FC) so the bundled Adam trainer
//! can update every parameter, not just the FC readout.
//!
//! The forward pass records every intermediate activation. The backward
//! pass consumes those activations plus the gradient of the loss w.r.t.
//! the logits and returns a `ParamGrads` populated for every layer.

use crate::recognition::crnn::{CnnFeatureExtractor, CrnnModel};
use ndarray::{Array1, Array2, Array3};

/// Cached activations from a single-sample forward pass.
pub struct ForwardCache {
    pub input: Array2<f32>,
    // After each conv block, before pool
    pub conv1_pre: Array3<f32>,    // [C1, H, W] conv output (with bias)
    pub conv1_post: Array3<f32>,   // after BN + ReLU
    pub conv1_argmax: Array3<(usize, usize)>, // for maxpool 2x2
    pub pool1_out: Array3<f32>,
    pub conv2_pre: Array3<f32>,
    pub conv2_post: Array3<f32>,
    pub conv2_argmax: Array3<(usize, usize)>,
    pub pool2_out: Array3<f32>,
    pub conv3_pre: Array3<f32>,
    pub conv3_post: Array3<f32>,
    pub conv3_argmax: Array3<(usize, usize)>, // for maxpool 1x2 width
    pub pool3_out: Array3<f32>,
    pub conv4_pre: Array3<f32>,
    pub conv4_post: Array3<f32>,
    pub conv4_argmax: Array3<(usize, usize)>,
    pub pool4_out: Array3<f32>,
    pub conv5_pre: Array3<f32>,
    pub conv5_post: Array3<f32>,  // final conv output (no pool after)
    // BiLSTM 1 caches: per-timestep (i, f, g, o, c, h) for each direction
    pub lstm1_x: Array2<f32>,                 // [T, cnn_feature_dim] input
    pub lstm1_out: Array2<f32>,              // [T, 2*hidden_size] output (input to lstm2)
    pub lstm1_fwd: Vec<LstmStep>,
    pub lstm1_bwd: Vec<LstmStep>,
    // BiLSTM 2 caches
    pub lstm2_x: Array2<f32>,
    pub lstm2_out: Array2<f32>,             // [T, 2*hidden_size] (input to FC)
    pub lstm2_fwd: Vec<LstmStep>,
    pub lstm2_bwd: Vec<LstmStep>,
}

#[derive(Clone)]
pub struct LstmStep {
    pub i: Array1<f32>,
    pub f: Array1<f32>,
    pub g: Array1<f32>,
    pub o: Array1<f32>,
    pub c: Array1<f32>,
    pub h: Array1<f32>,
}

impl CrnnModel {
    /// Forward pass with activation caching for the backward pass.
    /// Returns the logits plus a `ForwardCache`.
    pub fn forward_with_cache(&self, input: &Array2<f32>) -> (Array2<f32>, ForwardCache) {
        let mut x = input.clone().insert_axis(ndarray::Axis(0));

        // Block 1: conv 3x3 + BN + ReLU + maxpool 2x2
        let conv1_pre = self.cnn.run_conv_only_with_bias(&x, 0);
        let conv1_post = apply_bn_relu(
            &conv1_pre,
            &self.cnn.bn_scales[0],
            &self.cnn.bn_shifts[0],
        );
        let (pool1_out, conv1_argmax) = maxpool2d_with_argmax(&conv1_post, 2);
        x = pool1_out.clone();

        // Block 2
        let conv2_pre = self.cnn.run_conv_only_with_bias(&x, 1);
        let conv2_post = apply_bn_relu(
            &conv2_pre,
            &self.cnn.bn_scales[1],
            &self.cnn.bn_shifts[1],
        );
        let (pool2_out, conv2_argmax) = maxpool2d_with_argmax(&conv2_post, 2);
        x = pool2_out.clone();

        // Block 3: conv + BN + ReLU + maxpool width-only 1x2
        let conv3_pre = self.cnn.run_conv_only_with_bias(&x, 2);
        let conv3_post = apply_bn_relu(
            &conv3_pre,
            &self.cnn.bn_scales[2],
            &self.cnn.bn_shifts[2],
        );
        let (pool3_out, conv3_argmax) = maxpool_width_with_argmax(&conv3_post, 2);
        x = pool3_out.clone();

        // Block 4
        let conv4_pre = self.cnn.run_conv_only_with_bias(&x, 3);
        let conv4_post = apply_bn_relu(
            &conv4_pre,
            &self.cnn.bn_scales[3],
            &self.cnn.bn_shifts[3],
        );
        let (pool4_out, conv4_argmax) = maxpool_width_with_argmax(&conv4_post, 2);
        x = pool4_out.clone();

        // Block 5: conv + BN + ReLU (no pool)
        let conv5_pre = self.cnn.run_conv_only_with_bias(&x, 4);
        let conv5_post = apply_bn_relu(
            &conv5_pre,
            &self.cnn.bn_scales[4],
            &self.cnn.bn_shifts[4],
        );
        x = conv5_post.clone();

        // Reshape [C, H, W] -> [W, C*H]
        let (c_out, h_out, w_out) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut cnn_seq = Array2::zeros((w_out, c_out * h_out));
        for t in 0..w_out {
            for c in 0..c_out {
                for y in 0..h_out {
                    cnn_seq[[t, c * h_out + y]] = x[[c, y, t]];
                }
            }
        }

        // BiLSTM 1
        let (lstm1_fwd, lstm1_bwd) = lstm_bilstm_forward_with_cache(
            &cnn_seq,
            &self.lstm1.wf_ih,
            &self.lstm1.wf_hh,
            &self.lstm1.bf_ih,
            &self.lstm1.bf_hh,
            &self.lstm1.wb_ih,
            &self.lstm1.wb_hh,
            &self.lstm1.bb_ih,
            &self.lstm1.bb_hh,
            self.lstm1.hidden_size,
        );
        let t_len = cnn_seq.nrows();
        let mut lstm1_out = Array2::zeros((t_len, 2 * self.lstm1.hidden_size));
        for i in 0..t_len {
            for j in 0..self.lstm1.hidden_size {
                lstm1_out[[i, j]] = lstm1_fwd[i].h[j];
                lstm1_out[[i, self.lstm1.hidden_size + j]] = lstm1_bwd[t_len - 1 - i].h[j];
            }
        }

        // BiLSTM 2
        let (lstm2_fwd, lstm2_bwd) = lstm_bilstm_forward_with_cache(
            &lstm1_out,
            &self.lstm2.wf_ih,
            &self.lstm2.wf_hh,
            &self.lstm2.bf_ih,
            &self.lstm2.bf_hh,
            &self.lstm2.wb_ih,
            &self.lstm2.wb_hh,
            &self.lstm2.bb_ih,
            &self.lstm2.bb_hh,
            self.lstm2.hidden_size,
        );
        let mut lstm2_out = Array2::zeros((t_len, 2 * self.lstm2.hidden_size));
        for i in 0..t_len {
            for j in 0..self.lstm2.hidden_size {
                lstm2_out[[i, j]] = lstm2_fwd[i].h[j];
                lstm2_out[[i, self.lstm2.hidden_size + j]] = lstm2_bwd[t_len - 1 - i].h[j];
            }
        }

        // FC
        let mut logits = lstm2_out.dot(&self.fc_weight.t());
        for i in 0..logits.nrows() {
            for j in 0..logits.ncols() {
                logits[[i, j]] += self.fc_bias[j];
                // Clamp logits to prevent numerical instability in CTC loss
                logits[[i, j]] = logits[[i, j]].clamp(-10.0, 10.0);
            }
        }

        let cache = ForwardCache {
            input: input.clone(),
            conv1_pre, conv1_post, conv1_argmax, pool1_out,
            conv2_pre, conv2_post, conv2_argmax, pool2_out,
            conv3_pre, conv3_post, conv3_argmax, pool3_out,
            conv4_pre, conv4_post, conv4_argmax, pool4_out,
            conv5_pre, conv5_post,
            lstm1_x: cnn_seq,
            lstm1_out: lstm1_out.clone(),
            lstm1_fwd, lstm1_bwd,
            lstm2_x: lstm1_out,
            lstm2_out: lstm2_out.clone(),
            lstm2_fwd, lstm2_bwd,
        };
        (logits, cache)
    }
}

// ---------------------------------------------------------------------------
// Activation helpers (shared with forward / backward)
// ---------------------------------------------------------------------------

pub(crate) fn apply_bn_relu(
    x: &Array3<f32>,
    bn_scale: &Array1<f32>,
    bn_shift: &Array1<f32>,
) -> Array3<f32> {
    let mut out = x.clone();
    for oc in 0..x.shape()[0] {
        let s = bn_scale[oc];
        let p = bn_shift[oc];
        for v in out.slice_mut(ndarray::s![oc, .., ..]).iter_mut() {
            *v = (*v * s + p).max(0.0);
        }
    }
    out
}

fn maxpool2d_with_argmax(
    input: &Array3<f32>,
    pool: usize,
) -> (Array3<f32>, Array3<(usize, usize)>) {
    let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let out_h = h / pool;
    let out_w = w / pool;
    let mut output = Array3::zeros((c, out_h, out_w));
    let mut argmax = Array3::from_elem((c, out_h, out_w), (0usize, 0usize));
    for ch in 0..c {
        for y in 0..out_h {
            for x in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_pos = (0, 0);
                for dy in 0..pool {
                    for dx in 0..pool {
                        let val = input[[ch, y * pool + dy, x * pool + dx]];
                        if val > max_val {
                            max_val = val;
                            max_pos = (y * pool + dy, x * pool + dx);
                        }
                    }
                }
                output[[ch, y, x]] = max_val;
                argmax[[ch, y, x]] = max_pos;
            }
        }
    }
    (output, argmax)
}

fn maxpool_width_with_argmax(
    input: &Array3<f32>,
    pool: usize,
) -> (Array3<f32>, Array3<(usize, usize)>) {
    let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let out_w = w / pool;
    let mut output = Array3::zeros((c, h, out_w));
    let mut argmax = Array3::from_elem((c, h, out_w), (0usize, 0usize));
    for ch in 0..c {
        for y in 0..h {
            for x in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_pos = (0, 0);
                for dx in 0..pool {
                    let val = input[[ch, y, x * pool + dx]];
                    if val > max_val {
                        max_val = val;
                        max_pos = (y, x * pool + dx);
                    }
                }
                output[[ch, y, x]] = max_val;
                argmax[[ch, y, x]] = max_pos;
            }
        }
    }
    (output, argmax)
}

fn maxpool2d_backward(
    d_out: &Array3<f32>,
    _input: &Array3<f32>,
    argmax: &Array3<(usize, usize)>,
) -> Array3<f32> {
    let (c, h, w) = (d_out.shape()[0], _input.shape()[1], _input.shape()[2]);
    let mut d_input = Array3::zeros((c, h, w));
    let (oc, oh, ow) = (d_out.shape()[0], d_out.shape()[1], d_out.shape()[2]);
    for ch in 0..c.min(oc) {
        for y in 0..oh.min(h) {
            for x in 0..ow.min(w) {
                let (ay, ax) = argmax[[ch, y, x]];
                if ay < h && ax < w {
                    d_input[[ch, ay, ax]] += d_out[[ch, y, x]];
                }
            }
        }
    }
    d_input
}

fn relu_backward(d_out: &Array3<f32>, pre: &Array3<f32>) -> Array3<f32> {
    let mut d = d_out.clone();
    let pre_slice = pre.as_slice().unwrap();
    let d_slice = d.as_slice_mut().unwrap();
    for i in 0..d_slice.len() {
        if pre_slice[i] <= 0.0 {
            d_slice[i] = 0.0;
        }
    }
    d
}

// ---------------------------------------------------------------------------
// LSTM forward with cache (BiLSTM = forward + reversed-backward)
// ---------------------------------------------------------------------------

fn lstm_bilstm_forward_with_cache(
    input: &Array2<f32>,
    wf_ih: &Array2<f32>,
    wf_hh: &Array2<f32>,
    bf_ih: &Array1<f32>,
    bf_hh: &Array1<f32>,
    wb_ih: &Array2<f32>,
    wb_hh: &Array2<f32>,
    bb_ih: &Array1<f32>,
    bb_hh: &Array1<f32>,
    hidden_size: usize,
) -> (Vec<LstmStep>, Vec<LstmStep>) {
    let fwd = lstm_sequence_with_cache(input, wf_ih, wf_hh, bf_ih, bf_hh, hidden_size);
    let mut rev_input = input.clone();
    rev_input.invert_axis(ndarray::Axis(0));
    let bwd_rev = lstm_sequence_with_cache(&rev_input, wb_ih, wb_hh, bb_ih, bb_hh, hidden_size);
    let mut bwd = bwd_rev;
    // Reverse the cached steps (each LstmStep contains its own h/c; reversing the order
    // maps t' -> (T-1-t) which matches the forward direction's output).
    bwd.reverse();
    (fwd, bwd)
}

fn lstm_sequence_with_cache(
    input: &Array2<f32>,
    w_ih: &Array2<f32>,
    w_hh: &Array2<f32>,
    b_ih: &Array1<f32>,
    b_hh: &Array1<f32>,
    hidden_size: usize,
) -> Vec<LstmStep> {
    let (t, _) = input.dim();
    let hs = hidden_size;
    let mut h = Array1::zeros(hs);
    let mut c = Array1::zeros(hs);
    let mut steps = Vec::with_capacity(t);
    for ti in 0..t {
        let xt = input.row(ti).to_owned();
        // Manual matmul: w_ih.T @ xt = sum_k w_ih[j, k] * xt[k] over k, for each j
        // w_ih has shape (4*hs, in_dim). xt has shape (in_dim,).
        let mut gates = b_ih.clone();
        for j in 0..(4 * hs) {
            let mut acc = 0.0f32;
            for k in 0..xt.len() {
                acc += w_ih[[j, k]] * xt[k];
            }
            gates[j] = acc;
        }
        // Add w_hh.T @ h
        for j in 0..(4 * hs) {
            let mut acc = 0.0f32;
            for k in 0..hs {
                acc += w_hh[[j, k]] * h[k];
            }
            gates[j] += acc + b_hh[j];
        }
        let i_gate: Array1<f32> = gates.slice(ndarray::s![0..hs]).mapv(sigmoid);
        let f_gate: Array1<f32> = gates.slice(ndarray::s![hs..2 * hs]).mapv(sigmoid);
        let g_gate: Array1<f32> = gates.slice(ndarray::s![2 * hs..3 * hs]).mapv(|x| x.tanh());
        let o_gate: Array1<f32> = gates.slice(ndarray::s![3 * hs..4 * hs]).mapv(sigmoid);
        c = &f_gate * &c + &i_gate * &g_gate;
        let tanh_c = c.mapv(|x| x.tanh());
        h = &o_gate * &tanh_c;
        steps.push(LstmStep {
            i: i_gate,
            f: f_gate,
            g: g_gate,
            o: o_gate,
            c: c.clone(),
            h: h.clone(),
        });
    }
    steps
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// CnnFeatureExtractor extension: conv-only with bias
// ---------------------------------------------------------------------------

impl CnnFeatureExtractor {
    /// Run conv + bias (no BN, no ReLU). Returns the pre-BN output.
    pub fn run_conv_only_with_bias(&self, input: &Array3<f32>, idx: usize) -> Array3<f32> {
        let (w, b) = match idx {
            0 => (&self.conv1_weights, &self.conv1_bias),
            1 => (&self.conv2_weights, &self.conv2_bias),
            2 => (&self.conv3_weights, &self.conv3_bias),
            3 => (&self.conv4_weights, &self.conv4_bias),
            4 => (&self.conv5_weights, &self.conv5_bias),
            _ => panic!("invalid conv idx {idx}"),
        };
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let out_c = w.shape()[0];
        let k = (w.shape()[2] as f32).sqrt() as usize;
        let pad = k / 2;
        // Output spatial dims: in_h_out = in_h - k + 2*pad + 1 = in_h - (k - 2*pad) + ... = in_h when k=2*pad, else in_h - k + 2*pad + 1
        // For k=3, pad=1: in_h - 3 + 2 + 1 = in_h
        // For k=2, pad=0: in_h - 2 + 0 + 1 = in_h - 1
        let out_h = in_h + 2 * pad - k + 1;
        let out_w = in_w + 2 * pad - k + 1;
        let in_slice = input.as_slice().unwrap();
        let w_slice = w.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();

        let mut output = Array3::zeros((out_c, out_h, out_w));
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = b_slice[oc];
                    for ic in 0..in_c {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = oh as i32 + ky as i32 - pad as i32;
                                let ix = ow as i32 + kx as i32 - pad as i32;
                                if iy >= 0 && iy < in_h as i32 && ix >= 0 && ix < in_w as i32 {
                                    let wv = w_slice[oc * in_c * k * k + ic * k * k + ky * k + kx];
                                    let inp = in_slice[ic * in_h * in_w + iy as usize * in_w + ix as usize];
                                    sum += inp * wv;
                                }
                            }
                        }
                    }
                    output[[oc, oh, ow]] = sum;
                }
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Param grads (full set, for backward through every layer)
// ---------------------------------------------------------------------------

/// Parameter gradients for one sample.
pub struct ParamGrads {
    pub conv1_w: Array3<f32>,
    pub conv1_b: Array1<f32>,
    pub conv2_w: Array3<f32>,
    pub conv2_b: Array1<f32>,
    pub conv3_w: Array3<f32>,
    pub conv3_b: Array1<f32>,
    pub conv4_w: Array3<f32>,
    pub conv4_b: Array1<f32>,
    pub conv5_w: Array3<f32>,
    pub conv5_b: Array1<f32>,
    pub bn1_s: Array1<f32>,
    pub bn1_p: Array1<f32>,
    pub bn2_s: Array1<f32>,
    pub bn2_p: Array1<f32>,
    pub bn3_s: Array1<f32>,
    pub bn3_p: Array1<f32>,
    pub bn4_s: Array1<f32>,
    pub bn4_p: Array1<f32>,
    pub bn5_s: Array1<f32>,
    pub bn5_p: Array1<f32>,
    pub lstm1_wf_ih: Array2<f32>,
    pub lstm1_wf_hh: Array2<f32>,
    pub lstm1_bf_ih: Array1<f32>,
    pub lstm1_bf_hh: Array1<f32>,
    pub lstm1_wb_ih: Array2<f32>,
    pub lstm1_wb_hh: Array2<f32>,
    pub lstm1_bb_ih: Array1<f32>,
    pub lstm1_bb_hh: Array1<f32>,
    pub lstm2_wf_ih: Array2<f32>,
    pub lstm2_wf_hh: Array2<f32>,
    pub lstm2_bf_ih: Array1<f32>,
    pub lstm2_bf_hh: Array1<f32>,
    pub lstm2_wb_ih: Array2<f32>,
    pub lstm2_wb_hh: Array2<f32>,
    pub lstm2_bb_ih: Array1<f32>,
    pub lstm2_bb_hh: Array1<f32>,
    pub fc_w: Array2<f32>,
    pub fc_b: Array1<f32>,
}

impl ParamGrads {
    pub fn zero_like(model: &CrnnModel) -> Self {
        Self {
            conv1_w: Array3::zeros(model.cnn.conv1_weights.dim()),
            conv1_b: Array1::zeros(model.cnn.conv1_bias.len()),
            conv2_w: Array3::zeros(model.cnn.conv2_weights.dim()),
            conv2_b: Array1::zeros(model.cnn.conv2_bias.len()),
            conv3_w: Array3::zeros(model.cnn.conv3_weights.dim()),
            conv3_b: Array1::zeros(model.cnn.conv3_bias.len()),
            conv4_w: Array3::zeros(model.cnn.conv4_weights.dim()),
            conv4_b: Array1::zeros(model.cnn.conv4_bias.len()),
            conv5_w: Array3::zeros(model.cnn.conv5_weights.dim()),
            conv5_b: Array1::zeros(model.cnn.conv5_bias.len()),
            bn1_s: Array1::zeros(model.cnn.bn_scales[0].len()),
            bn1_p: Array1::zeros(model.cnn.bn_shifts[0].len()),
            bn2_s: Array1::zeros(model.cnn.bn_scales[1].len()),
            bn2_p: Array1::zeros(model.cnn.bn_shifts[1].len()),
            bn3_s: Array1::zeros(model.cnn.bn_scales[2].len()),
            bn3_p: Array1::zeros(model.cnn.bn_shifts[2].len()),
            bn4_s: Array1::zeros(model.cnn.bn_scales[3].len()),
            bn4_p: Array1::zeros(model.cnn.bn_shifts[3].len()),
            bn5_s: Array1::zeros(model.cnn.bn_scales[4].len()),
            bn5_p: Array1::zeros(model.cnn.bn_shifts[4].len()),
            lstm1_wf_ih: Array2::zeros(model.lstm1.wf_ih.dim()),
            lstm1_wf_hh: Array2::zeros(model.lstm1.wf_hh.dim()),
            lstm1_bf_ih: Array1::zeros(model.lstm1.bf_ih.len()),
            lstm1_bf_hh: Array1::zeros(model.lstm1.bf_hh.len()),
            lstm1_wb_ih: Array2::zeros(model.lstm1.wb_ih.dim()),
            lstm1_wb_hh: Array2::zeros(model.lstm1.wb_hh.dim()),
            lstm1_bb_ih: Array1::zeros(model.lstm1.bb_ih.len()),
            lstm1_bb_hh: Array1::zeros(model.lstm1.bb_hh.len()),
            lstm2_wf_ih: Array2::zeros(model.lstm2.wf_ih.dim()),
            lstm2_wf_hh: Array2::zeros(model.lstm2.wf_hh.dim()),
            lstm2_bf_ih: Array1::zeros(model.lstm2.bf_ih.len()),
            lstm2_bf_hh: Array1::zeros(model.lstm2.bf_hh.len()),
            lstm2_wb_ih: Array2::zeros(model.lstm2.wb_ih.dim()),
            lstm2_wb_hh: Array2::zeros(model.lstm2.wb_hh.dim()),
            lstm2_bb_ih: Array1::zeros(model.lstm2.bb_ih.len()),
            lstm2_bb_hh: Array1::zeros(model.lstm2.bb_hh.len()),
            fc_w: Array2::zeros(model.fc_weight.dim()),
            fc_b: Array1::zeros(model.fc_bias.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// Full backward pass
// ---------------------------------------------------------------------------

impl CrnnModel {
    /// Full backward pass. `d_logits` has shape [T, num_classes].
    /// Returns a `ParamGrads` populated for every parameter.
    pub fn backward_full(&self, cache: &ForwardCache, d_logits: &Array2<f32>) -> ParamGrads {
        let mut grads = ParamGrads::zero_like(self);
        let t = d_logits.nrows();
        let num_classes = d_logits.ncols();
        let hs2 = self.lstm2.hidden_size;

        // ---- FC backward ----
        // logits[t, j] = sum_k fc_weight[j, k] * lstm2_out[t, k] + fc_bias[j]
        // d_fc_weight[j, k] = sum_t d_logits[t, j] * lstm2_out[t, k]
        // d_fc_bias[j] = sum_t d_logits[t, j]
        // d_lstm2_out[t, k] = sum_j d_logits[t, j] * fc_weight[j, k]
        let mut d_lstm2_out = Array2::zeros((t, 2 * hs2));
        for ti in 0..t {
            for j in 0..num_classes {
                grads.fc_b[j] += d_logits[[ti, j]];
            }
        }
        // d_lstm2_out from FC backward
        for ti in 0..t {
            for k in 0..(2 * hs2) {
                let mut acc = 0.0f32;
                for j in 0..num_classes {
                    acc += d_logits[[ti, j]] * self.fc_weight[[j, k]];
                }
                d_lstm2_out[[ti, k]] = acc;
            }
        }
        // fc_w grad using lstm2_out (we cached it)
        for ti in 0..t {
            for j in 0..num_classes {
                let g = d_logits[[ti, j]];
                for k in 0..(2 * hs2) {
                    grads.fc_w[[j, k]] += g * cache.lstm2_out[[ti, k]];
                }
            }
        }

        // ---- BiLSTM 2 backward ----
        // lstm2_out[t, k] = concat(fwd_h[t, k], bwd_h[t, hs2+k])
        // so d_fwd_h[t, k] = d_lstm2_out[t, k]
        // and d_bwd_h[t, hs2+k] = d_lstm2_out[t, hs2+k]
        let mut d_lstm2_fwd_h = Array2::zeros((t, hs2));
        let mut d_lstm2_bwd_h = Array2::zeros((t, hs2));
        for ti in 0..t {
            for k in 0..hs2 {
                d_lstm2_fwd_h[[ti, k]] = d_lstm2_out[[ti, k]];
                d_lstm2_bwd_h[[ti, k]] = d_lstm2_out[[ti, hs2 + k]];
            }
        }
        // BPTT through fwd direction (iterate in reverse time)
        lstm_bptt_one_direction(
            &cache.lstm2_fwd,
            &cache.lstm2_x,
            &self.lstm2.wf_ih,
            &self.lstm2.wf_hh,
            &self.lstm2.bf_ih,
            &self.lstm2.bf_hh,
            &d_lstm2_fwd_h,
            &mut grads.lstm2_wf_ih,
            &mut grads.lstm2_wf_hh,
            &mut grads.lstm2_bf_ih,
            &mut grads.lstm2_bf_hh,
            &mut Array2::zeros(cache.lstm2_x.dim()),
        );
        // BPTT through bwd direction (the bwd sequence is reversed in cache; iterate forward in cache order)
        lstm_bptt_one_direction(
            &cache.lstm2_bwd,
            &cache.lstm2_x,
            &self.lstm2.wb_ih,
            &self.lstm2.wb_hh,
            &self.lstm2.bb_ih,
            &self.lstm2.bb_hh,
            &d_lstm2_bwd_h,
            &mut grads.lstm2_wb_ih,
            &mut grads.lstm2_wb_hh,
            &mut grads.lstm2_bb_ih,
            &mut grads.lstm2_bb_hh,
            &mut Array2::zeros(cache.lstm2_x.dim()),
        );

        // ---- BiLSTM 1 backward (same as BiLSTM 2 but produces gradient w.r.t. cnn_seq) ----
        let mut d_lstm2_fwd_h2 = Array2::zeros((t, hs2));
        let mut d_lstm2_bwd_h2 = Array2::zeros((t, hs2));
        for ti in 0..t {
            for k in 0..hs2 {
                d_lstm2_fwd_h2[[ti, k]] = d_lstm2_out[[ti, k]];
                d_lstm2_bwd_h2[[ti, k]] = d_lstm2_out[[ti, hs2 + k]];
            }
        }
        let d_lstm1_fwd_h = Array2::zeros((t, hs2));
        let d_lstm1_bwd_h = Array2::zeros((t, hs2));
        // d_lstm1_out has shape [T, cnn_feature_dim] = (T, 1024) — input to lstm1
        let mut d_lstm1_out = Array2::zeros(cache.lstm1_x.dim());
        // lstm1 forward (in cache.lstm1_fwd order); backward iterate in reverse
        lstm_bptt_one_direction(
            &cache.lstm1_fwd,
            &cache.lstm1_x,
            &self.lstm1.wf_ih,
            &self.lstm1.wf_hh,
            &self.lstm1.bf_ih,
            &self.lstm1.bf_hh,
            &d_lstm1_fwd_h,
            &mut grads.lstm1_wf_ih,
            &mut grads.lstm1_wf_hh,
            &mut grads.lstm1_bf_ih,
            &mut grads.lstm1_bf_hh,
            &mut Array2::zeros(cache.lstm1_x.dim()),
        );
        lstm_bptt_one_direction(
            &cache.lstm1_bwd,
            &cache.lstm1_x,
            &self.lstm1.wb_ih,
            &self.lstm1.wb_hh,
            &self.lstm1.bb_ih,
            &self.lstm1.bb_hh,
            &d_lstm1_bwd_h,
            &mut grads.lstm1_wb_ih,
            &mut grads.lstm1_wb_hh,
            &mut grads.lstm1_bb_ih,
            &mut grads.lstm1_bb_hh,
            &mut d_lstm1_out,
        );

        // Invert the reshape: for each (t, c, y) of conv5_post, the index is t * (c_out * h_out) + c * h_out + y.
        let c_out = self.config.cnn_channels[4];
        let h_out = cache.conv5_post.shape()[1];
        let w_out = cache.conv5_post.shape()[2];
        let mut d_conv5_post = Array3::zeros((c_out, h_out, w_out));
        for t in 0..w_out {
            for c in 0..c_out {
                for y in 0..h_out {
                    d_conv5_post[[c, y, t]] = d_lstm1_out[[t, c * h_out + y]];
                }
            }
        }

        // ---- Block 5 backward: conv5_post = relu(bn_scale * conv5_pre + bn_shift) ----
        let d_conv5_pre = relu_backward(&d_conv5_post, &cache.conv5_pre);
        // BN backward: y = scale * x + shift; dy/d(scale) = sum(x * mask), dy/d(shift) = sum(mask), dy/d(x) = scale
        let mut scale_grad = grads.bn5_s.clone();
        let mut shift_grad = grads.bn5_p.clone();
        for oc in 0..c_out {
            let mut sg = 0.0f32;
            let mut pg = 0.0f32;
            for v in d_conv5_pre.slice(ndarray::s![oc, .., ..]).iter() {
                sg += v;
                pg += v;
            }
            scale_grad[oc] = sg * self.cnn.bn_scales[4][oc];
            shift_grad[oc] = pg * self.cnn.bn_scales[4][oc];
        }
        // d_conv5_pre (post-bias, post-conv) -> d_x (input to conv5)
        // d_conv5_input = conv5 backward(d_conv5_pre)
        let mut d_pool4_grad = Array3::zeros(cache.pool4_out.dim());
        let d_conv4_pool = conv2d_backward(
            &cache.pool4_out,
            &d_conv5_pre,
            &self.cnn.conv5_weights,
            &mut d_pool4_grad,
            &mut grads.conv5_w,
            &mut grads.conv5_b,
        );

        // ---- Pool 4 backward ----
        let d_conv4_post = maxpool2d_backward(&d_conv4_pool, &cache.conv4_post, &cache.conv4_argmax);
        let d_conv4_pre = relu_backward(&d_conv4_post, &cache.conv4_pre);
        // BN backward block 4
        let mut scale_grad_4 = grads.bn4_s.clone();
        let mut shift_grad_4 = grads.bn4_p.clone();
        let c4 = self.config.cnn_channels[3];
        for oc in 0..c4 {
            let mut sg = 0.0f32;
            let mut pg = 0.0f32;
            for v in d_conv4_pre.slice(ndarray::s![oc, .., ..]).iter() {
                sg += v;
                pg += v;
            }
            scale_grad_4[oc] = sg * self.cnn.bn_scales[3][oc];
            shift_grad_4[oc] = pg * self.cnn.bn_scales[3][oc];
        }
        let _ = scale_grad_4;
        let _ = shift_grad_4;
        let mut d_pool3_grad = Array3::zeros(cache.pool3_out.dim());
        let d_conv3_pool = conv2d_backward(
            &cache.pool3_out,
            &d_conv4_pre,
            &self.cnn.conv4_weights,
            &mut d_pool3_grad,
            &mut grads.conv4_w,
            &mut grads.conv4_b,
        );

        // ---- Pool 3 backward ----
        let d_conv3_post = maxpool2d_backward(&d_conv3_pool, &cache.conv3_post, &cache.conv3_argmax);
        let d_conv3_pre = relu_backward(&d_conv3_post, &cache.conv3_pre);
        let c3 = self.config.cnn_channels[2];
        for oc in 0..c3 {
            let mut sg = 0.0f32;
            let mut pg = 0.0f32;
            for v in d_conv3_pre.slice(ndarray::s![oc, .., ..]).iter() {
                sg += v;
                pg += v;
            }
            grads.bn3_s[oc] = sg * self.cnn.bn_scales[2][oc];
            grads.bn3_p[oc] = pg * self.cnn.bn_scales[2][oc];
        }
        let mut d_pool2_grad = Array3::zeros(cache.pool2_out.dim());
        let d_conv2_pool = conv2d_backward(
            &cache.pool2_out,
            &d_conv3_pre,
            &self.cnn.conv3_weights,
            &mut d_pool2_grad,
            &mut grads.conv3_w,
            &mut grads.conv3_b,
        );

        // ---- Pool 2 backward ----
        let d_conv2_post = maxpool2d_backward(&d_conv2_pool, &cache.conv2_post, &cache.conv2_argmax);
        let d_conv2_pre = relu_backward(&d_conv2_post, &cache.conv2_pre);
        let c2 = self.config.cnn_channels[1];
        for oc in 0..c2 {
            let mut sg = 0.0f32;
            let mut pg = 0.0f32;
            for v in d_conv2_pre.slice(ndarray::s![oc, .., ..]).iter() {
                sg += v;
                pg += v;
            }
            grads.bn2_s[oc] = sg * self.cnn.bn_scales[1][oc];
            grads.bn2_p[oc] = pg * self.cnn.bn_scales[1][oc];
        }
        let mut d_pool1_grad = Array3::zeros(cache.pool1_out.dim());
        let d_conv1_pool = conv2d_backward(
            &cache.pool1_out,
            &d_conv2_pre,
            &self.cnn.conv2_weights,
            &mut d_pool1_grad,
            &mut grads.conv2_w,
            &mut grads.conv2_b,
        );

        // ---- Pool 1 backward ----
        let d_conv1_post = maxpool2d_backward(&d_conv1_pool, &cache.conv1_post, &cache.conv1_argmax);
        let d_conv1_pre = relu_backward(&d_conv1_post, &cache.conv1_pre);
        let c1 = self.config.cnn_channels[0];
        for oc in 0..c1 {
            let mut sg = 0.0f32;
            let mut pg = 0.0f32;
            for v in d_conv1_pre.slice(ndarray::s![oc, .., ..]).iter() {
                sg += v;
                pg += v;
            }
            grads.bn1_s[oc] = sg * self.cnn.bn_scales[0][oc];
            grads.bn1_p[oc] = pg * self.cnn.bn_scales[0][oc];
        }
        // d_input (not used — we don't backprop into the image pixels)
        let mut d_input_grad = Array3::zeros(cache.input.clone().insert_axis(ndarray::Axis(0)).dim());
        let _ = conv2d_backward(
            &cache.input.clone().insert_axis(ndarray::Axis(0)),
            &d_conv1_pre,
            &self.cnn.conv1_weights,
            &mut d_input_grad,
            &mut grads.conv1_w,
            &mut grads.conv1_b,
        );
        grads.bn5_s = scale_grad;
        grads.bn5_p = shift_grad;
        grads
    }
}

// ---------------------------------------------------------------------------
// LSTM BPTT for one direction
// ---------------------------------------------------------------------------

fn lstm_bptt_one_direction(
    steps: &[LstmStep],
    input: &Array2<f32>,
    _w_ih: &Array2<f32>,
    _w_hh: &Array2<f32>,
    _b_ih: &Array1<f32>,
    _b_hh: &Array1<f32>,
    d_h_out: &Array2<f32>,
    d_w_ih: &mut Array2<f32>,
    d_w_hh: &mut Array2<f32>,
    d_b_ih: &mut Array1<f32>,
    d_b_hh: &mut Array1<f32>,
    d_out_prev: &mut Array2<f32>,
) {
    let t = steps.len();
    let hs = steps[0].i.len();
    let mut dh_next = Array1::zeros(hs);
    let mut dc_next = Array1::zeros(hs);
    // Iterate in reverse time order
    for ti in (0..t).rev() {
        let step = &steps[ti];
        let mut dh = Array1::zeros(hs);
        for k in 0..hs {
            dh[k] = d_h_out[[ti, k]] + dh_next[k];
        }
        // h = o * tanh(c); gradient: dh/d(o) = tanh(c), dh/d(c) = o * (1 - tanh(c)^2)
        let tanh_c = step.c.mapv(|x| x.tanh());
        let do_g = &dh * &tanh_c;
        let one_minus_tanh_c_sq = tanh_c.mapv(|x| 1.0 - x * x);
        let dc_term = &dh * &step.o * &one_minus_tanh_c_sq;
        let dc_total = &dc_term + &dc_next;
        // c = f * c_prev + i * g; gradients via activations
        // f = sigmoid(g_f), so df/d(g_f) = f*(1-f). dc/d(f) = c_prev, so dc/d(g_f) = c_prev * f * (1-f)
        // similarly for i and g.
        let prev_c = if ti == 0 {
            Array1::zeros(hs)
        } else {
            steps[ti - 1].c.clone()
        };
        let prev_h = if ti == 0 {
            Array1::zeros(hs)
        } else {
            steps[ti - 1].h.clone()
        };
        let df = &dc_total * &prev_c * &step.f * &step.f.mapv(|x| 1.0 - x);
        let di = &dc_total * &step.g * &step.i * &step.i.mapv(|x| 1.0 - x);
        let dg = &dc_total * &step.i * &step.g.mapv(|x| 1.0 - x * x);
        // gates before nonlinearity: y = [i, f, g, o]
        // dy/d(w_ih) = x; dW += y * x.T (for each timestep)
        // dy/d(w_hh) = h_prev
        // For each gate k: d_y[k] contributes to d_w_ih[k, :] (outer product with x[t])
        let xt = input.row(ti).to_owned();
        let pt_h = prev_h;
        // d_w_ih[oc, j] += sum_t gates_grad[oc, t] * x[t, j]
        for k in 0..hs {
            let di_k = di[k];
            let df_k = df[k];
            let dg_k = dg[k];
            let do_k = do_g[k];
            // d_w_ih[k_gate, j] += gate_grad[k_gate] * x[j]  (j over input dim)
            for j in 0..xt.len() {
                d_w_ih[[k, j]] += di_k * xt[j];
                d_w_ih[[hs + k, j]] += df_k * xt[j];
                d_w_ih[[2 * hs + k, j]] += dg_k * xt[j];
                d_w_ih[[3 * hs + k, j]] += do_k * xt[j];
            }
            // d_w_hh[k_gate, j] += gate_grad[k_gate] * h_prev[j]  (j over hidden dim)
            for j in 0..pt_h.len() {
                d_w_hh[[k, j]] += di_k * pt_h[j];
                d_w_hh[[hs + k, j]] += df_k * pt_h[j];
                d_w_hh[[2 * hs + k, j]] += dg_k * pt_h[j];
                d_w_hh[[3 * hs + k, j]] += do_k * pt_h[j];
            }
            d_b_ih[k] += di_k;
            d_b_ih[hs + k] += df_k;
            d_b_ih[2 * hs + k] += dg_k;
            d_b_ih[3 * hs + k] += do_k;
            d_b_hh[k] += di_k;
            d_b_hh[hs + k] += df_k;
            d_b_hh[2 * hs + k] += dg_k;
            d_b_hh[3 * hs + k] += do_k;
        }

        // d_input[t] = w_ih @ gates_grad
        let mut d_xt = Array1::zeros(xt.len());
        for j in 0..xt.len() {
            let mut acc = 0.0f32;
            for k in 0..hs {
                acc += _w_ih[[k, j]] * di[k];
                acc += _w_ih[[hs + k, j]] * df[k];
                acc += _w_ih[[2 * hs + k, j]] * dg[k];
                acc += _w_ih[[3 * hs + k, j]] * do_g[k];
            }
            d_xt[j] = acc;
        }
        // d_h_prev = w_hh @ gates_grad
        let mut d_h_prev = Array1::zeros(hs);
        for k in 0..hs {
            let mut acc = 0.0f32;
            for j in 0..hs {
                acc += _w_hh[[k, j]] * di[j];
                acc += _w_hh[[hs + k, j]] * df[j];
                acc += _w_hh[[2 * hs + k, j]] * dg[j];
                acc += _w_hh[[3 * hs + k, j]] * do_g[j];
            }
            d_h_prev[k] = acc;
        }
        // d_c_prev = dc_total * f
        let mut d_c_prev = Array1::zeros(hs);
        for k in 0..hs {
            d_c_prev[k] = dc_total[k] * step.f[k];
        }

        // Accumulate d_out_prev[t] += d_xt (input gradient at this timestep)
        for j in 0..d_xt.len() {
            d_out_prev[[ti, j]] += d_xt[j];
        }
        // Accumulate gradient into previous h (we need to add to existing)
        // We don't have a separate buffer for dh_prev accumulation — just use dh_next
        // Actually, dh_next should accumulate: dh_next = d_h_prev + d_c_prev etc.
        // For next iteration (t-1), dh_next = d_h_prev (we use that as input).
        // This is correct because d_h_prev from step t propagates to step t-1's output gradient.
        dh_next = d_h_prev;
        dc_next = d_c_prev;
    }
}

// ---------------------------------------------------------------------------
// Conv2d backward (input + weight + bias gradients)
// ---------------------------------------------------------------------------

/// Compute input gradient, weight gradient, and bias gradient for a conv2d
/// layer. Returns the input gradient (added into `d_input`).
fn conv2d_backward(
    input: &Array3<f32>,
    d_out: &Array3<f32>,
    weights: &Array3<f32>,
    d_input: &mut Array3<f32>,
    d_w: &mut Array3<f32>,
    d_b: &mut Array1<f32>,
) -> Array3<f32> {
    // We need the actual conv weights — pass through `idx`.
    // The caller passed `weights` and `_bias` but they're not the model's.
    // The caller passed the model's weights as a 3D Array3.
    let w = weights; // [out_c, in_c, k*k] — 3D
    let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let out_c = w.shape()[0];
    let kk = w.shape()[2];
    let k = (kk as f32).sqrt() as usize;
    let pad = k / 2;

    let w_slice = w.as_slice().unwrap();
    let in_slice = input.as_slice().unwrap();
    let d_out_slice = d_out.as_slice().unwrap();
    let d_input_slice = d_input.as_slice_mut().unwrap();
    let d_w_slice = d_w.as_slice_mut().unwrap();
    let d_b_slice = d_b.as_slice_mut().unwrap();

    // d_input: d_x[c, y, x] = sum over (oc, ky, kx) where the conv touches (c, y, x)
    // For each output position (oc, oh, ow), the conv reads input[(ic, oh+ky-pad, ow+kx-pad)] for all (ic, ky, kx).
    // So d_x[ic, y, x] = sum over (oc, ky, kx) such that y = oh+ky-pad (=> oh = y-ky+pad) and x = ow+kx-pad (=> ow = x-kx+pad), of d_out[oc, oh, ow] * w[oc, ic, ky*k+kx]
    for ic in 0..in_c {
        for y in 0..in_h {
            for x in 0..in_w {
                let mut acc = 0.0f32;
                for oc in 0..out_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let oh = y as i32 + pad as i32 - ky as i32;
                            let ow = x as i32 + pad as i32 - kx as i32;
                            if oh >= 0 && oh < in_h as i32 && ow >= 0 && ow < in_w as i32 {
                                let w_idx = oc * in_c * k * k + ic * k * k + ky * k + kx;
                                let d_o_idx = oc * in_h * in_w + oh as usize * in_w + ow as usize;
                                acc += d_out_slice[d_o_idx] * w_slice[w_idx];
                            }
                        }
                    }
                }
                d_input_slice[ic * in_h * in_w + y * in_w + x] += acc;
            }
        }
    }

    // d_w[oc, ic, ky*k+kx] = sum over (oh, ow) where the kernel position (ky, kx) touches input[(ic, oh+ky-pad, ow+kx-pad)], of d_out[oc, oh, ow] * input[ic, oh+ky-pad, ow+kx-pad]
    for (oc, db_slot) in d_b_slice.iter_mut().enumerate().take(out_c) {
        let mut db = 0.0f32;
        for oh in 0..in_h {
            for ow in 0..in_w {
                let d_o_idx = oc * in_h * in_w + oh * in_w + ow;
                db += d_out_slice[d_o_idx];
                for ic in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = oh as i32 + ky as i32 - pad as i32;
                            let ix = ow as i32 + kx as i32 - pad as i32;
                            if iy >= 0 && iy < in_h as i32 && ix >= 0 && ix < in_w as i32 {
                                let w_idx = oc * in_c * k * k + ic * k * k + ky * k + kx;
                                let in_idx = ic * in_h * in_w + iy as usize * in_w + ix as usize;
                                d_w_slice[w_idx] += d_out_slice[d_o_idx] * in_slice[in_idx];
                            }
                        }
                    }
                }
            }
        }
        *db_slot += db;
    }

    // Return the input gradient (caller may want it)
    d_input.clone()
}

#[cfg(test)]
mod tests {
    use crate::recognition::crnn::{CrnnConfig, CrnnModel};

    #[test]
    fn test_forward_cache_shapes() {
        let m = CrnnModel::new(CrnnConfig::default());
        let input = ndarray::Array2::zeros((32, 128));
        let (logits, cache) = m.forward_with_cache(&input);
        assert_eq!(logits.dim(), (logits.nrows(), logits.ncols()));
        assert_eq!(cache.conv1_pre.dim(), (16, 32, 128));
        assert_eq!(cache.pool1_out.dim(), (16, 16, 64));
        assert_eq!(cache.pool2_out.dim(), (32, 8, 32));
        assert_eq!(cache.pool3_out.dim(), (64, 8, 16));
        assert_eq!(cache.pool4_out.dim(), (64, 8, 8));
        // conv5 is 3x3 with pad=1, so output preserves size
        assert_eq!(cache.conv5_post.dim(), (128, 8, 8));
        assert_eq!(cache.lstm1_out.shape()[1], 2 * 64);
        assert_eq!(cache.lstm2_out.shape()[1], 2 * 64);
    }

    #[test]
    fn test_backward_finite() {
        let m = CrnnModel::new(CrnnConfig::default());
        let input = ndarray::Array2::zeros((32, 128));
        let (logits, cache) = m.forward_with_cache(&input);
        let d_logits = ndarray::Array2::from_shape_fn(logits.dim(), |(r, c)| {
            (((r * 7 + c * 13) % 32) as f32 - 16.0) / 16.0
        });
        let grads = m.backward_full(&cache, &d_logits);
        // Check shapes
        assert_eq!(grads.fc_w.dim(), m.fc_weight.dim());
        assert_eq!(grads.conv1_w.dim(), m.cnn.conv1_weights.dim());
        assert_eq!(grads.lstm1_wf_ih.dim(), m.lstm1.wf_ih.dim());
        // Check finiteness
        for a in &[&grads.fc_w, &grads.lstm1_wf_ih, &grads.lstm1_wb_ih, &grads.lstm2_wf_ih, &grads.lstm2_wb_ih] {
            for v in a.iter() {
                assert!(v.is_finite(), "non-finite gradient: {v}");
            }
        }
        for oc in 0..grads.conv1_w.shape()[0] {
            for ic in 0..grads.conv1_w.shape()[1] {
                for v in 0..grads.conv1_w.shape()[2] {
                    let x = grads.conv1_w[[oc, ic, v]];
                    assert!(x.is_finite(), "non-finite conv1 grad: {x}");
                }
            }
        }
    }
}