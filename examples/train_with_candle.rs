//! Train the bundled CRNN from scratch with `candle`.
//!
//! This example:
//! 1. Generates synthetic text-line images with the in-crate generator
//! 2. Defines the CRNN architecture in `candle` (matching the inference
//!    `CrnnModel` in `src/recognition/crnn.rs`)
//! 3. Trains with Adam + frame-level cross-entropy for `--epochs`
//! 4. Extracts trained weights, converts to our binary model format, and
//!    writes `models/crnn-en.bin`
//!
//! Run with:
//!   cargo run --release --example train_with_candle -- --epochs 30
//!   cargo run --release --example train_with_candle -- --epochs 30 \
//!       --samples 5000 --batch-size 16 --output models/crnn-en.bin
//!
//! The architecture matches the inference path exactly so the trained
//! weights can be loaded by `recognition::loader::load_bundled`.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Conv2d, Linear, Module, Optimizer, RNN, VarBuilder, VarMap};
use image::GrayImage;
use ocr::recognition::crnn::{CrnnConfig, Vocabulary};
use ocr::recognition::model_format;
use ocr::synthetic::{DistortionConfig, TextLineGenerator};

const CNN_CHANNELS: [usize; 5] = [16, 32, 64, 64, 128];
const HIDDEN_SIZE: usize = 64;
const NUM_LSTM_LAYERS: usize = 2;
const INPUT_HEIGHT: usize = 32;
const INPUT_WIDTH_DEFAULT: usize = 128;

struct Args {
    epochs: usize,
    samples: usize,
    batch_size: usize,
    learning_rate: f64,
    output: String,
    max_text_len: usize,
    width: usize,
    distortion: String,
    device: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        epochs: 20,
        samples: 3000,
        batch_size: 16,
        learning_rate: 1e-3,
        output: "models/crnn-en.bin".to_string(),
        max_text_len: 16,
        width: INPUT_WIDTH_DEFAULT,
        distortion: "mild".to_string(),
        device: "cpu".to_string(),
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--epochs" => a.epochs = iter.next().unwrap().parse().unwrap(),
            "--samples" => a.samples = iter.next().unwrap().parse().unwrap(),
            "--batch-size" => a.batch_size = iter.next().unwrap().parse().unwrap(),
            "--learning-rate" | "--lr" => {
                a.learning_rate = iter.next().unwrap().parse().unwrap()
            }
            "--output" | "-o" => a.output = iter.next().unwrap(),
            "--max-len" => a.max_text_len = iter.next().unwrap().parse().unwrap(),
            "--width" => a.width = iter.next().unwrap().parse().unwrap(),
            "--distortion" => a.distortion = iter.next().unwrap(),
            "--device" => a.device = iter.next().unwrap(),
            "--help" | "-h" => {
                println!("{}", USAGE);
                std::process::exit(0);
            }
            other => panic!("unknown flag: {other}"),
        }
    }
    a
}

const USAGE: &str = "\
Train the bundled CRNN from scratch.

Usage:
  cargo run --release --example train_with_candle -- [OPTIONS]

Options:
  --epochs <N>            training epochs                  [default: 20]
  --samples <N>           synthetic samples per epoch     [default: 3000]
  --batch-size <N>        batch size                      [default: 16]
  --lr, --learning-rate   Adam learning rate              [default: 1e-3]
  -o, --output <PATH>     output model path               [default: models/crnn-en.bin]
  --max-len <N>           max text length                 [default: 16]
  --width <N>             input width                     [default: 128]
  --distortion <LEVEL>    clean|mild|heavy                [default: mild]
  --device <DEV>          cpu                             [default: cpu]
";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args();
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        other => panic!("unsupported device: {other}"),
    };
    println!("device: {:?}", device);

    let vocab = Vocabulary::from_ascii();
    let num_classes = vocab.size();
    println!("vocab size: {} (incl. blank)", num_classes);

    // Build the CRNN architecture in candle.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = CrnnCandle::new(
        &vb,
        num_classes,
        args.width,
        INPUT_HEIGHT,
    )?;
    println!("parameters: {}", varmap.all_vars().len());

    // Generate a synthetic dataset.
    let mut generator = TextLineGenerator::default();
    let distortion = match args.distortion.as_str() {
        "clean" => DistortionConfig::none(),
        "mild" => DistortionConfig::mild(),
        "heavy" => DistortionConfig::heavy(),
        _ => DistortionConfig::mild(),
    };

    // Build the validation set once.
    let val_texts = generator.generate_random_texts(200, args.max_text_len);
    let val_samples = generator.generate_batch(&val_texts);
    crate::synthetic_run_augment(&mut val_samples.clone(), &distortion);
    let val_tensors = build_batch_tensors(&val_samples, args.width, INPUT_HEIGHT, &vocab, &device)?;
    let (val_images, val_targets, val_target_lengths, _, _) = val_tensors;

    // Adam optimizer.
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), args.learning_rate)?;

    let mut best_cer = f32::INFINITY;
    let mut best_step: Option<usize> = None;

    for epoch in 1..=args.epochs {
        let texts = generator.generate_random_texts(args.samples, args.max_text_len);
        let mut samples = generator.generate_batch(&texts);
        crate::synthetic_run_augment(&mut samples, &distortion);

        // Shuffle and iterate in mini-batches.
        let n = samples.len();
        let mut indices: Vec<usize> = (0..n).collect();
        // simple Fisher-Yates with a deterministic seed
        for i in (1..n).rev() {
            let j = (i * 1103515245 + 12345 + epoch) % (i + 1);
            indices.swap(i, j);
        }

        let mut total_loss = 0.0f64;
        let mut n_batches = 0usize;
        let mut total_correct = 0usize;
        let mut total_chars = 0usize;

        for batch_start in (0..n).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(n);
            let batch: Vec<_> = indices[batch_start..batch_end]
                .iter()
                .map(|&i| samples[i].clone())
                .collect();

            let (images, targets, target_lengths, input_lengths, _) =
                build_batch_tensors(&batch, args.width, INPUT_HEIGHT, &vocab, &device)?;

            // Forward
            let logits = model.forward(&images)?;
            // logits: [B, T, C]
            let log_probs = candle_nn::ops::log_softmax(&logits, 2)?;

            // Build a [B, T] target tensor: at each (b, t) put the target
            // character index (spread evenly over the sequence), or 0 (blank)
            // if the sample has no text.
            let (b, t, c) = (log_probs.dim(0)?, log_probs.dim(1)?, log_probs.dim(2)?);
            let mut tgt_data = vec![0u32; b * t];
            let mut mask_data = vec![0f32; b * t];
            for bi in 0..b {
                let tgt = &targets[bi];
                let tgt_len = tgt.len().min(t);
                if tgt_len == 0 {
                    continue;
                }
                for ti in 0..t {
                    let slot = (ti * tgt_len) / t;
                    if slot < tgt_len {
                        tgt_data[bi * t + ti] = tgt[slot];
                        mask_data[bi * t + ti] = 1.0;
                    }
                }
            }
            let target_t = Tensor::from_vec(tgt_data, (b, t), &device)?;
            let mask = Tensor::from_vec(mask_data, (b, t), &device)?;

            // Per-(b,t) cross-entropy: gather log_probs at the target index
            // for each (b, t), giving us a [B, T] loss tensor.
            let lp_flat = log_probs.reshape((b * t, c))?;
            let target_flat = target_t.reshape(b * t)?;
            // Compute NLL manually: gather the log_softmax values at the target
            // indices, then negate. This gives per-element loss of shape [B*T].
            let lp_softmax = candle_nn::ops::log_softmax(&lp_flat, 1)?;
            let target_u32 = target_flat.to_dtype(DType::U32)?;
            let gathered = lp_softmax.gather(&target_u32.unsqueeze(1)?, 1)?;
            let per_pos_loss = gathered.squeeze(1)?.neg()?; // [B*T]
            let per_pos_loss = per_pos_loss.reshape((b, t))?;
            let mask_sum = mask.sum_all()?.maximum(1e-8)?;
            let masked = (per_pos_loss * mask)?;
            let loss = (masked.sum_all()? / mask_sum)?;
            opt.backward_step(&loss)?;

            // Decode for accuracy
            let lp_data: Vec<f32> = log_probs.flatten_all()?.to_vec1()?;
            let mut flat_correct = 0usize;
            let mut flat_total = 0usize;
            for bi in 0..b {
                let tgt_len = target_lengths[bi].min(t);
                for ti in 0..t {
                    if ti * tgt_len / t >= tgt_len {
                        continue;
                    }
                    let target_idx = targets[bi][ti * tgt_len / t] as usize;
                    let mut best = 0usize;
                    let mut best_v = f32::NEG_INFINITY;
                    for ci in 0..c {
                        let v = lp_data[(bi * t + ti) * c + ci];
                        if v > best_v {
                            best_v = v;
                            best = ci;
                        }
                    }
                    if best == target_idx {
                        flat_correct += 1;
                    }
                    flat_total += 1;
                }
            }
            let avg_loss: f32 = loss.to_scalar()?;
            total_loss += avg_loss as f64;
            n_batches += 1;
            total_correct += flat_correct;
            total_chars += flat_total;
        }

        // Validation
        let val_logits = model.forward(&val_images)?;
        let val_pred = decode_logits_batch(&val_logits, &vocab);

        let mut cer_correct = 0usize;
        let mut cer_total = 0usize;
        for (i, sample) in val_samples.iter().enumerate() {
            let pred_text = &val_pred[i];
            let gt = &sample.ground_truth;
            let max_len = pred_text.chars().count().max(gt.chars().count());
            if max_len == 0 {
                continue;
            }
            let dist = lev(pred_text, gt);
            cer_correct += max_len - dist.min(max_len);
            cer_total += max_len;
        }
        let val_cer = 1.0 - cer_correct as f32 / cer_total.max(1) as f32;

        println!(
            "epoch {epoch:>3}/{epochs}  train_loss={tl:.4}  train_acc={ta:.2}%  val_cer={vc:.2}%  ({n_b} batches)",
            epochs = args.epochs,
            tl = total_loss / n_batches.max(1) as f64,
            ta = total_correct as f32 / total_chars.max(1) as f32 * 100.0,
            vc = val_cer * 100.0,
            n_b = n_batches,
        );

        if val_cer < best_cer {
            best_cer = val_cer;
            best_step = Some(epoch);
            // Save best so far
            if let Err(e) = save_model(&varmap, &vocab, &args.output, INPUT_HEIGHT, args.width) {
                eprintln!("save error: {e}");
            } else {
                println!("  -> saved {} (val_cer={:.2}%)", args.output, val_cer * 100.0);
            }
        }
    }

    if let Some(step) = best_step {
        println!("best val_cer {:.2}% at epoch {}", best_cer * 100.0, step);
    } else {
        // Save anyway
        save_model(&varmap, &vocab, &args.output, INPUT_HEIGHT, args.width)?;
        println!("saved final model to {}", args.output);
    }
    Ok(())
}

fn save_model(
    varmap: &VarMap,
    vocab: &Vocabulary,
    output: &str,
    input_height: usize,
    input_width: usize,
) -> Result<()> {
    // Read the trained parameters and convert them into our existing
    // CrnnModel + binary format. We do this by building a CrnnModel with
    // matching shape and copying weights in.
    let data = varmap.data().lock().unwrap();
    let fetch = |name: &str| -> Tensor {
        data.get(name)
            .unwrap_or_else(|| panic!("missing var: {name}"))
            .as_tensor()
            .clone()
    };
    let conv_w: Vec<Tensor> = (0..5).map(|i| fetch(&format!("conv{}.weight", i + 1))).collect();
    let conv_b: Vec<Tensor> = (0..5).map(|i| fetch(&format!("conv{}.bias", i + 1))).collect();

    let lstm1_w_ih = fetch("lstm1.weight_ih_l0");
    let lstm1_w_hh = fetch("lstm1.weight_hh_l0");
    let lstm1_b_ih = fetch("lstm1.bias_ih_l0");
    let lstm1_b_hh = fetch("lstm1.bias_hh_l0");
    let lstm2_w_ih = fetch("lstm2.weight_ih_l0");
    let lstm2_w_hh = fetch("lstm2.weight_hh_l0");
    let lstm2_b_ih = fetch("lstm2.bias_ih_l0");
    let lstm2_b_hh = fetch("lstm2.bias_hh_l0");

    let fc_w = fetch("fc.weight");
    let fc_b = fetch("fc.bias");

    // The trained model has the same architecture as our inference model.
    // Build an inference model with these weights and serialize it.
    let cfg = CrnnConfig {
        input_height,
        input_channels: 1,
        num_classes: vocab.size(),
        hidden_size: HIDDEN_SIZE,
        num_lstm_layers: NUM_LSTM_LAYERS,
        cnn_channels: CNN_CHANNELS.to_vec(),
        dropout: 0.0,
        use_beam_search: true,
        beam_width: 10,
        dict_weight: 1.0,
        lm_weight: 0.5,
        confidence_temperature: 1.0,
    };
    let mut model = ocr::recognition::crnn::CrnnModel::new(cfg);
    let _ = input_width;

    // Copy CNN weights
    for i in 0..5 {
        let w = conv_w[i].flatten_all()?.to_vec1::<f32>()?;
        let b = conv_b[i].to_vec1::<f32>()?;
        let (oc, ic, kk) = match i {
            0 => (16usize, 1usize, 9usize),
            1 => (32, 16, 9),
            2 => (64, 32, 9),
            3 => (64, 64, 9),
            4 => (128, 64, 4),
            _ => unreachable!(),
        };
        let arr = ndarray::Array3::from_shape_vec((oc, ic, kk), w).unwrap();
        let bias = ndarray::Array1::from_vec(b);
        match i {
            0 => {
                model.cnn.conv1_weights = arr;
                model.cnn.conv1_bias = bias;
            }
            1 => {
                model.cnn.conv2_weights = arr;
                model.cnn.conv2_bias = bias;
            }
            2 => {
                model.cnn.conv3_weights = arr;
                model.cnn.conv3_bias = bias;
            }
            3 => {
                model.cnn.conv4_weights = arr;
                model.cnn.conv4_bias = bias;
            }
            4 => {
                model.cnn.conv5_weights = arr;
                model.cnn.conv5_bias = bias;
            }
            _ => {}
        }
    }

    // Copy LSTM weights
    let lstm1_ih_v: Vec<f32> = lstm1_w_ih.flatten_all()?.to_vec1()?;
    let lstm1_ih = ndarray::Array2::from_shape_vec((4 * HIDDEN_SIZE, 896), lstm1_ih_v).unwrap();
    let lstm1_hh_v: Vec<f32> = lstm1_w_hh.flatten_all()?.to_vec1()?;
    let lstm1_hh = ndarray::Array2::from_shape_vec((4 * HIDDEN_SIZE, HIDDEN_SIZE), lstm1_hh_v).unwrap();
    let lstm1_bih = ndarray::Array1::from_vec(lstm1_b_ih.to_vec1::<f32>()?);
    let lstm1_bhh = ndarray::Array1::from_vec(lstm1_b_hh.to_vec1::<f32>()?);
    let lstm2_ih_v: Vec<f32> = lstm2_w_ih.flatten_all()?.to_vec1()?;
    let lstm2_ih = ndarray::Array2::from_shape_vec((4 * HIDDEN_SIZE, 2 * HIDDEN_SIZE), lstm2_ih_v).unwrap();
    let lstm2_hh_v: Vec<f32> = lstm2_w_hh.flatten_all()?.to_vec1()?;
    let lstm2_hh = ndarray::Array2::from_shape_vec((4 * HIDDEN_SIZE, HIDDEN_SIZE), lstm2_hh_v).unwrap();
    let lstm2_bih = ndarray::Array1::from_vec(lstm2_b_ih.to_vec1::<f32>()?);
    let lstm2_bhh = ndarray::Array1::from_vec(lstm2_b_hh.to_vec1::<f32>()?);
    model.lstm1.wf_ih = lstm1_ih.clone();
    model.lstm1.wf_hh = lstm1_hh.clone();
    model.lstm1.bf_ih = lstm1_bih.clone();
    model.lstm1.bf_hh = lstm1_bhh.clone();
    model.lstm1.wb_ih = lstm1_ih;
    model.lstm1.wb_hh = lstm1_hh;
    model.lstm1.bb_ih = lstm1_bih;
    model.lstm1.bb_hh = lstm1_bhh;
    model.lstm2.wf_ih = lstm2_ih.clone();
    model.lstm2.wf_hh = lstm2_hh.clone();
    model.lstm2.bf_ih = lstm2_bih.clone();
    model.lstm2.bf_hh = lstm2_bhh.clone();
    model.lstm2.wb_ih = lstm2_ih;
    model.lstm2.wb_hh = lstm2_hh;
    model.lstm2.bb_ih = lstm2_bih;
    model.lstm2.bb_hh = lstm2_bhh;

    // Copy FC weights
    let fc_w_data = fc_w.to_vec2::<f32>()?;
    let mut fc_arr = ndarray::Array2::zeros((vocab.size(), 2 * HIDDEN_SIZE));
    for r in 0..vocab.size() {
        for c in 0..2 * HIDDEN_SIZE {
            fc_arr[[r, c]] = fc_w_data[r][c];
        }
    }
    model.fc_weight = fc_arr;
    model.fc_bias = ndarray::Array1::from_vec(fc_b.to_vec1::<f32>()?);

    let bytes = model_format::write_model(&model).expect("write_model");
    if let Some(parent) = std::path::Path::new(output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output, &bytes).expect("write model");
    println!("  wrote {} bytes to {}", bytes.len(), output);
    Ok(())
}

fn softmax_exp(log_probs: &Tensor) -> Result<Vec<f32>> {
    let v: Vec<f32> = log_probs.to_vec1()?;
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Ok(exps.iter().map(|x| x / sum).collect())
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

fn lev(a: &str, b: &str) -> usize {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=bc.len()).collect();
    let mut curr = vec![0usize; bc.len() + 1];
    for i in 1..=ac.len() {
        curr[0] = i;
        for j in 1..=bc.len() {
            let cost = if ac[i - 1] == bc[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[bc.len()]
}

// ---------------------------------------------------------------------------
// CRNN architecture in candle
// ---------------------------------------------------------------------------

struct CrnnCandle {
    convs: Vec<Conv2d>,
    lstm1: candle_nn::rnn::LSTM,
    lstm2: candle_nn::rnn::LSTM,
    fc: Linear,
    width: usize,
}

impl CrnnCandle {
    fn new(
        vb: &VarBuilder,
        num_classes: usize,
        width: usize,
        _input_height: usize,
    ) -> Result<Self> {
        let cfg_conv1 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let cfg_conv2 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let cfg_conv3 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let cfg_conv4 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let cfg_conv5 = candle_nn::Conv2dConfig {
            padding: 0, // 2x2 kernel, no padding
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d(1, CNN_CHANNELS[0], 3, cfg_conv1, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(CNN_CHANNELS[0], CNN_CHANNELS[1], 3, cfg_conv2, vb.pp("conv2"))?;
        let conv3 = candle_nn::conv2d(CNN_CHANNELS[1], CNN_CHANNELS[2], 3, cfg_conv3, vb.pp("conv3"))?;
        let conv4 = candle_nn::conv2d(CNN_CHANNELS[2], CNN_CHANNELS[3], 3, cfg_conv4, vb.pp("conv4"))?;
        let conv5 = candle_nn::conv2d(CNN_CHANNELS[3], CNN_CHANNELS[4], 2, cfg_conv5, vb.pp("conv5"))?;

        let lstm_cfg = candle_nn::rnn::LSTMConfig::default();
        let lstm1 = candle_nn::rnn::lstm(
            CNN_CHANNELS[4] * 7, // 32/4 - 1 = 7
            HIDDEN_SIZE,
            lstm_cfg.clone(),
            vb.pp("lstm1"),
        )?;
        let lstm2 = candle_nn::rnn::lstm(
            2 * HIDDEN_SIZE,
            HIDDEN_SIZE,
            lstm_cfg,
            vb.pp("lstm2"),
        )?;
        let fc = linear(2 * HIDDEN_SIZE, num_classes, vb.pp("fc"))?;
        Ok(Self {
            convs: vec![conv1, conv2, conv3, conv4, conv5],
            lstm1,
            lstm2,
            fc,
            width,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, 1, H, W]
        let mut x = xs.clone();
        // Block 1: conv 3x3 + maxpool 2x2
        x = self.convs[0].forward(&x)?.relu()?.max_pool2d(2)?;
        // Block 2: conv 3x3 + maxpool 2x2
        x = self.convs[1].forward(&x)?.relu()?.max_pool2d(2)?;
        // Block 3: conv 3x3 + maxpool width-only (kernel 1x2, stride 1x2)
        x = self.convs[2].forward(&x)?.relu()?.max_pool2d_with_stride((1, 2), (1, 2))?;
        // Block 4: conv 3x3 + maxpool width-only
        x = self.convs[3].forward(&x)?.relu()?.max_pool2d_with_stride((1, 2), (1, 2))?;
        // Block 5: conv 2x2 (no padding)
        x = self.convs[4].forward(&x)?.relu()?;

        // Reshape: [B, C, H, W] -> [B, W, C*H]
        let (b, c, h, w) = (x.dim(0)?, x.dim(1)?, x.dim(2)?, x.dim(3)?);
        let x = x.permute((0, 3, 1, 2))?.contiguous()?.reshape((b, w, c * h))?;
        let _ = self.width;

        // BiLSTM layer 1
        let x = bilstm_forward(&self.lstm1, &x)?;

        // BiLSTM layer 2
        let x = bilstm_forward(&self.lstm2, &x)?;

        // FC
        let logits = self.fc.forward(&x)?;
        Ok(logits)
    }
}

/// Run a bidirectional LSTM over a [B, T, F] input. Returns [B, T, 2H].
fn bilstm_forward(lstm: &candle_nn::rnn::LSTM, x: &Tensor) -> Result<Tensor> {
    let (_b, _t, _f) = x.dims3()?;
    // Forward direction
    let fwd_states = lstm.seq(x)?;
    let fwd_h: Vec<Tensor> = fwd_states.iter().map(|s| s.h.clone()).collect();
    let fwd_out = Tensor::stack(&fwd_h, 1)?; // [B, T, H]
    // Backward direction: reverse the time axis, run, then reverse back.
    let x_rev = x.flip(&[1])?.contiguous()?;
    let bwd_states = lstm.seq(&x_rev)?;
    let bwd_h: Vec<Tensor> = bwd_states.iter().map(|s| s.h.clone()).collect();
    let bwd_out = Tensor::stack(&bwd_h, 1)?.flip(&[1])?.contiguous()?; // [B, T, H]
    // Concatenate
    Tensor::cat(&[fwd_out, bwd_out], 2)
}

// ---------------------------------------------------------------------------
// Data pipeline
// ---------------------------------------------------------------------------

fn build_batch_tensors(
    samples: &[ocr::synthetic::SyntheticSample],
    width: usize,
    height: usize,
    vocab: &Vocabulary,
    device: &Device,
) -> Result<(Tensor, Vec<Vec<u32>>, Vec<usize>, Vec<usize>, Vec<String>)> {
    let n = samples.len();
    let mut img_data: Vec<f32> = Vec::with_capacity(n * height * width);
    let mut targets: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut target_lengths: Vec<usize> = Vec::with_capacity(n);
    let input_length = width / 4 - 1; // approximate
    let input_lengths = vec![input_length; n];
    let mut texts: Vec<String> = Vec::with_capacity(n);

    for sample in samples {
        let gray = sample.image.to_luma8();
        // Resize to (height, width)
        let resized = resize_gray(&gray, width as u32, height as u32);
        for px in resized.pixels() {
            img_data.push(px.0[0] as f32 / 255.0);
        }
        let indices: Vec<u32> = sample
            .ground_truth
            .chars()
            .map(|c| vocab.char_to_idx.get(&c).copied().unwrap_or(0) as u32)
            .collect();
        target_lengths.push(indices.len());
        targets.push(indices);
        texts.push(sample.ground_truth.clone());
    }

    let images = Tensor::from_vec(img_data, (n, 1, height, width), device)?;
    Ok((images, targets, target_lengths, input_lengths, texts))
}

fn resize_gray(img: &GrayImage, w: u32, h: u32) -> GrayImage {
    image::imageops::resize(img, w, h, image::imageops::FilterType::Lanczos3)
}

fn decode_logits_batch(logits: &Tensor, vocab: &Vocabulary) -> Vec<String> {
    // Greedy decode per row
    let logits = logits.detach();
    let (b, t, c) = (logits.dim(0).unwrap(), logits.dim(1).unwrap(), logits.dim(2).unwrap());
    let data: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let mut out = Vec::with_capacity(b);
    for bi in 0..b {
        let mut s = String::new();
        let mut prev = vocab.blank_idx;
        for ti in 0..t {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for ci in 0..c {
                let v = data[((bi * t + ti) * c + ci)];
                if v > best_v {
                    best_v = v;
                    best = ci;
                }
            }
            if best != prev && best != vocab.blank_idx {
                if let Some(ch) = vocab.idx_to_char(best) {
                    s.push(ch);
                }
            }
            prev = best;
        }
        out.push(s);
    }
    out
}

// Re-export the synthetic augmentation entry point to avoid an extra import
// path in the example. The real function lives in `synthetic::distortion`.
mod synthetic_helpers {
    use ocr::synthetic::{DistortionConfig, SyntheticSample};
    pub fn augment(samples: &mut Vec<SyntheticSample>, cfg: &DistortionConfig) {
        ocr::synthetic::distortion::augment_batch(samples, cfg);
    }
}
fn synthetic_run_augment(samples: &mut Vec<ocr::synthetic::SyntheticSample>, cfg: &ocr::synthetic::DistortionConfig) {
    synthetic_helpers::augment(samples, cfg);
}
