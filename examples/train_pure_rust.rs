//! Train the bundled CRNN using the in-crate pure-Rust CTC + Adam with
//! full BPTT through CNN + BiLSTM + FC (`CrnnTrainer::train_batch_full`).
//!
//! This example demonstrates that the training pipeline works end-to-end
//! with the bundled model and can write a re-trained `models/crnn-en.bin`.
//!
//! Run with:
//!   cargo run --release --example train_pure_rust -- --epochs 10
//!   cargo run --release --example train_pure_rust -- --epochs 10 \
//!       --samples 2000 --batch-size 16 --output models/crnn-en.bin

use ndarray::Array1;
use ocr::recognition::crnn::{CrnnConfig, Vocabulary};
use ocr::recognition::model_format;
use ocr::synthetic::{DistortionConfig, SyntheticSample, TextLineGenerator};
use ocr::training::crnn_trainer::{AdamState, CrnnTrainer, FullAdamState};
use ocr::utils::Result;

const INPUT_HEIGHT: usize = 32;
const INPUT_WIDTH: usize = 256;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    println!("device: ndarray (CPU)");
    println!("mode: {}", args.mode);
    let vocab = Vocabulary::from_ascii();
    let num_classes = vocab.size();
    println!("vocab size: {} (incl. blank)", num_classes);

    // Build a model with default config (random init weights)
    let cfg = CrnnConfig::default();
    let mut trainer = CrnnTrainer::new(ocr::recognition::crnn::CrnnModel::new(cfg))
        .with_learning_rate(args.learning_rate)
        .with_batch_size(args.batch_size);
    let mut full_adam = FullAdamState::new_like(&trainer.model);
    let mut fc_adam = AdamState::new_like(&trainer.model);
    println!("parameters: {}", trainer.model.parameter_count());

    // Synthetic data
    let generator = TextLineGenerator::default();
    let distortion = match args.distortion.as_str() {
        "clean" => DistortionConfig::none(),
        "mild" => DistortionConfig::mild(),
        "heavy" => DistortionConfig::heavy(),
        _ => DistortionConfig::mild(),
    };
    trainer.distortion = distortion.clone();

    let val_texts = generator.generate_random_texts(200, args.max_text_len);
    let val_samples = generator.generate_batch(&val_texts);
    ocr::synthetic::distortion::augment_batch(&mut val_samples.clone(), &distortion);

    let mut best_cer = f32::INFINITY;
    let mut best_loss = f32::INFINITY;
    for epoch in 1..=args.epochs {
        let texts = generator.generate_random_texts(args.samples, args.max_text_len);
        let mut samples = generator.generate_batch(&texts);
        ocr::synthetic::distortion::augment_batch(&mut samples, &distortion);

        let n = samples.len();
        let mut total_loss = 0.0f32;
        let mut n_batches = 0usize;
        let mut correct = 0usize;
        let mut chars = 0usize;
        for batch_start in (0..n).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(n);
            let batch: Vec<SyntheticSample> = samples[batch_start..batch_end].to_vec();
            if batch.is_empty() {
                continue;
            }
            let loss = if args.mode == "full" {
                trainer.train_batch_full(&batch, &mut full_adam)
            } else {
                trainer.train_batch_ctc(&batch, &mut fc_adam)
            };
            total_loss += loss;
            n_batches += 1;
            // Greedy decode for accuracy
            for s in &batch {
                let gray = s.image.to_luma8();
                let (w, h) = (gray.width() as usize, gray.height() as usize);
                let mut arr = ndarray::Array2::zeros((h, w));
                for y in 0..h {
                    for x in 0..w {
                        arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
                    }
                }
                if h != INPUT_HEIGHT || w != INPUT_WIDTH {
                    arr = resize_to(&arr, INPUT_WIDTH, INPUT_HEIGHT);
                }
                let logits = trainer.model.forward(&arr);
                let pred = trainer.model.decode_greedy_from_logits(&logits);
                chars += s.ground_truth.chars().count();
                correct += pred.chars().zip(s.ground_truth.chars()).filter(|(a, b)| a == b).count();
            }
        }

        // Validation CER (greedy decode)
        let mut cer_correct = 0usize;
        let mut cer_total = 0usize;
        for s in &val_samples {
            let gray = s.image.to_luma8();
            let (w, h) = (gray.width() as usize, gray.height() as usize);
            let mut arr = ndarray::Array2::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    arr[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
                }
            }
            if h != INPUT_HEIGHT || w != INPUT_WIDTH {
                arr = resize_to(&arr, INPUT_WIDTH, INPUT_HEIGHT);
            }
            let (logits, _) = trainer.model.forward_with_cache(&arr);
            let pred = trainer.model.decode_greedy_from_logits(&logits);
            let max_len = pred.chars().count().max(s.ground_truth.chars().count());
            if max_len == 0 {
                continue;
            }
            let dist = lev(&pred, &s.ground_truth);
            cer_correct += max_len - dist.min(max_len);
            cer_total += max_len;
        }
        let val_cer = 1.0 - cer_correct as f32 / cer_total.max(1) as f32;
        let epoch_loss = total_loss / n_batches.max(1) as f32;

        println!(
            "epoch {epoch:>3}/{}  train_loss={:.4}  train_acc={:.2}%  val_cer={:.2}%  ({} batches)",
            args.epochs,
            epoch_loss,
            correct as f32 / chars.max(1) as f32 * 100.0,
            val_cer * 100.0,
            n_batches,
        );

        // Checkpoint on val_cer improvement OR on train_loss improvement.
        // CTC accuracy often stays flat (100% CER) for many epochs while loss
        // descends, then falls off a cliff — loss-based checkpoints guarantee we
        // keep the best model during that pre-accuracy phase.
        let improved = val_cer < best_cer || epoch_loss < best_loss;
        if val_cer < best_cer {
            best_cer = val_cer;
        }
        if epoch_loss < best_loss {
            best_loss = epoch_loss;
        }
        if improved {
            save_model(&trainer.model, &vocab, &args.output);
        }
    }

    println!("best val_cer {:.2}%  best train_loss {:.4}", best_cer * 100.0, best_loss);
    Ok(())
}

fn save_model(model: &ocr::recognition::crnn::CrnnModel, _vocab: &Vocabulary, output: &str) {
    let bytes = model_format::write_model(model).expect("write_model");
    if let Some(parent) = std::path::Path::new(output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output, &bytes).expect("write model");
    println!("  -> saved {} ({} bytes)", output, bytes.len());
    let _ = Array1::<f32>::zeros(0); // unused, just to keep ndarray dep used
}

fn resize_to(arr: &ndarray::Array2<f32>, w: usize, h: usize) -> ndarray::Array2<f32> {
    use image::GrayImage;
    let (cur_h, cur_w) = arr.dim();
    let mut img = GrayImage::new(cur_w as u32, cur_h as u32);
    for y in 0..cur_h {
        for x in 0..cur_w {
            img.put_pixel(x as u32, y as u32, image::Luma([(arr[[y, x]] * 255.0) as u8]));
        }
    }
    let resized = image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3);
    let mut out = ndarray::Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = resized.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
        }
    }
    out
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

struct Args {
    epochs: usize,
    samples: usize,
    batch_size: usize,
    learning_rate: f32,
    output: String,
    max_text_len: usize,
    distortion: String,
    mode: String,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            epochs: 10,
            samples: 1000,
            batch_size: 16,
            learning_rate: 0.001,
            output: "models/crnn-en.bin".to_string(),
            max_text_len: 4,
            distortion: "mild".to_string(),
            mode: "fc".to_string(),
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
                "--distortion" => a.distortion = iter.next().unwrap(),
                "--mode" => a.mode = iter.next().unwrap(),
                "--help" | "-h" => {
                    println!("{}", USAGE);
                    std::process::exit(0);
                }
                other => panic!("unknown flag: {other}"),
            }
        }
        a
    }
}

const USAGE: &str = "\
Train the bundled CRNN with in-crate pure-Rust CTC + full BPTT (CNN + BiLSTM + FC).

Usage:
  cargo run --release --example train_pure_rust -- [OPTIONS]

Options:
  --epochs <N>            training epochs                  [default: 10]
  --samples <N>           synthetic samples per epoch     [default: 1000]
  --batch-size <N>        batch size                      [default: 16]
  --lr, --learning-rate   Adam learning rate              [default: 0.001]
  -o, --output <PATH>     output model path               [default: models/crnn-en.bin]
  --max-len <N>           max text length                 [default: 4]
  --distortion <LEVEL>    clean|mild|heavy                [default: mild]
  --mode <MODE>           fc|full (fc=FC-only, full=BPTT)  [default: fc]
";