use anyhow::Result;
use std::path::PathBuf;

use crate::cli::commands::helpers::parse_engine;
use ocr::core::config::RecognitionEngine;
use ocr::recognition::crnn::{CrnnConfig, CrnnModel};
use ocr::synthetic::{DistortionConfig, TextLineGenerator};
use ocr::training::crnn_trainer::CrnnTrainer;

pub async fn handle_train(
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
    engine: String,
    checkpoint_dir: Option<PathBuf>,
    distortion: String,
) -> Result<()> {
    let recognition_engine = parse_engine(&engine);

    match recognition_engine {
        RecognitionEngine::LSTM => {
            println!("Training CRNN model...");
            println!("  Epochs: {}", epochs);
            println!("  Batch size: {}", batch_size);
            println!("  Learning rate: {}", learning_rate);

            let config = CrnnConfig::default();
            let model = CrnnModel::new(config);
            let mut trainer = CrnnTrainer::new(model)
                .with_learning_rate(learning_rate)
                .with_batch_size(batch_size);

            let distortion_cfg = match distortion.as_str() {
                "clean" => DistortionConfig::none(),
                "mild" => DistortionConfig::mild(),
                "heavy" => DistortionConfig::heavy(),
                _ => DistortionConfig::mild(),
            };
            trainer = trainer.with_distortion(distortion_cfg);

            let generator = TextLineGenerator::default();
            let val_texts = generator.generate_random_texts(50, 20);
            let val_samples = generator.generate_batch(&val_texts);

            for epoch in 1..=epochs {
                let metrics = trainer.train_epoch(10, batch_size);
                let val_metrics = trainer.evaluate(&val_samples);

                println!(
                    "Epoch {}/{} | Loss: {:.4} | Train CER: {:.2}% | Val CER: {:.2}% | {:.1} samples/sec",
                    epoch,
                    epochs,
                    metrics.train_loss,
                    metrics.train_cer * 100.0,
                    val_metrics.val_cer * 100.0,
                    metrics.samples_per_sec,
                );

                if let Some(ref dir) = checkpoint_dir {
                    std::fs::create_dir_all(dir)?;
                    let path = dir.join(format!("crnn_epoch_{}.json", epoch));
                    trainer.save_checkpoint(&path)?;
                }
            }

            if let Some(ref dir) = checkpoint_dir {
                let final_path = dir.join("crnn_final.json");
                trainer.save_checkpoint(&final_path)?;
                println!("Final checkpoint saved to {}", final_path.display());
            }

            println!("Training complete.");
        }
        _ => {
            println!("Training is only supported for the LSTM (CRNN) engine.");
            println!("Use --engine lstm to train the CRNN model.");
        }
    }

    Ok(())
}
