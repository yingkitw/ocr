use anyhow::Result;

use ocr::lang::unicode::Script;
use ocr::recognition::crnn::ScriptModelRegistry;
use ocr::synthetic::multi_script::ScriptLineGenerator;
use ocr::training::crnn_trainer::{levenshtein_distance, word_error_distance};

/// Evaluate CRNN accuracy per script on synthetic test data
pub async fn handle_benchmark(
    samples_per_script: usize,
    distortion: String,
) -> Result<()> {
    println!("Running per-script accuracy benchmark...");
    println!("  Samples per script: {}", samples_per_script);
    println!("  Distortion level: {}", distortion);
    println!();

    let registry = ScriptModelRegistry::new();
    let scripts = [
        Script::Latin,
        Script::CJK,
        Script::Cyrillic,
        Script::Arabic,
        Script::Greek,
        Script::Hebrew,
        Script::Thai,
        Script::Devanagari,
    ];

    println!("{:<15} {:>8} {:>8} {:>8} {:>8}", "Script", "CER%", "WER%", "Samples", "Chars");
    println!("{}", "-".repeat(55));

    let mut total_char_errors = 0usize;
    let mut total_chars = 0usize;
    let mut total_word_errors = 0usize;
    let mut total_words = 0usize;

    for script in &scripts {
        let model = match registry.model_for(*script) {
            Some(m) => m,
            None => {
                println!("{:<15} {:>8} {:>8} {:>8} {:>8}", format!("{:?}", script), "N/A", "N/A", 0, 0);
                continue;
            }
        };

        let line_gen = ScriptLineGenerator::new(*script);

        let mut char_errors = 0usize;
        let mut chars = 0usize;
        let mut word_errors = 0usize;
        let mut words = 0usize;

        for _ in 0..samples_per_script {
            let text = line_gen.random_text(15);
            let sample = line_gen.generate(&text);

            let pred = model.recognize_from_sample(&sample);

            char_errors += levenshtein_distance(&text, &pred);
            chars += text.chars().count();
            word_errors += word_error_distance(&text, &pred);
            words += text.split_whitespace().count().max(1);
        }

        let cer = if chars > 0 { char_errors as f32 / chars as f32 * 100.0 } else { 0.0 };
        let wer = if words > 0 { word_errors as f32 / words as f32 * 100.0 } else { 0.0 };

        println!(
            "{:<15} {:>7.1}% {:>7.1}% {:>8} {:>8}",
            format!("{:?}", script),
            cer,
            wer,
            samples_per_script,
            chars,
        );

        total_char_errors += char_errors;
        total_chars += chars;
        total_word_errors += word_errors;
        total_words += words;
    }

    println!("{}", "-".repeat(55));
    let overall_cer = if total_chars > 0 { total_char_errors as f32 / total_chars as f32 * 100.0 } else { 0.0 };
    let overall_wer = if total_words > 0 { total_word_errors as f32 / total_words as f32 * 100.0 } else { 0.0 };
    println!(
        "{:<15} {:>7.1}% {:>7.1}% {:>8} {:>8}",
        "Overall",
        overall_cer,
        overall_wer,
        samples_per_script * scripts.len(),
        total_chars,
    );

    println!();
    println!("Note: These are untrained (random-weight) baseline scores.");
    println!("      Run `ocr train --engine lstm --epochs 50` to improve accuracy.");

    Ok(())
}
