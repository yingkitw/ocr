use std::path::Path;
use std::time::Instant;

use ocr::api::{Ocr, TextProcessor};
use ocr::core::config::{OcrConfig, PageSegMode, RecognitionEngine};
use ocr::lang::dictionary::Dictionary;

fn build_config(lang: &str, preprocess: bool, engine: &str, dict_correct: bool) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.recognition.language = lang.to_string();
    config.recognition.confidence_threshold = 0.3;
    config.recognition.engine = match engine {
        "lstm" => RecognitionEngine::LSTM,
        "hybrid" => RecognitionEngine::Hybrid,
        _ => RecognitionEngine::PatternMatching,
    };
    config.recognition.enable_dictionary_correction = dict_correct;
    config.image_processing.enable_preprocessing = preprocess;
    config.image_processing.enable_binarization = preprocess;
    config.image_processing.enable_noise_reduction = preprocess;
    config.image_processing.enable_contrast_enhancement = preprocess;
    config.image_processing.enable_deskewing = preprocess;
    config.layout_analysis.page_seg_mode = PageSegMode::Auto;
    config
}

fn apply_dict_correction(text: &mut ocr::core::text::TextResult, lang: &str) {
    let mut dict = Dictionary::new();
    match lang {
        "en" => dict.load_words(&[
            "the", "this", "that", "and", "for", "are", "was", "but", "not", "you",
            "from", "they", "been", "with", "their", "would", "about", "which",
            "there", "could", "should", "people", "hello", "world", "ocr", "text",
        ]),
        _ => {}
    }
    for word in text.words.iter_mut() {
        let w = word.text.trim();
        if w.len() > 2 && !dict.contains(w) {
            let corrected = dict.correct_word(w);
            if corrected != w {
                word.text = corrected;
            }
        }
    }
    text.text = text.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image-path> [engine] [lang] [preprocess] [dict-correct]",
                  args.first().map(|s| s.as_str()).unwrap_or("ocr_demo"));
        eprintln!("  engine: pattern (default), lstm, hybrid");
        eprintln!("  lang: en (default), zh, ja, ko");
        eprintln!("  preprocess: true (default), false");
        eprintln!("  dict-correct: true, false (default)");
        return Ok(());
    }

    let image_path = Path::new(&args[1]);
    let engine = args.get(2).map(|s| s.as_str()).unwrap_or("pattern");
    let lang = args.get(3).map(|s| s.as_str()).unwrap_or("en");
    let preprocess = args.get(4).map(|s| s == "true").unwrap_or(true);
    let dict_correct = args.get(5).map(|s| s == "true").unwrap_or(false);

    if !image_path.exists() {
        eprintln!("Error: image file not found: {:?}", image_path);
        return Ok(());
    }

    println!("OCR Demo");
    println!("========");
    println!("Image:     {:?}", image_path);
    println!("Engine:    {}", engine);
    println!("Language:  {}", lang);
    println!("Preprocess: {}", preprocess);
    println!("DictCorrect: {}", dict_correct);
    println!();

    let config = build_config(lang, preprocess, engine, dict_correct);
    let ocr = Ocr::with_config(config)?;
    let start = Instant::now();
    ocr.initialize().await?;
    println!("Initialized in {:?}", start.elapsed());

    let start = Instant::now();
    let mut result = ocr.recognize_text_from_file(image_path).await?;
    let elapsed = start.elapsed();
    println!("Recognized in {:?}", elapsed);

    if dict_correct {
        apply_dict_correction(&mut result, lang);
        println!("Dictionary correction applied");
    }

    println!();
    println!("Recognized text:");
    println!("{}", result.text);
    println!();
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    println!("Words: {}", result.words.len());
    println!("Characters: {}", result.characters.len());
    println!();

    let filtered = TextProcessor::filter_by_confidence(&result, 0.5);
    if !filtered.text.is_empty() && filtered.text != result.text {
        println!("High-confidence text (>=50%):");
        println!("{}", filtered.text);
    }

    let stats = TextProcessor::get_text_statistics(&result);
    println!("Statistics:");
    println!("  Lines: {}", stats.line_count);
    println!("  Words: {}", stats.word_count);
    println!("  Characters: {}", stats.character_count);
    println!("  Avg char confidence: {:.1}%", stats.avg_character_confidence * 100.0);
    println!("  Avg word confidence: {:.1}%", stats.avg_word_confidence * 100.0);

    Ok(())
}
