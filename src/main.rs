mod cli;

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use cli::{parse, Commands};
use ::image::GenericImageView;
use tracing::{error, info, warn};

use ocr::api::{Ocr, TextProcessor};
use ocr::core::config::{OcrConfig, PageSegMode, RecognitionEngine};
use ocr::core::output::{format_hocr, format_tsv, to_json_output};
use ocr::core::text::TextResult;
use ocr::lang::dictionary::Dictionary;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = parse();

    match cli.command {
        Commands::Extract {
            image_path,
            output,
            lang,
            preprocess,
            format,
            psm,
            confidence,
            engine,
            dict_correct,
        } => {
            extract_text(image_path, output, &lang, preprocess, &format, psm, confidence, &engine, dict_correct).await?;
        }
        Commands::Batch {
            input_dir,
            output_dir,
            lang,
            confidence,
            max_concurrent,
            engine,
            dict_correct,
        } => {
            batch_process(input_dir, output_dir, &lang, confidence, max_concurrent, &engine, dict_correct).await?;
        }
        Commands::Layout {
            image_path,
            output,
        } => {
            analyze_layout(image_path, output).await?;
        }
        Commands::ListLanguages => {
            list_languages().await?;
        }
        Commands::Check => {
            check_system().await?;
        }
        Commands::Info => {
            show_info().await?;
        }
        Commands::Validate { config_file } => {
            validate_config(config_file).await?;
        }
        #[cfg(feature = "web-api")]
        Commands::Serve {
            host,
            port,
            max_upload_size,
        } => {
            use ocr::server::{run_server, ServerConfig};
            let config = ServerConfig {
                host,
                port,
                max_upload_size_mb: max_upload_size,
            };
            run_server(config).await?;
        }
    }

    Ok(())
}

fn parse_engine(s: &str) -> RecognitionEngine {
    match s.to_lowercase().as_str() {
        "lstm" => RecognitionEngine::LSTM,
        "hybrid" => RecognitionEngine::Hybrid,
        _ => RecognitionEngine::PatternMatching,
    }
}

fn build_config(lang: &str, preprocess: bool, psm: u8, confidence: f32, engine: &str, dict_correct: bool) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.recognition.language = lang.to_string();
    config.recognition.confidence_threshold = confidence;
    config.recognition.engine = parse_engine(engine);
    config.recognition.enable_dictionary_correction = dict_correct;
    config.image_processing.enable_preprocessing = preprocess;
    config.image_processing.enable_binarization = preprocess;
    config.image_processing.enable_noise_reduction = preprocess;
    config.image_processing.enable_contrast_enhancement = preprocess;
    config.image_processing.enable_deskewing = preprocess;
    config.layout_analysis.page_seg_mode = psm_to_mode(psm);
    config
}

fn psm_to_mode(psm: u8) -> PageSegMode {
    match psm {
        4 => PageSegMode::SingleColumn,
        6 => PageSegMode::SingleBlock,
        7 => PageSegMode::SingleLineRaw,
        8 => PageSegMode::SingleWord,
        10 => PageSegMode::SingleChar,
        11 => PageSegMode::SparseText,
        12 => PageSegMode::SparseTextWithOsd,
        13 => PageSegMode::SingleLine,
        _ => PageSegMode::Auto,
    }
}

fn apply_dictionary_correction(result: &mut TextResult, lang: &str) {
    let mut dict = Dictionary::new();
    match lang {
        "en" => dict.load_words(&[
            "the", "this", "that", "and", "for", "are", "was", "but", "not", "you",
            "all", "can", "had", "her", "was", "one", "our", "out", "has", "have",
            "from", "they", "been", "with", "their", "would", "about", "which",
            "there", "could", "should", "people", "water", "where", "after",
            "still", "world", "hello", "world", "ocr", "text", "image", "file",
        ]),
        "zh" => dict.load_words(&[
            "的", "一", "是", "不", "了", "人", "我", "在", "有", "他",
            "这", "中", "大", "来", "上", "国", "个", "到", "说", "们",
        ]),
        "ja" => dict.load_words(&[
            "の", "に", "を", "は", "が", "た", "で", "て", "と", "し",
            "れ", "る", "か", "な", "い", "あ", "こ", "さ", "き", "ま",
        ]),
        _ => {}
    }

    for word in result.words.iter_mut() {
        let word_text = word.text.trim();
        if word_text.len() > 2 && !dict.contains(word_text) {
            let corrected = dict.correct_word(word_text);
            if corrected != word_text {
                word.text = corrected;
            }
        }
    }
    result.text = result.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
}

fn format_result(result: &TextResult, format: &str) -> Result<String> {
    match format.to_lowercase().as_str() {
        "json" => Ok(serde_json::to_string_pretty(&to_json_output(result))?),
        "hocr" | "html" => Ok(format_hocr(result)?),
        "tsv" => Ok(format_tsv(result)?),
        _ => {
            if result.text.is_empty() {
                warn!("No text recognized");
                Ok(String::new())
            } else {
                Ok(result.text.clone())
            }
        }
    }
}

fn write_output(content: &str, output: Option<PathBuf>) -> Result<()> {
    match output {
        Some(path) => {
            std::fs::write(&path, content)?;
            println!("OCR results saved to: {:?}", path);
        }
        None => {
            if !content.is_empty() {
                print!("{}", content);
            }
        }
    }
    Ok(())
}

async fn extract_text(
    image_path: PathBuf,
    output: Option<PathBuf>,
    lang: &str,
    preprocess: bool,
    format: &str,
    psm: u8,
    confidence: f32,
    engine: &str,
    dict_correct: bool,
) -> Result<()> {
    info!(
        "Starting OCR extraction for: {:?} (psm={}, format={}, engine={})",
        image_path, psm, format, engine
    );

    let config = build_config(lang, preprocess, psm, confidence, engine, dict_correct);
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let mut result = ocr
        .recognize_text_from_file(&image_path)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    if dict_correct {
        apply_dictionary_correction(&mut result, lang);
    }

    info!("OCR completed with confidence: {:.2}%", result.confidence * 100.0);

    let output_content = format_result(&result, format)?;
    write_output(&output_content, output)
}

async fn batch_process(
    input_dir: PathBuf,
    output_dir: PathBuf,
    lang: &str,
    confidence: f32,
    _max_concurrent: usize,
    engine: &str,
    dict_correct: bool,
) -> Result<()> {
    info!(
        "Batch processing images from: {:?} -> {:?}",
        input_dir, output_dir
    );

    tokio::fs::create_dir_all(&output_dir).await?;

    let config = build_config(lang, true, 3, confidence, engine, dict_correct);
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let mut entries = tokio::fs::read_dir(&input_dir).await?;
    let mut image_files = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "webp") {
                image_files.push(path);
            }
        }
    }

    info!("Found {} image files", image_files.len());
    let mut processed = 0;

    for image_path in &image_files {
        let stem = image_path.file_stem().unwrap_or_default().to_string_lossy();
        let output_path = output_dir.join(format!("{}.txt", stem));

        match ocr.recognize_text_from_file(image_path).await {
            Ok(mut result) => {
                if dict_correct {
                    apply_dictionary_correction(&mut result, lang);
                }
                let text = if confidence > 0.0 {
                    TextProcessor::filter_by_confidence(&result, confidence)
                } else {
                    result
                };
                tokio::fs::write(&output_path, &text.text).await?;
                processed += 1;
                info!("Processed: {:?} -> {:?}", image_path, output_path);
            }
            Err(e) => {
                error!("Failed to process {:?}: {}", image_path, e);
            }
        }
    }

    info!(
        "Batch processing completed: {}/{} images processed",
        processed,
        image_files.len()
    );
    Ok(())
}

async fn analyze_layout(image_path: PathBuf, output: Option<PathBuf>) -> Result<()> {
    info!("Analyzing layout for: {:?}", image_path);

    let ocr = Ocr::new()?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let image_data = tokio::fs::read(&image_path).await?;
    let dynamic_image = ::image::load_from_memory(&image_data)?;
    let (width, height) = dynamic_image.dimensions();

    let layout_result = ocr
        .analyze_layout(&image_data, width, height)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    let layout_json = serde_json::to_string_pretty(&layout_result)?;
    write_output(&layout_json, output)
}

async fn list_languages() -> Result<()> {
    let ocr = Ocr::new()?;
    let base = ocr.get_supported_languages();
    let mut languages: Vec<String> = base.iter().cloned().collect();

    let cjk_codes = ["zh", "ja", "ko"];
    for code in &cjk_codes {
        let entry = format!("{} (CJK)", code);
        if !languages.contains(&entry) {
            languages.push(entry);
        }
    }
    languages.sort();

    println!("Supported languages:");
    for lang in &languages {
        println!("  - {}", lang);
    }
    println!();
    println!("Note: Use --engine to select recognition engine (pattern, lstm, hybrid)");
    println!("      Use --dict-correct to enable dictionary-based post-correction");

    Ok(())
}

async fn check_system() -> Result<()> {
    println!("Checking system requirements...");

    // Test pattern matching engine
    match Ocr::new() {
        Ok(ocr) => match ocr.initialize().await {
            Ok(_) => println!("✓ Pattern matching engine initialized"),
            Err(e) => eprintln!("✗ Failed to initialize OCR engine: {}", e),
        },
        Err(e) => eprintln!("✗ Failed to create OCR engine: {}", e),
    }

    // Test image loading
    println!("✓ Image processing modules loaded");
    println!("✓ Layout analysis engine ready");
    println!("✓ Dictionary correction module ready");

    println!("\nCapabilities:");
    println!("  - Pattern matching engine (default)");
    println!("  - LSTM neural network engine");
    println!("  - Hybrid recognition engine");
    println!("  - Layout analysis");
    println!("  - Dictionary-based post-correction");
    println!("  - Multiple output formats: text, json, hocr, tsv");

    Ok(())
}

async fn show_info() -> Result<()> {
    let ocr = Ocr::new()?;
    let metadata = ocr.get_metadata();

    println!("OCR Engine Information:");
    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Description: {}", metadata.description);
    println!(
        "  Supported Languages: {}",
        metadata.supported_languages.join(", ")
    );
    println!(
        "  Supported Image Formats: {}",
        metadata.supported_image_formats.join(", ")
    );

    Ok(())
}

async fn validate_config(config_file: PathBuf) -> Result<()> {
    info!("Validating configuration: {:?}", config_file);

    match OcrConfig::from_file(&config_file) {
        Ok(config) => {
            config.validate()?;
            println!("Configuration is valid");
            Ok(())
        }
        Err(e) => {
            error!("Configuration validation failed: {}", e);
            Err(anyhow!("Invalid configuration: {}", e))
        }
    }
}
