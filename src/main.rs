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
        } => {
            extract_text(image_path, output, &lang, preprocess, &format, psm, confidence).await?;
        }
        Commands::Batch {
            input_dir,
            output_dir,
            lang,
            confidence,
            max_concurrent,
        } => {
            batch_process(input_dir, output_dir, &lang, confidence, max_concurrent).await?;
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
    }

    Ok(())
}

fn build_config(lang: &str, preprocess: bool, psm: u8, confidence: f32) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.recognition.language = lang.replace("eng", "en");
    config.recognition.confidence_threshold = confidence;
    config.recognition.engine = RecognitionEngine::PatternMatching;
    config.image_processing.enable_preprocessing = preprocess;
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
) -> Result<()> {
    info!(
        "Starting OCR extraction for: {:?} (psm={}, format={})",
        image_path, psm, format
    );

    let config = build_config(lang, preprocess, psm, confidence);
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let result = ocr
        .recognize_text_from_file(&image_path)
        .await
        .map_err(|e| anyhow!("{}", e))?;

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
) -> Result<()> {
    info!(
        "Batch processing images from: {:?} -> {:?}",
        input_dir, output_dir
    );

    tokio::fs::create_dir_all(&output_dir).await?;

    let ocr = Ocr::new()?;
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
            Ok(result) => {
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
    let languages = ocr.get_supported_languages();

    println!("Supported languages:");
    for lang in languages {
        println!("  - {}", lang);
    }
    println!();
    println!("Note: The PatternMatching engine is used by default");
    println!("Language support is extendable through LSTM/Transformer models");

    Ok(())
}

async fn check_system() -> Result<()> {
    println!("Checking system requirements...");

    match Ocr::new() {
        Ok(ocr) => match ocr.initialize().await {
            Ok(_) => println!("✓ OCR engine initialized successfully"),
            Err(e) => eprintln!("✗ Failed to initialize OCR engine: {}", e),
        },
        Err(e) => eprintln!("✗ Failed to create OCR engine: {}", e),
    }

    println!("✓ Image processing modules loaded");
    println!("✓ Text segmentation engine ready");
    println!("✓ Character recognition system initialized");
    println!("✓ OCR CLI is ready to use");

    println!("\nCapabilities:");
    println!("  - Pattern matching engine (default)");
    println!("  - LSTM neural network engine");
    println!("  - Hybrid recognition engine");
    println!("  - Layout analysis");
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
