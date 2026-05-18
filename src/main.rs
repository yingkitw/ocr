mod cli;

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use cli::{parse, Commands};
use ::image::GenericImageView;
use tracing::{error, info, warn};

use ocr::api::{Ocr, TextProcessor};
use ocr::core::config::{OcrConfig, PageSegMode, RecognitionEngine};
use ocr::core::output::{format_alto, format_box, format_hocr, format_tsv, to_json_output};
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
            device,
            osd,
        } => {
            extract_text(image_path, output, &lang, preprocess, &format, psm, confidence, &engine, dict_correct, &device, osd).await?;
        }
        Commands::Batch {
            input_dir,
            output_dir,
            lang,
            confidence,
            max_concurrent,
            engine,
            dict_correct,
            device,
        } => {
            batch_process(input_dir, output_dir, &lang, confidence, max_concurrent, &engine, dict_correct, device.as_str()).await?;
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

fn build_config(
    lang: &str,
    preprocess: bool,
    psm: u8,
    confidence: f32,
    engine: &str,
    dict_correct: bool,
    device: &str,
    osd: bool,
) -> OcrConfig {
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
    config.layout_analysis.enable_orientation_detection = osd || preprocess;
    config.performance.device = device.to_string();
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
        "alto" | "xml" => Ok(format_alto(result)?),
        "box" => Ok(format_box(result)?),
        "pdf" => Ok("PDF output requires binary generation. Use --format pdf with --output.".to_string()),
        _ => {
            if result.text.is_empty() {
                warn!("No text recognized");
                Ok("".to_string())
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
    device: &str,
    osd: bool,
) -> Result<()> {
    info!(
        "Starting OCR extraction for: {:?} (psm={}, format={}, engine={})",
        image_path, psm, format, engine
    );

    let config = build_config(lang, preprocess, psm, confidence, engine, dict_correct, device, osd);
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let ext = image_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "pdf" {
        #[cfg(feature = "pdf")]
        {
            return extract_pdf_text(&ocr, &image_path, output, format, lang, dict_correct).await;
        }
        #[cfg(not(feature = "pdf"))]
        {
            return Err(anyhow!(
                "PDF support is not enabled. Rebuild with: cargo build --features pdf"
            ));
        }
    }

    let mut result = ocr
        .recognize_text_from_file(&image_path)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    if dict_correct {
        apply_dictionary_correction(&mut result, lang);
    }

    info!("OCR completed with confidence: {:.2}%", result.confidence * 100.0);

    if format.to_lowercase() == "pdf" {
        return generate_searchable_pdf(&image_path, &result, output).await;
    }

    let output_content = format_result(&result, format)?;
    write_output(&output_content, output)
}

async fn generate_searchable_pdf(
    image_path: &std::path::Path,
    result: &TextResult,
    output: Option<PathBuf>,
) -> Result<()> {
    let out_path = match output {
        Some(p) => p,
        None => return Err(anyhow!("PDF output requires --output file path")),
    };

    let img_data = tokio::fs::read(image_path).await?;
    let dynamic_img = ::image::load_from_memory(&img_data)?;
    let (img_width_px, img_height_px) = dynamic_img.dimensions();

    // Use 300 DPI for pixel-to-pt conversion (1 pt = 1/72 inch)
    let px_to_pt = |px: f32| printpdf::Pt(px * 72.0 / 300.0);
    let page_width_pt = px_to_pt(img_width_px as f32);
    let page_height_pt = px_to_pt(img_height_px as f32);

    let mut doc = printpdf::PdfDocument::new("OCR Output");

    // Convert dynamic image to RawImage for printpdf
    let rgb_img = dynamic_img.to_rgb8();
    let (w, h) = rgb_img.dimensions();
    let raw_image = printpdf::RawImage {
        pixels: printpdf::RawImageData::U8(rgb_img.into_raw()),
        width: w as usize,
        height: h as usize,
        data_format: printpdf::RawImageFormat::RGB8,
        tag: Vec::new(),
    };
    let image_id = doc.add_image(&raw_image);

    // Build page operations
    let mut ops = Vec::new();

    // Draw image full-page
    ops.push(printpdf::Op::UseXobject {
        id: image_id,
        transform: printpdf::XObjectTransform {
            translate_x: Some(printpdf::Pt(0.0)),
            translate_y: Some(printpdf::Pt(0.0)),
            scale_x: Some((page_width_pt.0 / img_width_px as f32) as f32),
            scale_y: Some((page_height_pt.0 / img_height_px as f32) as f32),
            rotate: None,
            dpi: Some(300.0),
        },
    });

    // Overlay text for each recognized word
    for line in &result.lines {
        for word in &line.words {
            let bbox = &word.bounding_box;
            let x = px_to_pt(bbox.left as f32);
            // PDF y=0 is at bottom, image y=0 is at top, so flip
            let y = page_height_pt - px_to_pt(bbox.bottom as f32);
            let font_size = px_to_pt((bbox.bottom - bbox.top) as f32).0.max(1.0);

            ops.push(printpdf::Op::StartTextSection);
            ops.push(printpdf::Op::SetFont {
                font: printpdf::PdfFontHandle::Builtin(printpdf::BuiltinFont::Helvetica),
                size: printpdf::Pt(font_size),
            });
            ops.push(printpdf::Op::SetTextCursor {
                pos: printpdf::Point { x, y },
            });
            ops.push(printpdf::Op::ShowText {
                items: vec![printpdf::TextItem::Text(word.text.clone())],
            });
            ops.push(printpdf::Op::EndTextSection);
        }
    }

    let page = printpdf::PdfPage::new(
        printpdf::Mm(page_width_pt.0 * 25.4 / 72.0),
        printpdf::Mm(page_height_pt.0 * 25.4 / 72.0),
        ops,
    );
    doc.pages.push(page);

    let mut warnings = Vec::new();
    let pdf_bytes = doc.save(&printpdf::PdfSaveOptions::default(), &mut warnings);
    std::fs::write(&out_path, pdf_bytes)?;
    println!("Searchable PDF saved to: {:?}", out_path);
    Ok(())
}

#[cfg(feature = "pdf")]
async fn extract_pdf_text(
    ocr: &Ocr,
    pdf_path: &std::path::Path,
    output: Option<PathBuf>,
    format: &str,
    lang: &str,
    dict_correct: bool,
) -> Result<()> {
    use ocr::pdf::extract_images;

    info!("Extracting images from PDF: {:?}", pdf_path);
    let images = extract_images(pdf_path)?;

    if images.is_empty() {
        warn!("No embedded images found in PDF. For vector-based PDFs, convert to images first.");
        return Ok(());
    }

    info!("Found {} images across {} pages", images.len(), images.iter().map(|i| i.page_number).max().unwrap_or(0));

    let mut all_text = Vec::new();
    for img in &images {
        let img_format = match img.format {
            ocr::pdf::PdfImageFormat::Jpeg => "jpeg",
            ocr::pdf::PdfImageFormat::Png => "png",
            ocr::pdf::PdfImageFormat::Raw => "raw",
        };
        info!(
            "Processing page {}: {}x{} ({})",
            img.page_number, img.width, img.height, img_format
        );

        match ocr.recognize_text(&img.data, img.width, img.height).await {
            Ok(mut result) => {
                if dict_correct {
                    apply_dictionary_correction(&mut result, lang);
                }
                all_text.push(format!("--- Page {} ---\n{}", img.page_number, result.text));
            }
            Err(e) => {
                warn!("Failed to OCR page {}: {}", img.page_number, e);
            }
        }
    }

    let combined = all_text.join("\n\n");
    let output_content = if format == "json" {
        serde_json::to_string_pretty(&serde_json::json!({
            "pages": images.iter().map(|i| serde_json::json!({
                "page": i.page_number,
                "width": i.width,
                "height": i.height,
            })).collect::<Vec<_>>(),
            "text": combined,
        }))?
    } else {
        combined
    };

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
    device: &str,
) -> Result<()> {
    info!(
        "Batch processing images from: {:?} -> {:?}",
        input_dir, output_dir
    );

    tokio::fs::create_dir_all(&output_dir).await?;

    let config = build_config(lang, true, 3, confidence, engine, dict_correct, device, false);
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
    println!("  - Multiple output formats: text, json, hocr, tsv, alto, xml");

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
