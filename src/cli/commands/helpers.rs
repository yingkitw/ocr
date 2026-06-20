use std::path::PathBuf;

use anyhow::Result;
use tracing::info;

use ocr::api::Ocr;
use ocr::core::text::TextResult;
use ocr::lang::dictionary::DictionaryHandler;

pub fn apply_dictionary_correction(result: &mut TextResult, lang: &str) {
    let handler = DictionaryHandler::new_for_language(lang);

    let words: Vec<String> = result
        .text
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let mut corrected_words = Vec::new();
    let mut changed = false;

    for word in &words {
        if handler.is_word_valid(word) {
            corrected_words.push(word.clone());
        } else {
            let correction = handler.correct(word);
            if correction != *word {
                changed = true;
            }
            corrected_words.push(correction);
        }
    }

    if changed {
        result.text = corrected_words.join(" ");
        info!("Applied dictionary correction for language: {}", lang);
    }
}

fn parse_engine(s: &str) -> ocr::core::config::RecognitionEngine {
    match s.to_lowercase().as_str() {
        "lstm" => ocr::core::config::RecognitionEngine::LSTM,
        "hybrid" => ocr::core::config::RecognitionEngine::Hybrid,
        _ => ocr::core::config::RecognitionEngine::PatternMatching,
    }
}

pub fn build_config(
    lang: &str,
    preprocess: bool,
    psm: u8,
    confidence: f32,
    engine: &str,
    dict_correct: bool,
    device: &str,
    osd: bool,
) -> ocr::core::config::OcrConfig {
    use ocr::core::config::{OcrConfig, PageSegMode};

    let psm_mode = match psm {
        4 => PageSegMode::SingleColumn,
        6 => PageSegMode::SingleBlock,
        7 => PageSegMode::SingleLineRaw,
        8 => PageSegMode::SingleWord,
        10 => PageSegMode::SingleChar,
        11 => PageSegMode::SparseText,
        12 => PageSegMode::SparseTextWithOsd,
        13 => PageSegMode::SingleLine,
        _ => PageSegMode::Auto,
    };

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
    config.layout_analysis.page_seg_mode = psm_mode;
    config.layout_analysis.enable_orientation_detection = osd || preprocess;
    config.performance.device = device.to_string();
    config
}

pub fn format_result(result: &TextResult, format: &str) -> Result<String> {
    use ocr::core::output::{format_alto, format_box, format_hocr, format_tsv, to_json_output};
    use tracing::warn;

    match format.to_lowercase().as_str() {
        "json" => Ok(serde_json::to_string_pretty(&to_json_output(result))?),
        "hocr" | "html" => Ok(format_hocr(result)?),
        "tsv" => Ok(format_tsv(result)?),
        "alto" | "xml" => Ok(format_alto(result)?),
        "box" => Ok(format_box(result)?),
        "pdf" => Ok(
            "PDF output requires binary generation. Use --format pdf with --output.".to_string(),
        ),
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

pub async fn generate_searchable_pdf(
    image_path: &std::path::Path,
    result: &TextResult,
    output: Option<PathBuf>,
) -> Result<()> {
    let out_path = match output {
        Some(p) => p,
        None => return Err(anyhow::anyhow!("PDF output requires --output file path")),
    };

    let img_data = std::fs::read(image_path)?;
    let dynamic_img = image::load_from_memory(&img_data)?;
    let img_width_px = dynamic_img.width();
    let img_height_px = dynamic_img.height();

    let px_to_pt = |px: f32| printpdf::Pt(px * 72.0 / 300.0);
    let page_width_pt = px_to_pt(img_width_px as f32);
    let page_height_pt = px_to_pt(img_height_px as f32);

    let mut doc = printpdf::PdfDocument::new("OCR Output");

    let rgb_img = dynamic_img.to_rgb8();
    let (w, h) = (rgb_img.width(), rgb_img.height());
    let raw_image = printpdf::RawImage {
        pixels: printpdf::RawImageData::U8(rgb_img.into_raw()),
        width: w as usize,
        height: h as usize,
        data_format: printpdf::RawImageFormat::RGB8,
        tag: Vec::new(),
    };
    let image_id = doc.add_image(&raw_image);

    let mut ops = Vec::new();

    // Draw the original image as background
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

    // Overlay invisible text for searchability
    ops.push(printpdf::Op::SetTextRenderingMode {
        mode: printpdf::TextRenderingMode::Invisible,
    });

    for line in &result.lines {
        for word in &line.words {
            let bbox = &word.bounding_box;
            let x = px_to_pt(bbox.left as f32);
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

pub fn write_output(content: &str, output: Option<PathBuf>) -> Result<()> {
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
