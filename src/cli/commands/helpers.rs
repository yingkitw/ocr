use std::path::PathBuf;

use anyhow::Result;
use tracing::info;

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

pub fn parse_engine(s: &str) -> ocr::core::config::RecognitionEngine {
    // Only the CRNN engine exists; the legacy `--engine pattern` flag is
    // accepted and silently mapped to LSTM for back-compat.
    let _ = s;
    ocr::core::config::RecognitionEngine::LSTM
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
