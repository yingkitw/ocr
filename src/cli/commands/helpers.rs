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

pub fn parse_engine(s: &str) -> anyhow::Result<ocr::core::config::RecognitionEngine> {
    // `pattern` is the legacy flag, silently mapped to LSTM for back-compat.
    match s.to_ascii_lowercase().as_str() {
        "pattern" | "lstm" => Ok(ocr::core::config::RecognitionEngine::LSTM),
        "template" => Ok(ocr::core::config::RecognitionEngine::Template),
        other => Err(anyhow::anyhow!(
            "unknown engine '{other}' (valid: pattern, lstm, template)"
        )),
    }
}

pub fn build_config(
    lang: &str,
    preprocess: bool,
    psm: &str,
    confidence: f32,
    engine: &str,
    dict_correct: bool,
    device: &str,
    osd: bool,
    dpi: u32,
    whitelist: Option<&str>,
) -> anyhow::Result<ocr::core::config::OcrConfig> {
    use ocr::core::config::{OcrConfig, PageSegMode};

    // Tesseract-compatible PSM parsing: numeric (0-13) or symbolic names
    // (`auto`, `sparse_text`, …). Errors carry the valid spellings.
    let psm_mode = PageSegMode::parse_psm(psm).map_err(|e| anyhow::anyhow!(e))?;

    let mut config = OcrConfig::default();
    config.recognition.language = lang.to_string();
    config.recognition.confidence_threshold = confidence;
    config.recognition.engine = parse_engine(engine)?;
    config.recognition.enable_dictionary_correction = dict_correct;
    if let Some(chars) = whitelist {
        let parsed: Vec<char> = chars.chars().collect();
        if !parsed.is_empty() {
            config.recognition.character_whitelist = Some(parsed);
        }
    }
    config.recognition.parameters.insert("dpi".to_string(), dpi.to_string());
    config.image_processing.enable_preprocessing = preprocess;
    config.image_processing.enable_binarization = preprocess;
    config.image_processing.enable_noise_reduction = preprocess;
    config.image_processing.enable_contrast_enhancement = preprocess;
    config.image_processing.enable_deskewing = preprocess;
    config.image_processing.target_dpi = dpi;
    config.layout_analysis.page_seg_mode = psm_mode;
    // OSD modes always need orientation detection, matching Tesseract where
    // `--psm 0/1/12` forces OSD regardless of other flags.
    config.layout_analysis.enable_orientation_detection = osd
        || preprocess
        || matches!(
            psm_mode,
            PageSegMode::OsdOnly | PageSegMode::AutoOsd | PageSegMode::SparseTextWithOsd
        );
    config.performance.device = device.to_string();
    Ok(config)
}

pub fn format_result(result: &TextResult, format: &str) -> Result<String> {
    use ocr::core::output::{format_output, parse_output_format};

    // Route every format through the library dispatcher so CLI flags stay in
    // sync with `OutputFormat` (text, json, hocr, tsv, csv, alto, box, markdown,
    // structured-json, rag, page). Warn on empty results but still emit valid
    // structured output (e.g. `{"chunks": []}`).
    if result.text.is_empty() {
        tracing::warn!("No text recognized");
    }
    let fmt = parse_output_format(format)?;
    Ok(format_output(result, fmt)?)
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
