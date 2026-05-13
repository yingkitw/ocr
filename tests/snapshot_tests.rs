//! Snapshot tests for MiniOCR using insta
//!
//! These tests capture the output of various OCR operations and ensure
//! they remain consistent over time.

use insta::assert_snapshot;
use ocr::core::*;
use ocr::lang::{CJKLanguage, CJKProcessor as LangCJKProcessor};
use ocr::recognition::{ModelConfig, *};
use serde::{Deserialize, Serialize};
use serde_json;

// Define missing types for tests
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnicodeCategory {
    Latin,
    Greek,
    Cyrillic,
    Arabic,
    Hebrew,
    CJKUnifiedIdeographs,
    Hiragana,
    Katakana,
    HangulSyllables,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScriptType {
    Latin,
    Chinese,
    Japanese,
    Korean,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadingOrder {
    LeftToRight,
    RightToLeft,
    TopToBottom,
}

/// Test CJK character detection snapshots
#[test]
fn test_cjk_character_detection_snapshot() {
    let processor = LangCJKProcessor::new();

    let test_cases = vec![
        ('中', "Chinese character"),
        ('ひ', "Japanese Hiragana"),
        ('カ', "Japanese Katakana"),
        ('한', "Korean Hangul"),
        ('A', "Latin character"),
        ('1', "Digit"),
        ('!', "Punctuation"),
    ];

    let mut results = Vec::new();
    for (char, description) in test_cases {
        let is_cjk = LangCJKProcessor::is_cjk_character(char);
        let is_chinese = LangCJKProcessor::is_cjk_character(char)
            && char as u32 >= 0x4E00
            && char as u32 <= 0x9FFF;
        let is_japanese = LangCJKProcessor::is_cjk_character(char)
            && (char as u32 >= 0x3040 && char as u32 <= 0x309F
                || char as u32 >= 0x30A0 && char as u32 <= 0x30FF);
        let is_korean = LangCJKProcessor::is_cjk_character(char)
            && char as u32 >= 0xAC00
            && char as u32 <= 0xD7AF;
        let unicode_category = LangCJKProcessor::get_character_category(char);
        let script_type = "CJK";

        results.push(format!(
            "Character: {} ({})\n  CJK: {}\n  Chinese: {}\n  Japanese: {}\n  Korean: {}\n  Unicode Category: {:?}\n  Script Type: {:?}\n",
            char, description, is_cjk, is_chinese, is_japanese, is_korean, unicode_category, script_type
        ));
    }

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test CJK language detection snapshots
#[test]
fn test_cjk_language_detection_snapshot() {
    let processor = LangCJKProcessor::new();

    let test_texts = vec![
        "Hello World",
        "中文测试",
        "ひらがなカタカナ",
        "한글테스트",
        "Hello 世界",
        "こんにちは World",
        "안녕하세요 World",
        "Mixed 中文 and English",
        "日本語とEnglish混合",
        "한국어와English混合",
    ];

    let mut results = Vec::new();
    for text in test_texts {
        let scores = processor.detect_cjk_language(text);
        let mut score_strings: Vec<String> = scores
            .iter()
            .map(|(lang, score)| format!("{:?}: {:.3}", lang, score))
            .collect();
        score_strings.sort();

        results.push(format!(
            "Text: {}\nLanguage scores: {}\n",
            text,
            score_strings.join(", ")
        ));
    }

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test CJK text segmentation snapshots
#[test]
fn test_cjk_text_segmentation_snapshot() {
    let processor = LangCJKProcessor::new();

    let test_cases = vec![
        ("中文测试", CJKLanguage::ChineseSimplified),
        ("ひらがなカタカナ", CJKLanguage::Japanese),
        ("한글테스트", CJKLanguage::Korean),
        ("Hello World", CJKLanguage::ChineseSimplified), // Will be treated as Chinese
    ];

    let mut results = Vec::new();
    for (text, language) in test_cases {
        let result = processor.segment_text(text, language).unwrap();

        let mut segment_strings: Vec<String> = result
            .segments
            .iter()
            .map(|seg| {
                format!(
                    "Text: '{}', Category: {:?}, Confidence: {:.3}",
                    seg.text, seg.category, seg.confidence
                )
            })
            .collect();

        results.push(format!(
            "Input: '{}' (Language: {:?})\nSegments:\n{}\n",
            text,
            result.language,
            segment_strings.join("\n")
        ));
    }

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test geometry types snapshots
#[test]
fn test_geometry_types_snapshot() {
    let mut results = Vec::new();

    // Test ICoord
    let coord1 = ICoord::new(10, 20);
    let coord2 = ICoord::new(-5, 15);
    let coord3 = coord1 + coord2;
    let coord4 = coord1 - coord2;

    results.push(format!(
        "ICoord Tests:\n  coord1: {:?}\n  coord2: {:?}\n  coord1 + coord2: {:?}\n  coord1 - coord2: {:?}\n  coord1 length: {:.3}\n",
        coord1, coord2, coord3, coord4, coord1.length()
    ));

    // Test FCoord
    let fcoord1 = FCoord::new(10.5, 20.7);
    let fcoord2 = FCoord::new(-5.2, 15.3);
    let fcoord3 = fcoord1 + fcoord2;
    let fcoord4 = fcoord1 - fcoord2;

    results.push(format!(
        "FCoord Tests:\n  fcoord1: {:?}\n  fcoord2: {:?}\n  fcoord1 + fcoord2: {:?}\n  fcoord1 - fcoord2: {:?}\n  fcoord1 length: {:.3}\n",
        fcoord1, fcoord2, fcoord3, fcoord4, fcoord1.length()
    ));

    // Test TBox
    let bbox1 = TBox::new(0, 0, 100, 200);
    let bbox2 = TBox::new(50, 50, 150, 250);
    let bbox3 = bbox1.intersection(&bbox2);
    let bbox4 = bbox1.union(&bbox2);

    results.push(format!(
        "TBox Tests:\n  bbox1: {:?} (area: {})\n  bbox2: {:?} (area: {})\n  intersection: {:?}\n  union: {:?}\n  bbox1 contains (25, 25): {}\n  bbox1 overlaps bbox2: {}\n",
        bbox1, bbox1.area(),
        bbox2, bbox2.area(),
        bbox3,
        bbox4,
        bbox1.contains(ICoord::new(25, 25)),
        bbox1.overlaps(&bbox2)
    ));

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test text structures snapshots
#[test]
fn test_text_structures_snapshot() {
    let mut results = Vec::new();

    // Test BlobChoice
    let blob_choice = BlobChoice::new(
        65, // 'A'
        0.1,
        0.9,
        0, // Latin script
        10.0,
        20.0,
        0.0,
        BlobChoiceClassifier::StaticClassifier,
    );

    results.push(format!(
        "BlobChoice:\n  Character: '{}' (ID: {})\n  Rating: {:.3}\n  Certainty: {:.3}\n  Script: {}\n  X-height: {:.1}-{:.1}\n  Classifier: {:?}\n",
        blob_choice.unichar_id as u8 as char,
        blob_choice.unichar_id,
        blob_choice.rating,
        blob_choice.certainty,
        blob_choice.script_id,
        blob_choice.min_xheight,
        blob_choice.max_xheight,
        blob_choice.classifier
    ));

    // Test WordChoice
    let mut word_choice = WordChoice::new();
    word_choice.add_choice(vec![blob_choice.clone()]);
    word_choice.rating = 0.8;
    word_choice.certainty = 0.85;
    word_choice.blanks = 1;
    word_choice.script_id = 0;

    results.push(format!(
        "WordChoice:\n  Text: '{}'\n  Rating: {:.3}\n  Certainty: {:.3}\n  Blanks: {}\n  Script: {}\n  Choices: {}\n",
        word_choice.text(),
        word_choice.rating,
        word_choice.certainty,
        word_choice.blanks,
        word_choice.script_id,
        word_choice.choices.len()
    ));

    // Test Word
    let mut word = Word::new();
    word.set_flag(WordFlag::Bold, true);
    word.set_flag(WordFlag::Italic, false);
    word.set_flag(WordFlag::StartOfLine, true);
    word.correct_text = "Hello".to_string();
    word.blanks = 2;
    word.script_id = 0;

    // Sort flags for consistent output (HashSet order is non-deterministic)
    let mut sorted_flags: Vec<String> = word.flags.iter().map(|f| format!("{:?}", f)).collect();
    sorted_flags.sort();

    results.push(format!(
        "Word:\n  Text: '{}'\n  Blanks: {}\n  Script: {}\n  Bold: {}\n  Italic: {}\n  Start of Line: {}\n  Flags: {{{}}}\n",
        word.correct_text,
        word.blanks,
        word.script_id,
        word.has_flag(WordFlag::Bold),
        word.has_flag(WordFlag::Italic),
        word.has_flag(WordFlag::StartOfLine),
        sorted_flags.join(", ")
    ));

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test model configuration snapshots
#[test]
fn test_model_configuration_snapshot() {
    let configs = vec![
        ModelConfig {
            model_type: ModelType::LSTM,
            model_path: "tesseract_lstm.lstm".to_string(),
            supported_languages: vec![
                LanguageVariant::English,
                LanguageVariant::ChineseSimplified,
                LanguageVariant::ChineseTraditional,
            ],
            input_shape: (32, 128, 1),
            max_text_length: Some(100),
            confidence_threshold: 0.5,
            device: DeviceType::CPU,
            quantization: Some(QuantizationType::FP32),
        },
        ModelConfig {
            model_type: ModelType::Transformer,
            model_path: "trocr_model.onnx".to_string(),
            supported_languages: vec![
                LanguageVariant::English,
                LanguageVariant::ChineseSimplified,
                LanguageVariant::Japanese,
                LanguageVariant::Korean,
            ],
            input_shape: (224, 224, 3),
            max_text_length: Some(200),
            confidence_threshold: 0.8,
            device: DeviceType::GPU,
            quantization: Some(QuantizationType::FP16),
        },
        ModelConfig {
            model_type: ModelType::VisionTransformer,
            model_path: "vit_ocr_model.onnx".to_string(),
            supported_languages: vec![LanguageVariant::English],
            input_shape: (384, 384, 3),
            max_text_length: Some(150),
            confidence_threshold: 0.9,
            device: DeviceType::NPU,
            quantization: Some(QuantizationType::INT8),
        },
    ];

    let mut results = Vec::new();
    for (i, config) in configs.iter().enumerate() {
        results.push(format!(
            "Model Config {}:\n  Type: {:?}\n  Path: {}\n  Languages: {:?}\n  Input Shape: {:?}\n  Max Text Length: {:?}\n  Confidence Threshold: {:.1}\n  Device: {:?}\n  Quantization: {:?}\n",
            i + 1,
            config.model_type,
            config.model_path,
            config.supported_languages,
            config.input_shape,
            config.max_text_length,
            config.confidence_threshold,
            config.device,
            config.quantization
        ));
    }

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test recognition result snapshots
#[test]
fn test_recognition_result_snapshot() {
    let mut result = ocr::core::RecognitionResult::new("Hello World".to_string(), 0.95);
    result.model_type = Some(ModelType::LSTM);
    result.processing_time_ms = Some(150);
    result.language = Some("en".to_string());

    // Add character results
    result.character_results = vec![
        CharacterRecognition::new('H', 0.98),
        CharacterRecognition::new('e', 0.96),
        CharacterRecognition::new('l', 0.94),
        CharacterRecognition::new('l', 0.92),
        CharacterRecognition::new('o', 0.90),
    ];

    // Add word results
    result.word_results = vec![
        WordRecognition::new("Hello".to_string(), 0.94),
        WordRecognition::new("World".to_string(), 0.92),
    ];

    // Add line results
    result.line_results = vec![LineRecognition::new("Hello World".to_string(), 0.93)];

    // Serialize to JSON for snapshot
    let json = serde_json::to_string_pretty(&result).unwrap();
    assert_snapshot!(json);
}

/// Test CJK mixed script handling snapshots
#[test]
fn test_cjk_mixed_script_snapshot() {
    let processor = LangCJKProcessor::new();

    let mixed_texts = vec![
        "Hello 世界",
        "こんにちは World",
        "안녕하세요 World",
        "Mixed 中文 and English text",
        "日本語とEnglish混合テキスト",
        "한국어와English混合텍스트",
        "Complex 中文日本語한국어English混合",
    ];

    let mut results = Vec::new();
    for text in mixed_texts {
        // Language detection
        let language_scores = processor.detect_cjk_language(text);
        let mut score_strings: Vec<String> = language_scores
            .iter()
            .map(|(lang, score)| format!("{:?}: {:.3}", lang, score))
            .collect();
        score_strings.sort();

        // Character analysis
        let mut char_analysis = Vec::new();
        for ch in text.chars() {
            let is_cjk = LangCJKProcessor::is_cjk_character(ch);
            let category = LangCJKProcessor::get_character_category(ch);
            let script = "CJK";
            char_analysis.push(format!(
                "'{}': CJK={}, Category={:?}, Script={:?}",
                ch, is_cjk, category, script
            ));
        }

        results.push(format!(
            "Text: '{}'\nLanguage scores: {}\nCharacter analysis:\n{}\n",
            text,
            score_strings.join(", "),
            char_analysis.join("\n")
        ));
    }

    let output = results.join("\n");
    assert_snapshot!(output);
}

/// Test model type and device combinations snapshots
#[test]
fn test_model_type_device_combinations_snapshot() {
    let model_types = vec![
        ModelType::LSTM,
        ModelType::Transformer,
        ModelType::VisionTransformer,
        ModelType::CNN,
        ModelType::Hybrid,
        ModelType::EndToEnd,
        ModelType::Custom("CustomModel".to_string()),
    ];

    let device_types = vec![
        DeviceType::CPU,
        DeviceType::GPU,
        DeviceType::NPU,
        DeviceType::Auto,
    ];

    let quantization_types = vec![
        QuantizationType::FP32,
        QuantizationType::FP16,
        QuantizationType::INT8,
        QuantizationType::Dynamic,
    ];

    let mut results = Vec::new();

    for model_type in &model_types {
        for device_type in &device_types {
            for quantization in &quantization_types {
                let config = ModelConfig {
                    model_type: model_type.clone(),
                    model_path: format!(
                        "model_{:?}_{:?}_{:?}.onnx",
                        model_type, device_type, quantization
                    ),
                    supported_languages: vec![LanguageVariant::English],
                    input_shape: (224, 224, 3),
                    max_text_length: Some(100),
                    confidence_threshold: 0.8,
                    device: *device_type,
                    quantization: Some(*quantization),
                };

                results.push(format!(
                    "Model: {:?}, Device: {:?}, Quantization: {:?}\n  Path: {}\n  Input Shape: {:?}\n  Confidence: {:.1}\n",
                    model_type, device_type, quantization,
                    config.model_path,
                    config.input_shape,
                    config.confidence_threshold
                ));
            }
        }
    }

    let output = results.join("\n");
    assert_snapshot!(output);
}
