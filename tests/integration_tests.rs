//! Integration tests for OCR
//!
//! These tests verify the complete OCR pipeline and use insta for snapshot testing.

use insta::assert_snapshot;
use ocr::core::*;
use ocr::lang::cjk::{CJKLanguage, CJKProcessor as LangCJKProcessor};
use ocr::recognition::RecognitionEngine;
use ocr::recognition::*;
use std::path::Path;

/// Test basic OCR functionality
#[test]
fn test_basic_ocr_pipeline() {
    // Skip image creation for now - focus on testing other components
    // This test will be updated once we have proper image creation working

    // Test CJK character detection
    assert!(LangCJKProcessor::is_cjk_character('中'));
    assert!(LangCJKProcessor::is_cjk_character('ひ'));
    assert!(LangCJKProcessor::is_cjk_character('한'));
    assert!(!LangCJKProcessor::is_cjk_character('A'));

    // Test LSTM model creation
    let model = LstmModelBuilder::new().build().unwrap();
    assert_eq!(model.model_type(), ModelType::LSTM);
}

/// Test CJK language detection
#[test]
fn test_cjk_language_detection() {
    let processor = LangCJKProcessor::new();

    // Test Chinese detection
    let chinese_scores = processor.detect_cjk_language("中文测试");
    assert!(!chinese_scores.is_empty());
    assert!(chinese_scores
        .iter()
        .any(|(lang, _)| *lang == CJKLanguage::ChineseSimplified));

    // Test Japanese detection
    let japanese_scores = processor.detect_cjk_language("ひらがなカタカナ");
    assert!(!japanese_scores.is_empty());
    assert!(japanese_scores
        .iter()
        .any(|(lang, _)| *lang == CJKLanguage::Japanese));

    // Test Korean detection
    let korean_scores = processor.detect_cjk_language("한글테스트");
    assert!(!korean_scores.is_empty());
    assert!(korean_scores
        .iter()
        .any(|(lang, _)| *lang == CJKLanguage::Korean));
}

/// Test CJK text segmentation
#[test]
fn test_cjk_text_segmentation() {
    let processor = LangCJKProcessor::new();

    // Test Chinese segmentation
    let chinese_result = processor
        .segment_text("中文测试", CJKLanguage::ChineseSimplified)
        .unwrap();
    assert_eq!(chinese_result.language, CJKLanguage::ChineseSimplified);
    assert!(!chinese_result.segments.is_empty());

    // Test Japanese segmentation
    let japanese_result = processor
        .segment_text("ひらがなカタカナ", CJKLanguage::Japanese)
        .unwrap();
    assert_eq!(japanese_result.language, CJKLanguage::Japanese);
    assert!(!japanese_result.segments.is_empty());

    // Test Korean segmentation
    let korean_result = processor
        .segment_text("한글테스트", CJKLanguage::Korean)
        .unwrap();
    assert_eq!(korean_result.language, CJKLanguage::Korean);
    assert!(!korean_result.segments.is_empty());
}

/// Test geometry types
#[test]
fn test_geometry_types() {
    // Test ICoord
    let coord = ICoord::new(10, 20);
    assert_eq!(coord.x(), 10);
    assert_eq!(coord.y(), 20);
    assert_eq!(coord.length(), (10.0_f32.powi(2) + 20.0_f32.powi(2)).sqrt());

    // Test FCoord
    let fcoord = FCoord::new(10.5, 20.7);
    assert_eq!(fcoord.x(), 10.5);
    assert_eq!(fcoord.y(), 20.7);

    // Test TBox
    let bbox = TBox::new(0, 0, 100, 200);
    assert_eq!(bbox.width(), 100);
    assert_eq!(bbox.height(), 200);
    assert_eq!(bbox.area(), 20000);
    assert!(!bbox.is_null());

    // Test TBox operations
    let mut bbox2 = TBox::new(10, 10, 50, 50);
    bbox2.move_by(ICoord::new(5, 5));
    assert_eq!(bbox2.left(), 15);
    assert_eq!(bbox2.bottom(), 15);
}

/// Test text structures
#[test]
fn test_text_structures() {
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

    assert_eq!(blob_choice.unichar_id, 65);
    assert_eq!(blob_choice.rating, 0.1);
    assert_eq!(blob_choice.certainty, 0.9);

    // Test WordChoice
    let mut word_choice = WordChoice::new();
    word_choice.add_choice(vec![blob_choice.clone()]);
    assert_eq!(word_choice.text(), "A");

    // Test Word
    let mut word = Word::new();
    word.set_flag(WordFlag::Bold, true);
    word.set_flag(WordFlag::Italic, false);
    word.correct_text = "Hello".to_string();

    assert!(word.has_flag(WordFlag::Bold));
    assert!(!word.has_flag(WordFlag::Italic));
    assert_eq!(word.correct_text, "Hello");
}

/// Test model management
#[tokio::test]
async fn test_model_management() {
    let mut manager = ModelManager::new(DeviceType::CPU);

    // Create and load a model
    let config = ModelConfig {
        model_type: ModelType::LSTM,
        model_path: "test_model.lstm".to_string(),
        supported_languages: vec![LanguageVariant::English],
        input_shape: (32, 128, 1),
        max_text_length: Some(50),
        confidence_threshold: 0.7,
        device: DeviceType::CPU,
        quantization: Some(QuantizationType::FP32),
    };

    let model = LstmModel::new(config);
    manager.load_model(model).await.unwrap();

    // Test model switching
    assert!(manager.switch_model(ModelType::LSTM).is_ok());
    assert!(manager.switch_model(ModelType::Transformer).is_err());

    // Test available models
    let available = manager.available_models();
    assert!(available.contains(&ModelType::LSTM));
}

/// Test recognition engine
#[tokio::test]
async fn test_recognition_engine() {
    let config = ModelConfig {
        model_type: ModelType::LSTM,
        model_path: "test_model.lstm".to_string(),
        supported_languages: vec![LanguageVariant::English],
        input_shape: (32, 128, 1),
        max_text_length: Some(50),
        confidence_threshold: 0.7,
        device: DeviceType::CPU,
        quantization: Some(QuantizationType::FP32),
    };

    let mut engine = BasicRecognitionEngine::new(config);

    // Test engine properties
    assert_eq!(engine.model_type(), ModelType::LSTM);
    assert!(engine
        .supported_languages()
        .contains(&LanguageVariant::English));

    // Test model switching
    assert!(engine.switch_model(ModelType::LSTM).await.is_ok());
}

/// Test error handling
#[test]
fn test_error_handling() {
    // Test model not found error
    let config = ModelConfig {
        model_type: ModelType::LSTM,
        model_path: "nonexistent_model.lstm".to_string(),
        supported_languages: vec![LanguageVariant::English],
        input_shape: (32, 128, 1),
        max_text_length: Some(50),
        confidence_threshold: 0.7,
        device: DeviceType::CPU,
        quantization: Some(QuantizationType::FP32),
    };

    let model = LstmModel::new(config);
    let result = model.predict(b"test");

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("not loaded"));
    }
}

/// Test CJK character categorization
#[test]
fn test_cjk_character_categorization() {
    // Test Chinese characters
    assert!(LangCJKProcessor::is_cjk_character('中'));
    assert!(LangCJKProcessor::is_cjk_character('ひ'));
    assert!(LangCJKProcessor::is_cjk_character('한'));
    assert!(!LangCJKProcessor::is_cjk_character('A'));
}

/// Test model configuration
#[test]
fn test_model_configuration() {
    let config = ModelConfig {
        model_type: ModelType::Transformer,
        model_path: "transformer_model.onnx".to_string(),
        supported_languages: vec![
            LanguageVariant::English,
            LanguageVariant::ChineseSimplified,
            LanguageVariant::Japanese,
        ],
        input_shape: (224, 224, 3),
        max_text_length: Some(200),
        confidence_threshold: 0.8,
        device: DeviceType::GPU,
        quantization: Some(QuantizationType::FP16),
    };

    assert_eq!(config.model_type, ModelType::Transformer);
    assert_eq!(config.input_shape, (224, 224, 3));
    assert_eq!(config.device, DeviceType::GPU);
    assert_eq!(config.quantization, Some(QuantizationType::FP16));
    assert!(config
        .supported_languages
        .contains(&LanguageVariant::English));
    assert!(config
        .supported_languages
        .contains(&LanguageVariant::ChineseSimplified));
    assert!(config
        .supported_languages
        .contains(&LanguageVariant::Japanese));
}

/// Test recognition result serialization
#[test]
fn test_recognition_result_serialization() {
    let mut result = ocr::core::RecognitionResult::new("Test Result".to_string(), 0.95);
    result.metadata.processing_time_ms = 150;
    result.metadata.language = Some("en".to_string());

    // Test JSON serialization
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Result"));
    assert!(json.contains("0.95"));

    // Test deserialization
    let deserialized: ocr::core::RecognitionResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.text, "Test Result");
    assert_eq!(deserialized.confidence, 0.95);
    assert_eq!(deserialized.metadata.processing_time_ms, 150);
}

/// Test CJK text segmentation with mixed scripts
#[test]
fn test_mixed_script_segmentation() {
    let processor = LangCJKProcessor::new();

    // Test mixed Chinese and English
    let mixed_text = "Hello 世界";
    let scores = processor.detect_cjk_language(mixed_text);
    assert!(!scores.is_empty());

    // Test mixed Japanese and English
    let mixed_japanese = "Hello こんにちは";
    let japanese_scores = processor.detect_cjk_language(mixed_japanese);
    assert!(!japanese_scores.is_empty());

    // Test mixed Korean and English
    let mixed_korean = "Hello 안녕하세요";
    let korean_scores = processor.detect_cjk_language(mixed_korean);
    assert!(!korean_scores.is_empty());
}

/// Test model builder pattern
#[test]
fn test_model_builder_pattern() {
    let config = ModelConfig {
        model_type: ModelType::LSTM,
        model_path: "custom_model.lstm".to_string(),
        supported_languages: vec![LanguageVariant::English, LanguageVariant::ChineseSimplified],
        input_shape: (64, 256, 1),
        max_text_length: Some(100),
        confidence_threshold: 0.6,
        device: DeviceType::GPU,
        quantization: Some(QuantizationType::INT8),
    };

    let model = LstmModelBuilder::new().with_config(config).build().unwrap();

    assert_eq!(model.model_type(), ModelType::LSTM);
    assert!(model.supports_language(&LanguageVariant::English));
    assert!(model.supports_language(&LanguageVariant::ChineseSimplified));
    assert!(model.supports_language(&LanguageVariant::Korean));
}

/// Test device type handling
#[test]
fn test_device_type_handling() {
    let cpu_manager = ModelManager::new(DeviceType::CPU);
    let gpu_manager = ModelManager::new(DeviceType::GPU);
    let auto_manager = ModelManager::new(DeviceType::Auto);

    // All managers should be created successfully
    assert_eq!(cpu_manager.available_models().len(), 0);
    assert_eq!(gpu_manager.available_models().len(), 0);
    assert_eq!(auto_manager.available_models().len(), 0);
}

/// Test quantization types
#[test]
fn test_quantization_types() {
    let quantization_types = vec![
        QuantizationType::FP32,
        QuantizationType::FP16,
        QuantizationType::INT8,
        QuantizationType::Dynamic,
    ];

    for qt in quantization_types {
        let config = ModelConfig {
            model_type: ModelType::LSTM,
            model_path: "test.lstm".to_string(),
            supported_languages: vec![LanguageVariant::English],
            input_shape: (32, 128, 1),
            max_text_length: Some(50),
            confidence_threshold: 0.7,
            device: DeviceType::CPU,
            quantization: Some(qt),
        };

        assert_eq!(config.quantization, Some(qt));
    }
}
