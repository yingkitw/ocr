#[cfg(test)]
mod tests {
    use crate::core::ModelType;
    use crate::recognition::engine::{
        DeviceType, LanguageVariant, ModelConfig, ModelManager, QuantizationType,
    };
    use crate::recognition::lstm_model::LstmModel;
    use crate::recognition::pattern_model::PatternModel;

    #[tokio::test]
    async fn test_model_manager() {
        let mut manager = ModelManager::new(DeviceType::CPU);

        // Create LSTM config
        let lstm_config = ModelConfig {
            model_type: ModelType::LSTM,
            model_path: "test.lstm".to_string(),
            supported_languages: vec![LanguageVariant::English],
            input_shape: (32, 128, 1),
            max_text_length: Some(100),
            confidence_threshold: 0.5,
            device: DeviceType::CPU,
            quantization: Some(QuantizationType::FP32),
        };

        let lstm_model = LstmModel::new(lstm_config.clone());
        manager.load_model(lstm_model).await.unwrap();

        // Create Pattern config
        let pattern_config = ModelConfig {
            model_type: ModelType::Custom("PatternMatching".to_string()),
            model_path: "".to_string(),
            supported_languages: vec![LanguageVariant::English],
            input_shape: (0, 0, 0),
            max_text_length: None,
            confidence_threshold: 0.8,
            device: DeviceType::CPU,
            quantization: None,
        };

        let pattern_model = PatternModel::new(pattern_config);
        manager.load_model(pattern_model).await.unwrap();

        // Verify models are loaded
        assert!(manager.get_model(ModelType::LSTM).is_some());
        assert!(manager
            .get_model(ModelType::Custom("PatternMatching".to_string()))
            .is_some());

        // Verify switching
        manager.switch_model(ModelType::LSTM).unwrap();
        assert_eq!(
            manager.active_model().unwrap().model_type(),
            ModelType::LSTM
        );

        manager
            .switch_model(ModelType::Custom("PatternMatching".to_string()))
            .unwrap();
        match manager.active_model().unwrap().model_type() {
            ModelType::Custom(s) => assert_eq!(s, "PatternMatching"),
            _ => panic!("Wrong model type"),
        }
    }
}
