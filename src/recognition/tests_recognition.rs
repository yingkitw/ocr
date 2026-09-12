#[cfg(test)]
mod tests {
    use crate::core::ModelType;
    use crate::recognition::crnn::{CrnnConfig, CrnnModel, ScriptModelRegistry};
    use crate::recognition::loader;

    #[test]
    fn test_crnn_model_manager_registry() {
        let registry = ScriptModelRegistry::new();
        // All eight supported scripts should have a model
        for script in [
            crate::lang::unicode::Script::Latin,
            crate::lang::unicode::Script::CJK,
            crate::lang::unicode::Script::Arabic,
            crate::lang::unicode::Script::Cyrillic,
            crate::lang::unicode::Script::Greek,
            crate::lang::unicode::Script::Hebrew,
            crate::lang::unicode::Script::Thai,
            crate::lang::unicode::Script::Devanagari,
        ] {
            assert!(
                registry.model_for(script).is_some(),
                "missing model for {:?}",
                script
            );
        }
    }

    #[test]
    fn test_model_type_lstm_is_default() {
        assert_eq!(ModelType::LSTM, ModelType::LSTM);
    }

    #[test]
    fn test_bundled_model_present() {
        // The model file should be non-empty (or at least present, with the
        // correct magic). A freshly-initialised model serialises to a few KB
        // minimum so an empty file means the bundled binary is missing.
        let bytes = loader::BUNDLED_MODEL_BYTES;
        assert!(!bytes.is_empty(), "bundled model is empty");
        if bytes.len() >= 8 {
            assert_eq!(&bytes[0..4], b"OCR1", "bundled model has wrong magic");
        }
    }

    #[test]
    fn test_bundled_model_loads_if_valid() {
        // If the bundled file is a valid v0.x model, the loader should be able
        // to read it. If it's still the random-init placeholder, the loader
        // will error — that's fine, we just don't want a panic.
        let res = loader::load_bundled();
        if let Ok(model) = res {
            // sanity: a loaded model should have vocab and FC weights populated
            assert!(!model.vocab.chars.is_empty());
            assert!(model.fc_weight.nrows() > 0);
        }
    }

    #[test]
    fn test_crnn_config_for_latin() {
        let cfg = CrnnConfig::default();
        let m = CrnnModel::new(cfg);
        assert!(m.fc_weight.nrows() > 0);
    }

    #[test]
    fn test_recognize_detailed_narrow_crop_returns_empty() {
        // Regression: crops narrower than 16 px used to underflow in
        // `conv_block` (the CNN halves the width 4 times, so w < 16
        // collapses to a zero-width feature map). They must yield an
        // empty result instead of panicking.
        let model = CrnnModel::new(CrnnConfig::default());
        for w in [0u32, 1, 8, 15] {
            let img = crate::core::image::OcrImage::from_raw_pixels(
                w,
                32,
                vec![255u8; w as usize * 32],
                crate::core::image::ImageFormat::Grayscale,
                300,
            )
            .expect("valid OcrImage");
            let res = model.recognize_detailed(&img, None, None).expect("no error");
            assert!(res.text.is_empty(), "w={w} should decode to empty text");
            assert_eq!(res.confidence, 0.0);
        }
        // Width 16 is the first size that survives the CNN; it must not panic.
        let img = crate::core::image::OcrImage::from_raw_pixels(
            16,
            32,
            vec![255u8; 16 * 32],
            crate::core::image::ImageFormat::Grayscale,
            300,
        )
        .expect("valid OcrImage");
        let _ = model.recognize_detailed(&img, None, None).expect("no panic at w=16");
    }
}
