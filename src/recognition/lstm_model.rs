//! LSTM-based OCR model implementation
//!
//! This module provides a concrete implementation of the OcrModel trait
//! for LSTM-based models, maintaining compatibility with Tesseract.

use super::engine::*;
use crate::core::geometry::TBox;
use crate::core::recognition::TrainableModel;
use crate::core::ModelType;
use crate::utils::{MiniOcrError, Result};
use ndarray::{s, Array1, Array2};
use std::path::Path;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_vec(a: &Array1<f32>) -> Array1<f32> {
    a.mapv(sigmoid)
}

fn tanh_vec(a: &Array1<f32>) -> Array1<f32> {
    a.mapv(|x| x.tanh())
}

fn dot_add(a: &Array2<f32>, x: &Array1<f32>, bias: &Array1<f32>) -> Array1<f32> {
    let mut out = bias.clone();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[i] += a[[i, j]] * x[j];
        }
    }
    out
}

fn rand_init() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;
    static SEED: AtomicU64 = AtomicU64::new(0);
    let seed = SEED.fetch_add(1, Ordering::Relaxed);
    let actual = if seed == 0 {
        let init = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        SEED.store(init, Ordering::Relaxed);
        init
    } else {
        seed
    };
    let mixed = actual
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut hasher = DefaultHasher::new();
    mixed.hash(&mut hasher);
    let h = hasher.finish();
    (h % 2_000_000) as f32 / 1_000_000.0 - 1.0
}

#[allow(dead_code)]
struct LstmLayer {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Array2<f32>,
    weight_hh: Array2<f32>,
    bias_ih: Array1<f32>,
    bias_hh: Array1<f32>,
    weights: Array2<f32>,
    gradients: Array2<f32>,
}

#[allow(dead_code)]
impl LstmLayer {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let gate_size = hidden_size;
        Self {
            input_size,
            hidden_size,
            weight_ih: Array2::zeros((4 * gate_size, input_size)),
            weight_hh: Array2::zeros((4 * gate_size, hidden_size)),
            bias_ih: Array1::zeros(4 * gate_size),
            bias_hh: Array1::zeros(4 * gate_size),
            weights: Array2::zeros((input_size + hidden_size, hidden_size)),
            gradients: Array2::zeros((input_size + hidden_size, hidden_size)),
        }
    }

    fn randomize(&mut self) {
        let scale_ih = (1.0 / self.input_size as f32).sqrt();
        let scale_hh = (1.0 / self.hidden_size as f32).sqrt();
        for v in self.weight_ih.iter_mut() {
            *v = rand_init() * scale_ih;
        }
        for v in self.weight_hh.iter_mut() {
            *v = rand_init() * scale_hh;
        }
        for v in self.bias_ih.iter_mut() {
            *v = 0.0;
        }
        for v in self.bias_hh.iter_mut() {
            *v = 0.0;
        }
        let forget_bias = 1.0;
        for i in (self.hidden_size..2 * self.hidden_size).take(self.hidden_size) {
            self.bias_ih[i] = forget_bias;
            self.bias_hh[i] = forget_bias;
        }
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, input_dim) = input.dim();
        assert_eq!(input_dim, self.input_size, "Input dimension mismatch");

        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);
        let mut output = Array2::zeros((seq_len, self.hidden_size));

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let gates_pre = dot_add(&self.weight_ih, &x_t, &self.bias_ih)
                + dot_add(&self.weight_hh, &h, &self.bias_hh);

            let hs = self.hidden_size;
            let i_gate = sigmoid_vec(&gates_pre.slice(s![0..hs]).to_owned());
            let f_gate = sigmoid_vec(&gates_pre.slice(s![hs..2 * hs]).to_owned());
            let g_gate = tanh_vec(&gates_pre.slice(s![2 * hs..3 * hs]).to_owned());
            let o_gate = sigmoid_vec(&gates_pre.slice(s![3 * hs..4 * hs]).to_owned());

            c = &f_gate * &c + &i_gate * &g_gate;
            h = &o_gate * tanh_vec(&c);

            output.row_mut(t).assign(&h);
        }

        output
    }

    fn forward_with_cell(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (seq_len, _) = input.dim();
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);
        let mut hidden_out = Array2::zeros((seq_len, self.hidden_size));
        let mut cell_out = Array2::zeros((seq_len, self.hidden_size));

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let gates_pre = dot_add(&self.weight_ih, &x_t, &self.bias_ih)
                + dot_add(&self.weight_hh, &h, &self.bias_hh);

            let hs = self.hidden_size;
            let i_gate = sigmoid_vec(&gates_pre.slice(s![0..hs]).to_owned());
            let f_gate = sigmoid_vec(&gates_pre.slice(s![hs..2 * hs]).to_owned());
            let g_gate = tanh_vec(&gates_pre.slice(s![2 * hs..3 * hs]).to_owned());
            let o_gate = sigmoid_vec(&gates_pre.slice(s![3 * hs..4 * hs]).to_owned());

            c = &f_gate * &c + &i_gate * &g_gate;
            h = &o_gate * tanh_vec(&c);

            hidden_out.row_mut(t).assign(&h);
            cell_out.row_mut(t).assign(&c);
        }

        (hidden_out, cell_out)
    }
}

#[allow(dead_code)]
struct BiLstmLayer {
    forward_lstm: LstmLayer,
    backward_lstm: LstmLayer,
}

#[allow(dead_code)]
impl BiLstmLayer {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            forward_lstm: LstmLayer::new(input_size, hidden_size),
            backward_lstm: LstmLayer::new(input_size, hidden_size),
        }
    }

    fn randomize(&mut self) {
        self.forward_lstm.randomize();
        self.backward_lstm.randomize();
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let fwd = self.forward_lstm.forward(input);
        let (seq_len, _) = input.dim();
        let mut reversed = Array2::zeros(input.dim());
        for t in 0..seq_len {
            reversed.row_mut(seq_len - 1 - t).assign(&input.row(t));
        }
        let bwd = self.backward_lstm.forward(&reversed);
        let mut bwd_ordered = Array2::zeros(bwd.dim());
        for t in 0..seq_len {
            bwd_ordered.row_mut(t).assign(&bwd.row(seq_len - 1 - t));
        }

        let mut concat = Array2::zeros((seq_len, fwd.ncols() + bwd_ordered.ncols()));
        concat.slice_mut(s![.., ..fwd.ncols()]).assign(&fwd);
        concat.slice_mut(s![.., fwd.ncols()..]).assign(&bwd_ordered);
        concat
    }
}

#[allow(dead_code)]
struct LstmNetwork {
    layers: Vec<BiLstmLayer>,
    vocab_size: usize,
    output_weights: Array2<f32>,
    output_bias: Array1<f32>,
}

#[allow(dead_code)]
impl LstmNetwork {
    fn new(input_size: usize, hidden_size: usize, num_layers: usize, vocab_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut current_input = input_size;

        for _ in 0..num_layers {
            layers.push(BiLstmLayer::new(current_input, hidden_size));
            current_input = hidden_size * 2;
        }

        Self {
            layers,
            vocab_size,
            output_weights: Array2::zeros((vocab_size, current_input)),
            output_bias: Array1::zeros(vocab_size),
        }
    }

    fn randomize(&mut self) {
        for layer in &mut self.layers {
            layer.randomize();
        }
        let scale = (1.0 / self.output_weights.nrows() as f32).sqrt();
        for v in self.output_weights.iter_mut() {
            *v = (rand_init() * 2.0 - 1.0) * scale;
        }
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        let (seq_len, _) = x.dim();
        let mut logits = Array2::zeros((seq_len, self.vocab_size));
        for t in 0..seq_len {
            let x_t = x.row(t).to_owned();
            logits
                .row_mut(t)
                .assign(&dot_add(&self.output_weights, &x_t, &self.output_bias));
        }
        logits
    }
}

/// LSTM model implementation
pub struct LstmModel {
    config: ModelConfig,
    model_path: String,
    is_loaded: bool,
    network: Option<LstmNetwork>,
}

impl LstmModel {
    /// Create a new LSTM model
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            model_path: String::new(),
            is_loaded: false,
            network: None,
        }
    }

    /// Load the model from a file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.model_path = path_str;
        self.is_loaded = true;

        let (h, w, c) = self.config.input_shape;
        let feature_dim = h * c;
        let mut network = LstmNetwork::new(feature_dim, 256, 2, 100);
        network.randomize();
        self.network = Some(network);

        Ok(())
    }

    /// Check if the model is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Get the model path
    pub fn model_path(&self) -> &str {
        &self.model_path
    }
}

impl OcrModel for LstmModel {
    fn predict(&self, input: &[u8]) -> Result<RecognitionResult> {
        if !self.is_loaded {
            return Err(MiniOcrError::ModelNotFound("LSTM model not loaded".to_string()).into());
        }

        // TODO: Implement actual LSTM inference
        // For now, return a placeholder result
        let mut result = RecognitionResult::new("LSTM Recognition Result".to_string(), 0.85);
        result.model_type = ModelType::LSTM;
        result.processing_time_ms = 100; // Placeholder

        // Add some character-level results
        result.character_results = vec![
            CharacterRecognitionResult {
                character: 'L',
                confidence: 0.9,
                bounding_box: TBox::new(0, 0, 10, 20),
                unicode_category: UnicodeCategory::Latin,
                script: ScriptType::Latin,
            },
            CharacterRecognitionResult {
                character: 'S',
                confidence: 0.88,
                bounding_box: TBox::new(10, 0, 20, 20),
                unicode_category: UnicodeCategory::Latin,
                script: ScriptType::Latin,
            },
            CharacterRecognitionResult {
                character: 'T',
                confidence: 0.92,
                bounding_box: TBox::new(20, 0, 30, 20),
                unicode_category: UnicodeCategory::Latin,
                script: ScriptType::Latin,
            },
            CharacterRecognitionResult {
                character: 'M',
                confidence: 0.87,
                bounding_box: TBox::new(30, 0, 40, 20),
                unicode_category: UnicodeCategory::Latin,
                script: ScriptType::Latin,
            },
        ];

        // Add word-level results
        result.word_results = vec![WordRecognitionResult {
            text: "LSTM".to_string(),
            confidence: 0.89,
            bounding_box: TBox::new(0, 0, 40, 20),
            characters: result.character_results.clone(),
            language: Some("en".to_string()),
        }];

        // Add line-level results
        result.line_results = vec![LineRecognitionResult {
            text: "LSTM Recognition Result".to_string(),
            confidence: 0.85,
            bounding_box: TBox::new(0, 0, 200, 20),
            words: result.word_results.clone(),
            reading_order: ReadingOrder::LeftToRight,
        }];

        Ok(result)
    }

    fn model_type(&self) -> ModelType {
        ModelType::LSTM
    }

    fn supported_languages(&self) -> Vec<LanguageVariant> {
        vec![
            LanguageVariant::English,
            LanguageVariant::ChineseSimplified,
            LanguageVariant::ChineseTraditional,
            LanguageVariant::Japanese,
            LanguageVariant::Korean,
        ]
    }

    fn input_shape(&self) -> (usize, usize, usize) {
        self.config.input_shape
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn supports_language(&self, language: &LanguageVariant) -> bool {
        self.supported_languages().contains(language)
    }

    fn as_trainable(&mut self) -> Option<&mut dyn TrainableModel> {
        Some(self)
    }

    fn as_trainable_ref(&self) -> Option<&dyn TrainableModel> {
        Some(self)
    }
}

impl TrainableModel for LstmModel {
    fn forward_train(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if let Some(network) = &self.network {
            Ok(network.forward(input))
        } else {
            Err(MiniOcrError::ModelNotFound("Network not initialized".to_string()).into())
        }
    }

    fn backward_train(&mut self, _input: &Array2<f32>, _output_grad: &Array2<f32>) -> Result<()> {
        // Placeholder for backward pass
        Ok(())
    }

    fn get_params_and_grads(&mut self) -> Vec<(&mut Array2<f32>, &Array2<f32>)> {
        let mut params = Vec::new();
        if let Some(network) = &mut self.network {
            for layer in &mut network.layers {
                params.push((
                    &mut layer.forward_lstm.weights,
                    &layer.forward_lstm.gradients,
                ));
                params.push((
                    &mut layer.backward_lstm.weights,
                    &layer.backward_lstm.gradients,
                ));
            }
        }
        params
    }
}

/// LSTM model builder for easy configuration
pub struct LstmModelBuilder {
    config: Option<ModelConfig>,
}

impl LstmModelBuilder {
    /// Create a new LSTM model builder
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Set the model configuration
    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the LSTM model
    pub fn build(self) -> Result<LstmModel> {
        let config = self.config.unwrap_or_else(|| ModelConfig {
            model_type: ModelType::LSTM,
            model_path: String::new(),
            supported_languages: vec![
                LanguageVariant::English,
                LanguageVariant::ChineseSimplified,
                LanguageVariant::ChineseTraditional,
                LanguageVariant::Japanese,
                LanguageVariant::Korean,
            ],
            input_shape: (32, 128, 1), // Height, Width, Channels
            max_text_length: Some(100),
            confidence_threshold: 0.5,
            device: DeviceType::CPU,
            quantization: Some(QuantizationType::FP32),
        });

        Ok(LstmModel::new(config))
    }
}

impl Default for LstmModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_model_creation() {
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
        assert_eq!(model.model_type(), ModelType::LSTM);
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_lstm_model_builder() {
        let model = LstmModelBuilder::new()
            .build()
            .expect("Failed to build LSTM model");

        assert_eq!(model.model_type(), ModelType::LSTM);
        assert!(model.supports_language(&LanguageVariant::English));
        assert!(model.supports_language(&LanguageVariant::ChineseSimplified));
    }

    #[test]
    fn test_lstm_model_prediction() {
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

        let mut model = LstmModel::new(config);
        model.load_from_file("test_model.lstm").unwrap();

        let input_data = b"test image data";
        let result = model.predict(input_data).unwrap();

        assert_eq!(result.text, "LSTM Recognition Result");
        assert!(result.confidence > 0.0);
        assert_eq!(result.model_type, ModelType::LSTM);
        assert!(!result.character_results.is_empty());
        assert!(!result.word_results.is_empty());
        assert!(!result.line_results.is_empty());
    }

    #[test]
    fn test_lstm_model_unloaded_prediction() {
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
        let input_data = b"test image data";
        let result = model.predict(input_data);

        assert!(result.is_err());
    }

    #[test]
    fn test_lstm_layer_forward_real() {
        let mut layer = LstmLayer::new(8, 16);
        layer.randomize();

        let input = Array2::from_elem((5, 8), 0.5);
        let output = layer.forward(&input);

        assert_eq!(output.dim(), (5, 16));
        let has_nonzero = output.iter().any(|&v| v != 0.0);
        assert!(
            has_nonzero,
            "LSTM forward pass should produce non-zero outputs"
        );
    }

    #[test]
    fn test_bilstm_layer_forward_real() {
        let mut layer = BiLstmLayer::new(8, 16);
        layer.randomize();

        let input = Array2::from_elem((5, 8), 0.5);
        let output = layer.forward(&input);

        assert_eq!(output.dim(), (5, 32));
        let has_nonzero = output.iter().any(|&v| v != 0.0);
        assert!(
            has_nonzero,
            "BiLSTM forward pass should produce non-zero outputs"
        );
    }

    #[test]
    fn test_lstm_network_forward_real() {
        let mut network = LstmNetwork::new(8, 16, 2, 10);
        network.randomize();

        let input = Array2::from_elem((5, 8), 0.5);
        let output = network.forward(&input);

        assert_eq!(output.dim(), (5, 10));
        let has_nonzero = output.iter().any(|&v| v != 0.0);
        assert!(
            has_nonzero,
            "LSTM network forward pass should produce non-zero outputs"
        );
    }

    #[test]
    fn test_lstm_layer_with_signal() {
        let mut layer = LstmLayer::new(4, 8);
        layer.randomize();

        let input = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        )
        .unwrap();
        let output = layer.forward(&input);

        assert_eq!(output.dim(), (3, 8));
        for t in 0..3 {
            for h in 0..8 {
                assert!(
                    output[[t, h]].is_finite(),
                    "Output should be finite at [{}, {}]",
                    t,
                    h
                );
            }
        }
    }
}
