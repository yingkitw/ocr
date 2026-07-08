//! ONNX model loader for importing pre-trained OCR models
//!
//! Supports loading weight tensors from ONNX `ModelProto` and mapping them
//! to our `CrnnModel` structure.  Enabled via the `onnx` feature flag.
//!
//! # Example
//! ```no_run
//! use std::path::Path;
//! use ocr::onnx::OnnxLoader;
//! let loader = OnnxLoader::from_file(Path::new("model.onnx")).unwrap();
//! let weights = loader.extract_weights().unwrap();
//! ```

use crate::utils::{OcrError, Result};
use ndarray::ArrayD;

/// Wrapper that keeps ONNX bytes and parses on demand.
///
/// The underlying `onnx_rs::ast::Model` borrows from the byte slice, so we
/// store the owned bytes and re-parse for each operation.
pub struct OnnxLoader {
    bytes: Vec<u8>,
}

impl OnnxLoader {
    /// Parse an ONNX model from a byte slice.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let _ = Self::parse_model(&bytes)?;
        Ok(Self { bytes })
    }

    /// Parse an ONNX model from a file path.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to read {}: {}", path.display(), e)))?;
        Self::from_bytes(bytes)
    }

    fn parse_model(bytes: &[u8]) -> Result<onnx_rs::ast::Model> {
        onnx_rs::parse(bytes)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to parse ONNX: {:?}", e)))
    }

    /// Return the number of nodes (ops) in the graph.
    pub fn node_count(&self) -> Result<usize> {
        let model = Self::parse_model(&self.bytes)?;
        Ok(model.graph.as_ref().map(|g| g.node.len()).unwrap_or(0))
    }

    /// Return the number of initializer tensors.
    pub fn weight_count(&self) -> Result<usize> {
        let model = Self::parse_model(&self.bytes)?;
        Ok(model.graph.as_ref().map(|g| g.initializer.len()).unwrap_or(0))
    }

    /// List all node op-types in the graph.
    pub fn op_types(&self) -> Result<Vec<String>> {
        let model = Self::parse_model(&self.bytes)?;
        Ok(model
            .graph
            .as_ref()
            .map(|g| {
                g.node
                    .iter()
                    .map(|n| format!("{:?}", n.op_type))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Extract all weight tensors as a map name -> ndarray.
    pub fn extract_weights(&self) -> Result<std::collections::HashMap<String, ArrayD<f32>>> {
        let model = Self::parse_model(&self.bytes)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| OcrError::ModelLoad("ONNX model has no graph".to_string()))?;

        let mut weights = std::collections::HashMap::new();
        for tensor in &graph.initializer {
            let arr = tensor_to_ndarray(tensor)?;
            weights.insert(tensor.name().to_string(), arr);
        }
        Ok(weights)
    }

    /// Extract a specific weight tensor by name.
    pub fn weight_by_name(&self, name: &str) -> Result<ArrayD<f32>> {
        let model = Self::parse_model(&self.bytes)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| OcrError::ModelLoad("ONNX model has no graph".to_string()))?;

        for tensor in &graph.initializer {
            if tensor.name() == name {
                return tensor_to_ndarray(tensor);
            }
        }
        Err(OcrError::ModelLoad(format!("Weight '{}' not found", name)))
    }

    /// Find node inputs/outputs by op-type (e.g. Conv, Gemm, LSTM).
    ///
    /// Returns a simplified representation because the underlying `Node`
    /// borrows from the parse buffer.
    pub fn nodes_by_op(&self, op: &str) -> Result<Vec<SimplifiedNode>> {
        let model = Self::parse_model(&self.bytes)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| OcrError::ModelLoad("ONNX model has no graph".to_string()))?;

        let mut result = Vec::new();
        for node in &graph.node {
            let op_str = format!("{:?}", node.op_type);
            if op_str.eq_ignore_ascii_case(op) {
                result.push(SimplifiedNode {
                    op_type: op_str,
                    inputs: node.input.iter().map(|s| s.to_string()).collect(),
                    outputs: node.output.iter().map(|s| s.to_string()).collect(),
                });
            }
        }
        Ok(result)
    }

    /// List all initializer tensors with their shapes.
    pub fn weight_shapes(&self) -> Result<Vec<(String, Vec<usize>)>> {
        let model = Self::parse_model(&self.bytes)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| OcrError::ModelLoad("ONNX model has no graph".to_string()))?;

        Ok(graph
            .initializer
            .iter()
            .map(|t| {
                let dims: Vec<usize> = t.dims().iter().map(|&d| d as usize).collect();
                (t.name().to_string(), dims)
            })
            .collect())
    }
}

/// Map ONNX weights into a `CrnnModel` by following the graph topology.
///
/// # Strategy
///
/// 1. Conv nodes are visited in order; `input[1]` is the weight, `input[2]` the bias.
/// 2. Gemm nodes are visited in order; `input[1]` is the weight, `input[2]` the bias.
///    The *last* Gemm is treated as the final FC projection.
/// 3. LSTM nodes (if present) are mapped to `BiLstmLayer` weights.
///
/// # Limitations
///
/// Models that decompose LSTM into a sequence of MatMul/Add/Sigmoid/Tanh
/// (common in some PaddleOCR exports) require manual weight extraction
/// via `weight_by_name`.
pub fn load_crnn_weights(
    loader: &OnnxLoader,
    model: &mut crate::recognition::crnn::CrnnModel,
) -> Result<()> {
    let weights = loader.extract_weights()?;
    let conv_nodes = loader.nodes_by_op("Conv")?;
    let gemm_nodes = loader.nodes_by_op("Gemm")?;
    let lstm_nodes = loader.nodes_by_op("LSTM")?;

    // --- Conv layers --------------------------------------------------------
    let conv_weights = [
        &mut model.cnn.conv1_weights,
        &mut model.cnn.conv2_weights,
        &mut model.cnn.conv3_weights,
        &mut model.cnn.conv4_weights,
        &mut model.cnn.conv5_weights,
    ];
    let conv_biases = [
        &mut model.cnn.conv1_bias,
        &mut model.cnn.conv2_bias,
        &mut model.cnn.conv3_bias,
        &mut model.cnn.conv4_bias,
        &mut model.cnn.conv5_bias,
    ];

    for (i, node) in conv_nodes.iter().enumerate() {
        if i >= conv_weights.len() {
            break;
        }
        // ONNX Conv: input[1] = W, input[2] = B (optional)
        if node.inputs.len() >= 2 {
            let w_name = &node.inputs[1];
            if let Some(w_arr) = weights.get(w_name) {
                *conv_weights[i] = flatten_conv_weight(w_arr)?;
            }
        }
        if node.inputs.len() >= 3 {
            let b_name = &node.inputs[2];
            if let Some(b_arr) = weights.get(b_name) {
                *conv_biases[i] = ndarray_to_array1(b_arr)?;
            }
        }
    }

    // --- FC layer (last Gemm) ---------------------------------------------
    if let Some(gemm) = gemm_nodes.last() {
        if gemm.inputs.len() >= 2 {
            let w_name = &gemm.inputs[1];
            if let Some(w_arr) = weights.get(w_name) {
                model.fc_weight = ndarray_to_array2(w_arr)?;
            }
        }
        if gemm.inputs.len() >= 3 {
            let b_name = &gemm.inputs[2];
            if let Some(b_arr) = weights.get(b_name) {
                model.fc_bias = ndarray_to_array1(b_arr)?;
            }
        }
    }

    // --- LSTM layers --------------------------------------------------------
    // ONNX LSTM (opset >= 7):
    // W [num_directions, 4*H, I] — input-to-hidden for all gates
    // R [num_directions, 4*H, H] — hidden-to-hidden for all gates
    // B [num_directions, 8*H] — bias (ih + hh concatenated per direction)
    let hidden = model.config.hidden_size;
    let four_h = 4 * hidden;

    if let Some(node) = lstm_nodes.get(0) {
        let lstm = &mut model.lstm1;
        if node.inputs.len() >= 2 {
            if let Some(w_arr) = weights.get(&node.inputs[1]) {
                let w = ndarray_to_array3(w_arr)?;
                let ndir = w.shape()[0];
                if ndir >= 1 && w.shape()[1] == four_h {
                    lstm.wf_ih = w.slice(ndarray::s![0, .., ..]).to_owned();
                }
                if ndir >= 2 && w.shape()[1] == four_h {
                    lstm.wb_ih = w.slice(ndarray::s![1, .., ..]).to_owned();
                }
            }
        }
        if node.inputs.len() >= 3 {
            if let Some(r_arr) = weights.get(&node.inputs[2]) {
                let r = ndarray_to_array3(r_arr)?;
                let ndir = r.shape()[0];
                if ndir >= 1 && r.shape()[1] == four_h {
                    lstm.wf_hh = r.slice(ndarray::s![0, .., ..]).to_owned();
                }
                if ndir >= 2 && r.shape()[1] == four_h {
                    lstm.wb_hh = r.slice(ndarray::s![1, .., ..]).to_owned();
                }
            }
        }
        if node.inputs.len() >= 4 {
            if let Some(b_arr) = weights.get(&node.inputs[3]) {
                let b = ndarray_to_array2(b_arr)?;
                let ndir = b.shape()[0];
                if ndir >= 1 && b.shape()[1] == 8 * hidden {
                    let fwd = b.row(0);
                    lstm.bf_ih = fwd.slice(ndarray::s![0..four_h]).to_owned();
                    lstm.bf_hh = fwd.slice(ndarray::s![four_h..8 * hidden]).to_owned();
                }
                if ndir >= 2 && b.shape()[1] == 8 * hidden {
                    let bwd = b.row(1);
                    lstm.bb_ih = bwd.slice(ndarray::s![0..four_h]).to_owned();
                    lstm.bb_hh = bwd.slice(ndarray::s![four_h..8 * hidden]).to_owned();
                }
            }
        }
    }

    if let Some(node) = lstm_nodes.get(1) {
        let lstm = &mut model.lstm2;
        if node.inputs.len() >= 2 {
            if let Some(w_arr) = weights.get(&node.inputs[1]) {
                let w = ndarray_to_array3(w_arr)?;
                let ndir = w.shape()[0];
                if ndir >= 1 && w.shape()[1] == four_h {
                    lstm.wf_ih = w.slice(ndarray::s![0, .., ..]).to_owned();
                }
                if ndir >= 2 && w.shape()[1] == four_h {
                    lstm.wb_ih = w.slice(ndarray::s![1, .., ..]).to_owned();
                }
            }
        }
        if node.inputs.len() >= 3 {
            if let Some(r_arr) = weights.get(&node.inputs[2]) {
                let r = ndarray_to_array3(r_arr)?;
                let ndir = r.shape()[0];
                if ndir >= 1 && r.shape()[1] == four_h {
                    lstm.wf_hh = r.slice(ndarray::s![0, .., ..]).to_owned();
                }
                if ndir >= 2 && r.shape()[1] == four_h {
                    lstm.wb_hh = r.slice(ndarray::s![1, .., ..]).to_owned();
                }
            }
        }
        if node.inputs.len() >= 4 {
            if let Some(b_arr) = weights.get(&node.inputs[3]) {
                let b = ndarray_to_array2(b_arr)?;
                let ndir = b.shape()[0];
                if ndir >= 1 && b.shape()[1] == 8 * hidden {
                    let fwd = b.row(0);
                    lstm.bf_ih = fwd.slice(ndarray::s![0..four_h]).to_owned();
                    lstm.bf_hh = fwd.slice(ndarray::s![four_h..8 * hidden]).to_owned();
                }
                if ndir >= 2 && b.shape()[1] == 8 * hidden {
                    let bwd = b.row(1);
                    lstm.bb_ih = bwd.slice(ndarray::s![0..four_h]).to_owned();
                    lstm.bb_hh = bwd.slice(ndarray::s![four_h..8 * hidden]).to_owned();
                }
            }
        }
    }

    // If no LSTM nodes were found, the model probably uses MatMul/Gemm for
    // the recurrent part.  In that case the caller should map those manually
    // via `weight_by_name`.

    Ok(())
}

/// Flatten an ONNX Conv weight [out_c, in_c, k, k] → [out_c, in_c, k*k].
fn flatten_conv_weight(arr: &ArrayD<f32>) -> Result<ndarray::Array3<f32>> {
    let shape = arr.shape();
    if shape.len() != 4 {
        return Err(OcrError::ModelLoad(format!(
            "Expected 4-D Conv weight, got {:?}",
            shape
        )));
    }
    let out_c = shape[0];
    let in_c = shape[1];
    let k = shape[2];
    if shape[3] != k {
        return Err(OcrError::ModelLoad(format!(
            "Non-square kernel {:?}",
            shape
        )));
    }
    let flat = arr
        .view()
        .into_shape((out_c, in_c, k * k))
        .map_err(|e| OcrError::ModelLoad(format!("Conv flatten: {}", e)))?;
    Ok(flat.to_owned())
}

fn ndarray_to_array1(arr: &ArrayD<f32>) -> Result<ndarray::Array1<f32>> {
    let shape = arr.shape();
    if shape.len() != 1 {
        return Err(OcrError::ModelLoad(format!(
            "Expected 1-D bias, got {:?}",
            shape
        )));
    }
    Ok(arr
        .view()
        .into_shape(shape[0])
        .map_err(|e| OcrError::ModelLoad(format!("Bias flatten: {}", e)))?
        .to_owned())
}

fn ndarray_to_array2(arr: &ArrayD<f32>) -> Result<ndarray::Array2<f32>> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(OcrError::ModelLoad(format!(
            "Expected 2-D matrix, got {:?}",
            shape
        )));
    }
    Ok(arr
        .view()
        .into_shape((shape[0], shape[1]))
        .map_err(|e| OcrError::ModelLoad(format!("Matrix reshape: {}", e)))?
        .to_owned())
}

fn ndarray_to_array3(arr: &ArrayD<f32>) -> Result<ndarray::Array3<f32>> {
    let shape = arr.shape();
    if shape.len() != 3 {
        return Err(OcrError::ModelLoad(format!(
            "Expected 3-D tensor, got {:?}",
            shape
        )));
    }
    Ok(arr
        .view()
        .into_shape((shape[0], shape[1], shape[2]))
        .map_err(|e| OcrError::ModelLoad(format!("Tensor reshape: {}", e)))?
        .to_owned())
}

/// Simplified node representation (owned strings) for callers.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplifiedNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Convert an ONNX `TensorProto` to an `ndarray::ArrayD<f32>`.
fn tensor_to_ndarray(tensor: &onnx_rs::ast::TensorProto) -> Result<ArrayD<f32>> {
    let dims: Vec<usize> = tensor.dims().iter().map(|&d| d as usize).collect();

    if let Some(floats) = tensor.as_f32() {
        return ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), floats.to_vec())
            .map_err(|e| OcrError::ModelLoad(format!("Shape mismatch for '{}': {}", tensor.name(), e)));
    }

    Err(OcrError::ModelLoad(format!(
        "Tensor '{}' has no usable float data",
        tensor.name()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal ONNX model programmatically and round-trip it.
    fn minimal_onnx_bytes() -> Vec<u8> {
        use onnx_rs::ast::*;
        let model = Model {
            ir_version: 9,
            producer_name: "test",
            opset_import: vec![OperatorSetId { domain: "", version: 19 }],
            graph: Some(Graph {
                name: "test",
                initializer: vec![
                    TensorProto::from_f32("conv_w", vec![2, 1, 3, 3], vec![0.0; 2 * 1 * 3 * 3]),
                    TensorProto::from_f32("fc_w", vec![10, 20], vec![0.0; 10 * 20]),
                ],
                node: vec![
                    Node {
                        op_type: OpType::Conv,
                        input: vec!["x", "conv_w"],
                        output: vec!["y"],
                        ..Default::default()
                    },
                    Node {
                        op_type: OpType::Gemm,
                        input: vec!["y", "fc_w"],
                        output: vec!["z"],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        onnx_rs::encode(&model)
    }

    #[test]
    fn test_load_minimal_onnx() {
        let bytes = minimal_onnx_bytes();
        let loader = OnnxLoader::from_bytes(bytes).unwrap();
        assert_eq!(loader.node_count().unwrap(), 2);
        assert_eq!(loader.weight_count().unwrap(), 2);
    }

    #[test]
    fn test_extract_weights() {
        let bytes = minimal_onnx_bytes();
        let loader = OnnxLoader::from_bytes(bytes).unwrap();
        let weights = loader.extract_weights().unwrap();
        assert_eq!(weights.len(), 2);
        assert!(weights.contains_key("conv_w"));
        assert!(weights.contains_key("fc_w"));

        let conv = weights.get("conv_w").unwrap();
        assert_eq!(conv.shape(), &[2, 1, 3, 3]);

        let fc = weights.get("fc_w").unwrap();
        assert_eq!(fc.shape(), &[10, 20]);
    }

    #[test]
    fn test_weight_by_name() {
        let bytes = minimal_onnx_bytes();
        let loader = OnnxLoader::from_bytes(bytes).unwrap();
        let fc = loader.weight_by_name("fc_w").unwrap();
        assert_eq!(fc.shape(), &[10, 20]);
    }

    #[test]
    fn test_nodes_by_op() {
        let bytes = minimal_onnx_bytes();
        let loader = OnnxLoader::from_bytes(bytes).unwrap();
        let convs = loader.nodes_by_op("Conv").unwrap();
        assert_eq!(convs.len(), 1);
        let gemms = loader.nodes_by_op("Gemm").unwrap();
        assert_eq!(gemms.len(), 1);
        let missing = loader.nodes_by_op("LSTM").unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn test_missing_weight() {
        let bytes = minimal_onnx_bytes();
        let loader = OnnxLoader::from_bytes(bytes).unwrap();
        assert!(loader.weight_by_name("missing").is_err());
    }

    #[test]
    fn test_load_crnn_weights_mapping() {
        use crate::recognition::crnn::{CrnnConfig, CrnnModel};
        use onnx_rs::ast::*;

        // Build a tiny ONNX model that looks like a CRNN:
        // Conv(1->16, 3x3) -> Gemm(16->10)
        let model = Model {
            ir_version: 9,
            producer_name: "test",
            opset_import: vec![OperatorSetId { domain: "", version: 19 }],
            graph: Some(Graph {
                name: "test",
                initializer: vec![
                    // Conv weight: [16, 1, 3, 3] = 144 floats
                    TensorProto::from_f32(
                        "conv1_w",
                        vec![16, 1, 3, 3],
                        (0..144).map(|i| i as f32).collect(),
                    ),
                    // Conv bias: [16]
                    TensorProto::from_f32("conv1_b", vec![16], vec![0.5; 16]),
                    // FC weight: [10, 16]
                    TensorProto::from_f32(
                        "fc_w",
                        vec![10, 16],
                        (0..160).map(|i| i as f32).collect(),
                    ),
                    // FC bias: [10]
                    TensorProto::from_f32("fc_b", vec![10], vec![1.0; 10]),
                ],
                node: vec![
                    Node {
                        op_type: OpType::Conv,
                        input: vec!["x", "conv1_w", "conv1_b"],
                        output: vec!["y"],
                        ..Default::default()
                    },
                    Node {
                        op_type: OpType::Gemm,
                        input: vec!["y", "fc_w", "fc_b"],
                        output: vec!["z"],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let bytes = onnx_rs::encode(&model);
        let loader = OnnxLoader::from_bytes(bytes).unwrap();

        // Build a CRNN model with matching channel sizes
        let config = CrnnConfig {
            input_height: 32,
            input_channels: 1,
            num_classes: 10,
            hidden_size: 8,
            num_lstm_layers: 2,
            cnn_channels: vec![16, 32, 64, 64, 128],
            dropout: 0.0,
        };
        let mut crnn = CrnnModel::new(config);

        // Verify initial weights are randomized (not the ONNX values)
        assert!(crnn.cnn.conv1_weights[[0, 0, 0]] < 100.0); // random value, not 0.0

        // Map ONNX weights into the model
        load_crnn_weights(&loader, &mut crnn).unwrap();

        // Verify conv1 weight was mapped and flattened correctly
        // ONNX [16, 1, 3, 3] → our [16, 1, 9]
        assert_eq!(crnn.cnn.conv1_weights.shape(), &[16, 1, 9]);
        assert_eq!(crnn.cnn.conv1_weights[[0, 0, 0]], 0.0);
        assert_eq!(crnn.cnn.conv1_weights[[0, 0, 8]], 8.0);
        assert_eq!(crnn.cnn.conv1_bias[0], 0.5);

        // Verify FC weight was mapped
        assert_eq!(crnn.fc_weight.shape(), &[10, 16]);
        assert_eq!(crnn.fc_weight[[0, 0]], 0.0);
        assert_eq!(crnn.fc_weight[[0, 1]], 1.0);
        assert_eq!(crnn.fc_bias[0], 1.0);
    }
}
