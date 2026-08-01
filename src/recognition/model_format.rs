//! Binary model format for the CRNN.
//!
//! Layout:
//!   - 4 bytes: magic "OCR1"
//!   - 4 bytes: format version (u32 LE)
//!   - config block (counts first, then payloads in a fixed order)
//!
//! The format is self-describing per layer (count + payload), so adding new
//! layers in future versions is safe as long as existing counts stay stable.

use crate::recognition::crnn::{BiLstmLayer, CnnFeatureExtractor, CrnnConfig, CrnnModel, Vocabulary};
use crate::utils::{OcrError, Result};
use ndarray::{Array1, Array2, Array3, ShapeBuilder};

const MAGIC: &[u8; 4] = b"OCR1";
const VERSION: u32 = 1;

/// Serialize a CRNN model to bytes.
pub fn write_model(model: &CrnnModel) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());

    write_config(&model.config, &mut buf);
    write_vocab(&model.vocab, &mut buf);
    write_cnn(&model.cnn, &mut buf);
    write_lstm(&model.lstm1, &mut buf);
    write_lstm(&model.lstm2, &mut buf);
    write_fc(&model.fc_weight, &model.fc_bias, &mut buf);

    Ok(buf)
}

/// Deserialize a CRNN model from bytes.
pub fn read_model(bytes: &[u8]) -> Result<CrnnModel> {
    if bytes.len() < 8 {
        return Err(OcrError::Configuration("model too small".into()));
    }
    if &bytes[0..4] != MAGIC {
        return Err(OcrError::Configuration("bad magic".into()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(OcrError::Configuration(format!(
            "unsupported model version {version}"
        )));
    }

    let mut cursor = Cursor::new(&bytes[8..]);
    let config = read_config(&mut cursor)?;
    let vocab = read_vocab(&mut cursor)?;
    let cnn = read_cnn(&config, &mut cursor)?;
    let lstm1 = read_lstm(cnn_output_dim(&config), config.hidden_size, &mut cursor)?;
    let lstm2 = read_lstm(2 * config.hidden_size, config.hidden_size, &mut cursor)?;
    let (fc_weight, fc_bias) = read_fc(vocab.size(), 2 * config.hidden_size, &mut cursor)?;

    let model = CrnnModel {
        config,
        vocab,
        cnn,
        lstm1,
        lstm2,
        fc_weight,
        fc_bias,
        fc_weight_quantized: None,
    };
    Ok(model)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn read_u32(&mut self) -> Result<u32> {
        let n = self.read_array::<4>()?;
        Ok(u32::from_le_bytes(n))
    }
    fn read_f32(&mut self) -> Result<f32> {
        let n = self.read_array::<4>()?;
        Ok(f32::from_le_bytes(n))
    }
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.remaining() < N {
            return Err(OcrError::Configuration(format!(
                "unexpected end of model: need {N} bytes, have {} (at offset {})",
                self.remaining(),
                self.pos
            )));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }
    fn read_f32_vec(&mut self, count: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.read_f32()?);
        }
        Ok(out)
    }
    fn read_u32_vec(&mut self, count: usize) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.read_u32()?);
        }
        Ok(out)
    }
}

fn write_f32_vec(buf: &mut Vec<u8>, data: &[f32]) {
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_f32_array3(buf: &mut Vec<u8>, a: &Array3<f32>) {
    let (oc, ic, kk) = a.dim();
    buf.extend_from_slice(&(oc as u32).to_le_bytes());
    buf.extend_from_slice(&(ic as u32).to_le_bytes());
    buf.extend_from_slice(&(kk as u32).to_le_bytes());
    write_f32_vec(buf, a.as_slice().unwrap());
}

fn write_f32_array2(buf: &mut Vec<u8>, a: &Array2<f32>) {
    let (r, c) = a.dim();
    buf.extend_from_slice(&(r as u32).to_le_bytes());
    buf.extend_from_slice(&(c as u32).to_le_bytes());
    write_f32_vec(buf, a.as_slice().unwrap());
}

fn write_f32_array1(buf: &mut Vec<u8>, a: &Array1<f32>) {
    buf.extend_from_slice(&(a.len() as u32).to_le_bytes());
    write_f32_vec(buf, a.as_slice().unwrap());
}

// ---------------------------------------------------------------------------
// Per-section writers / readers
// ---------------------------------------------------------------------------

fn write_config(cfg: &CrnnConfig, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(cfg.input_height as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.input_channels as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.num_classes as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.hidden_size as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.num_lstm_layers as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.cnn_channels.len() as u32).to_le_bytes());
    for c in &cfg.cnn_channels {
        buf.extend_from_slice(&(*c as u32).to_le_bytes());
    }
    buf.extend_from_slice(&cfg.dropout.to_le_bytes());
    buf.extend_from_slice(&(cfg.use_beam_search as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.beam_width as u32).to_le_bytes());
    buf.extend_from_slice(&cfg.dict_weight.to_le_bytes());
    buf.extend_from_slice(&cfg.lm_weight.to_le_bytes());
    buf.extend_from_slice(&cfg.confidence_temperature.to_le_bytes());
}

fn read_config(c: &mut Cursor) -> Result<CrnnConfig> {
    let input_height = c.read_u32()? as usize;
    let input_channels = c.read_u32()? as usize;
    let num_classes = c.read_u32()? as usize;
    let hidden_size = c.read_u32()? as usize;
    let num_lstm_layers = c.read_u32()? as usize;
    let n_channels = c.read_u32()? as usize;
    let channels_raw = c.read_u32_vec(n_channels)?;
    let cnn_channels: Vec<usize> = channels_raw.iter().map(|&x| x as usize).collect();
    let dropout = c.read_f32()?;
    let use_beam_search = c.read_u32()? != 0;
    let beam_width = c.read_u32()? as usize;
    let dict_weight = c.read_f32()?;
    let lm_weight = c.read_f32()?;
    let confidence_temperature = c.read_f32()?;

    Ok(CrnnConfig {
        input_height,
        input_channels,
        num_classes,
        hidden_size,
        num_lstm_layers,
        cnn_channels,
        dropout,
        use_beam_search,
        beam_width,
        dict_weight,
        lm_weight,
        confidence_temperature,
    })
}

fn write_vocab(vocab: &Vocabulary, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(vocab.chars.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(vocab.blank_idx as u32).to_le_bytes());
    for ch in &vocab.chars {
        buf.extend_from_slice(&(*ch as u32).to_le_bytes());
    }
}

fn read_vocab(c: &mut Cursor) -> Result<Vocabulary> {
    let n = c.read_u32()? as usize;
    let blank_idx = c.read_u32()? as usize;
    let codepoints = c.read_u32_vec(n)?;
    let chars: Vec<char> = codepoints
        .into_iter()
        .filter_map(std::char::from_u32)
        .collect();
    let mut char_to_idx = std::collections::HashMap::new();
    for (i, &ch) in chars.iter().enumerate() {
        char_to_idx.insert(ch, i + 1);
    }
    Ok(Vocabulary {
        chars,
        char_to_idx,
        blank_idx,
    })
}

fn write_cnn(cnn: &CnnFeatureExtractor, buf: &mut Vec<u8>) {
    // Each conv block: 3x3 weights (we encode the kernel side as sqrt(k*k))
    write_f32_array3(buf, &cnn.conv1_weights);
    write_f32_array1(buf, &cnn.conv1_bias);
    write_f32_array1(buf, &cnn.bn_scales[0]);
    write_f32_array1(buf, &cnn.bn_shifts[0]);

    write_f32_array3(buf, &cnn.conv2_weights);
    write_f32_array1(buf, &cnn.conv2_bias);
    write_f32_array1(buf, &cnn.bn_scales[1]);
    write_f32_array1(buf, &cnn.bn_shifts[1]);

    write_f32_array3(buf, &cnn.conv3_weights);
    write_f32_array1(buf, &cnn.conv3_bias);
    write_f32_array1(buf, &cnn.bn_scales[2]);
    write_f32_array1(buf, &cnn.bn_shifts[2]);

    write_f32_array3(buf, &cnn.conv4_weights);
    write_f32_array1(buf, &cnn.conv4_bias);
    write_f32_array1(buf, &cnn.bn_scales[3]);
    write_f32_array1(buf, &cnn.bn_shifts[3]);

    write_f32_array3(buf, &cnn.conv5_weights);
    write_f32_array1(buf, &cnn.conv5_bias);
    write_f32_array1(buf, &cnn.bn_scales[4]);
    write_f32_array1(buf, &cnn.bn_shifts[4]);
}

fn read_cnn(config: &CrnnConfig, c: &mut Cursor) -> Result<CnnFeatureExtractor> {
    let conv1 = read_f32_array3(c)?;
    let bias1 = read_f32_array1(c)?;
    let bn1s = read_f32_array1(c)?;
    let bn1p = read_f32_array1(c)?;

    let conv2 = read_f32_array3(c)?;
    let bias2 = read_f32_array1(c)?;
    let bn2s = read_f32_array1(c)?;
    let bn2p = read_f32_array1(c)?;

    let conv3 = read_f32_array3(c)?;
    let bias3 = read_f32_array1(c)?;
    let bn3s = read_f32_array1(c)?;
    let bn3p = read_f32_array1(c)?;

    let conv4 = read_f32_array3(c)?;
    let bias4 = read_f32_array1(c)?;
    let bn4s = read_f32_array1(c)?;
    let bn4p = read_f32_array1(c)?;

    let conv5 = read_f32_array3(c)?;
    let bias5 = read_f32_array1(c)?;
    let bn5s = read_f32_array1(c)?;
    let bn5p = read_f32_array1(c)?;

    let mut cnn = CnnFeatureExtractor::new(config);
    cnn.conv1_weights = conv1;
    cnn.conv1_bias = bias1;
    cnn.bn_scales[0] = bn1s;
    cnn.bn_shifts[0] = bn1p;

    cnn.conv2_weights = conv2;
    cnn.conv2_bias = bias2;
    cnn.bn_scales[1] = bn2s;
    cnn.bn_shifts[1] = bn2p;

    cnn.conv3_weights = conv3;
    cnn.conv3_bias = bias3;
    cnn.bn_scales[2] = bn3s;
    cnn.bn_shifts[2] = bn3p;

    cnn.conv4_weights = conv4;
    cnn.conv4_bias = bias4;
    cnn.bn_scales[3] = bn4s;
    cnn.bn_shifts[3] = bn4p;

    cnn.conv5_weights = conv5;
    cnn.conv5_bias = bias5;
    cnn.bn_scales[4] = bn5s;
    cnn.bn_shifts[4] = bn5p;
    Ok(cnn)
}

fn read_f32_array3(c: &mut Cursor) -> Result<Array3<f32>> {
    let oc = c.read_u32()? as usize;
    let ic = c.read_u32()? as usize;
    let kk = c.read_u32()? as usize;
    let data = c.read_f32_vec(oc * ic * kk)?;
    let a = Array3::from_shape_vec((oc, ic, kk).strides((ic * kk, kk, 1)), data)
        .map_err(|e| OcrError::Configuration(format!("array3 shape: {e}")))?;
    Ok(a)
}

fn read_f32_array2(c: &mut Cursor) -> Result<Array2<f32>> {
    let r = c.read_u32()? as usize;
    let cols = c.read_u32()? as usize;
    let data = c.read_f32_vec(r * cols)?;
    let a = Array2::from_shape_vec((r, cols), data)
        .map_err(|e| OcrError::Configuration(format!("array2 shape: {e}")))?;
    Ok(a)
}

fn read_f32_array1(c: &mut Cursor) -> Result<Array1<f32>> {
    let n = c.read_u32()? as usize;
    let data = c.read_f32_vec(n)?;
    Ok(Array1::from_vec(data))
}

fn write_lstm(lstm: &BiLstmLayer, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(lstm.input_size as u32).to_le_bytes());
    buf.extend_from_slice(&(lstm.hidden_size as u32).to_le_bytes());
    write_f32_array2(buf, &lstm.wf_ih);
    write_f32_array2(buf, &lstm.wf_hh);
    write_f32_array1(buf, &lstm.bf_ih);
    write_f32_array1(buf, &lstm.bf_hh);
    write_f32_array2(buf, &lstm.wb_ih);
    write_f32_array2(buf, &lstm.wb_hh);
    write_f32_array1(buf, &lstm.bb_ih);
    write_f32_array1(buf, &lstm.bb_hh);
}

fn read_lstm(input_size: usize, hidden_size: usize, c: &mut Cursor) -> Result<BiLstmLayer> {
    let _in = c.read_u32()? as usize;
    let _hs = c.read_u32()? as usize;
    let wf_ih = read_f32_array2(c)?;
    let wf_hh = read_f32_array2(c)?;
    let bf_ih = read_f32_array1(c)?;
    let bf_hh = read_f32_array1(c)?;
    let wb_ih = read_f32_array2(c)?;
    let wb_hh = read_f32_array2(c)?;
    let bb_ih = read_f32_array1(c)?;
    let bb_hh = read_f32_array1(c)?;

    Ok(BiLstmLayer {
        input_size,
        hidden_size,
        wf_ih,
        wf_hh,
        bf_ih,
        bf_hh,
        wb_ih,
        wb_hh,
        bb_ih,
        bb_hh,
    })
}

fn write_fc(w: &Array2<f32>, b: &Array1<f32>, buf: &mut Vec<u8>) {
    write_f32_array2(buf, w);
    write_f32_array1(buf, b);
}

fn read_fc(
    expected_rows: usize,
    expected_cols: usize,
    c: &mut Cursor,
) -> Result<(Array2<f32>, Array1<f32>)> {
    let w = read_f32_array2(c)?;
    if w.dim() != (expected_rows, expected_cols) {
        return Err(OcrError::Configuration(format!(
            "FC weight shape mismatch: expected {}x{}, got {}x{}",
            expected_rows,
            expected_cols,
            w.nrows(),
            w.ncols()
        )));
    }
    let b = read_f32_array1(c)?;
    if b.len() != expected_rows {
        return Err(OcrError::Configuration(format!(
            "FC bias length mismatch: expected {}, got {}",
            expected_rows,
            b.len()
        )));
    }
    Ok((w, b))
}

fn cnn_output_dim(config: &CrnnConfig) -> usize {
    config.cnn_channels[4] * (config.input_height / 4 - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recognition::crnn::{CrnnConfig, CrnnModel};

    #[test]
    fn test_roundtrip_small_model() {
        let m = CrnnModel::new(CrnnConfig::default());
        let bytes = write_model(&m).unwrap();
        let r = read_model(&bytes).unwrap();
        assert_eq!(r.fc_weight.dim(), m.fc_weight.dim());
        assert_eq!(r.fc_bias.len(), m.fc_bias.len());
        assert_eq!(r.cnn.conv1_weights.dim(), m.cnn.conv1_weights.dim());
        assert_eq!(r.lstm1.wf_ih.dim(), m.lstm1.wf_ih.dim());
        assert_eq!(r.lstm2.wf_ih.dim(), m.lstm2.wf_ih.dim());
        assert_eq!(r.vocab.chars, m.vocab.chars);
    }

    #[test]
    fn test_roundtrip_preserves_weights() {
        let mut m = CrnnModel::new(CrnnConfig::default());
        m.fc_weight[[0, 0]] = 0.12345;
        m.fc_bias[2] = -0.6789;
        m.cnn.conv1_weights[[0, 0, 0]] = 0.5;
        m.lstm1.wf_ih[[0, 0]] = -0.25;

        let bytes = write_model(&m).unwrap();
        let r = read_model(&bytes).unwrap();
        assert!((r.fc_weight[[0, 0]] - 0.12345).abs() < 1e-6);
        assert!((r.fc_bias[2] - -0.6789).abs() < 1e-6);
        assert!((r.cnn.conv1_weights[[0, 0, 0]] - 0.5).abs() < 1e-6);
        assert!((r.lstm1.wf_ih[[0, 0]] - -0.25).abs() < 1e-6);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let bytes = vec![0u8; 8];
        let res = read_model(&bytes);
        assert!(res.is_err());
    }
}
