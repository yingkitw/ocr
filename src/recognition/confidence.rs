//! Confidence extraction and calibration for OCR recognition.
//!
//! Derives per-character probabilities from CTC logits (temperature-scaled
//! softmax), then aggregates to word and line confidence. Mature engines
//! (Tesseract, PaddleOCR, EasyOCR) all expose calibrated scores for filtering.

use ndarray::Array2;

/// Calibrates raw CTC class probabilities into downstream confidence scores.
#[derive(Debug, Clone)]
pub struct ConfidenceCalibrator {
    /// Softmax temperature. `T > 1` softens (more conservative); `T < 1` sharpens.
    pub temperature: f32,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl ConfidenceCalibrator {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(1e-6),
        }
    }

    /// Softmax a single logit row with temperature scaling.
    pub fn softmax_row(&self, logits: &[f32]) -> Vec<f32> {
        let t = self.temperature;
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| ((x - max) / t).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            let n = logits.len().max(1) as f32;
            return vec![1.0 / n; logits.len()];
        }
        exps.into_iter().map(|e| e / sum).collect()
    }

    /// Map a raw probability into a calibrated confidence in `[0, 1]`.
    ///
    /// Uses probability power scaling `p^(1/T)` which is equivalent to
    /// temperature scaling when the distribution is peaked.
    pub fn calibrate_prob(&self, raw: f32) -> f32 {
        let p = raw.clamp(0.0, 1.0);
        if self.temperature <= 1e-6 {
            return p;
        }
        p.powf(1.0 / self.temperature).clamp(0.0, 1.0)
    }
}

/// Per-character confidence after CTC decode.
#[derive(Debug, Clone, PartialEq)]
pub struct CharConfidence {
    pub character: char,
    /// Calibrated confidence in `[0, 1]`
    pub confidence: f32,
    /// Raw softmax probability before calibration
    pub raw_prob: f32,
}

/// Full decode confidence breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeConfidence {
    pub chars: Vec<CharConfidence>,
    /// Mean of calibrated character confidences (0 if empty)
    pub overall: f32,
}

impl DecodeConfidence {
    pub fn empty() -> Self {
        Self {
            chars: Vec::new(),
            overall: 0.0,
        }
    }

    /// Word-level confidences by splitting on non-alphanumeric boundaries.
    /// Returns `(word, confidence)` pairs.
    pub fn word_confidences(&self) -> Vec<(String, f32)> {
        let mut words = Vec::new();
        let mut current: Vec<&CharConfidence> = Vec::new();

        for ch in &self.chars {
            if ch.character.is_alphanumeric() {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    words.push(aggregate_word(&current));
                    current.clear();
                }
            }
        }
        if !current.is_empty() {
            words.push(aggregate_word(&current));
        }
        words
    }
}

fn aggregate_word(chars: &[&CharConfidence]) -> (String, f32) {
    let text: String = chars.iter().map(|c| c.character).collect();
    let mean = chars.iter().map(|c| c.confidence).sum::<f32>() / chars.len() as f32;
    (text, mean)
}

/// Extract per-character confidences along the CTC greedy best-path.
///
/// For each timestep the argmax label is taken; when a non-blank label is
/// emitted (and differs from the previous), its softmax probability is recorded.
pub fn greedy_path_confidence(
    logits: &Array2<f32>,
    vocab: &[char],
    calibrator: &ConfidenceCalibrator,
) -> DecodeConfidence {
    let (seq_len, num_classes) = logits.dim();
    if seq_len == 0 || num_classes == 0 {
        return DecodeConfidence::empty();
    }

    let mut chars = Vec::new();
    let mut prev_label = 0usize;

    for t in 0..seq_len {
        let row: Vec<f32> = (0..num_classes).map(|c| logits[[t, c]]).collect();
        let probs = calibrator.softmax_row(&row);
        let best_label = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if best_label != 0 && best_label != prev_label {
            if best_label > 0 && best_label - 1 < vocab.len() {
                let raw = probs[best_label];
                chars.push(CharConfidence {
                    character: vocab[best_label - 1],
                    confidence: calibrator.calibrate_prob(raw),
                    raw_prob: raw,
                });
            }
        }
        prev_label = best_label;
    }

    let overall = if chars.is_empty() {
        0.0
    } else {
        chars.iter().map(|c| c.confidence).sum::<f32>() / chars.len() as f32
    };

    DecodeConfidence { chars, overall }
}

/// Estimate confidence for an already-decoded hypothesis by averaging the
/// max non-blank softmax probability at each timestep (alignment-free).
///
/// Used after beam search when the exact emission frames are not retained.
pub fn hypothesis_confidence(
    logits: &Array2<f32>,
    text: &str,
    calibrator: &ConfidenceCalibrator,
) -> DecodeConfidence {
    let (seq_len, num_classes) = logits.dim();
    if seq_len == 0 || text.is_empty() || num_classes == 0 {
        return DecodeConfidence::empty();
    }

    // Collect peak non-blank probabilities per frame, then assign evenly to chars.
    let mut frame_peaks = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let row: Vec<f32> = (0..num_classes).map(|c| logits[[t, c]]).collect();
        let probs = calibrator.softmax_row(&row);
        let peak = probs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 0)
            .map(|(_, &p)| p)
            .fold(0.0f32, f32::max);
        if peak > 0.0 {
            frame_peaks.push(peak);
        }
    }

    let text_chars: Vec<char> = text.chars().collect();
    if text_chars.is_empty() {
        return DecodeConfidence::empty();
    }

    let chars: Vec<CharConfidence> = if frame_peaks.is_empty() {
        text_chars
            .iter()
            .map(|&c| CharConfidence {
                character: c,
                confidence: 0.0,
                raw_prob: 0.0,
            })
            .collect()
    } else {
        // Map character i to a frame window and take the mean peak in that window.
        let n = text_chars.len();
        text_chars
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let start = i * frame_peaks.len() / n;
                let end = ((i + 1) * frame_peaks.len() / n).max(start + 1);
                let slice = &frame_peaks[start.min(frame_peaks.len())..end.min(frame_peaks.len())];
                let raw = if slice.is_empty() {
                    0.0
                } else {
                    slice.iter().sum::<f32>() / slice.len() as f32
                };
                CharConfidence {
                    character: c,
                    confidence: calibrator.calibrate_prob(raw),
                    raw_prob: raw,
                }
            })
            .collect()
    };

    let overall = chars.iter().map(|c| c.confidence).sum::<f32>() / chars.len() as f32;
    DecodeConfidence { chars, overall }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_softmax_sums_to_one() {
        let cal = ConfidenceCalibrator::default();
        let probs = cal.softmax_row(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_temperature_softens() {
        let sharp = ConfidenceCalibrator::new(0.5);
        let soft = ConfidenceCalibrator::new(2.0);
        let logits = [0.0, 5.0, 0.0];
        let p_sharp = sharp.softmax_row(&logits);
        let p_soft = soft.softmax_row(&logits);
        // Softened distribution should be less peaked
        assert!(p_soft[1] < p_sharp[1]);
    }

    #[test]
    fn test_calibrate_prob_in_range() {
        let cal = ConfidenceCalibrator::new(1.5);
        assert!((0.0..=1.0).contains(&cal.calibrate_prob(0.9)));
        assert!((0.0..=1.0).contains(&cal.calibrate_prob(-1.0)));
        assert!((0.0..=1.0).contains(&cal.calibrate_prob(2.0)));
    }

    #[test]
    fn test_greedy_path_confidence_high_on_clear_signal() {
        let vocab = vec!['a', 'b'];
        let cal = ConfidenceCalibrator::default();
        // Strong 'a', blank, strong 'b'
        let logits = Array2::from_shape_vec(
            (3, 3),
            vec![
                -10.0, 10.0, -10.0, // a
                10.0, -10.0, -10.0, // blank
                -10.0, -10.0, 10.0, // b
            ],
        )
        .unwrap();
        let conf = greedy_path_confidence(&logits, &vocab, &cal);
        assert_eq!(conf.chars.len(), 2);
        assert_eq!(conf.chars[0].character, 'a');
        assert_eq!(conf.chars[1].character, 'b');
        assert!(conf.chars[0].confidence > 0.9);
        assert!(conf.chars[1].confidence > 0.9);
        assert!(conf.overall > 0.9);
    }

    #[test]
    fn test_word_confidences_split() {
        let conf = DecodeConfidence {
            chars: vec![
                CharConfidence {
                    character: 'h',
                    confidence: 0.9,
                    raw_prob: 0.9,
                },
                CharConfidence {
                    character: 'i',
                    confidence: 0.7,
                    raw_prob: 0.7,
                },
                CharConfidence {
                    character: ' ',
                    confidence: 0.5,
                    raw_prob: 0.5,
                },
                CharConfidence {
                    character: 'a',
                    confidence: 0.8,
                    raw_prob: 0.8,
                },
            ],
            overall: 0.75,
        };
        let words = conf.word_confidences();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].0, "hi");
        assert!((words[0].1 - 0.8).abs() < 1e-5);
        assert_eq!(words[1].0, "a");
        assert!((words[1].1 - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_hypothesis_confidence_assigns_all_chars() {
        let cal = ConfidenceCalibrator::default();
        let logits = Array2::from_shape_vec(
            (4, 3),
            vec![
                -5.0, 5.0, -5.0, -5.0, 4.0, -5.0, -5.0, -5.0, 5.0, 5.0, -5.0, -5.0,
            ],
        )
        .unwrap();
        let conf = hypothesis_confidence(&logits, "ab", &cal);
        assert_eq!(conf.chars.len(), 2);
        assert!(conf.overall > 0.5);
    }
}
