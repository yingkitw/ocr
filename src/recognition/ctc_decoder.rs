//! CTC (Connectionist Temporal Classification) decoder
//!
//! Decodes LSTM output logits into text using CTC best-path decoding
//! and optional beam search with dictionary / LM rescoring.
//! Compatible with Tesseract's CTC approach.

use crate::lang::dictionary::Dictionary;
use crate::lang::NGramModel;
use ndarray::Array2;

const BLANK_LABEL: usize = 0;

/// A single CTC beam hypothesis after decoding.
#[derive(Debug, Clone)]
pub struct DecodeHypothesis {
    pub text: String,
    pub acoustic_log_prob: f32,
}

/// Scores beam hypotheses with dictionary hits and/or an n-gram LM.
#[derive(Clone)]
pub struct DictLmRescorer<'a> {
    pub dictionary: Option<&'a Dictionary>,
    pub ngram: Option<&'a NGramModel>,
    pub dict_weight: f32,
    pub lm_weight: f32,
}

impl<'a> DictLmRescorer<'a> {
    pub fn new(
        dictionary: Option<&'a Dictionary>,
        ngram: Option<&'a NGramModel>,
        dict_weight: f32,
        lm_weight: f32,
    ) -> Self {
        Self {
            dictionary,
            ngram,
            dict_weight,
            lm_weight,
        }
    }

    /// Additive log-space bonus for a hypothesis text.
    pub fn bonus(&self, text: &str) -> f32 {
        let mut bonus = 0.0f32;
        if let Some(dict) = self.dictionary {
            bonus += self.dict_weight * dictionary_log_bonus(dict, text);
        }
        if let Some(ngram) = self.ngram {
            bonus += self.lm_weight * (ngram.score_text(text) as f32);
        }
        bonus
    }
}

fn dictionary_log_bonus(dict: &Dictionary, text: &str) -> f32 {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return 0.0;
    }
    let mut score = 0.0f32;
    for word in &words {
        if dict.contains(word) {
            score += 1.0;
        } else {
            score -= 0.5;
        }
    }
    score / words.len() as f32
}

pub struct CtcDecoder {
    beam_width: usize,
    /// Logit gap from the max at each timestep beyond which labels are pruned.
    prune_threshold: f32,
}

impl CtcDecoder {
    pub fn new() -> Self {
        Self {
            beam_width: 10,
            prune_threshold: 20.0,
        }
    }

    pub fn with_beam_width(beam_width: usize) -> Self {
        Self {
            beam_width: beam_width.max(1),
            prune_threshold: 20.0,
        }
    }

    pub fn with_prune_threshold(mut self, prune_threshold: f32) -> Self {
        self.prune_threshold = prune_threshold.max(0.0);
        self
    }

    pub fn beam_width(&self) -> usize {
        self.beam_width
    }

    pub fn greedy_decode(&self, logits: &Array2<f32>, vocab: &[char]) -> String {
        let (seq_len, num_classes) = logits.dim();
        let mut output = Vec::new();
        let mut prev_label = BLANK_LABEL;

        for t in 0..seq_len {
            let best_label = (0..num_classes)
                .max_by(|&a, &b| {
                    logits[[t, a]]
                        .partial_cmp(&logits[[t, b]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(BLANK_LABEL);

            if best_label != BLANK_LABEL && best_label != prev_label {
                if best_label > 0 && best_label - 1 < vocab.len() {
                    output.push(vocab[best_label - 1]);
                }
            }
            prev_label = best_label;
        }

        output.into_iter().collect()
    }

    /// Beam search decoding; returns the top acoustic hypothesis.
    pub fn beam_search_decode(&self, logits: &Array2<f32>, vocab: &[char]) -> String {
        self.beam_search_nbest(logits, vocab)
            .into_iter()
            .next()
            .map(|h| h.text)
            .unwrap_or_default()
    }

    /// Beam search with dictionary / LM rescoring.
    ///
    /// Final score = acoustic_log_prob + rescorer.bonus(text).
    pub fn beam_search_decode_rescored(
        &self,
        logits: &Array2<f32>,
        vocab: &[char],
        rescorer: &DictLmRescorer<'_>,
    ) -> String {
        let mut nbest = self.beam_search_nbest(logits, vocab);
        if nbest.is_empty() {
            return String::new();
        }
        nbest.sort_by(|a, b| {
            let score_a = a.acoustic_log_prob + rescorer.bonus(&a.text);
            let score_b = b.acoustic_log_prob + rescorer.bonus(&b.text);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nbest.into_iter().next().map(|h| h.text).unwrap_or_default()
    }

    /// Return up to `beam_width` unique hypotheses, best acoustic score first.
    pub fn beam_search_nbest(
        &self,
        logits: &Array2<f32>,
        vocab: &[char],
    ) -> Vec<DecodeHypothesis> {
        let (seq_len, num_classes) = logits.dim();
        if seq_len == 0 {
            return Vec::new();
        }

        let mut beam = vec![CtcBeam::new(num_classes)];

        for t in 0..seq_len {
            let mut new_beam = Vec::new();

            for b in &beam {
                let max_logit = (0..num_classes)
                    .map(|c| logits[[t, c]])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut pruned = Vec::new();
                for c in 0..num_classes {
                    let log_prob = logits[[t, c]];
                    if max_logit - log_prob > self.prune_threshold {
                        continue;
                    }
                    pruned.push((c, log_prob));
                }
                if pruned.is_empty() {
                    pruned = (0..num_classes).map(|c| (c, logits[[t, c]])).collect();
                }

                for &(label, log_prob) in &pruned {
                    let new_label = if label == BLANK_LABEL {
                        None
                    } else {
                        Some(label)
                    };

                    let same_last = b.last_label == new_label;
                    let new_total = b.log_prob + log_prob;

                    if label == BLANK_LABEL {
                        let mut new_beam_entry = b.clone();
                        let via_non_blank = b.log_prob + log_prob;
                        let via_blank = b.blank_log_prob + log_prob;
                        new_beam_entry.blank_log_prob = log_sum_exp(via_non_blank, via_blank);
                        new_beam.push(new_beam_entry);
                    } else if same_last {
                        let mut new_beam_entry = b.clone();
                        let via_blank = b.blank_log_prob + log_prob;
                        new_beam_entry.log_prob = log_sum_exp(new_total, via_blank);
                        new_beam_entry.blank_log_prob = f32::NEG_INFINITY; // emitted non-blank
                        new_beam_entry.log_probs_t[label] =
                            (new_beam_entry.log_probs_t[label]).max(new_beam_entry.log_prob);
                        new_beam.push(new_beam_entry);
                    } else {
                        let mut new_beam_entry = b.clone();
                        let via_blank = b.blank_log_prob + log_prob;
                        new_beam_entry.log_prob = log_sum_exp(new_total, via_blank);
                        new_beam_entry.blank_log_prob = f32::NEG_INFINITY; // emitted non-blank
                        new_beam_entry.labels.push(label);
                        new_beam_entry.last_label = Some(label);
                        new_beam_entry.log_probs_t[label] =
                            (new_beam_entry.log_probs_t[label]).max(new_beam_entry.log_prob);
                        new_beam.push(new_beam_entry);
                    }
                }
            }

            beam = Self::merge_beams(new_beam, num_classes, self.beam_width);
        }

        let mut hypotheses: Vec<DecodeHypothesis> = beam
            .into_iter()
            .map(|b| {
                let text: String = b
                    .labels
                    .iter()
                    .filter_map(|&label| {
                        if label > 0 && label - 1 < vocab.len() {
                            Some(vocab[label - 1])
                        } else {
                            None
                        }
                    })
                    .collect();
                DecodeHypothesis {
                    text,
                    acoustic_log_prob: b.total_log_prob(),
                }
            })
            .collect();

        // Deduplicate identical texts, keeping the best acoustic score.
        let mut best_by_text: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        for h in hypotheses {
            best_by_text
                .entry(h.text)
                .and_modify(|p| *p = (*p).max(h.acoustic_log_prob))
                .or_insert(h.acoustic_log_prob);
        }
        hypotheses = best_by_text
            .into_iter()
            .map(|(text, acoustic_log_prob)| DecodeHypothesis {
                text,
                acoustic_log_prob,
            })
            .collect();
        hypotheses.sort_by(|a, b| {
            b.acoustic_log_prob
                .partial_cmp(&a.acoustic_log_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hypotheses.truncate(self.beam_width);
        hypotheses
    }

    fn merge_beams(beams: Vec<CtcBeam>, num_classes: usize, beam_width: usize) -> Vec<CtcBeam> {
        let mut by_prefix: std::collections::HashMap<Vec<usize>, CtcBeam> =
            std::collections::HashMap::new();

        for b in beams {
            let key = b.labels.clone();
            match by_prefix.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let entry = e.get_mut();
                    entry.log_prob = entry.log_prob.max(b.log_prob);
                    entry.blank_log_prob = entry.blank_log_prob.max(b.blank_log_prob);
                    entry.last_label = b.last_label;
                    for (c, &p) in b.log_probs_t.iter().enumerate() {
                        entry.log_probs_t[c] = entry.log_probs_t[c].max(p);
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(b);
                }
            }
        }

        let mut merged: Vec<CtcBeam> = by_prefix.into_values().collect();
        merged.sort_by(|a, b| {
            b.total_log_prob()
                .partial_cmp(&a.total_log_prob())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut log_probs_t = vec![f32::NEG_INFINITY; num_classes];
        for b in &merged {
            for (c, &p) in b.log_probs_t.iter().enumerate() {
                log_probs_t[c] = log_probs_t[c].max(p);
            }
        }

        for b in &mut merged {
            b.log_probs_t = log_probs_t.clone();
        }

        merged.truncate(beam_width);
        merged
    }
}

impl Default for CtcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
struct CtcBeam {
    labels: Vec<usize>,
    log_prob: f32,
    blank_log_prob: f32,
    last_label: Option<usize>,
    log_probs_t: Vec<f32>,
}

impl CtcBeam {
    fn new(num_classes: usize) -> Self {
        Self {
            labels: Vec::new(),
            log_prob: f32::NEG_INFINITY, // probability of ending in non-blank (0 at start)
            blank_log_prob: 0.0,         // probability of ending in blank (1.0 at start)
            last_label: None,
            log_probs_t: vec![f32::NEG_INFINITY; num_classes],
        }
    }

    fn total_log_prob(&self) -> f32 {
        log_sum_exp(self.log_prob, self.blank_log_prob)
    }
}

fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

pub fn default_vocab() -> Vec<char> {
    let mut vocab: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"()-/ "
        .chars()
        .collect();
    vocab.sort();
    vocab.dedup();
    vocab
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_decode_simple() {
        let vocab = vec!['a', 'b', 'c'];
        let decoder = CtcDecoder::new();

        let logits = Array2::from_shape_vec(
            (5, 4),
            vec![
                -10.0, 0.0, -10.0, -10.0, 0.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, -10.0, 0.0,
                -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0,
            ],
        )
        .unwrap();

        let result = decoder.greedy_decode(&logits, &vocab);
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_greedy_decode_with_repeats() {
        let vocab = vec!['a', 'b'];
        let decoder = CtcDecoder::new();

        let logits = Array2::from_shape_vec(
            (5, 3),
            vec![
                -10.0, 0.0, -10.0, -10.0, 0.0, -10.0, 0.0, -10.0, -10.0, -10.0, 0.0, -10.0, -10.0,
                0.0, -10.0,
            ],
        )
        .unwrap();

        let result = decoder.greedy_decode(&logits, &vocab);
        assert_eq!(result, "aa");
    }

    #[test]
    fn test_greedy_decode_with_blanks() {
        let vocab = vec!['a'];
        let decoder = CtcDecoder::new();

        let logits = Array2::from_shape_vec(
            (6, 2),
            vec![
                -10.0, 0.0, 0.0, -10.0, -10.0, 0.0, 0.0, -10.0, -10.0, 0.0, 0.0, -10.0,
            ],
        )
        .unwrap();

        let result = decoder.greedy_decode(&logits, &vocab);
        assert_eq!(result, "aaa");
    }

    #[test]
    fn test_greedy_decode_empty() {
        let vocab = vec!['a', 'b'];
        let decoder = CtcDecoder::new();

        let logits = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, -10.0, -10.0, 0.0, -10.0, -10.0, 0.0, -10.0, -10.0],
        )
        .unwrap();

        let result = decoder.greedy_decode(&logits, &vocab);
        assert_eq!(result, "");
    }

    #[test]
    fn test_beam_search_decode_simple() {
        let vocab = vec!['a', 'b'];
        let decoder = CtcDecoder::with_beam_width(5);

        // Logits shape: (seq_len=3, num_classes=3)
        // Classes: 0=blank, 1='a', 2='b'
        // We want: t=0 -> 'a', t=1 -> 'b', t=2 -> blank
        let logits = Array2::from_shape_vec(
            (3, 3),
            vec![
                // t=0: favor 'a' (class 1)
                -10.0, 10.0, -10.0, // t=1: favor 'b' (class 2)
                -10.0, -10.0, 10.0, // t=2: favor blank (class 0)
                10.0, -10.0, -10.0,
            ],
        )
        .unwrap();

        let result = decoder.beam_search_decode(&logits, &vocab);
        assert_eq!(result, "ab");
    }

    #[test]
    fn test_beam_search_nbest_returns_ranked() {
        let vocab = vec!['a', 'b'];
        let decoder = CtcDecoder::with_beam_width(5);
        let logits = Array2::from_shape_vec(
            (3, 3),
            vec![
                -10.0, 10.0, -10.0, -10.0, -10.0, 10.0, 10.0, -10.0, -10.0,
            ],
        )
        .unwrap();

        let nbest = decoder.beam_search_nbest(&logits, &vocab);
        assert!(!nbest.is_empty());
        assert_eq!(nbest[0].text, "ab");
        for window in nbest.windows(2) {
            assert!(window[0].acoustic_log_prob >= window[1].acoustic_log_prob);
        }
    }

    #[test]
    fn test_beam_search_dictionary_rescoring_prefers_known_word() {
        // Vocab: blank + t,h,e  → classes 0,1,2,3
        let vocab = vec!['t', 'h', 'e'];
        let decoder = CtcDecoder::with_beam_width(8);

        // Ambiguous sequence: "the" vs "teh" with "teh" slightly preferred acoustically.
        // Timesteps emit: t, h/e ambiguous, e/h ambiguous, blank
        // Shape (4, 4): blank, t, h, e
        let logits = Array2::from_shape_vec(
            (4, 4),
            vec![
                // t=0: strong 't'
                -10.0, 8.0, -10.0, -10.0, // t=1: 'e' slightly ahead of 'h' ( favors "te...")
                -10.0, -10.0, 5.0, 5.5, // t=2: 'h' slightly ahead of 'e' ( favors "...eh" → "teh")
                -10.0, -10.0, 5.5, 5.0, // t=3: blank
                8.0, -10.0, -10.0, -10.0,
            ],
        )
        .unwrap();

        let acoustic = decoder.beam_search_decode(&logits, &vocab);
        // Acoustic-only may prefer "teh"; either way rescoring must pick "the".
        let mut dict = Dictionary::new();
        dict.load_words(&["the"]);
        let rescorer = DictLmRescorer::new(Some(&dict), None, 5.0, 0.0);
        let rescored = decoder.beam_search_decode_rescored(&logits, &vocab, &rescorer);
        assert_eq!(
            rescored, "the",
            "dictionary rescoring should prefer 'the' (acoustic was '{acoustic}')"
        );
    }

    #[test]
    fn test_default_vocab() {
        let vocab = default_vocab();
        assert!(vocab.contains(&'a'));
        assert!(vocab.contains(&'z'));
        assert!(vocab.contains(&'0'));
        assert!(vocab.contains(&' '));
    }

    #[test]
    fn test_log_sum_exp() {
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - (2.0_f32).ln()).abs() < 1e-6);

        assert_eq!(log_sum_exp(f32::NEG_INFINITY, 5.0), 5.0);
        assert_eq!(log_sum_exp(3.0, f32::NEG_INFINITY), 3.0);
    }
}
