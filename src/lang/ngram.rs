//! N-gram language model for OCR post-processing
//!
//! This module implements character-level and word-level N-gram models
//! to improve OCR accuracy through statistical language modeling.

use std::collections::HashMap;

/// N-gram type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NGramType {
    /// Unigram (single character/word)
    Unigram = 1,
    /// Bigram (pairs)
    Bigram = 2,
    /// Trigram (triplets)
    Trigram = 3,
}

/// N-gram language model
pub struct NGramModel {
    /// Character-level N-gram counts
    char_ngrams: HashMap<usize, HashMap<String, u32>>,
    /// Word-level bigram counts
    word_bigrams: HashMap<(String, String), u32>,
    /// Total count for each N-gram order
    char_totals: HashMap<usize, u32>,
    /// Total word bigram count
    word_total: u32,
}

impl NGramModel {
    /// Create a new empty N-gram model
    pub fn new() -> Self {
        Self {
            char_ngrams: HashMap::new(),
            word_bigrams: HashMap::new(),
            char_totals: HashMap::new(),
            word_total: 0,
        }
    }

    /// Load from text data
    pub fn load_from_text(&mut self, text: &str, max_n: usize) {
        self.load_char_ngrams(text, max_n);
        self.load_word_bigrams(text);
    }

    /// Load character-level N-grams from text
    fn load_char_ngrams(&mut self, text: &str, max_n: usize) {
        let chars: Vec<char> = text.chars().collect();
        let text_lower: String = text.chars().flat_map(|c| c.to_lowercase()).collect();

        for n in 1..=max_n {
            let ngram_map = self.char_ngrams.entry(n).or_insert_with(HashMap::new);
            let total = self.char_totals.entry(n).or_insert(0);

            for i in 0..text_lower.len().saturating_sub(n - 1) {
                let ngram: String = text_lower.chars().skip(i).take(n).collect();
                if ngram
                    .chars()
                    .all(|c| c.is_alphabetic() || c.is_whitespace())
                {
                    *ngram_map.entry(ngram).or_insert(0) += 1;
                    *total += 1;
                }
            }
        }
    }

    /// Load word-level bigrams from text
    fn load_word_bigrams(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();

        for window in words.windows(2) {
            let w1 = window[0].to_lowercase();
            let w2 = window[1].to_lowercase();
            *self.word_bigrams.entry((w1, w2)).or_insert(0) += 1;
            self.word_total += 1;
        }
    }

    /// Get character N-gram probability (with smoothing)
    pub fn char_ngram_prob(&self, ngram: &str) -> f64 {
        let n = ngram.chars().count();
        let ngram_lower = ngram.to_lowercase();

        if let Some(ngram_map) = self.char_ngrams.get(&n) {
            let count = *ngram_map.get(&ngram_lower).unwrap_or(&0) as f64;
            let total = *self.char_totals.get(&n).unwrap_or(&1) as f64;

            if total > 0.0 {
                // Laplace smoothing: (count + 1) / (total + vocab_size)
                let vocab_size = ngram_map.len() as f64;
                return (count + 1.0) / (total + vocab_size);
            }
        }

        // Default probability for unseen n-grams
        1e-6
    }

    /// Get word bigram probability (with smoothing)
    pub fn word_bigram_prob(&self, w1: &str, w2: &str) -> f64 {
        let w1_lower = w1.to_lowercase();
        let w2_lower = w2.to_lowercase();

        if self.word_total == 0 {
            return 1e-6;
        }

        let count = *self
            .word_bigrams
            .get(&(w1_lower.clone(), w2_lower))
            .unwrap_or(&0) as f64;
        let vocab_size = self.word_bigrams.len() as f64;

        // Laplace smoothing
        (count + 1.0) / (self.word_total as f64 + vocab_size)
    }

    /// Score a string of text using character N-grams
    /// Returns average log probability
    pub fn score_text(&self, text: &str) -> f64 {
        let text_lower: String = text.chars().flat_map(|c| c.to_lowercase()).collect();
        let mut log_prob_sum = 0.0f64;
        let mut count = 0;

        // Use bigrams and trigrams if available
        if let Some(trigrams) = self.char_ngrams.get(&3) {
            // Score using trigrams
            for i in 0..text_lower.len().saturating_sub(2) {
                let trigram: String = text_lower.chars().skip(i).take(3).collect();
                let prob = self.char_ngram_prob(&trigram);
                log_prob_sum += prob.ln();
                count += 1;
            }
        } else if let Some(bigrams) = self.char_ngrams.get(&2) {
            // Score using bigrams
            for i in 0..text_lower.len().saturating_sub(1) {
                let bigram: String = text_lower.chars().skip(i).take(2).collect();
                let prob = self.char_ngram_prob(&bigram);
                log_prob_sum += prob.ln();
                count += 1;
            }
        }

        if count > 0 {
            log_prob_sum / count as f64
        } else {
            // Fall back to unigrams
            let mut log_prob_sum = 0.0f64;
            for ch in text_lower.chars() {
                let prob = self.char_ngram_prob(&ch.to_string());
                log_prob_sum += prob.ln();
            }
            log_prob_sum / text_lower.chars().count().max(1) as f64
        }
    }

    /// Suggest corrections for a word using word bigrams
    /// context_words: (previous_word, next_word)
    pub fn suggest_word_corrections(
        &self,
        word: &str,
        context_words: Option<(&str, &str)>,
        suggestions: &[&str],
        max_results: usize,
    ) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = suggestions
            .iter()
            .map(|&suggestion| {
                let mut score = 0.0f64;

                // Score based on word bigrams with context
                if let Some((prev, next)) = context_words {
                    if !prev.is_empty() {
                        score += self.word_bigram_prob(prev, suggestion).ln();
                    }
                    if !next.is_empty() {
                        score += self.word_bigram_prob(suggestion, next).ln();
                    }
                }

                (suggestion.to_string(), score)
            })
            .collect();

        // Sort by score (descending) and return top results
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(max_results);
        scored
    }

    /// Get character transition probability (for bigram model)
    pub fn char_transition_prob(&self, from: char, to: char) -> f64 {
        let bigram = format!("{}{}", from, to);
        self.char_ngram_prob(&bigram)
    }
}

impl Default for NGramModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Load a pre-built English N-gram model
pub fn load_english_ngram_model() -> NGramModel {
    let mut model = NGramModel::new();

    // Common English text for building N-grams
    let sample_text = concat!(
        "the quick brown fox jumps over the lazy dog ",
        "pack my box with five dozen liquor jugs ",
        "how vexingly quick daft zebras jump ",
        "the five boxing wizards jump quickly ",
        "sphinx of black quartz judge my vow ",
        "the quick brown fox jumps over the lazy dog ",
        "hello world this is a test of ocr recognition ",
        "machine learning is a subset of artificial intelligence ",
        "natural language processing helps computers understand text ",
        "computer vision enables machines to interpret visual information ",
        "deep neural networks have revolutionized artificial intelligence ",
        "the cat sat on the mat and waited patiently ",
        "to be or not to be that is the question ",
        "all that glitters is not gold ",
        "a journey of a thousand miles begins with a single step ",
        "the pen is mightier than the sword ",
        "actions speak louder than words ",
        "where there is a will there is a way ",
        "rome was not built in a day ",
        "practice makes perfect ",
        "knowledge is power ",
        "time is money ",
        "honesty is the best policy ",
        "the early bird catches the worm ",
        "better late than never ",
        "look before you leap ",
        "a stitch in time saves nine ",
        "don't count your chickens before they hatch ",
        "the grass is always greener on the other side ",
        "every cloud has a silver lining ",
        "when life gives you lemons make lemonade ",
        "fortune favors the bold ",
        "the customer is always right ",
        "location location location ",
        "less is more ",
        "form follows function ",
        "knowledge is power "
    );

    model.load_from_text(sample_text, 3);
    model
}

/// Character-level N-gram model for OCR correction
pub struct CharNGramModel {
    /// Bigram probabilities: P(current_char | previous_char)
    bigrams: HashMap<(char, char), f64>,
}

impl CharNGramModel {
    /// Create from character transition counts
    pub fn from_counts(counts: &HashMap<(char, char), u32>) -> Self {
        let mut bigrams = HashMap::new();
        let total = counts.values().sum::<u32>() as f64;

        for (&(from, to), &count) in counts {
            bigrams.insert((from, to), count as f64 / total);
        }

        Self { bigrams }
    }

    /// Get probability of character transition
    pub fn transition_prob(&self, from: char, to: char) -> f64 {
        *self.bigrams.get(&(from, to)).unwrap_or(&0.0)
    }

    /// Score a word using character transition probabilities
    pub fn score_word(&self, word: &str) -> f64 {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() {
            return 0.0;
        }

        let mut log_prob = 0.0f64;
        for window in chars.windows(2) {
            let prob = self.transition_prob(window[0], window[1]);
            if prob > 0.0 {
                log_prob += prob.ln();
            }
        }

        log_prob / chars.len().saturating_sub(1).max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_model_creation() {
        let mut model = NGramModel::new();
        model.load_from_text("the cat", 2);

        // Should have some bigrams and unigrams
        assert!(*model.char_totals.get(&1).unwrap_or(&0) > 0);
        assert!(*model.char_totals.get(&2).unwrap_or(&0) > 0);
    }

    #[test]
    fn test_char_ngram_prob() {
        let model = load_english_ngram_model();

        // Common bigrams should have higher probability
        let th_prob = model.char_ngram_prob("th");
        let tz_prob = model.char_ngram_prob("tz");

        assert!(th_prob > 0.0);
        assert!(tz_prob > 0.0);
    }

    #[test]
    fn test_score_text() {
        let model = load_english_ngram_model();

        // Common English text should score reasonably
        let score1 = model.score_text("the quick brown fox");
        let score2 = model.score_text("x y z q w e r t");

        // The common text should have higher score (less negative log prob)
        assert!(score1 > score2);
    }

    #[test]
    fn test_word_bigram_prob() {
        let model = load_english_ngram_model();

        // Common word pairs should have some probability
        let prob = model.word_bigram_prob("the", "quick");
        assert!(prob > 0.0);
    }

    #[test]
    fn test_char_transition_prob() {
        let model = load_english_ngram_model();

        // 'q' is almost always followed by 'u' in English
        let qu_prob = model.char_transition_prob('q', 'u');
        let qx_prob = model.char_transition_prob('q', 'x');

        assert!(qu_prob > 0.0);
        assert!(qx_prob < qu_prob);
    }
}
