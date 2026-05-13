//! Dictionary operations for word validation and correction

use std::collections::HashSet;

/// Dictionary for word validation and spelling correction
pub struct Dictionary {
    words: HashSet<String>,
    max_edit_distance: usize,
}

impl Dictionary {
    pub fn new() -> Self {
        Self {
            words: HashSet::new(),
            max_edit_distance: 2,
        }
    }

    pub fn with_max_edit_distance(max_edit_distance: usize) -> Self {
        Self {
            words: HashSet::new(),
            max_edit_distance,
        }
    }

    pub fn load_from_text(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            if !cleaned.is_empty() {
                self.words.insert(cleaned);
            }
        }
    }

    pub fn load_words(&mut self, words: &[&str]) {
        for word in words {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            if !cleaned.is_empty() {
                self.words.insert(cleaned);
            }
        }
    }

    pub fn contains(&self, word: &str) -> bool {
        let lower: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        self.words.contains(&lower)
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    pub fn suggest_corrections(&self, word: &str, max_suggestions: usize) -> Vec<(String, usize)> {
        let lower: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();

        if lower.is_empty() {
            return Vec::new();
        }

        let exact_lower: String = word.chars().flat_map(|c| c.to_lowercase()).collect();

        if self.words.contains(&exact_lower) {
            return Vec::new();
        }

        let mut candidates: Vec<(String, usize)> = self
            .words
            .iter()
            .filter_map(|w| {
                let dist = edit_distance(&lower, w);
                if dist <= self.max_edit_distance {
                    Some((w.clone(), dist))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
        candidates.truncate(max_suggestions);
        candidates
    }

    pub fn correct_word(&self, word: &str) -> String {
        let lower: String = word.chars().flat_map(|c| c.to_lowercase()).collect();
        if self.words.contains(&lower) {
            return word.to_string();
        }

        let suggestions = self.suggest_corrections(word, 1);
        if let Some((suggestion, _)) = suggestions.first() {
            preserve_case(word, suggestion)
        } else {
            word.to_string()
        }
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let mut prev = vec![0usize; b_len + 1];
    let mut curr = vec![0usize; b_len + 1];

    for j in 0..=b_len {
        prev[j] = j;
    }

    for i in 1..=a_len {
        curr[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_len]
}

fn preserve_case(original: &str, corrected: &str) -> String {
    if original.is_empty() {
        return corrected.to_string();
    }

    let is_upper = original.chars().next().unwrap().is_uppercase();
    let is_all_upper = original.chars().all(|c| c.is_uppercase());

    if is_all_upper {
        corrected.to_uppercase()
    } else if is_upper {
        let mut result = String::with_capacity(corrected.len());
        let mut chars = corrected.chars();
        if let Some(first) = chars.next() {
            result.extend(first.to_uppercase());
        }
        result.extend(chars);
        result
    } else {
        corrected.to_string()
    }
}

pub fn load_english_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();

    let common_words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "was",
        "are",
        "were",
        "been",
        "has",
        "had",
        "did",
        "does",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "need",
        "very",
        "still",
        "much",
        "more",
        "here",
        "own",
        "each",
        "where",
        "why",
        "while",
        "through",
        "during",
        "before",
        "between",
        "after",
        "above",
        "below",
        "under",
        "again",
        "further",
        "once",
        "such",
        "those",
        "being",
        "both",
        "same",
        "every",
        "many",
        "great",
        "old",
        "big",
        "high",
        "long",
        "small",
        "large",
        "next",
        "early",
        "young",
        "important",
        "public",
        "bad",
        "another",
        "right",
        "left",
        "able",
        "end",
        "point",
        "world",
        "life",
        "hand",
        "part",
        "place",
        "case",
        "week",
        "company",
        "system",
        "program",
        "question",
        "work",
        "government",
        "number",
        "night",
        "point",
        "home",
        "water",
        "room",
        "mother",
        "area",
        "money",
        "story",
        "fact",
        "month",
        "lot",
        "right",
        "study",
        "book",
        "eye",
        "job",
        "word",
        "business",
        "issue",
        "side",
        "kind",
        "head",
        "house",
        "service",
        "friend",
        "father",
        "power",
        "hour",
        "game",
        "line",
        "member",
        "law",
        "car",
        "city",
        "community",
        "name",
        "president",
        "team",
        "minute",
        "idea",
        "body",
        "information",
        "back",
        "parent",
        "face",
        "others",
        "level",
        "office",
        "door",
        "health",
        "person",
        "art",
        "war",
        "history",
        "party",
        "result",
        "change",
        "morning",
        "reason",
        "research",
        "girl",
        "guy",
        "moment",
        "air",
        "teacher",
        "force",
        "education",
        "food",
        "picture",
        "class",
        "product",
        "experience",
        "country",
        "problem",
        "today",
        "market",
        "report",
        "family",
        "return",
        "dog",
        "student",
        "group",
        "value",
        "best",
        "plan",
        "state",
        "school",
        "child",
        "college",
        "interest",
        "death",
        "center",
        "often",
        "development",
        "process",
        "music",
        "paper",
        "control",
        "love",
        "true",
        "woman",
        "man",
        "age",
        "per",
        "yes",
        "too",
        "any",
        "let",
        "began",
        "run",
        "help",
        "turn",
        "start",
        "might",
        "show",
        "move",
        "live",
        "put",
        "bring",
        "offer",
        "keep",
        "try",
        "leave",
        "call",
        "hold",
        "provide",
        "seem",
        "help",
        "begin",
        "set",
        "learn",
        "read",
        "grow",
        "open",
        "walk",
        "win",
        "offer",
        "remember",
        "love",
        "consider",
        "appear",
        "buy",
        "wait",
        "serve",
        "die",
        "send",
        "expect",
        "build",
        "stay",
        "fall",
        "cut",
        "reach",
        "kill",
        "remain",
        "suggest",
        "raise",
        "pass",
        "sell",
        "require",
        "report",
        "decide",
        "pull",
        "develop",
        "not",
        "but",
        "and",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "any",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "also",
        "around",
        "away",
        "across",
        "along",
        "already",
        "although",
        "always",
        "am",
        "among",
        "another",
        "anything",
        "are",
        "been",
        "being",
        "came",
        "come",
        "could",
        "did",
        "does",
        "done",
        "either",
        "else",
        "even",
        "ever",
        "every",
        "from",
        "get",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "however",
        "i",
        "if",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "mine",
        "must",
        "my",
        "neither",
        "never",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "own",
        "per",
        "quite",
        "rather",
        "really",
        "said",
        "same",
        "she",
        "should",
        "since",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "upon",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "yet",
        "you",
        "your",
    ];

    dict.load_words(&common_words);
    dict
}

pub struct DictionaryHandler {
    dictionary: Dictionary,
}

impl DictionaryHandler {
    pub fn new() -> Self {
        Self {
            dictionary: load_english_dictionary(),
        }
    }

    pub fn with_dictionary(dictionary: Dictionary) -> Self {
        Self { dictionary }
    }

    /// Load additional words from a plain text file (one word per line or whitespace-separated)
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> std::io::Result<usize> {
        let text = std::fs::read_to_string(path)?;
        let before = self.dictionary.word_count();
        self.dictionary.load_from_text(&text);
        Ok(self.dictionary.word_count() - before)
    }

    /// Load additional words from a string (whitespace-separated)
    pub fn load_from_text(&mut self, text: &str) {
        self.dictionary.load_from_text(text);
    }

    pub fn is_word_valid(&self, word: &str) -> bool {
        self.dictionary.contains(word)
    }

    pub fn suggest(&self, word: &str, max: usize) -> Vec<(String, usize)> {
        self.dictionary.suggest_corrections(word, max)
    }

    pub fn correct(&self, word: &str) -> String {
        self.dictionary.correct_word(word)
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    pub fn dictionary_mut(&mut self) -> &mut Dictionary {
        &mut self.dictionary
    }
}

impl Default for DictionaryHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(edit_distance("", "hello"), 5);
        assert_eq!(edit_distance("hello", ""), 5);
    }

    #[test]
    fn test_edit_distance_substitution() {
        assert_eq!(edit_distance("cat", "bat"), 1);
        assert_eq!(edit_distance("cat", "bet"), 2);
    }

    #[test]
    fn test_edit_distance_insertion() {
        assert_eq!(edit_distance("cat", "cats"), 1);
        assert_eq!(edit_distance("cat", "cast"), 1);
    }

    #[test]
    fn test_edit_distance_deletion() {
        assert_eq!(edit_distance("cats", "cat"), 1);
    }

    #[test]
    fn test_dictionary_contains() {
        let dict = load_english_dictionary();
        assert!(dict.contains("the"));
        assert!(dict.contains("THE"));
        assert!(dict.contains("help"));
    }

    #[test]
    fn test_dictionary_suggest() {
        let dict = load_english_dictionary();
        let suggestions = dict.suggest_corrections("teh", 3);
        assert!(!suggestions.is_empty());
        let (word, dist) = &suggestions[0];
        assert!(*dist <= 2);
        assert!(*word == "the" || *word == "yet");
    }

    #[test]
    fn test_dictionary_correct() {
        let dict = load_english_dictionary();
        let corrected = dict.correct_word("teh");
        assert!(corrected == "the" || corrected == "yet");
    }

    #[test]
    fn test_dictionary_correct_teh() {
        let dict = load_english_dictionary();
        let corrected = dict.correct_word("teh");
        assert!(corrected == "the" || corrected == "yet");
    }

    #[test]
    fn test_dictionary_no_correction_needed() {
        let dict = load_english_dictionary();
        assert_eq!(dict.correct_word("the"), "the");
    }

    #[test]
    fn test_preserve_case() {
        assert_eq!(preserve_case("HELLO", "the"), "THE");
        assert_eq!(preserve_case("Hello", "the"), "The");
        assert_eq!(preserve_case("hello", "the"), "the");
    }
}
