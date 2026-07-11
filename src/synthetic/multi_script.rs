//! Multi-script synthetic text generation
//!
//! Extends the basic Latin-only generator to support CJK, Arabic,
//! Cyrillic, Greek, Hebrew, Thai, and Devanagari scripts.

use crate::lang::unicode::Script;
use crate::synthetic::generator::TextLineGenerator;
use rand::Rng;

/// Character pools for major world scripts
pub struct ScriptCharPool;

impl ScriptCharPool {
    /// Common CJK characters (simplified subset of most frequent chars)
    pub fn cjk() -> Vec<char> {
        // Mix of CJK Unified Ideographs (common), Hiragana, Katakana, and Hangul
        let ideographs: Vec<char> = (
            "的一是了我不人在他有这个上大来们中为到说国和也出时年"
            .to_string()
            + "会作对开发好用能就工过地行学小多天然于心面看当"
            + "只些想还去法以都可那得无如全定又路它现"
            + "最下家长力里前已但公者从前很动"
            + "日事明其分什高次回被手活"
            + "尔问气因比自什或"
        )
        .chars()
        .collect();

        let hiragana: Vec<char> = (0x3040..=0x309F).map(|c| c as u8 as char).collect();
        let katakana: Vec<char> = (0x30A0..=0x30FF).map(|c| c as u8 as char).collect();
        let hangul: Vec<char> = (0xAC00..=0xAC1F).map(|c| c as u8 as char).collect();

        [ideographs, hiragana, katakana, hangul].concat()
    }

    /// Arabic script characters
    pub fn arabic() -> Vec<char> {
        (0x0620..=0x064A)
            .filter_map(|cp| {
                let ch = std::char::from_u32(cp)?;
                if ch.is_alphabetic() {
                    Some(ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Cyrillic characters (Russian + neighboring languages)
    pub fn cyrillic() -> Vec<char> {
        (0x0400..=0x0450)
            .filter_map(|cp| std::char::from_u32(cp))
            .collect()
    }

    /// Greek characters
    pub fn greek() -> Vec<char> {
        (0x0391..=0x03C9)
            .filter_map(|cp| {
                let ch = std::char::from_u32(cp)?;
                if ch.is_alphabetic() {
                    Some(ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Hebrew characters
    pub fn hebrew() -> Vec<char> {
        (0x05D0..=0x05EA)
            .filter_map(|cp| std::char::from_u32(cp))
            .collect()
    }

    /// Thai characters
    pub fn thai() -> Vec<char> {
        (0x0E01..=0x0E5B)
            .filter_map(|cp| {
                let ch = std::char::from_u32(cp)?;
                if ch.is_alphabetic() {
                    Some(ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Devanagari characters (Hindi, Sanskrit, Marathi)
    pub fn devanagari() -> Vec<char> {
        (0x0904..=0x0939)
            .chain(0x093E..=0x094D)
            .chain(0x0950..=0x0957)
            .filter_map(|cp| std::char::from_u32(cp))
            .collect()
    }

    /// Get character pool for a script
    pub fn for_script(script: Script) -> Vec<char> {
        match script {
            Script::Latin => ('a'..='z').chain('A'..='Z').chain('0'..='9').collect(),
            Script::CJK => Self::cjk(),
            Script::Arabic => Self::arabic(),
            Script::Cyrillic => Self::cyrillic(),
            Script::Greek => Self::greek(),
            Script::Hebrew => Self::hebrew(),
            Script::Thai => Self::thai(),
            Script::Devanagari => Self::devanagari(),
            Script::Other => ('a'..='z').collect(),
        }
    }
}

/// Generates random text lines in a specific script
pub struct ScriptLineGenerator {
    script: Script,
    pool: Vec<char>,
    generator: TextLineGenerator,
}

impl ScriptLineGenerator {
    /// Create a new script-specific line generator
    pub fn new(script: Script) -> Self {
        let pool = ScriptCharPool::for_script(script);
        Self {
            script,
            pool,
            generator: TextLineGenerator::default(),
        }
    }

    /// Set font size and image height
    pub fn with_size(mut self, font_size: f32, image_height: u32) -> Self {
        self.generator = TextLineGenerator::with_size(font_size, image_height);
        self
    }

    /// Add a font for rendering this script
    pub fn add_font(&mut self, font_data: Vec<u8>) {
        self.generator.add_font(font_data);
    }

    /// Generate a random text string in this script
    pub fn random_text(&self, length: usize) -> String {
        if self.pool.is_empty() {
            return String::new();
        }
        (0..length)
            .map(|_| self.pool[rand::thread_rng().gen_range(0..self.pool.len())])
            .collect()
    }

    /// Generate a single synthetic sample in this script
    pub fn generate(&self, text: &str) -> crate::synthetic::generator::SyntheticSample {
        self.generator.generate(text)
    }

    /// Generate a batch of random samples
    pub fn generate_batch(&self, count: usize, text_length: usize) -> Vec<crate::synthetic::generator::SyntheticSample> {
        let texts: Vec<String> = (0..count)
            .map(|_| self.random_text(text_length))
            .collect();
        self.generator.generate_batch(&texts)
    }

    /// Verify that the generated text is classified as the expected script
    pub fn verify_script(&self, text: &str) -> bool {
        Script::detect(text) == self.script
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyrillic_pool_non_empty() {
        let pool = ScriptCharPool::cyrillic();
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_greek_pool_non_empty() {
        let pool = ScriptCharPool::greek();
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_hebrew_pool_non_empty() {
        let pool = ScriptCharPool::hebrew();
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_thai_pool_non_empty() {
        let pool = ScriptCharPool::thai();
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_devanagari_pool_non_empty() {
        let pool = ScriptCharPool::devanagari();
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_script_generator_random_text() {
        let generator = ScriptLineGenerator::new(Script::Cyrillic);
        let text = generator.random_text(20);
        assert_eq!(text.chars().count(), 20);
        // All chars should be in the Cyrillic pool
        let pool = ScriptCharPool::cyrillic();
        for ch in text.chars() {
            assert!(pool.contains(&ch), "Char {} not in Cyrillic pool", ch);
        }
    }

    #[test]
    fn test_script_detection_cyrillic() {
        let generator = ScriptLineGenerator::new(Script::Cyrillic);
        let text = generator.random_text(30);
        assert!(generator.verify_script(&text), "Generated text should be detected as Cyrillic");
    }

    #[test]
    fn test_script_detection_greek() {
        let generator = ScriptLineGenerator::new(Script::Greek);
        let text = generator.random_text(30);
        assert!(generator.verify_script(&text), "Generated text should be detected as Greek");
    }

    #[test]
    fn test_script_detection_hebrew() {
        let generator = ScriptLineGenerator::new(Script::Hebrew);
        let text = generator.random_text(30);
        assert!(generator.verify_script(&text), "Generated text should be detected as Hebrew");
    }

    #[test]
    fn test_script_detection_thai() {
        let generator = ScriptLineGenerator::new(Script::Thai);
        let text = generator.random_text(30);
        assert!(generator.verify_script(&text), "Generated text should be detected as Thai");
    }

    #[test]
    fn test_generate_cyrillic_sample() {
        let generator = ScriptLineGenerator::new(Script::Cyrillic);
        let text = generator.random_text(10);
        let sample = generator.generate(&text);
        assert_eq!(sample.ground_truth, text);
    }
}
