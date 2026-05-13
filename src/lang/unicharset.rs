//! Unicode character set handling
//!
//! Ported from Tesseract's unicharset.h/cpp
//! Manages character sets, properties, and character mappings for OCR

use crate::utils::{OcrError, Result};
use std::collections::HashMap;

/// Special unichar codes (keep in sync with SpecialUnicharCodes enum)
const SPECIAL_UNICHAR_CODES: &[&str] = &[" ", "Joined", "|Broken|0|1"];

/// Unicode character set for OCR
///
/// Manages the set of characters that can be recognized,
/// their properties, and mappings
pub struct Unicharset {
    /// Map from character representation to ID
    ids: HashMap<String, u32>,
    /// Map from ID to character representation
    chars: HashMap<u32, String>,
    /// Character properties
    properties: HashMap<u32, UnicharProperties>,
    /// Next available ID
    next_id: u32,
    /// Whether old style characters are included
    old_style_included: bool,
}

/// Properties of a Unicode character
#[derive(Debug, Clone)]
pub struct UnicharProperties {
    /// Is alphabetic
    pub is_alpha: bool,
    /// Is lowercase
    pub is_lower: bool,
    /// Is uppercase
    pub is_upper: bool,
    /// Is digit
    pub is_digit: bool,
    /// Is punctuation
    pub is_punctuation: bool,
    /// Is n-gram
    pub is_ngram: bool,
    /// Is enabled
    pub enabled: bool,
    /// Script ID
    pub script_id: u32,
    /// Other case ID
    pub other_case: u32,
    /// Normalized representation
    pub normed: String,
    /// Text direction
    pub direction: TextDirection,
}

/// Text direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextDirection {
    /// Left to right
    LeftToRight,
    /// Right to left
    RightToLeft,
    /// Top to bottom
    TopToBottom,
    /// Mixed
    Mixed,
}

impl UnicharProperties {
    /// Create default properties
    fn new() -> Self {
        Self {
            is_alpha: false,
            is_lower: false,
            is_upper: false,
            is_digit: false,
            is_punctuation: false,
            is_ngram: false,
            enabled: true,
            script_id: 0,
            other_case: 0,
            normed: String::new(),
            direction: TextDirection::LeftToRight,
        }
    }

    /// Set properties from character
    fn from_char(ch: char) -> Self {
        let mut props = Self::new();
        props.is_alpha = ch.is_alphabetic();
        props.is_lower = ch.is_lowercase();
        props.is_upper = ch.is_uppercase();
        props.is_digit = ch.is_ascii_digit();
        props.is_punctuation = ch.is_ascii_punctuation();
        props.normed = ch.to_string();
        props
    }
}

impl Unicharset {
    /// Create a new unicharset
    pub fn new() -> Self {
        let mut set = Self {
            ids: HashMap::new(),
            chars: HashMap::new(),
            properties: HashMap::new(),
            next_id: 0,
            old_style_included: false,
        };

        // Insert special characters
        for special in SPECIAL_UNICHAR_CODES {
            set.unichar_insert(special).unwrap();
        }

        set
    }

    /// Insert a character into the set
    ///
    /// Returns the ID of the character
    pub fn unichar_insert(&mut self, unichar: &str) -> Result<u32> {
        let cleaned = if self.old_style_included {
            unichar.to_string()
        } else {
            self.cleanup_string(unichar)
        };

        if let Some(&id) = self.ids.get(&cleaned) {
            return Ok(id);
        }

        let id = self.next_id;
        self.ids.insert(cleaned.clone(), id);
        self.chars.insert(id, cleaned.clone());

        // Set properties
        let props = if cleaned.len() == 1 {
            if let Some(ch) = cleaned.chars().next() {
                UnicharProperties::from_char(ch)
            } else {
                UnicharProperties::new()
            }
        } else {
            UnicharProperties::new()
        };

        self.properties.insert(id, props);
        self.next_id += 1;

        Ok(id)
    }

    /// Get character ID from representation
    pub fn unichar_to_id(&self, unichar: &str) -> Option<u32> {
        let cleaned = if self.old_style_included {
            unichar.to_string()
        } else {
            self.cleanup_string(unichar)
        };

        self.ids.get(&cleaned).copied()
    }

    /// Get character representation from ID
    pub fn id_to_unichar(&self, id: u32) -> Option<&String> {
        self.chars.get(&id)
    }

    /// Get properties for a character ID
    pub fn get_properties(&self, id: u32) -> Option<&UnicharProperties> {
        self.properties.get(&id)
    }

    /// Get mutable properties for a character ID
    pub fn get_properties_mut(&mut self, id: u32) -> Option<&mut UnicharProperties> {
        self.properties.get_mut(&id)
    }

    /// Get the size of the character set
    pub fn size(&self) -> usize {
        self.ids.len()
    }

    /// Check if character set contains a character
    pub fn contains(&self, unichar: &str) -> bool {
        self.unichar_to_id(unichar).is_some()
    }

    /// Cleanup string representation
    ///
    /// Removes ligatures and normalizes characters
    fn cleanup_string(&self, s: &str) -> String {
        let mut result = s.to_string();

        // Remove TATWEEL (U+0640)
        result = result.replace("\u{0640}", "");

        // Replace ligatures
        result = result.replace("\u{FB01}", "fi"); // fi ligature
        result = result.replace("\u{FB02}", "fl"); // fl ligature

        result
    }

    /// Set whether character is enabled
    pub fn set_enabled(&mut self, id: u32, enabled: bool) -> Result<()> {
        if let Some(props) = self.properties.get_mut(&id) {
            props.enabled = enabled;
            Ok(())
        } else {
            Err(OcrError::InvalidInput(format!(
                "Invalid character ID: {}",
                id
            )))
        }
    }

    /// Set whether character is n-gram
    pub fn set_is_ngram(&mut self, id: u32, is_ngram: bool) -> Result<()> {
        if let Some(props) = self.properties.get_mut(&id) {
            props.is_ngram = is_ngram;
            Ok(())
        } else {
            Err(OcrError::InvalidInput(format!(
                "Invalid character ID: {}",
                id
            )))
        }
    }

    /// Get all enabled character IDs
    pub fn get_enabled_ids(&self) -> Vec<u32> {
        self.properties
            .iter()
            .filter(|(_, props)| props.enabled)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all characters
    pub fn get_all_chars(&self) -> Vec<String> {
        self.chars.values().cloned().collect()
    }

    /// Check if character set has x-height
    ///
    /// Determines if the character set distinguishes between
    /// capital and lowercase letters
    pub fn has_xheight(&self) -> bool {
        let mut cap_count = 0;
        let mut x_count = 0;
        let mut alpha_count = 0;

        for props in self.properties.values() {
            if props.is_alpha {
                alpha_count += 1;
                if props.is_upper {
                    cap_count += 1;
                } else if props.is_lower {
                    x_count += 1;
                }
            }
        }

        if alpha_count == 0 {
            return false;
        }

        // Check if we have enough distinction between caps and lowercase
        const MIN_X_HEIGHT_FRACTION: f64 = 0.25;
        const MIN_CAP_HEIGHT_FRACTION: f64 = 0.05;

        (x_count as f64 / alpha_count as f64 > MIN_X_HEIGHT_FRACTION
            && cap_count as f64 / alpha_count as f64 > MIN_CAP_HEIGHT_FRACTION)
            || (cap_count + x_count) as f64 / alpha_count as f64 > 0.5
    }
}

impl Default for Unicharset {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicharset_creation() {
        let set = Unicharset::new();
        assert!(set.size() >= SPECIAL_UNICHAR_CODES.len());
    }

    #[test]
    fn test_unichar_insert() {
        let mut set = Unicharset::new();
        let id = set.unichar_insert("A").unwrap();
        assert!(set.contains("A"));
        assert_eq!(set.id_to_unichar(id), Some(&"A".to_string()));
    }

    #[test]
    fn test_character_properties() {
        let mut set = Unicharset::new();
        let id_a = set.unichar_insert("A").unwrap();
        let id_b = set.unichar_insert("b").unwrap();
        let id_1 = set.unichar_insert("1").unwrap();

        let props_a = set.get_properties(id_a).unwrap();
        assert!(props_a.is_alpha);
        assert!(props_a.is_upper);
        assert!(!props_a.is_lower);

        let props_b = set.get_properties(id_b).unwrap();
        assert!(props_b.is_alpha);
        assert!(props_b.is_lower);
        assert!(!props_b.is_upper);

        let props_1 = set.get_properties(id_1).unwrap();
        assert!(props_1.is_digit);
        assert!(!props_1.is_alpha);
    }
}
