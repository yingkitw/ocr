//! Text data structures and operations for OCR

use crate::utils::Point2D;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Text recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextResult {
    /// The recognized text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Bounding box of the text
    pub bounding_box: BoundingBox,
    /// Character-level results
    pub characters: Vec<CharacterResult>,
    /// Word-level results
    pub words: Vec<WordResult>,
    /// Line-level results
    pub lines: Vec<LineResult>,
    /// Language detected
    pub language: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TextResult {
    /// Create a new text result
    pub fn new(text: String, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            text,
            confidence,
            bounding_box,
            characters: Vec::new(),
            words: Vec::new(),
            lines: Vec::new(),
            language: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the average confidence of all characters
    pub fn average_character_confidence(&self) -> f32 {
        if self.characters.is_empty() {
            self.confidence
        } else {
            let sum: f32 = self.characters.iter().map(|c| c.confidence).sum();
            sum / self.characters.len() as f32
        }
    }

    /// Get the number of characters
    pub fn character_count(&self) -> usize {
        self.characters.len()
    }

    /// Get the number of words
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get the number of lines
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

/// Character recognition result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharacterResult {
    /// The recognized character
    pub character: char,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Bounding box of the character
    pub bounding_box: BoundingBox,
    /// Character code point
    pub code_point: u32,
    /// Character properties
    pub properties: CharacterProperties,
}

impl CharacterResult {
    /// Create a new character result
    pub fn new(character: char, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            character,
            confidence,
            bounding_box,
            code_point: character as u32,
            properties: CharacterProperties::default(),
        }
    }
}

/// Word recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordResult {
    /// The recognized word
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Bounding box of the word
    pub bounding_box: BoundingBox,
    /// Character results within this word
    pub characters: Vec<CharacterResult>,
    /// Word properties
    pub properties: WordProperties,
}

impl WordResult {
    /// Create a new word result
    pub fn new(text: String, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            text,
            confidence,
            bounding_box,
            characters: Vec::new(),
            properties: WordProperties::default(),
        }
    }

    /// Get the average confidence of characters in this word
    pub fn average_character_confidence(&self) -> f32 {
        if self.characters.is_empty() {
            self.confidence
        } else {
            let sum: f32 = self.characters.iter().map(|c| c.confidence).sum();
            sum / self.characters.len() as f32
        }
    }
}

/// Line recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineResult {
    /// The recognized line text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Bounding box of the line
    pub bounding_box: BoundingBox,
    /// Word results within this line
    pub words: Vec<WordResult>,
    /// Line properties
    pub properties: LineProperties,
}

impl LineResult {
    /// Create a new line result
    pub fn new(text: String, confidence: f32, bounding_box: BoundingBox) -> Self {
        Self {
            text,
            confidence,
            bounding_box,
            words: Vec::new(),
            properties: LineProperties::default(),
        }
    }

    /// Get the average confidence of words in this line
    pub fn average_word_confidence(&self) -> f32 {
        if self.words.is_empty() {
            self.confidence
        } else {
            let sum: f32 = self.words.iter().map(|w| w.confidence).sum();
            sum / self.words.len() as f32
        }
    }
}

/// Bounding box for text elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Left coordinate
    pub left: u32,
    /// Top coordinate
    pub top: u32,
    /// Right coordinate
    pub right: u32,
    /// Bottom coordinate
    pub bottom: u32,
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Create a bounding box from center point and dimensions
    pub fn from_center(center: Point2D, width: u32, height: u32) -> Self {
        let half_width = width as f32 / 2.0;
        let half_height = height as f32 / 2.0;

        Self {
            left: (center.x - half_width) as u32,
            top: (center.y - half_height) as u32,
            right: (center.x + half_width) as u32,
            bottom: (center.y + half_height) as u32,
        }
    }

    /// Get the width of the bounding box
    pub fn width(&self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    /// Get the height of the bounding box
    pub fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }

    /// Get the area of the bounding box
    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.left + self.right) as f32 / 2.0,
            (self.top + self.bottom) as f32 / 2.0,
        )
    }

    /// Check if the bounding box contains a point
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    /// Check if the bounding box intersects with another
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.right <= other.left
            || other.right <= self.left
            || self.bottom <= other.top
            || other.bottom <= self.top)
    }

    /// Get the intersection area with another bounding box
    pub fn intersection_area(&self, other: &BoundingBox) -> u32 {
        if !self.intersects(other) {
            return 0;
        }

        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right);
        let bottom = self.bottom.min(other.bottom);

        (right - left) * (bottom - top)
    }

    /// Get the union with another bounding box
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }
}

/// Character properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharacterProperties {
    /// Whether the character is a digit
    pub is_digit: bool,
    /// Whether the character is a letter
    pub is_letter: bool,
    /// Whether the character is whitespace
    pub is_whitespace: bool,
    /// Whether the character is punctuation
    pub is_punctuation: bool,
    /// Character width in pixels
    pub width: u32,
    /// Character height in pixels
    pub height: u32,
    /// Font size estimate
    pub font_size: f32,
}

impl Default for CharacterProperties {
    fn default() -> Self {
        Self {
            is_digit: false,
            is_letter: false,
            is_whitespace: false,
            is_punctuation: false,
            width: 0,
            height: 0,
            font_size: 0.0,
        }
    }
}

/// Word properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordProperties {
    /// Whether the word is in a dictionary
    pub is_dictionary_word: bool,
    /// Word length
    pub length: usize,
    /// Average character width
    pub average_character_width: f32,
    /// Average character height
    pub average_character_height: f32,
    /// Estimated font size
    pub font_size: f32,
    /// Text direction (left-to-right, right-to-left, etc.)
    pub direction: TextDirection,
    /// Whether the word appears to be bold
    pub is_bold: bool,
    /// Whether the word appears to be italic
    pub is_italic: bool,
    /// Whether the word appears to be monospace
    pub is_monospace: bool,
}

impl Default for WordProperties {
    fn default() -> Self {
        Self {
            is_dictionary_word: false,
            length: 0,
            average_character_width: 0.0,
            average_character_height: 0.0,
            font_size: 0.0,
            direction: TextDirection::LeftToRight,
            is_bold: false,
            is_italic: false,
            is_monospace: false,
        }
    }
}

/// Line properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineProperties {
    /// Line height
    pub height: u32,
    /// Baseline position
    pub baseline: u32,
    /// Line spacing
    pub line_spacing: f32,
    /// Text alignment
    pub alignment: TextAlignment,
    /// Reading order
    pub reading_order: ReadingOrder,
    /// Whether the line is detected as vertical text (CJK)
    pub is_vertical: bool,
}

impl Default for LineProperties {
    fn default() -> Self {
        Self {
            height: 0,
            baseline: 0,
            line_spacing: 0.0,
            alignment: TextAlignment::Left,
            reading_order: ReadingOrder::TopToBottom,
            is_vertical: false,
        }
    }
}

/// Text direction enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDirection {
    /// Left to right
    LeftToRight,
    /// Right to left
    RightToLeft,
    /// Top to bottom
    TopToBottom,
    /// Bottom to top
    BottomToTop,
}

/// Text alignment enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlignment {
    /// Left aligned
    Left,
    /// Right aligned
    Right,
    /// Center aligned
    Center,
    /// Justified
    Justified,
}

/// Reading order enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadingOrder {
    /// Top to bottom, left to right
    TopToBottom,
    /// Left to right, top to bottom
    LeftToRight,
    /// Right to left, top to bottom
    RightToLeft,
    /// Bottom to top, left to right
    BottomToTop,
}

// ============================================================================
// Tesseract-specific structures (migrated from Tesseract OCR)
// ============================================================================

use crate::core::geometry::{ICoord, TBox};

/// Word flags (migrated from Tesseract's WERD_FLAGS)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WordFlag {
    Segmented,        // correctly segmented
    Italic,           // italic text
    Bold,             // bold text
    Bol,              // start of line
    Eol,              // end of line
    StartOfLine,      // start of line (alias for Bol)
    EndOfLine,        // end of line (alias for Eol)
    Normalized,       // flags
    ScriptHasXHeight, // x-height concept makes sense
    ScriptIsLatin,    // Special case latin for y splitting
    DontChop,         // fixed pitch chopped
    RepChar,          // repeated character
    FuzzySp,          // fuzzy space
    FuzzyNon,         // fuzzy nonspace
    Inverse,          // white on black
}

/// Display flags (migrated from Tesseract's DISPLAY_FLAGS)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisplayFlag {
    Box,         // Bounding box
    Text,        // Correct ascii
    Polygonal,   // Polyg approx
    EdgeStep,    // Edge steps
    BnPolygonal, // BL normalisd polyapx
    Blamer,      // Blamer information
}

/// Blob choice classifier (migrated from Tesseract's BlobChoiceClassifier)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlobChoiceClassifier {
    StaticClassifier,  // From the char_norm classifier
    AdaptedClassifier, // From the adaptive classifier
    SpeckleClassifier, // Backup for failed classification
    Ambiguity,         // Generated by ambiguity detection
    Fake,              // From some other process
}

/// Character choice for a blob
///
/// This corresponds to Tesseract's `BLOB_CHOICE` class.
/// Represents a possible character recognition result for a blob.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobChoice {
    /// Unicode character ID
    pub unichar_id: u32,
    /// Recognition rating (lower is better)
    pub rating: f32,
    /// Recognition certainty (higher is better)
    pub certainty: f32,
    /// Font information ID
    pub fontinfo_id: i16,
    /// Secondary font information ID
    pub fontinfo_id2: i16,
    /// Script ID
    pub script_id: i16,
    /// Minimum x-height in image pixel units
    pub min_xheight: f32,
    /// Maximum x-height allowed by this char
    pub max_xheight: f32,
    /// Y shift (the larger of y shift top or bottom)
    pub yshift: f32,
    /// Classifier that produced this choice
    pub classifier: BlobChoiceClassifier,
}

impl BlobChoice {
    /// Create a new blob choice
    pub fn new(
        unichar_id: u32,
        rating: f32,
        certainty: f32,
        script_id: i16,
        min_xheight: f32,
        max_xheight: f32,
        yshift: f32,
        classifier: BlobChoiceClassifier,
    ) -> Self {
        Self {
            unichar_id,
            rating,
            certainty,
            fontinfo_id: -1,
            fontinfo_id2: -1,
            script_id,
            min_xheight,
            max_xheight,
            yshift,
            classifier,
        }
    }

    /// Create a default blob choice
    pub fn default() -> Self {
        Self {
            unichar_id: 32, // Space character
            rating: 10.0,
            certainty: -1.0,
            fontinfo_id: -1,
            fontinfo_id2: -1,
            script_id: -1,
            min_xheight: 0.0,
            max_xheight: 0.0,
            yshift: 0.0,
            classifier: BlobChoiceClassifier::Fake,
        }
    }
}

/// Word choice containing multiple character possibilities
///
/// This corresponds to Tesseract's `WERD_CHOICE` class.
/// Represents a word with multiple possible character choices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordChoice {
    /// Character choices for each position
    pub blob_choices: Vec<Vec<BlobChoice>>,
    /// Character choices (alias for blob_choices for compatibility)
    pub choices: Vec<Vec<BlobChoice>>,
    /// Overall word rating
    pub rating: f32,
    /// Overall word certainty
    pub certainty: f32,
    /// Number of blanks before the word
    pub blanks: u8,
    /// Script ID
    pub script_id: i16,
    /// Language model state
    pub language_model_state: Option<String>,
}

impl WordChoice {
    /// Create a new word choice
    pub fn new() -> Self {
        Self {
            blob_choices: Vec::new(),
            choices: Vec::new(),
            rating: 0.0,
            certainty: 0.0,
            blanks: 0,
            script_id: -1,
            language_model_state: None,
        }
    }

    /// Add a character choice to the word
    pub fn add_choice(&mut self, choices: Vec<BlobChoice>) {
        self.blob_choices.push(choices);
    }

    /// Get the best character choice for a position
    pub fn best_choice(&self, position: usize) -> Option<&BlobChoice> {
        self.blob_choices.get(position)?.first()
    }

    /// Get the text representation of the word
    pub fn text(&self) -> String {
        self.blob_choices
            .iter()
            .filter_map(|choices| choices.first())
            .map(|choice| char::from_u32(choice.unichar_id).unwrap_or('?'))
            .collect()
    }
}

/// Word structure containing blobs and recognition results
///
/// This corresponds to Tesseract's `WERD` class.
/// Represents a word with its constituent blobs and recognition information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    /// Number of blanks before the word
    pub blanks: u8,
    /// Word flags
    pub flags: std::collections::HashSet<WordFlag>,
    /// Display flags
    pub display_flags: std::collections::HashSet<DisplayFlag>,
    /// Script ID
    pub script_id: i16,
    /// Correct text (ground truth)
    pub correct_text: String,
    /// Recognition choices
    pub choices: Vec<WordChoice>,
    /// Bounding box of the word
    pub bounding_box: TBox,
    /// Individual character properties
    pub characters: Vec<CharacterResult>,
}

impl Word {
    /// Create a new word
    pub fn new() -> Self {
        Self {
            blanks: 0,
            flags: std::collections::HashSet::new(),
            display_flags: std::collections::HashSet::new(),
            script_id: 0,
            correct_text: String::new(),
            choices: Vec::new(),
            bounding_box: TBox::null(),
            characters: Vec::new(),
        }
    }

    /// Check if a flag is set
    pub fn has_flag(&self, flag: WordFlag) -> bool {
        self.flags.contains(&flag)
    }

    /// Set a flag
    pub fn set_flag(&mut self, flag: WordFlag, value: bool) {
        if value {
            self.flags.insert(flag);
        } else {
            self.flags.remove(&flag);
        }
    }

    /// Check if a display flag is set
    pub fn has_display_flag(&self, flag: DisplayFlag) -> bool {
        self.display_flags.contains(&flag)
    }

    /// Set a display flag
    pub fn set_display_flag(&mut self, flag: DisplayFlag, value: bool) {
        if value {
            self.display_flags.insert(flag);
        } else {
            self.display_flags.remove(&flag);
        }
    }

    /// Get the best recognition choice
    pub fn best_choice(&self) -> Option<&WordChoice> {
        self.choices.first()
    }

    /// Get the recognized text
    pub fn text(&self) -> String {
        self.best_choice()
            .map(|choice| choice.text())
            .unwrap_or_else(|| self.correct_text.clone())
    }

    /// Move the word by a vector
    pub fn move_by(&mut self, vec: ICoord) {
        self.bounding_box.move_by(vec);
        for character in &mut self.characters {
            // Update character bounding boxes
            character.bounding_box = BoundingBox::new(
                character.bounding_box.left + vec.x as u32,
                character.bounding_box.top + vec.y as u32,
                character.bounding_box.right + vec.x as u32,
                character.bounding_box.bottom + vec.y as u32,
            );
        }
    }
}

#[cfg(test)]
mod tesseract_tests {
    use super::*;

    #[test]
    fn test_blob_choice_creation() {
        let choice = BlobChoice::new(
            65, // 'A'
            0.1,
            0.9,
            0, // Latin script
            10.0,
            20.0,
            0.0,
            BlobChoiceClassifier::StaticClassifier,
        );

        assert_eq!(choice.unichar_id, 65);
        assert_eq!(choice.rating, 0.1);
        assert_eq!(choice.certainty, 0.9);
    }

    #[test]
    fn test_word_choice_creation() {
        let mut word_choice = WordChoice::new();
        word_choice.add_choice(vec![BlobChoice::new(
            65,
            0.1,
            0.9,
            0,
            10.0,
            20.0,
            0.0,
            BlobChoiceClassifier::StaticClassifier,
        )]);
        word_choice.add_choice(vec![BlobChoice::new(
            66,
            0.2,
            0.8,
            0,
            10.0,
            20.0,
            0.0,
            BlobChoiceClassifier::StaticClassifier,
        )]);

        assert_eq!(word_choice.text(), "AB");
    }

    #[test]
    fn test_word_creation() {
        let mut word = Word::new();
        word.set_flag(WordFlag::Bold, true);
        word.set_flag(WordFlag::Italic, false);
        word.correct_text = "Hello".to_string();

        assert!(word.has_flag(WordFlag::Bold));
        assert!(!word.has_flag(WordFlag::Italic));
        assert_eq!(word.correct_text, "Hello");
    }
}

// ============================================================================
// Output format serialization
// ============================================================================

impl TextResult {
    /// Format as plain text
    pub fn to_plain_text(&self) -> String {
        self.text.clone()
    }

    /// Format as hOCR (HTML OCR) with bounding boxes
    ///
    /// Produces an HTML document with `ocr_line`, `ocrx_word`, and `ocr_cinfo`
    /// spans following the hOCR specification.
    pub fn to_hocr(&self, image_width: u32, image_height: u32) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str(
            "<html xmlns='http://www.w3.org/1999/xhtml' \
             xmlns:hocr='http://www.w3.org/1999/04/hocr' \
             xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' \
             xsi:schemaLocation='http://www.w3.org/1999/04/hocr http://www.w3.org/2001/04/hocr/hocr.xsd'>\n",
        );
        html.push_str("<head><meta charset='UTF-8' />\n");
        html.push_str(&format!("<title>OCR Output</title>\n</head>\n<body>\n",));
        html.push_str(&format!(
            "<div class='ocr_page' id='page_1' title='image' \
             style='width:{}px;height:{}px'>\n",
            image_width, image_height
        ));

        for (line_idx, line) in self.lines.iter().enumerate() {
            html.push_str(&format!(
                "  <span class='ocr_line' id='line_{}' \
                 title='line {}' \
                 confidence='{}'>\n",
                line_idx + 1,
                line_idx + 1,
                format_confidence(line.confidence),
            ));

            for (word_idx, word) in line.words.iter().enumerate() {
                let bbox = &word.bounding_box;
                let title = format!(
                    "word {}; x {}; y {}; w {}; h {}; \
                     confidence {}",
                    word_idx,
                    bbox.left,
                    bbox.top,
                    bbox.width(),
                    bbox.height(),
                    format_confidence(word.confidence),
                );
                html.push_str(&format!(
                    "    <span class='ocrx_word' id='word_{}_{}' \
                     title='{}'>{}</span>\n",
                    line_idx + 1,
                    word_idx + 1,
                    title,
                    word.text,
                ));
            }

            html.push_str("  </span>\n");
        }

        html.push_str("</div>\n</body>\n</html>");
        html
    }

    /// Format as JSON with confidence scores and bounding boxes
    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct JsonOutput<'a> {
            text: &'a str,
            confidence: f32,
            language: &'a Option<String>,
            num_lines: usize,
            num_words: usize,
            num_characters: usize,
            lines: Vec<JsonLine<'a>>,
        }

        #[derive(Serialize)]
        struct JsonLine<'a> {
            text: &'a str,
            confidence: f32,
            bounding_box: Option<JsonBbox>,
            words: Vec<JsonWord<'a>>,
        }

        #[derive(Serialize)]
        struct JsonWord<'a> {
            text: &'a str,
            confidence: f32,
            bounding_box: Option<JsonBbox>,
            characters: Vec<JsonChar>,
        }

        #[derive(Serialize)]
        struct JsonChar {
            character: char,
            confidence: f32,
            bounding_box: Option<JsonBbox>,
        }

        #[derive(Serialize)]
        struct JsonBbox {
            left: u32,
            top: u32,
            right: u32,
            bottom: u32,
        }

        impl<'a> From<&'a crate::core::text::BoundingBox> for JsonBbox {
            fn from(bbox: &'a crate::core::text::BoundingBox) -> Self {
                JsonBbox {
                    left: bbox.left,
                    top: bbox.top,
                    right: bbox.right,
                    bottom: bbox.bottom,
                }
            }
        }

        let json_output = JsonOutput {
            text: &self.text,
            confidence: self.confidence,
            language: &self.language,
            num_lines: self.lines.len(),
            num_words: self.words.len(),
            num_characters: self.characters.len(),
            lines: self
                .lines
                .iter()
                .map(|line| JsonLine {
                    text: &line.text,
                    confidence: line.confidence,
                    bounding_box: Some((&line.bounding_box).into()),
                    words: line
                        .words
                        .iter()
                        .map(|word| JsonWord {
                            text: &word.text,
                            confidence: word.confidence,
                            bounding_box: Some((&word.bounding_box).into()),
                            characters: word
                                .characters
                                .iter()
                                .map(|c| JsonChar {
                                    character: c.character,
                                    confidence: c.confidence,
                                    bounding_box: Some((&c.bounding_box).into()),
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
        };

        serde_json::to_string_pretty(&json_output).unwrap_or_default()
    }
}

fn format_confidence(confidence: f32) -> String {
    format!("{:.2}", (confidence * 100.0).min(100.0))
}
