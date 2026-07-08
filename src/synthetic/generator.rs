//! Text-line image generator for synthetic OCR training data

use image::{DynamicImage, GrayImage, Luma};
use imageproc::drawing::draw_text_mut;
use rusttype::{Font, Scale};

/// A single synthetic training sample
#[derive(Debug, Clone)]
pub struct SyntheticSample {
    /// Generated image containing text
    pub image: DynamicImage,
    /// Ground truth text string
    pub ground_truth: String,
}

/// Generator for synthetic text-line images
pub struct TextLineGenerator {
    /// Loaded fonts for rendering
    fonts: Vec<Font<'static>>,
    /// Font size in pixels
    font_size: f32,
    /// Target image height
    image_height: u32,
    /// Background color (white = 255)
    background_color: Luma<u8>,
    /// Text color (black = 0)
    text_color: Luma<u8>,
    /// Horizontal padding in pixels
    padding: u32,
    /// Whether to use variable spacing between characters
    variable_spacing: bool,
}

impl Default for TextLineGenerator {
    fn default() -> Self {
        let mut fonts = Vec::new();
        // Load built-in monospace font data if available, otherwise use empty vec
        if let Some(font_data) = Self::default_font_data() {
            if let Some(font) = Font::try_from_vec(font_data) {
                fonts.push(font);
            }
        }

        Self {
            fonts,
            font_size: 32.0,
            image_height: 64,
            background_color: Luma([255]),
            text_color: Luma([0]),
            padding: 16,
            variable_spacing: false,
        }
    }
}

impl TextLineGenerator {
    /// Create with a specific font size and image height
    pub fn with_size(font_size: f32, image_height: u32) -> Self {
        let mut gen = Self::default();
        gen.font_size = font_size;
        gen.image_height = image_height;
        gen
    }

    /// Add a font from raw TTF/OTF bytes
    pub fn add_font(&mut self, font_data: Vec<u8>) {
        if let Some(font) = Font::try_from_vec(font_data) {
            self.fonts.push(font);
        }
    }

    /// Number of loaded fonts (0 = bitmap fallback)
    pub fn font_count(&self) -> usize {
        self.fonts.len()
    }

    /// Set padding
    pub fn with_padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    /// Enable variable character spacing
    pub fn with_variable_spacing(mut self) -> Self {
        self.variable_spacing = true;
        self
    }

    /// Generate a single text-line image
    pub fn generate(&self, text: &str) -> SyntheticSample {
        self.generate_with_options(text, None)
    }

    /// Generate with a specific font index
    pub fn generate_with_font(&self, text: &str, font_index: usize) -> SyntheticSample {
        self.generate_with_options(text, Some(font_index))
    }

    fn generate_with_options(&self, text: &str, font_index: Option<usize>) -> SyntheticSample {
        // Use TTF font rendering if fonts are available
        if let Some(font) = font_index
            .and_then(|idx| self.fonts.get(idx))
            .or_else(|| self.fonts.first())
        {
            return self.render_with_ttf(text, font);
        }

        // Fallback: bitmap font rendering (no TTF fonts available)
        let image = crate::synthetic::bitmap_font::render_text_bitmap(
            text,
            self.image_height,
            self.background_color,
            self.text_color,
            self.padding,
        );

        SyntheticSample {
            image,
            ground_truth: text.to_string(),
        }
    }

    fn render_with_ttf(&self, text: &str, font: &Font) -> SyntheticSample {
        let scale = Scale::uniform(self.font_size);

        // Calculate text width using font metrics
        let text_width = self.measure_text_width(text, font, scale);
        let img_width = text_width + self.padding * 2;
        let img_height = self.image_height;

        // Create grayscale image with background color
        let mut image = GrayImage::from_pixel(img_width, img_height, self.background_color);

        // Calculate vertical position to center text
        let v_metrics = font.v_metrics(scale);
        let ascent = v_metrics.ascent;
        let baseline_y = (img_height as f32 + ascent * 0.7) / 2.0;

        // Draw text
        draw_text_mut(
            &mut image,
            self.text_color,
            self.padding as i32,
            baseline_y as i32 - self.font_size as i32,
            scale,
            font,
            text,
        );

        SyntheticSample {
            image: DynamicImage::ImageLuma8(image),
            ground_truth: text.to_string(),
        }
    }

    /// Generate a batch of samples from a list of texts
    pub fn generate_batch(&self, texts: &[String]) -> Vec<SyntheticSample> {
        let mut samples = Vec::with_capacity(texts.len());
        for text in texts {
            samples.push(self.generate(text));
        }
        samples
    }

    /// Generate samples with random font selection per text
    pub fn generate_batch_mixed_fonts(&self, texts: &[String]) -> Vec<SyntheticSample> {
        if self.fonts.is_empty() {
            return self.generate_batch(texts);
        }

        let mut samples = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let font_idx = i % self.fonts.len();
            samples.push(self.generate_with_font(text, font_idx));
        }
        samples
    }

    /// Generate random text samples of given length
    pub fn generate_random_texts(&self, count: usize, text_length: usize) -> Vec<String> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let chars: Vec<char> = ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .chain(" .,!?;-:()\"'".chars())
            .collect();

        let mut rng = thread_rng();
        let mut texts = Vec::with_capacity(count);

        for _ in 0..count {
            let text: String = (0..text_length)
                .map(|_| *chars.choose(&mut rng).unwrap())
                .collect();
            texts.push(text);
        }

        texts
    }

    fn measure_text_width(&self, text: &str, font: &Font, scale: Scale) -> u32 {
        let mut width = 0.0f32;
        for glyph in font.layout(text, scale, rusttype::point(0.0, 0.0)) {
            if let Some(bb) = glyph.pixel_bounding_box() {
                width = width.max(bb.max.x as f32);
            }
        }
        (width as u32).max(1)
    }

    fn default_font_data() -> Option<Vec<u8>> {
        // Try to load a system monospace font
        let candidates = [
            "/System/Library/Fonts/Monaco.dfont",
            "/System/Library/Fonts/Courier.dfont",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ];

        for path in &candidates {
            if let Ok(data) = std::fs::read(path) {
                return Some(data);
            }
        }
        None
    }
}

/// Character-level synthetic sample for training per-character classifiers
#[derive(Debug, Clone)]
pub struct CharacterSample {
    pub image: GrayImage,
    pub character: char,
    pub font_index: usize,
}

/// Generator for individual character images (useful for pattern-matching template training)
pub struct CharacterGenerator {
    generator: TextLineGenerator,
    target_size: (u32, u32),
}

impl CharacterGenerator {
    pub fn new(target_size: (u32, u32)) -> Self {
        Self {
            generator: TextLineGenerator::with_size(28.0, target_size.1),
            target_size,
        }
    }

    pub fn add_font(&mut self, font_data: Vec<u8>) {
        self.generator.add_font(font_data);
    }

    /// Generate an image for a single character, centered and normalized
    pub fn generate(&self, character: char, font_index: usize) -> CharacterSample {
        let text = character.to_string();
        let sample = self.generator.generate_with_font(&text, font_index);

        // Resize to target size
        let resized = sample.image.resize_exact(
            self.target_size.0,
            self.target_size.1,
            image::imageops::FilterType::Lanczos3,
        );

        let gray = resized.to_luma8();

        CharacterSample {
            image: gray,
            character,
            font_index,
        }
    }

    /// Generate all ASCII printable characters for each loaded font
    pub fn generate_all_ascii(&self) -> Vec<CharacterSample> {
        let mut samples = Vec::new();
        let chars: Vec<char> = (' '..='~').collect();

        for font_idx in 0..self.generator.fonts.len() {
            for &ch in &chars {
                samples.push(self.generate(ch, font_idx));
            }
        }

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_simple_text() {
        let gen = TextLineGenerator::default();
        let sample = gen.generate("Hello");
        assert_eq!(sample.ground_truth, "Hello");
        assert!(sample.image.width() > 0);
        assert!(sample.image.height() > 0);
    }

    #[test]
    fn test_generate_batch() {
        let gen = TextLineGenerator::default();
        let texts = vec![
            "Hello".to_string(),
            "World".to_string(),
            "Test123".to_string(),
        ];
        let samples = gen.generate_batch(&texts);
        assert_eq!(samples.len(), 3);
        for (i, sample) in samples.iter().enumerate() {
            assert_eq!(sample.ground_truth, texts[i]);
        }
    }

    #[test]
    fn test_generate_random_texts() {
        let gen = TextLineGenerator::default();
        let texts = gen.generate_random_texts(10, 20);
        assert_eq!(texts.len(), 10);
        for text in &texts {
            assert_eq!(text.len(), 20);
        }
    }

    #[test]
    fn test_character_generator() {
        let gen = CharacterGenerator::new((28, 28));
        let sample = gen.generate('A', 0);
        assert_eq!(sample.character, 'A');
        assert_eq!(sample.image.width(), 28);
        assert_eq!(sample.image.height(), 28);
    }
}
