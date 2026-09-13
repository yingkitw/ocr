//! Generate test images with varying complexity for OCR testing.
//!
//! Uses real TTF font rendering via `rusttype` (same engine as the synthetic
//  training-data generator) instead of pseudo-random pixel patterns. Loads
//  multiple system fonts when available and renders diverse text content at
//  different sizes, styles, and layouts.
//
//  Run: cargo run --example generate_test_images

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use rusttype::{Font, Scale};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("test_images")?;
    std::fs::create_dir_all("examples/images")?;

    let fonts = load_fonts();
    println!("Loaded {} font(s)", fonts.len());
    println!("Generating test images...");

    // examples/images/ — the pre-bundled fixture images used by examples and docs
    generate_hello_world(&fonts, "examples/images/hello_world.png")?;
    generate_alphabet(&fonts, "examples/images/alphabet.png")?;
    generate_lowercase(&fonts, "examples/images/lowercase.png")?;
    generate_digits(&fonts, "examples/images/digits.png")?;
    generate_punctuation(&fonts, "examples/images/punctuation.png")?;
    generate_sentence(&fonts, "examples/images/sentence.png")?;
    generate_multiline(&fonts, "examples/images/multiline.png")?;
    generate_mixed(&fonts, "examples/images/mixed.png")?;

    // test_images/ — richer set for integration tests
    generate_simple_text(&fonts, "test_images/simple_text.png")?;
    generate_multiline_text(&fonts, "test_images/multiline_text.png")?;
    generate_mixed_case(&fonts, "test_images/mixed_case.png")?;
    generate_numbers_special(&fonts, "test_images/numbers_special.png")?;
    generate_small_text(&fonts, "test_images/small_text.png")?;
    generate_large_text(&fonts, "test_images/large_text.png")?;
    generate_dark_background(&fonts, "test_images/dark_background.png")?;
    generate_column_layout(&fonts, "test_images/column_layout.png")?;
    generate_noisy_background(&fonts, "test_images/noisy_background.png")?;
    generate_dense_text(&fonts, "test_images/dense_text.png")?;
    generate_mixed_languages(&fonts, "test_images/mixed_languages.png")?;
    generate_heading_paragraph(&fonts, "test_images/heading_paragraph.png")?;

    // Tesseract-inspired test images (phototest, eurotext, training digits)
    generate_phototest(&fonts, "test_images/phototest.png")?;
    generate_eurotext(&fonts, "test_images/eurotext.png")?;
    generate_training_digits(&fonts, "test_images/training_digits.png")?;

    // GOCR-inspired test images (ligatures, thin font, framed, snowy, 200dpi)
    generate_ligatures(&fonts, "test_images/ligatures.png")?;
    generate_thin_font(&fonts, "test_images/thin_font.png")?;
    generate_framed_document(&fonts, "test_images/framed_document.png")?;
    generate_snowy(&fonts, "test_images/snowy.png")?;
    generate_200dpi(&fonts, "test_images/text_200dpi.png")?;

    println!("✓ All test images generated successfully!");
    Ok(())
}

// ---------------------------------------------------------------------------
// Font loading
// ---------------------------------------------------------------------------

/// Font category for deliberate variety across test images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FontCategory {
    SansSerif,
    Serif,
    Monospace,
    Decorative,
}

struct LoadedFont {
    font: Font<'static>,
    category: FontCategory,
    name: &'static str,
}

/// Load all available system fonts for rendering variety, organized by category.
fn load_fonts() -> Vec<LoadedFont> {
    // (path, category, name) — ordered so the first match per category wins
    // as a stable default, but all available fonts are loaded.
    let candidates: &[(&str, FontCategory, &str)] = &[
        // --- Sans-serif ---
        ("/System/Library/Fonts/Helvetica.ttc", FontCategory::SansSerif, "Helvetica"),
        ("/System/Library/Fonts/Avenir.ttc", FontCategory::SansSerif, "Avenir"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", FontCategory::SansSerif, "Arial"),
        ("/System/Library/Fonts/Supplemental/Verdana.ttf", FontCategory::SansSerif, "Verdana"),
        ("/System/Library/Fonts/Supplemental/Trebuchet MS.ttf", FontCategory::SansSerif, "Trebuchet MS"),
        ("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf", FontCategory::SansSerif, "Arial Rounded"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FontCategory::SansSerif, "DejaVu Sans"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", FontCategory::SansSerif, "Liberation Sans"),
        ("C:/Windows/Fonts/arial.ttf", FontCategory::SansSerif, "Arial"),
        ("C:/Windows/Fonts/verdana.ttf", FontCategory::SansSerif, "Verdana"),
        // --- Serif ---
        ("/System/Library/Fonts/Times.ttc", FontCategory::Serif, "Times"),
        ("/System/Library/Fonts/Supplemental/Georgia.ttf", FontCategory::Serif, "Georgia"),
        ("/System/Library/Fonts/Supplemental/Baskerville.ttf", FontCategory::Serif, "Baskerville"),
        ("/System/Library/Fonts/Supplemental/Palatino.ttf", FontCategory::Serif, "Palatino"),
        ("/System/Library/Fonts/Supplemental/Hoefler Text.ttf", FontCategory::Serif, "Hoefler Text"),
        ("/System/Library/Fonts/Supplemental/Cochin.ttf", FontCategory::Serif, "Cochin"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", FontCategory::Serif, "DejaVu Serif"),
        ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", FontCategory::Serif, "Liberation Serif"),
        ("C:/Windows/Fonts/times.ttf", FontCategory::Serif, "Times New Roman"),
        ("C:/Windows/Fonts/georgia.ttf", FontCategory::Serif, "Georgia"),
        // --- Monospace ---
        ("/System/Library/Fonts/Courier.ttc", FontCategory::Monospace, "Courier"),
        ("/System/Library/Fonts/Menlo.ttc", FontCategory::Monospace, "Menlo"),
        ("/System/Library/Fonts/Monaco.dfont", FontCategory::Monospace, "Monaco"),
        ("/System/Library/Fonts/Supplemental/Courier New.ttf", FontCategory::Monospace, "Courier New"),
        ("/System/Library/Fonts/Supplemental/Andale Mono.ttf", FontCategory::Monospace, "Andale Mono"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", FontCategory::Monospace, "DejaVu Sans Mono"),
        ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", FontCategory::Monospace, "Liberation Mono"),
        ("C:/Windows/Fonts/cour.ttf", FontCategory::Monospace, "Courier New"),
        ("C:/Windows/Fonts/consola.ttf", FontCategory::Monospace, "Consolas"),
        // --- Decorative / handwriting ---
        ("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf", FontCategory::Decorative, "Bradley Hand"),
        ("/System/Library/Fonts/Supplemental/Brush Script.ttf", FontCategory::Decorative, "Brush Script"),
        ("/System/Library/Fonts/Supplemental/Chalkduster.ttf", FontCategory::Decorative, "Chalkduster"),
        ("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf", FontCategory::Decorative, "Comic Sans MS"),
        ("/System/Library/Fonts/Supplemental/Herculanum.ttf", FontCategory::Decorative, "Herculanum"),
        ("/System/Library/Fonts/Supplemental/Apple Chancery.ttf", FontCategory::Decorative, "Apple Chancery"),
    ];

    let mut loaded = Vec::new();
    for (path, category, name) in candidates {
        if let Ok(data) = std::fs::read(path)
            && let Some(font) = Font::try_from_vec(data)
        {
            loaded.push(LoadedFont { font, category: *category, name });
        }
    }

    if loaded.is_empty() {
        eprintln!("Warning: no system fonts found; images will use placeholder glyphs");
    } else {
        let mut counts = [0usize; 4];
        for f in &loaded {
            counts[f.category as usize] += 1;
        }
        println!(
            "  sans-serif={} serif={} mono={} decorative={}",
            counts[0], counts[1], counts[2], counts[3]
        );
        for f in &loaded {
            println!("    - {} ({:?})", f.name, f.category);
        }
    }

    loaded
}

/// Pick the first loaded font of a given category.
fn font_by_category(fonts: &[LoadedFont], cat: FontCategory) -> Option<&Font<'static>> {
    fonts.iter().find(|f| f.category == cat).map(|f| &f.font)
}

/// Pick the n-th loaded font of a given category (wraps around).
fn font_by_category_idx(
    fonts: &[LoadedFont],
    cat: FontCategory,
    idx: usize,
) -> Option<&Font<'static>> {
    let of_cat: Vec<&LoadedFont> = fonts.iter().filter(|f| f.category == cat).collect();
    if of_cat.is_empty() {
        None
    } else {
        Some(&of_cat[idx % of_cat.len()].font)
    }
}

/// Pick any font by simple index (for backward-compat with the draw_line helper).
#[allow(dead_code)]
fn pick_font(fonts: &[LoadedFont], idx: usize) -> Option<&Font<'static>> {
    if fonts.is_empty() {
        None
    } else {
        Some(&fonts[idx % fonts.len()].font)
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

fn white_gray(w: u32, h: u32) -> GrayImage {
    GrayImage::from_pixel(w, h, Luma([255]))
}

fn white_rgb(w: u32, h: u32) -> RgbImage {
    RgbImage::from_pixel(w, h, Rgb([255, 255, 255]))
}

fn render_line_gray(
    img: &mut GrayImage,
    text: &str,
    x: i32,
    y: i32,
    font_size: f32,
    font: &Font,
) {
    let scale = Scale::uniform(font_size);
    draw_text_mut(img, Luma([0]), x, y, scale, font, text);
}

fn render_line_rgb(
    img: &mut RgbImage,
    text: &str,
    x: i32,
    y: i32,
    font_size: f32,
    font: &Font,
    color: Rgb<u8>,
) {
    let scale = Scale::uniform(font_size);
    draw_text_mut(img, color, x, y, scale, font, text);
}

fn measure_width(text: &str, font: &Font, font_size: f32) -> u32 {
    let scale = Scale::uniform(font_size);
    let mut width = 0.0f32;
    for glyph in font.layout(text, scale, rusttype::point(0.0, 0.0)) {
        if let Some(bb) = glyph.pixel_bounding_box() {
            width = width.max(bb.max.x as f32);
        }
    }
    (width as u32).max(1)
}

/// Fallback: draw a filled rectangle as a placeholder when no font is available.
fn placeholder_line(img: &mut GrayImage, x: u32, y: u32, text: &str, font_size: u32) {
    let w = (text.len() as u32 * font_size / 2).max(20);
    let h = (font_size / 2).max(8);
    for dy in 0..h {
        for dx in 0..w {
            if x + dx < img.width() && y + dy < img.height() {
                img.put_pixel(x + dx, y + dy, Luma([0]));
            }
        }
    }
}

/// Render a single line of text, using TTF if available, placeholder otherwise.
#[allow(dead_code)]
fn draw_line(
    img: &mut GrayImage,
    text: &str,
    x: u32,
    y: u32,
    font_size: f32,
    fonts: &[LoadedFont],
    font_idx: usize,
) {
    if let Some(font) = pick_font(fonts, font_idx) {
        render_line_gray(img, text, x as i32, y as i32, font_size, font);
    } else {
        placeholder_line(img, x, y, text, font_size as u32);
    }
}

/// Render a single line using a specific font category.
fn draw_line_cat(
    img: &mut GrayImage,
    text: &str,
    x: u32,
    y: u32,
    font_size: f32,
    fonts: &[LoadedFont],
    cat: FontCategory,
    idx: usize,
) {
    if let Some(font) = font_by_category_idx(fonts, cat, idx) {
        render_line_gray(img, text, x as i32, y as i32, font_size, font);
    } else {
        placeholder_line(img, x, y, text, font_size as u32);
    }
}

// ---------------------------------------------------------------------------
// examples/images/ generators (the pre-bundled fixtures)
// ---------------------------------------------------------------------------

fn generate_hello_world(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(400, 80);
    draw_line_cat(&mut img, "Hello World", 30, 15, 36.0, fonts, FontCategory::SansSerif, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_alphabet(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 80);
    draw_line_cat(&mut img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 20, 15, 28.0, fonts, FontCategory::Serif, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_lowercase(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 80);
    draw_line_cat(&mut img, "abcdefghijklmnopqrstuvwxyz", 20, 15, 28.0, fonts, FontCategory::Serif, 1);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_digits(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(400, 80);
    draw_line_cat(&mut img, "0123456789", 20, 15, 36.0, fonts, FontCategory::Monospace, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_punctuation(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(500, 80);
    draw_line_cat(&mut img, "!\"#$%&'()*+,-./:;<=>?@[\\]^_{|}~", 20, 15, 28.0, fonts, FontCategory::Monospace, 1);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_sentence(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(700, 80);
    draw_line_cat(&mut img, "The quick brown fox jumps over the lazy dog.", 20, 15, 28.0, fonts, FontCategory::SansSerif, 1);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_multiline(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 200);
    let lines = [
        "First line of text content",
        "Second line with numbers 12345",
        "Third line: mixed CASE and !@#",
    ];
    // Each line in a different font category for variety
    let cats = [FontCategory::SansSerif, FontCategory::Serif, FontCategory::Monospace];
    let mut y = 20;
    for (i, line) in lines.iter().enumerate() {
        draw_line_cat(&mut img, line, 20, y, 28.0, fonts, cats[i], 0);
        y += 50;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_mixed(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 200);
    // One line per font category — sans, serif, mono, decorative
    let lines: [(&str, f32, FontCategory, usize); 4] = [
        ("Hello World", 32.0, FontCategory::SansSerif, 0),
        ("1234567890", 28.0, FontCategory::Monospace, 0),
        ("Mixed CASE Text!@#", 24.0, FontCategory::Serif, 0),
        ("Handwritten note", 26.0, FontCategory::Decorative, 0),
    ];
    let mut y = 10;
    for (text, size, cat, fi) in lines {
        draw_line_cat(&mut img, text, 20, y, size, fonts, cat, fi);
        y += 45;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// test_images/ generators (richer set for integration tests)
// ---------------------------------------------------------------------------

fn generate_simple_text(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(400, 100);
    draw_line_cat(&mut img, "Hello World", 20, 25, 32.0, fonts, FontCategory::SansSerif, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_multiline_text(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(500, 200);
    let lines = [
        ("Line 1: The quick brown fox", FontCategory::Serif),
        ("Line 2: jumps over the lazy dog", FontCategory::SansSerif),
        ("Line 3: 1234567890", FontCategory::Monospace),
    ];
    let mut y = 25;
    for (line, cat) in lines.iter() {
        draw_line_cat(&mut img, line, 20, y, 24.0, fonts, *cat, 0);
        y += 45;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_mixed_case(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 150);
    draw_line_cat(&mut img, "MiXeD cAsE: AbCdEfG 123 xYz", 20, 45, 28.0, fonts, FontCategory::SansSerif, 1);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_numbers_special(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(500, 180);
    let lines = [
        ("Numbers: 0123456789", FontCategory::Monospace),
        ("Special: !@#$%^&*()", FontCategory::Monospace, ),
        ("Math: +-*/=<>?", FontCategory::Monospace),
    ];
    let mut y = 20;
    for (line, cat) in lines.iter() {
        draw_line_cat(&mut img, line, 20, y, 24.0, fonts, *cat, 0);
        y += 45;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_small_text(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(300, 80);
    draw_line_cat(&mut img, "Small text: 8pt font size", 10, 25, 14.0, fonts, FontCategory::SansSerif, 2);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_large_text(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(800, 200);
    draw_line_cat(&mut img, "LARGE TEXT 72PT", 20, 60, 64.0, fonts, FontCategory::SansSerif, 3);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_dark_background(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = GrayImage::from_pixel(500, 150, Luma([30]));
    if let Some(font) = font_by_category(fonts, FontCategory::SansSerif) {
        render_line_gray(&mut img, "White text on dark", 20, 45, 32.0, font);
    } else {
        // placeholder on dark bg — draw light rectangles
        let w = 200;
        for dy in 0..20 {
            for dx in 0..w {
                if 20 + dx < img.width() && 45 + dy < img.height() {
                    img.put_pixel(20 + dx, 45 + dy, Luma([255]));
                }
            }
        }
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_column_layout(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(800, 400);
    let left = ["Column 1:", "Item A", "Item B", "Item C"];
    let right = ["Column 2:", "Data X", "Data Y", "Data Z"];
    let mut y = 30;
    for (i, line) in left.iter().enumerate() {
        draw_line_cat(&mut img, line, 50, y, 20.0, fonts, FontCategory::Serif, i % 2);
        y += 35;
    }
    y = 30;
    for (i, line) in right.iter().enumerate() {
        draw_line_cat(&mut img, line, 450, y, 20.0, fonts, FontCategory::SansSerif, i % 2);
        y += 35;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_noisy_background(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(500, 150);
    // Add noise
    let mut rng = rand::thread_rng();
    use rand::Rng;
    for pixel in img.pixels_mut() {
        let noise = rng.gen_range(200..255);
        *pixel = Luma([noise]);
    }
    draw_line_cat(&mut img, "Text on noisy background", 20, 45, 32.0, fonts, FontCategory::Serif, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_dense_text(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let paragraph = "This is a dense paragraph of text that contains multiple sentences. \
                     It tests how well the OCR system handles continuous text without breaks. \
                     The text includes various punctuation marks, numbers like 123, and mixed \
                     case letters. This simulates real-world document scanning scenarios.";

    // Word-wrap the paragraph into lines that fit within max_width.
    let font_size = 20.0f32;
    let max_width = 560u32;
    let line_height = 30u32;
    let mut img = white_gray(600, 300);

    let words: Vec<&str> = paragraph.split_whitespace().collect();
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    for word in &words {
        let trial = if current.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current, word)
        };
        let w = if let Some(font) = font_by_category(fonts, FontCategory::Serif) {
            measure_width(&trial, font, font_size)
        } else {
            trial.len() as u32 * 10
        };
        if w <= max_width {
            current = trial;
        } else {
            if !current.is_empty() {
                lines.push(current.clone());
            }
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }

    let mut y = 20u32;
    for line in lines.iter() {
        draw_line_cat(&mut img, line, 20, y, font_size, fonts, FontCategory::Serif, 0);
        y += line_height;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_mixed_languages(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_rgb(600, 250);
    let lines = [
        ("English: Hello World", FontCategory::SansSerif),
        ("Francais: Bonjour le monde", FontCategory::Serif),
        ("Deutsch: Guten Tag", FontCategory::SansSerif),
        ("Espanol: Hola mundo", FontCategory::Serif),
        ("Italiano: Ciao mondo", FontCategory::SansSerif),
    ];
    let mut y = 25;
    for (line, cat) in lines.iter() {
        if let Some(font) = font_by_category(fonts, *cat) {
            render_line_rgb(&mut img, line, 20, y, 22.0, font, Rgb([0, 0, 0]));
        }
        y += 40;
    }
    // Save as grayscale for consistency with other test images
    let gray = DynamicImage::ImageRgb8(img).to_luma8();
    gray.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_heading_paragraph(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 350);
    // Heading (larger font, sans-serif)
    draw_line_cat(&mut img, "Document Title", 20, 20, 32.0, fonts, FontCategory::SansSerif, 0);
    // Paragraph (smaller font, serif, multiple lines)
    let para = "This is a paragraph of body text that follows the heading. \
                It contains multiple sentences and tests the layout analysis \
                pipeline's ability to distinguish headings from body text \
                based on font size differences.";
    let words: Vec<&str> = para.split_whitespace().collect();
    let font_size = 20.0f32;
    let max_width = 560u32;
    let line_height = 28u32;

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in &words {
        let trial = if current.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current, word)
        };
        let w = if let Some(font) = font_by_category(fonts, FontCategory::Serif) {
            measure_width(&trial, font, font_size)
        } else {
            trial.len() as u32 * 10
        };
        if w <= max_width {
            current = trial;
        } else {
            if !current.is_empty() {
                lines.push(current.clone());
            }
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }

    let mut y = 70u32;
    for line in &lines {
        draw_line_cat(&mut img, line, 20, y, font_size, fonts, FontCategory::Serif, 0);
        y += line_height;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tesseract-inspired test images
// ---------------------------------------------------------------------------

/// Canonical mixed-content test image, inspired by Tesseract's `phototest.tif`.
/// Contains a heading, a paragraph, a numbered list, and a short digit line —
/// the kind of mixed layout Tesseract uses for its basic regression test.
fn generate_phototest(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(700, 400);

    // Heading (sans-serif, larger)
    draw_line_cat(&mut img, "Test Page", 30, 20, 32.0, fonts, FontCategory::SansSerif, 0);

    // Paragraph (serif, body size)
    let para = "This is a test page for OCR. It contains a paragraph of text \
                that mixes upper and lower case letters, digits like 42, and \
                punctuation marks such as commas, periods, and semicolons; \
                the quick brown fox jumps over the lazy dog.";
    let words: Vec<&str> = para.split_whitespace().collect();
    let font_size = 22.0f32;
    let max_width = 640u32;
    let line_height = 30u32;
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in &words {
        let trial = if current.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current, word)
        };
        let w = if let Some(font) = font_by_category(fonts, FontCategory::Serif) {
            measure_width(&trial, font, font_size)
        } else {
            trial.len() as u32 * 12
        };
        if w <= max_width {
            current = trial;
        } else {
            if !current.is_empty() {
                lines.push(current.clone());
            }
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    let mut y = 65u32;
    for line in &lines {
        draw_line_cat(&mut img, line, 30, y, font_size, fonts, FontCategory::Serif, 0);
        y += line_height;
    }

    // Numbered list (monospace)
    y += 10;
    let list = ["1. First item with text", "2. Second item here", "3. Third item end"];
    for line in list {
        draw_line_cat(&mut img, line, 30, y, 20.0, fonts, FontCategory::Monospace, 0);
        y += 28;
    }

    // Digit line (monospace, larger)
    y += 10;
    draw_line_cat(&mut img, "136 256 410 432 540", 30, y, 28.0, fonts, FontCategory::Monospace, 0);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

/// Multi-language European text with accented characters, inspired by
/// Tesseract's `eurotext.tif`. Tests extended-Latin glyph coverage.
fn generate_eurotext(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_rgb(700, 300);
    let lines = [
        ("English: The quick brown fox", FontCategory::SansSerif),
        ("Francais: cafe creme naive eleve", FontCategory::Serif),
        ("Deutsch: uber gross schon Munchen", FontCategory::Serif),
        ("Espanol: nino pina agudo senor", FontCategory::SansSerif),
        ("Italiano: perche citta uovo", FontCategory::Serif),
        ("Portugues: avenida coracao nacao", FontCategory::SansSerif),
    ];
    let mut y = 20;
    for (line, cat) in lines.iter() {
        if let Some(font) = font_by_category(fonts, *cat) {
            render_line_rgb(&mut img, line, 20, y, 24.0, font, Rgb([0, 0, 0]));
        }
        y += 42;
    }
    let gray = DynamicImage::ImageRgb8(img).to_luma8();
    gray.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

/// Short digit-only lines, mirroring Tesseract's training pages
/// (136.tif, 256.tif, 410.tif, etc.). Tests digit recognition in isolation.
fn generate_training_digits(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(400, 300);
    let lines = ["136", "256", "410", "432", "540", "692", "779", "793"];
    let mut y = 20;
    for (i, line) in lines.iter().enumerate() {
        draw_line_cat(&mut img, line, 30, y, 28.0, fonts, FontCategory::Monospace, i % 2);
        y += 32;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// GOCR-inspired test images
// ---------------------------------------------------------------------------

/// Ligature-prone text in serif fonts. GOCR identifies connected characters
/// (fi, ff, fl, ffi, ffl) in italic fonts as the most common recognition
/// error source. This image deliberately includes ligature-heavy words to
/// stress-test character segmentation.
fn generate_ligatures(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 250);
    let lines = [
        "ff fi fl ffi ffl",
        "field office fliffon",
        "afford effort affable",
        "difficult fluent suffix",
    ];
    let mut y = 20;
    for (i, line) in lines.iter().enumerate() {
        draw_line_cat(&mut img, line, 20, y, 28.0, fonts, FontCategory::Serif, i % 2);
        y += 50;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

/// Thin-font text, inspired by GOCR's `meraji1` test (marked "thinn").
/// Thin strokes are harder to binarize and segment — tests the thresholding
/// and character-detection pipeline at low stroke width.
fn generate_thin_font(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 150);
    draw_line_cat(&mut img, "Thin font text sample 12345", 20, 30, 18.0, fonts, FontCategory::Serif, 1);
    draw_line_cat(&mut img, "The quick brown fox jumps", 20, 70, 18.0, fonts, FontCategory::Serif, 1);
    draw_line_cat(&mut img, "over the lazy dog 67890", 20, 110, 18.0, fonts, FontCategory::Serif, 1);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

/// Framed document with a border, inspired by GOCR's `paraguay1` (marked
/// "frame"). Tests that the layout pipeline doesn't mistake the border for
/// text or let it interfere with text region detection.
fn generate_framed_document(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 300);
    let margin = 15u32;
    let thickness = 2u32;
    for t in 0..thickness {
        for x in margin..(img.width() - margin) {
            img.put_pixel(x, margin + t, Luma([0]));
            img.put_pixel(x, img.height() - margin - 1 - t, Luma([0]));
        }
        for y in margin..(img.height() - margin) {
            img.put_pixel(margin + t, y, Luma([0]));
            img.put_pixel(img.width() - margin - 1 - t, y, Luma([0]));
        }
    }
    draw_line_cat(&mut img, "Framed Document", 40, 40, 28.0, fonts, FontCategory::SansSerif, 0);
    draw_line_cat(&mut img, "Line one with some text", 40, 80, 22.0, fonts, FontCategory::Serif, 0);
    draw_line_cat(&mut img, "Line two with numbers 42", 40, 115, 22.0, fonts, FontCategory::Serif, 0);
    draw_line_cat(&mut img, "Line three ends here.", 40, 150, 22.0, fonts, FontCategory::Serif, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

/// Snowy/noisy image with heavy salt-and-pepper noise, inspired by GOCR's
/// `g300b2`/`g300b3` (marked "snowy"). Tests the noise-reduction and
/// binarization pipeline under heavy noise.
fn generate_snowy(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(600, 200);
    let mut rng = rand::thread_rng();
    use rand::Rng;
    for pixel in img.pixels_mut() {
        let r = rng.gen_range(0..100);
        if r < 15 {
            *pixel = Luma([0]);
        } else if r < 30 {
            *pixel = Luma([128]);
        } else {
            *pixel = Luma([255]);
        }
    }
    draw_line_cat(&mut img, "Snowy text recognition test", 20, 40, 28.0, fonts, FontCategory::SansSerif, 0);
    draw_line_cat(&mut img, "Line 2: 1234567890", 20, 90, 24.0, fonts, FontCategory::Monospace, 0);
    draw_line_cat(&mut img, "Line 3: end of test", 20, 130, 24.0, fonts, FontCategory::Serif, 0);
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

/// 200 DPI variant of a standard text block, inspired by GOCR's `font2.pbm`
/// (200dpi). GOCR tests at both 200 and 300 DPI — lower DPI produces smaller
/// glyphs that stress the super-resolution and character-detection pipeline.
fn generate_200dpi(fonts: &[LoadedFont], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = white_gray(500, 200);
    let lines = [
        "This is text at 200 DPI",
        "Lower resolution glyphs",
        "Numbers: 0123456789",
    ];
    let mut y = 25;
    for (i, line) in lines.iter().enumerate() {
        draw_line_cat(&mut img, line, 20, y, 24.0, fonts, FontCategory::Serif, i % 2);
        y += 45;
    }
    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}
