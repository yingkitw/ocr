//! Generate test images with varying complexity for OCR testing

use image::{Rgb, RgbImage};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure test_images directory exists
    std::fs::create_dir_all("test_images")?;

    // Try to load a font - use a simple approach without external font file
    // We'll use imageproc's built-in text rendering which works without fonts
    println!("Generating test images...");

    // 1. Simple single line text
    generate_simple_text("test_images/simple_text.png")?;

    // 2. Multi-line text
    generate_multiline_text("test_images/multiline_text.png")?;

    // 3. Mixed case text
    generate_mixed_case("test_images/mixed_case.png")?;

    // 4. Numbers and special characters
    generate_numbers_special("test_images/numbers_special.png")?;

    // 5. Small text (low resolution)
    generate_small_text("test_images/small_text.png")?;

    // 6. Large text (high resolution)
    generate_large_text("test_images/large_text.png")?;

    // 7. Dark background
    generate_dark_background("test_images/dark_background.png")?;

    // 8. Complex layout (columns)
    generate_column_layout("test_images/column_layout.png")?;

    // 9. Noisy background
    generate_noisy_background("test_images/noisy_background.png")?;

    // 10. Handwritten style (simulated with varying sizes)
    generate_handwritten_style("test_images/handwritten_style.png")?;

    // 11. Dense text (paragraph)
    generate_dense_text("test_images/dense_text.png")?;

    // 12. Mixed languages
    generate_mixed_languages("test_images/mixed_languages.png")?;

    println!("✓ All test images generated successfully!");
    Ok(())
}

fn create_white_image(width: u32, height: u32) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }
    img
}

fn generate_simple_text(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(400, 100);

    // Draw simple text using pixel manipulation
    draw_text_simple(&mut img, "Hello World", 20, 30, 32);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_multiline_text(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(500, 200);

    let lines = vec![
        "Line 1: The quick brown fox",
        "Line 2: jumps over the lazy dog",
        "Line 3: 1234567890",
    ];

    let mut y = 30;
    for line in lines {
        draw_text_simple(&mut img, line, 20, y, 24);
        y += 40;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_mixed_case(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(600, 150);

    let text = "MiXeD cAsE: AbCdEfG 123 xYz";
    draw_text_simple(&mut img, text, 20, 50, 28);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_numbers_special(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(500, 150);

    let lines = vec![
        "Numbers: 0123456789",
        "Special: !@#$%^&*()",
        "Math: +-*/=<>?",
    ];

    let mut y = 30;
    for line in lines {
        draw_text_simple(&mut img, line, 20, y, 22);
        y += 35;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_small_text(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(300, 80);

    draw_text_simple(&mut img, "Small text: 8pt font size", 10, 30, 14);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_large_text(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(800, 200);

    draw_text_simple(&mut img, "LARGE TEXT 72PT", 20, 80, 64);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_dark_background(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = RgbImage::new(500, 150);
    // Dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([30, 30, 30]);
    }

    draw_text_simple_white(&mut img, "White text on dark", 20, 50, 32);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_column_layout(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(800, 400);

    // Left column
    let left_text = vec!["Column 1:", "Item A", "Item B", "Item C"];
    let mut y = 30;
    for line in left_text {
        draw_text_simple(&mut img, line, 50, y, 20);
        y += 35;
    }

    // Right column
    let right_text = vec!["Column 2:", "Data X", "Data Y", "Data Z"];
    y = 30;
    for line in right_text {
        draw_text_simple(&mut img, line, 450, y, 20);
        y += 35;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_noisy_background(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = RgbImage::new(500, 150);

    // Create noisy background
    let mut rng = rand::thread_rng();
    for pixel in img.pixels_mut() {
        let noise = rng.gen_range(200..255);
        *pixel = Rgb([noise, noise, noise]);
    }

    draw_text_simple(&mut img, "Text on noisy background", 20, 50, 32);

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_handwritten_style(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(600, 200);

    // Simulate handwritten style with varying sizes and positions
    let text = "Handwritten style text";
    let chars: Vec<char> = text.chars().collect();
    let mut x = 20;
    let base_y = 80;
    let base_size = 28;

    let mut rng = rand::thread_rng();

    for ch in chars {
        let size_variation = rng.gen_range(-3..3);
        let y_variation = rng.gen_range(-5..5);
        let x_variation = rng.gen_range(-2..2);

        let size = (base_size as i32 + size_variation).max(12) as u32;
        draw_char_simple(
            &mut img,
            ch,
            (x as i32 + x_variation) as u32,
            (base_y as i32 + y_variation) as u32,
            size,
        );

        // Estimate character width
        x += size as i32 + 2;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_dense_text(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(600, 300);

    let paragraph = "This is a dense paragraph of text that contains multiple sentences. \
                     It tests how well the OCR system handles continuous text without breaks. \
                     The text includes various punctuation marks, numbers like 123, and mixed \
                     case letters. This simulates real-world document scanning scenarios.";

    let words: Vec<&str> = paragraph.split_whitespace().collect();
    let mut x = 20;
    let mut y = 30;
    let line_height = 25;
    let max_width = 560;

    for word in words {
        let word_width = word.len() as u32 * 8; // Rough estimate
        if x + word_width > max_width {
            x = 20;
            y += line_height;
        }
        draw_text_simple(&mut img, word, x, y, 18);
        x += word_width + 8; // Space between words
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

fn generate_mixed_languages(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = create_white_image(600, 250);

    let lines = vec![
        "English: Hello World",
        "中文: 你好世界",
        "日本語: こんにちは",
        "한국어: 안녕하세요",
        "Français: Bonjour",
        "Deutsch: Guten Tag",
    ];

    let mut y = 30;
    for line in lines {
        draw_text_simple(&mut img, line, 20, y, 22);
        y += 35;
    }

    img.save(path)?;
    println!("  ✓ {}", path);
    Ok(())
}

// Simple text drawing using pixel manipulation (fallback when fonts aren't available)
fn draw_text_simple(img: &mut RgbImage, text: &str, x: u32, y: u32, size: u32) {
    draw_text_simple_color(img, text, x, y, size, Rgb([0, 0, 0]));
}

fn draw_text_simple_white(img: &mut RgbImage, text: &str, x: u32, y: u32, size: u32) {
    draw_text_simple_color(img, text, x, y, size, Rgb([255, 255, 255]));
}

fn draw_text_simple_color(
    img: &mut RgbImage,
    text: &str,
    x: u32,
    y: u32,
    size: u32,
    color: Rgb<u8>,
) {
    let char_width = size / 2;
    let _char_height = size;
    let mut current_x = x;

    for ch in text.chars() {
        draw_char_simple_color(img, ch, current_x, y, size, color);
        current_x += char_width;
    }
}

fn draw_char_simple(img: &mut RgbImage, ch: char, x: u32, y: u32, size: u32) {
    draw_char_simple_color(img, ch, x, y, size, Rgb([0, 0, 0]));
}

fn draw_char_simple_color(img: &mut RgbImage, ch: char, x: u32, y: u32, size: u32, color: Rgb<u8>) {
    let char_width = (size / 2).max(6);
    let char_height = size;
    let _thickness = (size / 10).max(2);

    // Simple character rendering - draw a box pattern for each character
    // This is a simplified approach; in production you'd use proper font rendering
    for dy in 0..char_height.min(img.height().saturating_sub(y)) {
        for dx in 0..char_width.min(img.width().saturating_sub(x)) {
            // Create a simple pattern based on character code
            let pattern = (ch as u32 + dx + dy) % 3;
            if pattern == 0 && dx < char_width && dy < char_height {
                if x + dx < img.width() && y + dy < img.height() {
                    img.put_pixel(x + dx, y + dy, color);
                }
            }
        }
    }
}
