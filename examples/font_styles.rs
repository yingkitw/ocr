//! Generate and recognize test images with different font styles
//!
//! This example creates synthetic images simulating bold, italic, and
//! normal text, then runs OCR and reports font attribute detection.
//!
//! Run: cargo run --example font_styles

use image::{Rgb, RgbImage};
use ocr::api::Ocr;
use ocr::core::config::OcrConfig;

type Drawer = fn(&str) -> RgbImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("examples/output")?;

    let mut config = OcrConfig::default();
    config.recognition.confidence_threshold = 0.2;
    config.recognition.enable_font_attribute_detection = true;
    let ocr = Ocr::with_config(config)?;

    let styles: Vec<(&str, &str, Drawer)> = vec![
        ("normal", "The quick brown fox", draw_normal),
        ("bold", "THE QUICK BROWN FOX", draw_bold),
        ("italic", "The quick brown fox", draw_italic),
        ("large", "HELLO WORLD", draw_large),
        ("small", "tiny text", draw_small),
        ("monospace", "ABC DEF GHI", draw_monospace),
    ];

    println!("Font Style Recognition Demo");
    println!("==========================");

    for (name, text, drawer) in styles {
        let path = format!("examples/output/{}.png", name);
        let img = drawer(text);
        img.save(&path)?;

        let gray = image::DynamicImage::ImageRgb8(img).to_luma8();
        let (w, h) = gray.dimensions();
        let raw: Vec<u8> = gray.pixels().map(|p| p[0]).collect();

        let rt = tokio::runtime::Runtime::new()?;
        let result = rt.block_on(async {
            ocr.initialize().await.ok();
            ocr.recognize_text(&raw, w, h).await
        })?;

        println!("\n[{}] text: '{}'", name, text);
        println!("  OCR result: '{}'", result.text);
        println!("  Confidence: {:.1}%", result.confidence * 100.0);

        for line in &result.lines {
            for word in &line.words {
                let p = &word.properties;
                print!(
                    "  Word: '{}' bold={} italic={} mono={}",
                    word.text, p.is_bold, p.is_italic, p.is_monospace
                );
                println!();
            }
        }
    }

    println!("\nImages saved to examples/output/");
    Ok(())
}

fn draw_normal(text: &str) -> RgbImage {
    let mut img = RgbImage::new(400, 80);
    fill_white(&mut img);
    draw_text(&mut img, text, 20, 30, 24, false, false);
    img
}

fn draw_bold(text: &str) -> RgbImage {
    let mut img = RgbImage::new(400, 80);
    fill_white(&mut img);
    draw_text(&mut img, text, 20, 30, 24, true, false);
    img
}

fn draw_italic(text: &str) -> RgbImage {
    let mut img = RgbImage::new(400, 80);
    fill_white(&mut img);
    draw_text(&mut img, text, 20, 30, 24, false, true);
    img
}

fn draw_large(text: &str) -> RgbImage {
    let mut img = RgbImage::new(600, 120);
    fill_white(&mut img);
    draw_text(&mut img, text, 20, 40, 48, false, false);
    img
}

fn draw_small(text: &str) -> RgbImage {
    let mut img = RgbImage::new(300, 60);
    fill_white(&mut img);
    draw_text(&mut img, text, 10, 20, 14, false, false);
    img
}

fn draw_monospace(text: &str) -> RgbImage {
    let mut img = RgbImage::new(400, 80);
    fill_white(&mut img);
    // Fixed character width for monospace look
    let char_w = 18;
    let size = 24;
    for (i, ch) in text.chars().enumerate() {
        let x = 20 + (i as u32 * char_w);
        draw_char(&mut img, ch, x, 30, size, false, false, Rgb([0, 0, 0]));
    }
    img
}

fn fill_white(img: &mut RgbImage) {
    for p in img.pixels_mut() {
        *p = Rgb([255, 255, 255]);
    }
}

fn draw_text(img: &mut RgbImage, text: &str, x: u32, y: u32, size: u32, bold: bool, italic: bool) {
    let mut cx = x;
    for ch in text.chars() {
        draw_char(img, ch, cx, y, size, bold, italic, Rgb([0, 0, 0]));
        cx += size / 2 + 2;
    }
}

fn draw_char(
    img: &mut RgbImage,
    ch: char,
    x: u32,
    y: u32,
    size: u32,
    bold: bool,
    italic: bool,
    color: Rgb<u8>,
) {
    let pattern = glyph_pattern(ch);
    let scale = (size / 7).max(1);
    let thickness: u32 = if bold { scale + 1 } else { scale };
    let slant: i32 = if italic { (scale as i32) / 2 } else { 0 };

    for gy in 0..7u32 {
        for gx in 0..5u32 {
            let bit = pattern[gy as usize] >> (4 - gx) & 1;
            if bit == 1 {
                let base_x = x + gx * scale;
                let base_y = y + gy * scale;
                let offset_x = (gy as i32 * slant) / 7;
                for dy in 0..thickness {
                    for dx in 0..thickness {
                        let px = (base_x as i32 + dx as i32 + offset_x) as u32;
                        let py = base_y + dy;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
    }
}

fn glyph_pattern(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '?' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
        _ => [0b11111, 0b11111, 0b11111, 0b11111, 0b11111, 0b11111, 0b11111],
    }
}
