//! OCR with system TrueType fonts
//!
//! Attempts to load a system TTF font and renders text with it.
//! Falls back to the built-in bitmap renderer if no font is found.
//!
//! macOS:  /System/Library/Fonts/Helvetica.ttc
//! Linux:  /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
//!
//! Run: cargo run --example system_font

use image::{Rgb, RgbImage};
use ocr::api::Ocr;
use ocr::core::config::OcrConfig;

#[cfg(target_os = "macos")]
const SYSTEM_FONT_PATHS: &[&str] = &[
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
];

#[cfg(target_os = "linux")]
const SYSTEM_FONT_PATHS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
];

#[cfg(target_os = "windows")]
const SYSTEM_FONT_PATHS: &[&str] = &[
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\calibri.ttf",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("examples/output")?;

    let text = "Hello World 123";
    let size = 48u32;

    let (img, source) = match load_system_font() {
        Some(font_data) => {
            let img = render_with_ttf(&font_data, text, size)?;
            (img, "system TTF")
        }
        None => {
            let img = render_bitmap(text, size);
            (img, "built-in bitmap")
        }
    };

    let path = "examples/output/system_font.png";
    img.save(path)?;

    let config = OcrConfig::default();
    let ocr = Ocr::with_config(config)?;

    let gray = image::DynamicImage::ImageRgb8(img).to_luma8();
    let (w, h) = gray.dimensions();
    let raw: Vec<u8> = gray.pixels().map(|p| p[0]).collect();

    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(async {
        ocr.initialize().await.ok();
        ocr.recognize_text(&raw, w, h).await
    })?;

    println!("System Font OCR Demo");
    println!("====================");
    println!("Font source: {}", source);
    println!("Text:        '{}'", text);
    println!("OCR result:  '{}'", result.text);
    println!("Confidence:  {:.1}%", result.confidence * 100.0);
    println!("Image saved: {}", path);

    Ok(())
}

fn load_system_font() -> Option<Vec<u8>> {
    for path in SYSTEM_FONT_PATHS {
        if std::path::Path::new(path).exists() {
            if let Ok(data) = std::fs::read(path) {
                println!("Loaded font: {}", path);
                return Some(data);
            }
        }
    }
    println!("No system font found, falling back to bitmap renderer");
    None
}

fn render_with_ttf(font_data: &[u8], text: &str, size: u32) -> Result<RgbImage, Box<dyn std::error::Error>> {
    use imageproc::drawing::draw_text_mut;
    use rusttype::{Font, Scale};

    let font = Font::try_from_bytes(font_data)
        .ok_or("Failed to parse font")?;

    let scale = Scale::uniform(size as f32);
    let (img_w, img_h) = (600, 150);
    let mut img = RgbImage::new(img_w, img_h);
    for p in img.pixels_mut() {
        *p = Rgb([255, 255, 255]);
    }

    draw_text_mut(
        &mut img,
        Rgb([0, 0, 0]),
        20,
        40,
        scale,
        &font,
        text,
    );

    Ok(img)
}

fn render_bitmap(text: &str, size: u32) -> RgbImage {
    let scale = (size / 7).max(1);
    let width = text.len() as u32 * (5 * scale + 4) + 40;
    let height = 7 * scale + 60;
    let mut img = RgbImage::new(width, height);
    for p in img.pixels_mut() {
        *p = Rgb([255, 255, 255]);
    }

    let mut cx = 20u32;
    for ch in text.chars() {
        draw_bitmap_char(&mut img, ch, cx, 40, scale);
        cx += 5 * scale + 4;
    }
    img
}

fn draw_bitmap_char(img: &mut RgbImage, ch: char, x: u32, y: u32, scale: u32) {
    let pattern = glyph_pattern(ch);
    for gy in 0..7u32 {
        for gx in 0..5u32 {
            let bit = pattern[gy as usize] >> (4 - gx) & 1;
            if bit == 1 {
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x + gx * scale + dx;
                        let py = y + gy * scale + dy;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, Rgb([0, 0, 0]));
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
        ' ' => [0b00000; 7],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        _ => [0b11111; 7],
    }
}
