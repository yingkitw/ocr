//! Generate test images with text at different font sizes
//!
//! Creates a grid of images from 8px to 72px to test OCR accuracy
//! across scales. Also measures recognition speed per size.
//!
//! Run: cargo run --example font_sizes

use image::{Rgb, RgbImage};
use ocr::api::Ocr;
use ocr::core::config::OcrConfig;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("examples/output")?;

    let config = OcrConfig::default();
    let ocr = Ocr::with_config(config)?;

    let text = "ABCD 1234";
    let sizes = vec![8u32, 12, 16, 20, 24, 32, 48, 72];

    println!("Font Size OCR Benchmark");
    println!("======================");
    println!(
        "{:<8} {:<15} {:<12} {:<10}",
        "Size", "OCR Text", "Conf %", "Time ms"
    );
    println!("{}", "-".repeat(55));

    for size in sizes {
        let path = format!("examples/output/size_{}px.png", size);
        let img = draw_size_sample(text, size);
        img.save(&path)?;

        let gray = image::DynamicImage::ImageRgb8(img).to_luma8();
        let (w, h) = gray.dimensions();
        let raw: Vec<u8> = gray.pixels().map(|p| p[0]).collect();

        let rt = tokio::runtime::Runtime::new()?;
        let start = Instant::now();
        let result = rt.block_on(async {
            ocr.initialize().await.ok();
            ocr.recognize_text(&raw, w, h).await
        })?;
        let elapsed = start.elapsed();

        println!(
            "{:>3}px    {:<15} {:>6.1}%    {:>6.2}",
            size,
            result.text.replace('\n', " "),
            result.confidence * 100.0,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\nImages saved to examples/output/size_*px.png");
    Ok(())
}

fn draw_size_sample(text: &str, size: u32) -> RgbImage {
    let padding = size;
    let width = text.len() as u32 * (size / 2 + 4) + padding * 2;
    let height = size * 2 + padding * 2;
    let mut img = RgbImage::new(width.max(100), height.max(60));
    for p in img.pixels_mut() {
        *p = Rgb([255, 255, 255]);
    }

    let mut cx = padding;
    for ch in text.chars() {
        draw_char_scaled(&mut img, ch, cx, padding, size);
        cx += size / 2 + 4;
    }
    img
}

fn draw_char_scaled(img: &mut RgbImage, ch: char, x: u32, y: u32, size: u32) {
    let scale = (size / 7).max(1);
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
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        ' ' => [0b00000; 7],
        _ => [0b11111; 7],
    }
}
