//! Benchmark tests for OCR performance using generated synthetic images
//!
//! No external test files required. All images are generated programmatically.
//! Run with: cargo test --test test_images_benchmark -- --nocapture

use ocr::api::Ocr;
use ocr::utils::{Result, SimdImageOps};
use std::time::Instant;

fn generate_synthetic_text_image(text: &str, font_size: f32) -> Vec<u8> {
    use image::{GrayImage, Luma};
    let width = 800u32;
    let height = 200u32;
    let mut img = GrayImage::from_pixel(width, height, Luma([255u8]));

    let chars: Vec<char> = text.chars().collect();
    let char_w = (width as f32 / chars.len().max(1) as f32).min(font_size * 1.2);
    let x_start = 20.0;
    let y_center = height as f32 / 2.0;

    for (i, ch) in chars.iter().enumerate() {
        let x = x_start + i as f32 * char_w;
        let glyph_idx = *ch as u32;
        let _scale = font_size / 14.0;
        let gx = (x as u32).min(width - 1);
        let gy = (y_center as u32).min(height - 1);

        let glyph_pattern: u32 = glyph_idx.wrapping_mul(0x45d9f3b).wrapping_add(0x1b873593);
        for dy in 0..((font_size * 0.7) as u32) {
            for dx in 0..((char_w * 0.6) as u32) {
                let px = gx.saturating_add(dx);
                let py = gy
                    .saturating_sub((font_size * 0.35) as u32)
                    .saturating_add(dy);
                if px < width && py < height {
                    let bit = ((glyph_pattern >> ((dx + dy * 3) % 32)) & 1) == 1;
                    let edge = dx == 0
                        || dy == 0
                        || dx >= (char_w * 0.6) as u32 - 1
                        || dy >= (font_size * 0.7) as u32 - 1;
                    if bit || edge {
                        img.put_pixel(px, py, Luma([0u8]));
                    }
                }
            }
        }
    }

    let mut buf = Vec::new();
    let dynamic = image::DynamicImage::ImageLuma8(img);
    dynamic
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .ok();
    buf
}

fn generate_noisy_image(width: u32, height: u32) -> Vec<u8> {
    use image::{GrayImage, Luma};
    let mut img = GrayImage::from_pixel(width, height, Luma([128u8]));
    for y in 0..height {
        for x in 0..width {
            let noise = ((x
                .wrapping_mul(y.wrapping_add(1))
                .wrapping_mul(1103515245)
                .wrapping_add(12345)
                >> 16)
                & 0xFF) as u8;
            img.put_pixel(x, y, Luma([noise]));
        }
    }
    let mut buf = Vec::new();
    let dynamic = image::DynamicImage::ImageLuma8(img);
    dynamic
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .ok();
    buf
}

#[tokio::test]
async fn benchmark_synthetic_text_recognition() -> Result<()> {
    let png_data = generate_synthetic_text_image("Hello World Test OCR Benchmark", 24.0);
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::write(tmp.path(), &png_data)?;

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let start = Instant::now();
    let result = ocr.recognize_text_from_file(tmp.path()).await?;
    let duration = start.elapsed();

    println!("Synthetic text recognition:");
    println!("  Time: {:?}", duration);
    println!("  Text: {}", result.text);
    println!("  Confidence: {:.2}%", result.confidence * 100.0);

    assert!(
        duration.as_millis() < 10000,
        "Should complete in reasonable time"
    );
    Ok(())
}

#[tokio::test]
async fn benchmark_noisy_image_recognition() -> Result<()> {
    let png_data = generate_noisy_image(400, 200);
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::write(tmp.path(), &png_data)?;

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let start = Instant::now();
    let result = ocr.recognize_text_from_file(tmp.path()).await?;
    let duration = start.elapsed();

    println!("Noisy image recognition:");
    println!("  Time: {:?}", duration);
    println!("  Text length: {}", result.text.len());

    assert!(
        duration.as_millis() < 10000,
        "Should complete in reasonable time"
    );
    Ok(())
}

#[test]
fn benchmark_simd_contrast_adjust() {
    let size = 1024 * 1024;
    let pixels: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();
    let result = SimdImageOps::contrast_adjust(&pixels, 1.5);
    let duration = start.elapsed();

    assert_eq!(result.len(), size);
    println!("SIMD contrast_adjust ({} pixels): {:?}", size, duration);
}

#[test]
fn benchmark_simd_threshold() {
    let size = 1024 * 1024;
    let pixels: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();
    let result = SimdImageOps::threshold(&pixels, 128);
    let duration = start.elapsed();

    assert_eq!(result.len(), size);
    for &p in &result {
        assert!(p == 0 || p == 255, "Threshold output must be binary");
    }
    println!("SIMD threshold ({} pixels): {:?}", size, duration);
}

#[test]
fn test_simd_contrast_consistency() {
    let pixels: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let result = SimdImageOps::contrast_adjust(&pixels, 1.0);
    assert_eq!(result.len(), pixels.len());
    for (i, (&p, &r)) in pixels.iter().zip(result.iter()).enumerate() {
        assert!(
            (p as i16 - r as i16).abs() <= 1,
            "Pixel {} differs: input={}, output={}",
            i,
            p,
            r
        );
    }
}

#[test]
fn test_simd_threshold_consistency() {
    let pixels: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let result = SimdImageOps::threshold(&pixels, 128);
    assert_eq!(result.len(), pixels.len());
    for (i, (&p, &r)) in pixels.iter().zip(result.iter()).enumerate() {
        let expected = if p >= 128 { 255 } else { 0 };
        assert_eq!(
            r, expected,
            "Pixel {}: input={}, expected={}, got={}",
            i, p, expected, r
        );
    }
}

#[test]
fn test_simd_projections() {
    let width = 64usize;
    let height = 32usize;
    let data: Vec<u8> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let y = i / width;
            if x < width / 2 && y < height / 2 {
                0u8
            } else {
                255u8
            }
        })
        .collect();

    let (h_proj, v_proj) = SimdImageOps::compute_projections(&data, width, height);
    assert_eq!(h_proj.len(), height);
    assert_eq!(v_proj.len(), width);

    for y in 0..height {
        if y < height / 2 {
            assert!(h_proj[y] > 0, "Row {} should have dark pixels", y);
        } else {
            assert_eq!(h_proj[y], 0, "Row {} should have no dark pixels", y);
        }
    }
    for x in 0..width {
        if x < width / 2 {
            assert!(v_proj[x] > 0, "Col {} should have dark pixels", x);
        } else {
            assert_eq!(v_proj[x], 0, "Col {} should have no dark pixels", x);
        }
    }
}

#[test]
fn test_simd_ops_basic() {
    use ocr::utils::SimdOps;
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let r = SimdOps::add_f32x8(a, b);
    assert_eq!(r, [9.0; 8]);
    let m = SimdOps::mul_f32x8(a, b);
    assert_eq!(m, [8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0]);
}

#[test]
fn test_profiler_basic() {
    use ocr::utils::Profiler;
    let mut profiler = Profiler::new();
    profiler.start_operation("test_op");
    std::thread::sleep(std::time::Duration::from_millis(5));
    profiler.stop_operation();

    let stats = profiler.get_stats("test_op");
    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert!(stats.count > 0);
    assert!(stats.average_ms() > 0.0);
}
