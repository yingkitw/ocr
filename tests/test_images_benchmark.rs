//! Benchmark tests for OCR performance with different test images
//!
//! These tests measure OCR performance and can be used to track
//! improvements over time.

use ocr::api::Ocr;
use ocr::utils::Result;
use std::path::Path;
use std::time::Instant;

/// Benchmark simple text recognition
#[tokio::test]
#[ignore] // Ignore by default, run with: cargo test -- --ignored
async fn benchmark_simple_text() -> Result<()> {
    let test_image = Path::new("test_images/simple_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let start = Instant::now();
    let result = ocr.recognize_text_from_file(test_image).await?;
    let duration = start.elapsed();

    println!("Simple text recognition:");
    println!("  Time: {:?}", duration);
    println!("  Text: {}", result.text);
    println!("  Confidence: {:.2}%", result.confidence * 100.0);

    // Simple text should be fast
    assert!(
        duration.as_millis() < 5000,
        "Should complete in reasonable time"
    );

    Ok(())
}

/// Benchmark dense text recognition
#[tokio::test]
#[ignore]
async fn benchmark_dense_text() -> Result<()> {
    let test_image = Path::new("test_images/dense_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let start = Instant::now();
    let result = ocr.recognize_text_from_file(test_image).await?;
    let duration = start.elapsed();

    println!("Dense text recognition:");
    println!("  Time: {:?}", duration);
    println!("  Text length: {}", result.text.len());
    println!("  Words: {}", result.words.len());
    println!("  Confidence: {:.2}%", result.confidence * 100.0);

    Ok(())
}

/// Benchmark complex document recognition
#[tokio::test]
#[ignore]
async fn benchmark_complex_document() -> Result<()> {
    let test_image = Path::new("test_images/complex_document.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let start = Instant::now();
    let result = ocr.recognize_text_from_file(test_image).await?;
    let duration = start.elapsed();

    println!("Complex document recognition:");
    println!("  Time: {:?}", duration);
    println!("  Text length: {}", result.text.len());
    println!("  Words: {}", result.words.len());
    println!("  Lines: {}", result.lines.len());
    println!("  Confidence: {:.2}%", result.confidence * 100.0);

    Ok(())
}

/// Benchmark all images
#[tokio::test]
#[ignore]
async fn benchmark_all_images() -> Result<()> {
    let test_dir = Path::new("test_images");
    if !test_dir.exists() {
        return Ok(());
    }

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let mut total_time = std::time::Duration::new(0, 0);
    let mut count = 0;

    let entries = std::fs::read_dir(test_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
            let start = Instant::now();
            match ocr.recognize_text_from_file(&path).await {
                Ok(result) => {
                    let duration = start.elapsed();
                    total_time += duration;
                    count += 1;
                    println!(
                        "{}: {:?} (confidence: {:.2}%)",
                        path.file_name().unwrap().to_string_lossy(),
                        duration,
                        result.confidence * 100.0
                    );
                }
                Err(e) => {
                    println!(
                        "{}: ERROR - {}",
                        path.file_name().unwrap().to_string_lossy(),
                        e
                    );
                }
            }
        }
    }

    if count > 0 {
        let avg_time = total_time / count;
        println!("\nAverage time per image: {:?}", avg_time);
        println!("Total images processed: {}", count);
    }

    Ok(())
}
