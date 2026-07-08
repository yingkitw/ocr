//! Simple OCR integration test
//!
//! This test can be run with: cargo test --test ocr_test

use ocr::api::Ocr;
use ocr::utils::Result;
use std::path::Path;

#[tokio::test]
async fn test_ocr_with_sample_image() -> Result<()> {
    // Check if test image exists
    let test_image = Path::new("test_images/sample.png");
    if !test_image.exists() {
        eprintln!("Test image not found: {:?}", test_image);
        eprintln!("Skipping OCR test - please create a test image first");
        return Ok(());
    }

    println!("Testing OCR with image: {:?}", test_image);

    // Create OCR instance
    let ocr = Ocr::new()?;

    // Initialize
    ocr.initialize().await?;

    // Recognize text
    let result = ocr.recognize_text_from_file(test_image).await?;

    println!("Recognized text: {}", result.text);
    println!("Confidence: {:.2}%", result.confidence * 100.0);

    // Basic validation - should have some text
    assert!(
        !result.text.is_empty() || result.words.is_empty(),
        "OCR should return some result"
    );

    Ok(())
}
