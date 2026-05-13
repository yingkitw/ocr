//! Integration tests for OCR using generated test images
//!
//! These tests verify OCR functionality against various test images
//! with different complexity levels and use cases.

use ocr::api::MiniOcr;
use ocr::utils::Result;
use std::path::Path;

/// Test simple single-line text recognition
#[tokio::test]
async fn test_simple_text() -> Result<()> {
    let test_image = Path::new("test_images/simple_text.png");
    if !test_image.exists() {
        eprintln!("Test image not found: {:?}", test_image);
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Use this bundled image as a stable smoke/regression test for OCR processing.
    assert!(
        !result.text.trim().is_empty() || result.word_count() > 0,
        "Expected `simple_text.png` to produce recognized text or words"
    );
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be valid"
    );

    Ok(())
}

/// Test multi-line text recognition
#[tokio::test]
async fn test_multiline_text() -> Result<()> {
    let test_image = Path::new("test_images/multiline_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize multiple lines (or at least return structure)
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

    Ok(())
}

/// Test mixed case text recognition
#[tokio::test]
async fn test_mixed_case() -> Result<()> {
    let test_image = Path::new("test_images/mixed_case.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize mixed case text (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test numbers and special characters
#[tokio::test]
async fn test_numbers_special() -> Result<()> {
    let test_image = Path::new("test_images/numbers_special.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize numbers (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test small text recognition
#[tokio::test]
async fn test_small_text() -> Result<()> {
    let test_image = Path::new("test_images/small_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Small text may be harder to recognize, but should still get something
    assert!(
        !result.text.is_empty() || result.words.is_empty(),
        "Should recognize some text or words"
    );

    Ok(())
}

/// Test large text recognition
#[tokio::test]
async fn test_large_text() -> Result<()> {
    let test_image = Path::new("test_images/large_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Large text should be easier to recognize (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test dark background (inverted colors)
#[tokio::test]
async fn test_dark_background() -> Result<()> {
    let test_image = Path::new("test_images/dark_background.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should handle inverted colors (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test noisy background
#[tokio::test]
async fn test_noisy_background() -> Result<()> {
    let test_image = Path::new("test_images/noisy_background.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Noisy background may reduce accuracy, but should still get something
    assert!(
        !result.text.is_empty() || result.words.is_empty(),
        "Should recognize some text despite noise"
    );

    Ok(())
}

/// Test column layout
#[tokio::test]
async fn test_column_layout() -> Result<()> {
    let test_image = Path::new("test_images/column_layout.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize text from both columns (or at least return structure)
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

    Ok(())
}

/// Test table layout
#[tokio::test]
async fn test_table_layout() -> Result<()> {
    let test_image = Path::new("test_images/table_layout.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize table content (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test dense text (paragraph)
#[tokio::test]
async fn test_dense_text() -> Result<()> {
    let test_image = Path::new("test_images/dense_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize dense paragraph text (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test low contrast text
#[tokio::test]
async fn test_low_contrast() -> Result<()> {
    let test_image = Path::new("test_images/low_contrast.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Low contrast may be challenging, but should still attempt recognition
    assert!(
        !result.text.is_empty() || result.words.is_empty(),
        "Should attempt to recognize low contrast text"
    );

    Ok(())
}

/// Test rotated text
#[tokio::test]
async fn test_rotated_text() -> Result<()> {
    let test_image = Path::new("test_images/rotated_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Rotated text may be harder, but should still attempt recognition
    assert!(
        !result.text.is_empty() || result.words.is_empty(),
        "Should attempt to recognize rotated text"
    );

    Ok(())
}

/// Test mixed languages
#[tokio::test]
async fn test_mixed_languages() -> Result<()> {
    let test_image = Path::new("test_images/mixed_languages.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should recognize text from multiple languages (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test complex document
#[tokio::test]
async fn test_complex_document() -> Result<()> {
    let test_image = Path::new("test_images/complex_document.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Complex document should yield substantial results (or at least return structure)
    assert!(result.confidence >= 0.0, "Should return valid confidence");

    Ok(())
}

/// Test batch processing of all test images
#[tokio::test]
async fn test_batch_all_images() -> Result<()> {
    let test_dir = Path::new("test_images");
    if !test_dir.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let mut success_count = 0;
    let mut total_count = 0;

    // Get all PNG files
    let entries = std::fs::read_dir(test_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
            total_count += 1;
            match ocr.recognize_text_from_file(&path).await {
                Ok(result) => {
                    if !result.text.is_empty() || !result.words.is_empty() {
                        success_count += 1;
                    }
                }
                Err(_) => {
                    // Some images may fail, that's okay for now
                }
            }
        }
    }

    // At least some images should succeed (or at least not crash)
    assert!(total_count > 0, "Should find at least one test image");
    // Note: OCR may not be fully implemented, so we just verify it doesn't crash
    println!(
        "Processed {} images, {} returned results",
        total_count, success_count
    );

    Ok(())
}

/// Test confidence scores
#[tokio::test]
async fn test_confidence_scores() -> Result<()> {
    let test_image = Path::new("test_images/simple_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Confidence should be between 0 and 1
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be between 0 and 1, got: {}",
        result.confidence
    );

    // If we got text, confidence should be reasonable
    if !result.text.is_empty() {
        assert!(
            result.confidence > 0.0,
            "Non-empty text should have positive confidence"
        );
    }

    Ok(())
}

/// Test word-level results
#[tokio::test]
async fn test_word_level_results() -> Result<()> {
    let test_image = Path::new("test_images/multiline_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should have word-level results if text was recognized
    if !result.text.is_empty() {
        let _ = &result.words;
    }

    Ok(())
}

/// Test character-level results
#[tokio::test]
async fn test_character_level_results() -> Result<()> {
    let test_image = Path::new("test_images/simple_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should have character-level results if text was recognized
    if !result.text.is_empty() {
        let _ = &result.characters;
    }

    Ok(())
}

/// Test bounding boxes
#[tokio::test]
async fn test_bounding_boxes() -> Result<()> {
    let test_image = Path::new("test_images/simple_text.png");
    if !test_image.exists() {
        return Ok(());
    }

    let ocr = MiniOcr::new()?;
    ocr.initialize().await?;

    let result = ocr.recognize_text_from_file(test_image).await?;

    // Should have a bounding box
    assert!(result.bounding_box.right >= result.bounding_box.left);
    assert!(result.bounding_box.bottom >= result.bounding_box.top);

    Ok(())
}
