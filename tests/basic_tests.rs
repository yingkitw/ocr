//! Basic tests for OCR

use ocr::api::Ocr;
use ocr::utils::Result;

#[tokio::test]
async fn test_ocr_creation() -> Result<()> {
    let ocr = Ocr::new()?;
    assert!(!ocr.is_initialized().await);
    Ok(())
}

#[tokio::test]
async fn test_ocr_initialization() -> Result<()> {
    let ocr = Ocr::new()?;
    ocr.initialize().await?;
    assert!(ocr.is_initialized().await);
    Ok(())
}

#[tokio::test]
async fn test_supported_languages() -> Result<()> {
    let ocr = Ocr::new()?;
    let languages = ocr.get_supported_languages();
    assert!(!languages.is_empty());
    assert!(languages.iter().any(|lang| lang == "en"));
    Ok(())
}

#[tokio::test]
async fn test_supported_image_formats() -> Result<()> {
    let ocr = Ocr::new()?;
    let formats = ocr.get_supported_image_formats();
    assert!(!formats.is_empty());
    assert!(formats.iter().any(|fmt| fmt == "png"));
    assert!(formats.iter().any(|fmt| fmt == "jpg"));
    Ok(())
}

#[tokio::test]
async fn test_metadata() -> Result<()> {
    let ocr = Ocr::new()?;
    let metadata = ocr.get_metadata();
    assert!(!metadata.name.is_empty());
    assert!(!metadata.version.is_empty());
    assert!(!metadata.description.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_configuration() -> Result<()> {
    let ocr = Ocr::new()?;
    let config = ocr.get_config();
    assert_eq!(config.recognition.language, "en");
    assert!(config.recognition.confidence_threshold > 0.0);
    assert!(config.recognition.confidence_threshold <= 1.0);
    Ok(())
}

#[tokio::test]
async fn test_statistics() -> Result<()> {
    let ocr = Ocr::new()?;
    let stats = ocr.get_statistics().await?;
    assert_eq!(stats.total_images_processed, 0);
    assert_eq!(stats.total_text_recognized, 0);
    assert_eq!(stats.total_processing_time_ms, 0);
    Ok(())
}

#[tokio::test]
async fn test_cache_operations() -> Result<()> {
    let ocr = Ocr::new()?;

    // Clear cache (should not error)
    ocr.clear_cache().await?;

    // Reset statistics (should not error)
    ocr.reset_statistics().await?;

    Ok(())
}

#[tokio::test]
async fn test_recognize_text_from_bytes() -> Result<()> {
    use image::{ImageBuffer, Luma};
    use std::io::Cursor;

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    // Create a simple 200x50 white image with two black rectangles
    let mut img = ImageBuffer::from_pixel(200, 50, Luma([255u8]));
    for y in 10..40 {
        for x in 10..20 {
            img.put_pixel(x, y, Luma([0u8]));
        }
    }
    for y in 10..40 {
        for x in 25..35 {
            img.put_pixel(x, y, Luma([0u8]));
        }
    }

    // Encode as PNG bytes
    let mut png_bytes = Cursor::new(Vec::new());
    img.write_to(&mut png_bytes, image::ImageFormat::Png)
        .map_err(|e| ocr::utils::OcrError::Internal(e.to_string()))?;
    let png_bytes = png_bytes.into_inner();

    // recognize_text_from_bytes should accept encoded image bytes and return a result
    let result = ocr.recognize_text_from_bytes(&png_bytes).await?;
    // The method should complete without error; the image has no real text,
    // so we just verify the result object is returned
    let _ = result.text;

    Ok(())
}

#[tokio::test]
async fn test_recognize_text_from_bytes_multi_page_png() -> Result<()> {
    use image::{ImageBuffer, Luma};
    use std::io::Cursor;

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    // Create a simple PNG image (single-page, should return 1-element vec)
    let img = ImageBuffer::from_pixel(200, 50, Luma([255u8]));
    let mut png_bytes = Cursor::new(Vec::new());
    img.write_to(&mut png_bytes, image::ImageFormat::Png)
        .map_err(|e| ocr::utils::OcrError::Internal(e.to_string()))?;
    let png_bytes = png_bytes.into_inner();

    let results = ocr
        .recognize_text_from_bytes_multi_page(&png_bytes)
        .await?;
    // PNG is single-page → exactly 1 result
    assert_eq!(results.len(), 1);

    Ok(())
}

#[tokio::test]
async fn test_recognize_text_from_bytes_multi_page_tiff() -> Result<()> {
    use image::{ImageBuffer, Luma};
    use std::io::Cursor;

    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    // Create a 2-page TIFF using the tiff crate directly
    let page1 = ImageBuffer::from_pixel(200, 50, Luma([255u8]));
    let page2 = ImageBuffer::from_pixel(200, 50, Luma([255u8]));

    let mut tiff_bytes = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut tiff_bytes)
            .map_err(|e| ocr::utils::OcrError::Internal(e.to_string()))?;
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(
                page1.width(),
                page1.height(),
                page1.as_raw(),
            )
            .map_err(|e| ocr::utils::OcrError::Internal(e.to_string()))?;
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(
                page2.width(),
                page2.height(),
                page2.as_raw(),
            )
            .map_err(|e| ocr::utils::OcrError::Internal(e.to_string()))?;
    }
    let tiff_bytes = tiff_bytes.into_inner();

    let results = ocr
        .recognize_text_from_bytes_multi_page(&tiff_bytes)
        .await?;
    // 2-page TIFF → exactly 2 results
    assert_eq!(results.len(), 2);

    Ok(())
}
