//! Basic tests for OCR

use ocr::api::Ocr;
use ocr::utils::Result;

#[tokio::test]
async fn test_ocr_creation() -> Result<()> {
    let ocr = Ocr::new()?;
    assert!(ocr.is_initialized().await);
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
