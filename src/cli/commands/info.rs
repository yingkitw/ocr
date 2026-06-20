use anyhow::Result;
use ocr::api::Ocr;

pub async fn handle_info() -> Result<()> {
    let ocr = Ocr::new()?;
    let metadata = ocr.get_metadata();

    println!("OCR Engine Information:");
    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Description: {}", metadata.description);
    println!(
        "  Supported Languages: {}",
        metadata.supported_languages.join(", ")
    );
    println!(
        "  Supported Image Formats: {}",
        metadata.supported_image_formats.join(", ")
    );

    Ok(())
}
