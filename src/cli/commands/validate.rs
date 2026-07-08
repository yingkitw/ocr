use std::path::PathBuf;

use anyhow::Result;

use ocr::core::config::OcrConfig;

pub async fn handle_validate(config_file: PathBuf) -> Result<()> {
    let config = OcrConfig::from_file(&config_file)?;

    config.validate()?;
    println!("Configuration is valid");
    Ok(())
}
