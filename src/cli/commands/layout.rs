use std::path::PathBuf;

use anyhow::{anyhow, Result};
use image::GenericImageView;
use tracing::info;

use ocr::api::Ocr;

use super::helpers::write_output;

pub async fn handle_layout(image_path: PathBuf, output: Option<PathBuf>) -> Result<()> {
    info!("Analyzing layout for: {:?}", image_path);

    let ocr = Ocr::new()?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let image_data = tokio::fs::read(&image_path).await?;
    let dynamic_image = ::image::load_from_memory(&image_data)?;
    let (width, height) = dynamic_image.dimensions();

    let layout_result = ocr
        .analyze_layout(&image_data, width, height)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    let layout_json = serde_json::to_string_pretty(&layout_result)?;
    write_output(&layout_json, output)
}
