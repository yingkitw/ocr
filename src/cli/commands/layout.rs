use std::path::PathBuf;

use anyhow::{anyhow, Result};
use tracing::info;

use ocr::api::Ocr;

use super::helpers::write_output;

pub async fn handle_layout(image_path: PathBuf, output: Option<PathBuf>) -> Result<()> {
    info!("Analyzing layout for: {:?}", image_path);

    let ocr = Ocr::new()?;
    ocr.initialize().await.map_err(|e| anyhow!("{}", e))?;

    let image_data = tokio::fs::read(&image_path).await?;
    let dynamic_image = ::image::load_from_memory(&image_data)?;

    // `analyze_layout` takes RAW grayscale pixels (not encoded bytes), so
    // convert the decoded image to Luma8 before handing it over.
    let gray = dynamic_image.to_luma8();
    let (width, height) = gray.dimensions();

    let layout_result = ocr
        .analyze_layout(gray.as_raw(), width, height)
        .await
        .map_err(|e| anyhow!("{}", e))?;

    let layout_json = serde_json::to_string_pretty(&layout_result)?;
    write_output(&layout_json, output)
}
