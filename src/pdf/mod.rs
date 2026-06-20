//! PDF input support for OCR
//!
//! Extracts embedded images from PDF files for OCR processing.
//! Enabled via the `pdf` feature flag.

use crate::utils::Result;
use pdf::enc::StreamFilter;
use pdf::file::FileOptions;
use pdf::object::Resolve;
use std::path::Path;

/// Extracted image from a PDF page
#[derive(Debug, Clone)]
pub struct PdfImage {
    pub page_number: u32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: PdfImageFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfImageFormat {
    Jpeg,
    Png,
    Raw,
}

/// Extract all embedded images from a PDF file
pub fn extract_images(path: &Path) -> Result<Vec<PdfImage>> {
    let file = FileOptions::uncached()
        .open(path)
        .map_err(|e| crate::utils::OcrError::Internal(format!("Failed to open PDF: {}", e)))?;
    let resolver = file.resolver();
    let mut images = Vec::new();

    for (page_num, page) in file.pages().enumerate() {
        let page = match page {
            Ok(p) => p,
            Err(_) => continue,
        };
        let resources = match page.resources() {
            Ok(r) => r,
            Err(_) => continue,
        };

        for (_name, &xobject_ref) in resources.xobjects.iter() {
            let Ok(xobject) = resolver.get(xobject_ref) else {
                continue;
            };

            if let pdf::object::XObject::Image(ref image) = *xobject {
                let width = image.width;
                let height = image.height;

                let (data, format) = match image.raw_image_data(&resolver) {
                    Ok((raw_data, filter)) => {
                        let format = match filter {
                            Some(StreamFilter::DCTDecode(_)) => PdfImageFormat::Jpeg,
                            Some(StreamFilter::JPXDecode) => PdfImageFormat::Jpeg,
                            Some(StreamFilter::FlateDecode(_)) => PdfImageFormat::Png,
                            _ => PdfImageFormat::Raw,
                        };
                        (raw_data.to_vec(), format)
                    }
                    Err(_) => continue,
                };

                images.push(PdfImage {
                    page_number: (page_num + 1) as u32,
                    width,
                    height,
                    data,
                    format,
                });
            }
        }
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_images_no_file() {
        let result = extract_images(Path::new("nonexistent.pdf"));
        assert!(result.is_err());
    }
}
