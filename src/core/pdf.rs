//! Minimal searchable PDF writer (zero external dependencies).
//!
//! Produces a PDF with:
//! - The original image as a full-page XObject
//! - An invisible text layer (text rendering mode 3) positioned at word
//!   bounding boxes, so the text is selectable/searchable in any PDF viewer
//!
//! This is a minimal implementation of the PDF 1.4 format — just enough
//! for searchable OCR output. It uses the built-in Helvetica font
//! (PDF base-14, no embedding needed) with WinAnsiEncoding for Latin text.

use crate::core::text::TextResult;
use crate::utils::Result;
use std::io::Write;

/// Encode an image as JPEG bytes for embedding in the PDF.
fn encode_image_as_jpeg(image_data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::load_from_memory(image_data)?;
    let width = img.width();
    let height = img.height();
    let rgb = img.to_rgb8();
    let mut jpeg_bytes = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_bytes, 85);
    encoder.encode(&rgb, width, height, image::ColorType::Rgb8)?;
    Ok((jpeg_bytes, width, height))
}

/// Convert WinAnsi-encodable characters for the text layer.
/// Non-Latin characters are skipped (the text layer is invisible —
/// searchability is best-effort for non-Latin scripts).
fn winansi_bytes(text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in text.chars() {
        let code = ch as u32;
        if code < 0x80 {
            out.push(code as u8);
        } else if let Some(b) = char_to_winansi(ch) {
            out.push(b);
        }
        // Skip characters not in WinAnsi (invisible text — searchability is best-effort)
    }
    out
}

/// Map common Latin-1 supplement characters to WinAnsi byte values.
fn char_to_winansi(ch: char) -> Option<u8> {
    let code = ch as u32;
    // WinAnsi = ISO-8859-1 + Windows extensions; for the basic Latin-1
    // supplement (0xA0–0xFF), the byte value equals the code point.
    if (0xA0..=0xFF).contains(&code) {
        Some(code as u8)
    } else {
        // Common Windows extensions (smart quotes, etc.)
        match ch {
            '\u{2018}' => Some(0x91), // left single quote
            '\u{2019}' => Some(0x92), // right single quote
            '\u{201C}' => Some(0x93), // left double quote
            '\u{201D}' => Some(0x94), // right double quote
            '\u{2013}' => Some(0x96), // en dash
            '\u{2014}' => Some(0x97), // em dash
            '\u{2026}' => Some(0x85), // horizontal ellipsis
            '\u{20AC}' => Some(0x80), // euro sign
            _ => None,
        }
    }
}

/// Build the PDF content stream for one page: draw image + invisible text.
fn build_content_stream(
    page_width: f32,
    page_height: f32,
    image_name: &str,
    result: &TextResult,
) -> String {
    let mut stream = String::new();

    // Draw the image full-page
    // q = save state, cm = set transform matrix, Do = draw XObject, Q = restore
    stream.push_str(&format!(
        "q\n{pw} 0 0 {ph} 0 0 cm\n/{im} Do\nQ\n",
        pw = page_width,
        ph = page_height,
        im = image_name,
    ));

    // Invisible text layer (rendering mode 3 = invisible)
    if !result.words.is_empty() {
        stream.push_str("BT\n3 Tr\n/F1 1 Tf\n");
        for word in &result.words {
            let wb = &word.bounding_box;
            let word_w = (wb.right - wb.left).max(1) as f32;
            let word_h = (wb.bottom - wb.top).max(1) as f32;
            // PDF origin is bottom-left; image origin is top-left.
            // Y coordinate: page_height - bottom
            let x = wb.left as f32;
            let y = page_height - wb.bottom as f32;
            // Text matrix: [scaleX 0 0 scaleY x y]
            // Scale text to match word bounding box dimensions
            let scale_x = word_w;
            let scale_y = word_h;
            let bytes = winansi_bytes(&word.text);
            if bytes.is_empty() {
                continue;
            }
            let hex: String = bytes.iter().map(|b| format!("{:02X}", b)).collect();
            stream.push_str(&format!(
                "{sx} 0 0 {sy} {x} {y} Tm\n<{hex}> Tj\n",
                sx = scale_x,
                sy = scale_y,
                x = x,
                y = y,
                hex = hex,
            ));
        }
        stream.push_str("ET\n");
    }

    stream
}

/// Write a searchable PDF for a single page.
///
/// Takes the original image bytes (PNG/JPEG/etc.) and the OCR result
/// with word-level bounding boxes. Produces a PDF where:
/// - The image is drawn full-page
/// - The recognized text is placed invisibly at word positions
fn write_searchable_pdf(
    image_data: &[u8],
    result: &TextResult,
    out: &mut Vec<u8>,
) -> Result<()> {
    // Encode image as JPEG for embedding
    let (jpeg_bytes, img_w, img_h) = encode_image_as_jpeg(image_data)?;

    // PDF uses points (1/72 inch). We map the image pixels 1:1 to points.
    let page_w = img_w as f32;
    let page_h = img_h as f32;

    let content = build_content_stream(page_w, page_h, "Im0", result);
    let content_bytes = content.as_bytes();

    // Build PDF objects
    // Object numbering:
    // 1: Catalog
    // 2: Pages
    // 3: Page
    // 4: Content stream
    // 5: Image XObject
    // 6: Font

    let mut offsets: Vec<usize> = Vec::new();

    // Header
    out.extend_from_slice(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"); // binary marker

    // Object 1: Catalog
    offsets.push(out.len());
    writeln!(out, "1 0 obj")?;
    writeln!(out, "<< /Type /Catalog /Pages 2 0 R >>")?;
    writeln!(out, "endobj")?;

    // Object 2: Pages
    offsets.push(out.len());
    writeln!(out, "2 0 obj")?;
    writeln!(out, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")?;
    writeln!(out, "endobj")?;

    // Object 3: Page
    offsets.push(out.len());
    writeln!(out, "3 0 obj")?;
    writeln!(
        out,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources << /XObject << /Im0 5 0 R >> /Font << /F1 6 0 R >> >> /Contents 4 0 R >>",
        page_w, page_h
    )?;
    writeln!(out, "endobj")?;

    // Object 4: Content stream
    offsets.push(out.len());
    writeln!(out, "4 0 obj")?;
    writeln!(out, "<< /Length {} >>", content_bytes.len())?;
    writeln!(out, "stream")?;
    out.extend_from_slice(content_bytes);
    out.extend_from_slice(b"\nendstream\n");
    writeln!(out, "endobj")?;

    // Object 5: Image XObject (JPEG)
    offsets.push(out.len());
    writeln!(out, "5 0 obj")?;
    writeln!(
        out,
        "<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>",
        img_w, img_h, jpeg_bytes.len()
    )?;
    writeln!(out, "stream")?;
    out.extend_from_slice(&jpeg_bytes);
    out.extend_from_slice(b"\nendstream\n");
    writeln!(out, "endobj")?;

    // Object 6: Font (built-in Helvetica with WinAnsiEncoding)
    offsets.push(out.len());
    writeln!(out, "6 0 obj")?;
    writeln!(
        out,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )?;
    writeln!(out, "endobj")?;

    // xref table
    let xref_offset = out.len();
    writeln!(out, "xref")?;
    writeln!(out, "0 {}", offsets.len() + 1)?;
    writeln!(out, "0000000000 65535 f ")?;
    for off in &offsets {
        writeln!(out, "{:010} 00000 n ", off)?;
    }

    // Trailer
    writeln!(out, "trailer")?;
    writeln!(out, "<< /Size {} /Root 1 0 R >>", offsets.len() + 1)?;
    writeln!(out, "startxref")?;
    writeln!(out, "{}", xref_offset)?;
    writeln!(out, "%%EOF")?;

    Ok(())
}

/// Write a multi-page searchable PDF.
///
/// Each page gets its own image + invisible text layer.
pub fn format_searchable_pdf(pages: &[(&[u8], &TextResult)]) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    if pages.is_empty() {
        // Empty PDF with a single blank page
        let empty = TextResult::new(String::new(), 0.0, crate::core::text::BoundingBox::new(0, 0, 1, 1));
        write_searchable_pdf(&[0u8; 3], &empty, &mut out)?;
        return Ok(out);
    }

    if pages.len() == 1 {
        write_searchable_pdf(pages[0].0, pages[0].1, &mut out)?;
        return Ok(out);
    }

    // Multi-page: build all objects then write xref
    let mut offsets: Vec<usize> = Vec::new();
    let mut next_obj = 1; // next object number to allocate

    // Header
    out.extend_from_slice(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");

    // Object 1: Catalog (allocated first, but we need the Pages object number)
    let catalog_obj = next_obj;
    next_obj += 1;
    let pages_obj = next_obj;
    next_obj += 1;

    // For each page, allocate: Page, Content, Image, (Font is shared)
    let font_obj = next_obj; // shared font
    next_obj += 1;

    let mut page_obj_nums: Vec<usize> = Vec::new();
    let mut content_obj_nums: Vec<usize> = Vec::new();
    let mut image_obj_nums: Vec<usize> = Vec::new();

    for _ in 0..pages.len() {
        page_obj_nums.push(next_obj);
        next_obj += 1;
        content_obj_nums.push(next_obj);
        next_obj += 1;
        image_obj_nums.push(next_obj);
        next_obj += 1;
    }

    let page_refs: Vec<String> = page_obj_nums.iter().map(|n| format!("{} 0 R", n)).collect();

    // Write Catalog
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", catalog_obj)?;
    writeln!(out, "<< /Type /Catalog /Pages {} 0 R >>", pages_obj)?;
    writeln!(out, "endobj")?;

    // Write Pages
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", pages_obj)?;
    writeln!(
        out,
        "<< /Type /Pages /Kids [{}] /Count {} >>",
        page_refs.join(" "),
        pages.len()
    )?;
    writeln!(out, "endobj")?;

    // Write shared Font
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", font_obj)?;
    writeln!(
        out,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )?;
    writeln!(out, "endobj")?;

    // Write each page's objects
    for (i, (image_data, result)) in pages.iter().enumerate() {
        let (jpeg_bytes, img_w, img_h) = encode_image_as_jpeg(image_data)?;
        let page_w = img_w as f32;
        let page_h = img_h as f32;
        let image_name = format!("Im{}", i);
        let content = build_content_stream(page_w, page_h, &image_name, result);
        let content_bytes = content.as_bytes();

        // Page object
        offsets.push(out.len());
        writeln!(out, "{} 0 obj", page_obj_nums[i])?;
        writeln!(
            out,
            "<< /Type /Page /Parent {} 0 R /MediaBox [0 0 {} {}] /Resources << /XObject << /{} {} 0 R >> /Font << /F1 {} 0 R >> >> /Contents {} 0 R >>",
            pages_obj, page_w, page_h, image_name, image_obj_nums[i], font_obj, content_obj_nums[i]
        )?;
        writeln!(out, "endobj")?;

        // Content stream
        offsets.push(out.len());
        writeln!(out, "{} 0 obj", content_obj_nums[i])?;
        writeln!(out, "<< /Length {} >>", content_bytes.len())?;
        writeln!(out, "stream")?;
        out.extend_from_slice(content_bytes);
        out.extend_from_slice(b"\nendstream\n");
        writeln!(out, "endobj")?;

        // Image XObject
        offsets.push(out.len());
        writeln!(out, "{} 0 obj", image_obj_nums[i])?;
        writeln!(
            out,
            "<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>",
            img_w, img_h, jpeg_bytes.len()
        )?;
        writeln!(out, "stream")?;
        out.extend_from_slice(&jpeg_bytes);
        out.extend_from_slice(b"\nendstream\n");
        writeln!(out, "endobj")?;
    }

    // xref table
    let xref_offset = out.len();
    writeln!(out, "xref")?;
    writeln!(out, "0 {}", offsets.len() + 1)?;
    writeln!(out, "0000000000 65535 f ")?;
    for off in &offsets {
        writeln!(out, "{:010} 00000 n ", off)?;
    }

    // Trailer
    writeln!(out, "trailer")?;
    writeln!(out, "<< /Size {} /Root {} 0 R >>", offsets.len() + 1, catalog_obj)?;
    writeln!(out, "startxref")?;
    writeln!(out, "{}", xref_offset)?;
    writeln!(out, "%%EOF")?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::text::{BoundingBox, LineResult, TextResult, WordResult};

    fn make_test_result() -> TextResult {
        let mut result = TextResult::new(
            "Hello World".to_string(),
            0.9,
            BoundingBox::new(0, 0, 200, 50),
        );
        let mut line = LineResult::new(
            "Hello World".to_string(),
            0.9,
            BoundingBox::new(0, 0, 200, 20),
        );
        line.words.push(WordResult::new(
            "Hello".to_string(),
            0.95,
            BoundingBox::new(10, 5, 60, 25),
        ));
        line.words.push(WordResult::new(
            "World".to_string(),
            0.85,
            BoundingBox::new(70, 5, 130, 25),
        ));
        result.lines.push(line);
        result.words.push(WordResult::new(
            "Hello".to_string(),
            0.95,
            BoundingBox::new(10, 5, 60, 25),
        ));
        result.words.push(WordResult::new(
            "World".to_string(),
            0.85,
            BoundingBox::new(70, 5, 130, 25),
        ));
        result
    }

    fn make_test_image() -> Vec<u8> {
        // 100x50 white PNG
        let img = image::RgbImage::from_pixel(100, 50, image::Rgb([255, 255, 255]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn test_searchable_pdf_single_page() {
        let image = make_test_image();
        let result = make_test_result();
        let pdf = format_searchable_pdf(&[(&image[..], &result)]).unwrap();
        // PDF header
        assert!(pdf.starts_with(b"%PDF-1.4"));
        // PDF trailer
        assert!(pdf.ends_with(b"%%EOF\n") || pdf.ends_with(b"%%EOF"));
        // Contains image XObject
        let pdf_str = String::from_utf8_lossy(&pdf);
        assert!(pdf_str.contains("/Subtype /Image"), "PDF should contain an image XObject");
        // Contains invisible text (Tr 3)
        assert!(pdf_str.contains("3 Tr"), "PDF should have invisible text layer");
        // Contains the word text in hex
        // "Hello" in hex = 48656C6C6F
        assert!(pdf_str.contains("48656C6C6F"), "PDF should contain 'Hello' in hex");
    }

    #[test]
    fn test_searchable_pdf_multi_page() {
        let image = make_test_image();
        let result = make_test_result();
        let pdf = format_searchable_pdf(&[(&image[..], &result), (&image[..], &result)]).unwrap();
        assert!(pdf.starts_with(b"%PDF-1.4"));
        // Should have 2 pages
        let pdf_str = String::from_utf8_lossy(&pdf);
        assert!(pdf_str.contains("/Count 2"), "PDF should have 2 pages");
    }

    #[test]
    fn test_winansi_bytes_ascii() {
        let bytes = winansi_bytes("Hello");
        assert_eq!(bytes, vec![b'H', b'e', b'l', b'l', b'o']);
    }

    #[test]
    fn test_winansi_bytes_latin1() {
        let bytes = winansi_bytes("café");
        assert_eq!(bytes, vec![b'c', b'a', b'f', 0xE9]); // é = 0xE9 in Latin-1
    }

    #[test]
    fn test_winansi_bytes_skips_non_latin() {
        // CJK characters are not in WinAnsi — should be skipped
        let bytes = winansi_bytes("日本");
        assert!(bytes.is_empty(), "CJK chars should be skipped");
    }

}
