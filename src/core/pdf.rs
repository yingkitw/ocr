//! Minimal searchable PDF writer (zero external dependencies).
//!
//! Produces a PDF with:
//! - The original image as a full-page XObject
//! - An invisible text layer (text rendering mode 3) positioned at word
//!   bounding boxes, so the text is selectable/searchable in any PDF viewer
//!
//! This is a minimal implementation of the PDF 1.4 format — just enough
//! for searchable OCR output. Latin text uses the built-in Helvetica font
//! (PDF base-14, no embedding needed) with WinAnsiEncoding. Words that
//! contain non-WinAnsi characters (CJK, etc.) are drawn with a Type0 /
//! Identity-H CID font plus a `/ToUnicode` CMap — the same scheme
//! Tesseract uses — so the full Unicode text remains selectable and
//! searchable even though no glyph outlines are embedded.

use crate::core::text::TextResult;
use crate::utils::Result;
use std::collections::BTreeMap;
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

/// Can this character be encoded with the WinAnsi `/F1` font?
fn is_winansi(ch: char) -> bool {
    let code = ch as u32;
    code < 0x80 || char_to_winansi(ch).is_some()
}

/// A word goes wholly to the CID font when any character is outside
/// WinAnsi — no mid-word font switching.
fn word_needs_cid(word: &str) -> bool {
    word.chars().any(|c| !is_winansi(c))
}

/// Maps every character used by `/F2` words to a 2-byte CID.
///
/// BMP characters use their code point as the CID, so the content-stream
/// hex for a word is simply its UTF-16BE form. Astral characters
/// (> U+FFFF, e.g. CJK Extension B) get sequential CIDs in the private
/// range 0xF000+; the `/ToUnicode` CMap maps them back to a UTF-16
/// surrogate pair, so extraction still yields the true character.
struct CidPlan {
    char_to_cid: BTreeMap<u32, u16>,
}

impl CidPlan {
    /// Collect the CID table for all `/F2`-destined words. `None` when
    /// every word is WinAnsi — the common all-Latin case emits zero
    /// extra objects.
    fn collect(results: &[&TextResult]) -> Option<CidPlan> {
        let mut char_to_cid: BTreeMap<u32, u16> = BTreeMap::new();
        let mut next_private = 0xF000u16;
        for result in results {
            for word in &result.words {
                if !word_needs_cid(&word.text) {
                    continue;
                }
                for ch in word.text.chars() {
                    let cp = ch as u32;
                    if char_to_cid.contains_key(&cp) {
                        continue;
                    }
                    let cid = if cp <= 0xFFFF {
                        cp as u16
                    } else {
                        let c = next_private;
                        next_private += 1;
                        c
                    };
                    char_to_cid.insert(cp, cid);
                }
            }
        }
        if char_to_cid.is_empty() {
            None
        } else {
            Some(CidPlan { char_to_cid })
        }
    }

    /// Hex-encode a word as a sequence of 2-byte CIDs.
    fn hex(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() * 4);
        for ch in text.chars() {
            let cid = self.char_to_cid[&(ch as u32)];
            out.push_str(&format!("{:04X}", cid));
        }
        out
    }

    /// Serialize the `/ToUnicode` CMap stream: every used CID → its
    /// Unicode scalar value (UTF-16BE, surrogate pairs for astral).
    fn to_unicode_cmap(&self) -> Vec<u8> {
        let mut s = String::from(
            "/CIDInit /ProcSet findresource begin\n\
             12 dict begin\n\
             begincmap\n\
             /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
             /CMapName /Adobe-Identity-UCS def\n\
             /CMapType 2 def\n\
             1 begincodespacerange\n\
             <0000> <FFFF>\n\
             endcodespacerange\n",
        );
        // PDF limits bfchar blocks to 100 entries.
        let entries: Vec<(&u32, &u16)> = self.char_to_cid.iter().collect();
        for chunk in entries.chunks(100) {
            s.push_str(&format!("{} beginbfchar\n", chunk.len()));
            for (cp, cid) in chunk {
                let ch = char::from_u32(**cp).unwrap_or('\u{FFFD}');
                let mut utf16 = [0u16; 2];
                let encoded = ch.encode_utf16(&mut utf16);
                let dst: String = encoded.iter().map(|u| format!("{:04X}", u)).collect();
                s.push_str(&format!("<{:04X}> <{}>\n", cid, dst));
            }
            s.push_str("endbfchar\n");
        }
        s.push_str(
            "endcmap\n\
             CMapName currentdict /CMap defineresource pop\n\
             end\n\
             end\n",
        );
        s.into_bytes()
    }
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
    cids: Option<&CidPlan>,
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
            if word_needs_cid(&word.text) {
                let plan = match cids {
                    Some(p) => p,
                    None => continue,
                };
                // CID font advance is /DW = 1000 units per glyph at font
                // size 1, so the string advances exactly n text-space
                // units — scale X by word_w / n for a true bbox fit.
                let n = word.text.chars().count().max(1) as f32;
                let hex = plan.hex(&word.text);
                stream.push_str(&format!(
                    "{sx} 0 0 {sy} {x} {y} Tm\n/F2 1 Tf\n<{hex}> Tj\n/F1 1 Tf\n",
                    sx = word_w / n,
                    sy = scale_y,
                    x = x,
                    y = y,
                    hex = hex,
                ));
                continue;
            }
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

/// Emit the 4 objects for the `/F2` CID font and return the Type0 font's
/// object number. Caller must allocate the four consecutive numbers
/// `base..base+4` and register a `/F2 <base> 0 R` resource.
fn write_cjk_font_objects(
    out: &mut Vec<u8>,
    offsets: &mut Vec<usize>,
    base: usize,
    plan: &CidPlan,
) -> Result<()> {
    // Type0 composite font — Identity-H so each 2-byte code is a CID.
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", base)?;
    writeln!(
        out,
        "<< /Type /Font /Subtype /Type0 /BaseFont /GlyphLessFont /Encoding /Identity-H /DescendantFonts [{} 0 R] /ToUnicode {} 0 R >>",
        base + 1,
        base + 3
    )?;
    writeln!(out, "endobj")?;

    // Descendant CIDFont — /DW 1000 gives every glyph a 1-em advance,
    // which lets the text matrix fit the string exactly to the word bbox.
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", base + 1)?;
    writeln!(
        out,
        "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GlyphLessFont /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor {} 0 R /DW 1000 /CIDToGIDMap /Identity >>",
        base + 2
    )?;
    writeln!(out, "endobj")?;

    // FontDescriptor without an embedded FontFile — the text layer is
    // invisible (3 Tr), so viewers substitute any face; extraction uses
    // ToUnicode, not outlines.
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", base + 2)?;
    writeln!(
        out,
        "<< /Type /FontDescriptor /FontName /GlyphLessFont /Flags 4 /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >>"
    )?;
    writeln!(out, "endobj")?;

    // /ToUnicode CMap — what makes copy/search work for every Unicode char.
    let cmap = plan.to_unicode_cmap();
    offsets.push(out.len());
    writeln!(out, "{} 0 obj", base + 3)?;
    writeln!(out, "<< /Length {} >>", cmap.len())?;
    writeln!(out, "stream")?;
    out.extend_from_slice(&cmap);
    out.extend_from_slice(b"endstream\n");
    writeln!(out, "endobj")?;

    Ok(())
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

    let cids = CidPlan::collect(&[result]);
    let content = build_content_stream(page_w, page_h, "Im0", result, cids.as_ref());
    let content_bytes = content.as_bytes();

    // Build PDF objects
    // Object numbering:
    // 1: Catalog
    // 2: Pages
    // 3: Page
    // 4: Content stream
    // 5: Image XObject
    // 6: Font (Helvetica /F1)
    // 7..10: CID font quartet (/F2), only when non-WinAnsi words exist

    let cjk_base = if cids.is_some() { Some(7) } else { None };
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
    let font_resources = match cjk_base {
        Some(b) => format!("/F1 6 0 R /F2 {} 0 R", b),
        None => "/F1 6 0 R".to_string(),
    };
    writeln!(
        out,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources << /XObject << /Im0 5 0 R >> /Font << {} >> >> /Contents 4 0 R >>",
        page_w, page_h, font_resources
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

    // Objects 7-10: Type0/Identity-H CID font for non-WinAnsi words
    if let (Some(b), Some(plan)) = (cjk_base, cids.as_ref()) {
        write_cjk_font_objects(out, &mut offsets, b, plan)?;
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

    // Shared CID font quartet (/F2), allocated only when any page has
    // non-WinAnsi words.
    let all_results: Vec<&TextResult> = pages.iter().map(|p| p.1).collect();
    let cids = CidPlan::collect(&all_results);
    let cjk_base = if cids.is_some() {
        let b = next_obj;
        next_obj += 4;
        Some(b)
    } else {
        None
    };

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

    // Write shared CID font objects (/F2) when needed
    if let (Some(b), Some(plan)) = (cjk_base, cids.as_ref()) {
        write_cjk_font_objects(&mut out, &mut offsets, b, plan)?;
    }

    // Write each page's objects
    for (i, (image_data, result)) in pages.iter().enumerate() {
        let (jpeg_bytes, img_w, img_h) = encode_image_as_jpeg(image_data)?;
        let page_w = img_w as f32;
        let page_h = img_h as f32;
        let image_name = format!("Im{}", i);
        let content = build_content_stream(page_w, page_h, &image_name, result, cids.as_ref());
        let content_bytes = content.as_bytes();

        let font_resources = match cjk_base {
            Some(b) => format!("/F1 {} 0 R /F2 {} 0 R", font_obj, b),
            None => format!("/F1 {} 0 R", font_obj),
        };

        // Page object
        offsets.push(out.len());
        writeln!(out, "{} 0 obj", page_obj_nums[i])?;
        writeln!(
            out,
            "<< /Type /Page /Parent {} 0 R /MediaBox [0 0 {} {}] /Resources << /XObject << /{} {} 0 R >> /Font << {} >> >> /Contents {} 0 R >>",
            pages_obj, page_w, page_h, image_name, image_obj_nums[i], font_resources, content_obj_nums[i]
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
        // CJK characters are not in WinAnsi — /F1 skips them; the CID
        // font (/F2) is what makes them selectable.
        let bytes = winansi_bytes("日本");
        assert!(bytes.is_empty(), "CJK chars should be skipped");
    }

    fn make_cjk_result() -> TextResult {
        let mut result = TextResult::new(
            "日本語 test".to_string(),
            0.9,
            BoundingBox::new(0, 0, 200, 50),
        );
        result.words.push(WordResult::new(
            "日本語".to_string(),
            0.9,
            BoundingBox::new(10, 5, 70, 25),
        ));
        result.words.push(WordResult::new(
            "test".to_string(),
            0.95,
            BoundingBox::new(80, 5, 130, 25),
        ));
        result
    }

    #[test]
    fn test_cid_plan_collects_only_when_needed() {
        assert!(CidPlan::collect(&[&make_test_result()]).is_none());
        let plan = CidPlan::collect(&[&make_cjk_result()]).unwrap();
        // '日' = U+65E5 → CID = code point
        assert_eq!(plan.char_to_cid[&0x65E5], 0x65E5);
        assert_eq!(plan.hex("日本語"), "65E5672C8A9E");
    }

    #[test]
    fn test_to_unicode_cmap_maps_cids_back() {
        let plan = CidPlan::collect(&[&make_cjk_result()]).unwrap();
        let cmap = String::from_utf8(plan.to_unicode_cmap()).unwrap();
        assert!(cmap.contains("begincmap"));
        assert!(cmap.contains("/CMapName /Adobe-Identity-UCS def"));
        assert!(cmap.contains("<65E5> <65E5>"), "CID 0x65E5 → U+65E5 日");
        assert!(cmap.contains("<8A9E> <8A9E>"), "CID 0x8A9E → U+8A9E 語");
    }

    #[test]
    fn test_cid_plan_astral_chars_get_private_cids() {
        let mut result = TextResult::new(
            "𠀀".to_string(), // U+20000, CJK Ext B — astral
            0.9,
            BoundingBox::new(0, 0, 50, 25),
        );
        result.words.push(WordResult::new(
            "𠀀".to_string(),
            0.9,
            BoundingBox::new(5, 5, 30, 25),
        ));
        let plan = CidPlan::collect(&[&result]).unwrap();
        let cid = plan.char_to_cid[&0x20000];
        assert_eq!(cid, 0xF000, "astral chars get private-range CIDs");
        // ToUnicode maps the private CID back to the surrogate pair
        // U+20000 → UTF-16BE D840 DC00
        let cmap = String::from_utf8(plan.to_unicode_cmap()).unwrap();
        assert!(cmap.contains("<F000> <D840DC00>"));
    }

    #[test]
    fn test_searchable_pdf_cjk_text_layer() {
        let image = make_test_image();
        let result = make_cjk_result();
        let pdf = format_searchable_pdf(&[(&image[..], &result)]).unwrap();
        let pdf_str = String::from_utf8_lossy(&pdf);
        assert!(pdf_str.contains("/Subtype /Type0"), "CID Type0 font present");
        assert!(pdf_str.contains("/Encoding /Identity-H"), "Identity-H encoding");
        assert!(pdf_str.contains("/ToUnicode"), "ToUnicode CMap present");
        assert!(pdf_str.contains("/F2"), "F2 resource registered");
        // 日本語 as UTF-16BE CIDs in the content stream
        assert!(pdf_str.contains("65E5672C8A9E"), "CJK text in content stream");
        // Latin word still goes through WinAnsi /F1
        assert!(pdf_str.contains("74657374"), "'test' stays on /F1");
    }

    #[test]
    fn test_searchable_pdf_latin_only_has_no_cid_font() {
        let image = make_test_image();
        let result = make_test_result();
        let pdf = format_searchable_pdf(&[(&image[..], &result)]).unwrap();
        let pdf_str = String::from_utf8_lossy(&pdf);
        assert!(
            !pdf_str.contains("/Type0"),
            "all-Latin pages must not emit the CID font objects"
        );
    }

    #[test]
    fn test_searchable_pdf_multi_page_shares_cid_font() {
        let image = make_test_image();
        let latin = make_test_result();
        let cjk = make_cjk_result();
        let pdf = format_searchable_pdf(&[(&image[..], &latin), (&image[..], &cjk)]).unwrap();
        let pdf_str = String::from_utf8_lossy(&pdf);
        assert_eq!(
            pdf_str.matches("/Subtype /Type0").count(),
            1,
            "one shared Type0 font across pages"
        );
        assert!(pdf_str.contains("65E5672C8A9E"), "CJK page text present");
    }

}
