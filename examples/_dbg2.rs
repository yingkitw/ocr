//! Temporary diagnostic.

use ocr::webocr::GrayImage;
use rusttype::Font;

fn render(text: &str, font: &Font, px: u32, pad: u32) -> GrayImage {
    let scale = rusttype::Scale::uniform(px as f32);
    let v = font.v_metrics(scale);
    let ascent = v.ascent;
    let descent = -v.descent;
    let mut width = 0u32;
    for g in font.layout(text, scale, rusttype::point(0.0, 0.0)) {
        if let Some(bb) = g.pixel_bounding_box() {
            width = width.max(bb.max.x as u32);
        }
    }
    let w = width + 2 * pad;
    let h = (ascent + descent).ceil() as u32 + 2 * pad;
    let mut data = vec![255u8; (w * h) as usize];
    for g in font.layout(text, scale, rusttype::point(pad as f32, pad as f32 + ascent)) {
        if let Some(bb) = g.pixel_bounding_box() {
            let xo = (bb.min.x as i32).max(0) as u32;
            let yo = (bb.min.y as i32).max(0) as u32;
            g.draw(|x, y, v| {
                let xi = (xo + x) as usize;
                let yi = (yo + y) as usize;
                if xi < w as usize && yi < h as usize {
                    let s = &mut data[yi * w as usize + xi];
                    let ink = (v * 255.0).round() as u8;
                    *s = 255u8.saturating_sub(ink).min(*s);
                }
            });
        }
    }
    GrayImage {
        width: w as usize,
        height: h as usize,
        data,
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/System/Library/Fonts/Supplemental/Arial.ttf".to_string()
    });
    let data = std::fs::read(&path).unwrap();
    let font = Font::try_from_bytes(&data).expect("font parses");
    let owned = rusttype::Font::try_from_vec(data.clone());
    println!("try_from_bytes: OK; try_from_vec: {}", owned.is_some());
    let text = "The quick brown fox jumps over the 42 lazy dogs";
    let ocr = ocr::webocr::TemplateOcr::new().unwrap();
    let g = render(text, &font, 30, 15);
    eprintln!("DBG {}x{} sum={} otsu={}", g.width, g.height, g.data.iter().map(|v| *v as u64).sum::<u64>(), ocr::webocr::binarize::otsu_threshold(&g.data));
    let r = ocr.recognize(&g);
    println!("{} -> {:?}", path, r.text);
}
