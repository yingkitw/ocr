use ocr::webocr::{GrayImage, TemplateOcr};

fn render(text: &str, font_data: &[u8], px: u32) -> (Vec<u8>, u32, u32) {
    use rusttype::{Font, Scale};
    let font = Font::try_from_bytes(font_data).unwrap();
    let scale = Scale::uniform(px as f32);
    let pad = px / 2;
    let v = font.v_metrics(scale);
    let mut width = 0u32;
    for g in font.layout(text, scale, rusttype::point(0.0, 0.0)) {
        if let Some(bb) = g.pixel_bounding_box() {
            width = width.max(bb.max.x as u32);
        }
    }
    let text_h = (v.ascent - v.descent).ceil() as u32;
    let w = width + 2 * pad;
    let h = text_h + 2 * pad;
    let mut data = vec![255u8; (w * h) as usize];
    for g in font.layout(text, scale, rusttype::point(pad as f32, pad as f32 + v.ascent)) {
        if let Some(bb) = g.pixel_bounding_box() {
            let (xo, yo) = (bb.min.x, bb.min.y);
            g.draw(|x, y, c| {
                let (xi, yi) = ((xo + x as i32) as usize, (yo + y as i32) as usize);
                if xi < w as usize && yi < h as usize {
                    let s = &mut data[yi * w as usize + xi];
                    *s = 255u8.saturating_sub((c * 255.0).round() as u8).min(*s);
                }
            });
        }
    }
    (data, w, h)
}

fn main() {
    let ocr = TemplateOcr::new().unwrap();
    let serif = std::fs::read("/System/Library/Fonts/Times.ttc")
        .or_else(|_| std::fs::read("/System/Library/Fonts/Supplemental/Georgia.ttf"))
        .unwrap();
    let sans = std::fs::read("/System/Library/Fonts/Helvetica.ttc").unwrap();
    for (name, font_data) in [("serif", &serif), ("sans", &sans)] {
        for text in [
            "cafe creme naive eleve",
            "e c",
            "ee ee",
            "Francais: cafe creme naive eleve",
            "uber gross schon Munchen",
            "perche citta uovo",
        ] {
            let (b, w, h) = render(text, font_data, 24);
            let r = ocr.recognize(&GrayImage {
                width: w as usize,
                height: h as usize,
                data: b,
            });
            let out: String = r.lines.iter().map(|l| l.text.as_str()).collect();
            println!("{name} {text:>45} -> {out:?}");
        }
    }
}
