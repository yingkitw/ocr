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
    let han_kana = std::fs::read("/System/Library/Fonts/Hiragino Sans GB.ttc").unwrap();
    let hangul = std::fs::read("/System/Library/Fonts/AppleSDGothicNeo.ttc").unwrap();

    eprintln!("building CJK union library...");
    let t0 = std::time::Instant::now();
    let ocr = TemplateOcr::with_system_cjk().unwrap();
    eprintln!(
        "library: {} templates in {:?}",
        ocr.template_count(),
        t0.elapsed()
    );

    for (text, font_data) in [
        ("中", &han_kana),
        ("中文字", &han_kana),
        ("中文字测试", &han_kana),
        ("こんにちは", &han_kana),
        ("한국어", &hangul),
        ("中文测试 日本語", &han_kana),
    ] {
        let (b, w, h) = render(text, font_data, 48);
        let r = ocr.recognize(&GrayImage {
            width: w as usize,
            height: h as usize,
            data: b,
        });
        for line in &r.lines {
            let chars: Vec<String> = line
                .chars
                .iter()
                .map(|c| format!("{}:{:.2}", c.char, c.score))
                .collect();
            println!("{text:>12} -> {:?} | {}", line.text, chars.join(" "));
        }
        if r.lines.is_empty() {
            println!("{text:>12} -> <no lines>");
        }
    }
}
