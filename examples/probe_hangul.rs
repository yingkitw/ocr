use ocr::webocr::{GrayImage, RustTypeRenderer};

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
    let (w, h) = (width + 2 * pad, text_h + 2 * pad);
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
    let hangul_font = std::fs::read("/System/Library/Fonts/AppleSDGothicNeo.ttc").unwrap();
    let (b, w, h) = render("한국어", &hangul_font, 48);

    // Small library: just the three hangul chars + a few hanzi distractors.
    let renderer = RustTypeRenderer::for_cjk();
    let charset = "한국어中文字测试";
    let templates = ocr::webocr::templates::build_templates_for_options(
        &renderer,
        charset,
        ocr::webocr::templates::CJK_SIZES,
        true,
        true,
    );
    eprintln!("small lib: {} templates", templates.len());
    let r = ocr::webocr::run_ocr(
        &GrayImage {
            width: w as usize,
            height: h as usize,
            data: b.clone(),
        },
        &templates,
        ocr::webocr::RunOptions {
            cjk: true,
            ..Default::default()
        },
    );
    for line in &r.lines {
        let chars: Vec<String> = line
            .chars
            .iter()
            .map(|c| format!("{}:{:.3}", c.char, c.score))
            .collect();
        println!("small-lib -> {:?} | {}", line.text, chars.join(" "));
    }
}
