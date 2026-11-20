//! Render a short CJK string to a PNG for exercising the searchable-PDF
//! text layer end-to-end (`ocr extract img.png --engine template -f pdf`).

fn main() {
    let candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ];
    let font_data = candidates
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .expect("no CJK font found");

    use rusttype::{Font, Scale};
    let font = Font::try_from_bytes(&font_data).unwrap();
    let text = "中文字测试";
    let px = 48.0f32;
    let scale = Scale::uniform(px);
    let v = font.v_metrics(scale);
    let mut w = 0u32;
    for g in font.layout(text, scale, rusttype::point(0.0, 0.0)) {
        if let Some(bb) = g.pixel_bounding_box() {
            w = w.max(bb.max.x as u32);
        }
    }
    let (w, h) = (w + 40, 100u32);
    let mut img = image::GrayImage::from_pixel(w, h, image::Luma([255u8]));
    for g in font.layout(text, scale, rusttype::point(20.0, 20.0 + v.ascent)) {
        if let Some(bb) = g.pixel_bounding_box() {
            let (ox, oy) = (bb.min.x, bb.min.y);
            g.draw(|x, y, cov| {
                let (xi, yi) = (ox + x as i32, oy + y as i32);
                if xi >= 0 && yi >= 0 && xi < w as i32 && yi < h as i32 {
                    let p = img.get_pixel_mut(xi as u32, yi as u32);
                    p[0] = 255u8.saturating_sub((cov * 255.0) as u8).min(p[0]);
                }
            });
        }
    }
    img.save("/tmp/cjk_test.png").unwrap();
    println!("wrote /tmp/cjk_test.png ({}x{})", w, h);
}
