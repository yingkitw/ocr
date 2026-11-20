use ocr::webocr::{GrayImage, TemplateOcr};
use std::sync::OnceLock;

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

fn accuracy(actual: &str, expected: &str) -> f32 {
    1.0 - (levenshtein(actual, expected) as f32 / expected.chars().count().max(1) as f32)
}

fn render_lines(lines: &[&str], font_data: &[u8], px: u32, line_gap: u32) -> Option<(Vec<u8>, u32, u32)> {
    use rusttype::{Font, Scale};
    let font = Font::try_from_bytes(font_data)?;
    let scale = Scale::uniform(px as f32);
    let pad = px / 2;
    let v = font.v_metrics(scale);
    let text_h = (v.ascent - v.descent).ceil() as u32;
    let mut width = 0u32;
    for line in lines {
        let mut w = 0u32;
        for g in font.layout(line, scale, rusttype::point(0.0, 0.0)) {
            if let Some(bb) = g.pixel_bounding_box() {
                w = w.max(bb.max.x as u32);
            }
        }
        width = width.max(w);
    }
    let w = width + 2 * pad;
    let h = text_h * lines.len() as u32 + line_gap * (lines.len() as u32 - 1).max(0) + 2 * pad;
    let mut data = vec![255u8; (w * h) as usize];
    for (li, line) in lines.iter().enumerate() {
        let y0 = pad + li as u32 * (text_h + line_gap);
        for g in font.layout(line, scale, rusttype::point(pad as f32, y0 as f32 + v.ascent)) {
            if let Some(bb) = g.pixel_bounding_box() {
                let (x_off, y_off) = (bb.min.x, bb.min.y);
                g.draw(|x, y, cov| {
                    let xi = (x_off + x as i32) as usize;
                    let yi = (y_off + y as i32) as usize;
                    if xi < w as usize && yi < h as usize {
                        let slot = &mut data[yi * w as usize + xi];
                        *slot = 255u8.saturating_sub((cov * 255.0).round() as u8).min(*slot);
                    }
                });
            }
        }
    }
    Some((data, w, h))
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn speckle(buf: &mut [u8], prob: f32, seed: u64) {
    let mut s = seed.max(1);
    for p in buf.iter_mut() {
        let roll = (xorshift(&mut s) % 10000) as f32 / 10000.0;
        if roll < prob / 2.0 {
            *p = 0;
        } else if roll < prob {
            *p = 255;
        }
    }
}

fn box_blur3(buf: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0u32;
            let mut n = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xi, yi) = (x as i32 + dx, y as i32 + dy);
                    if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                        sum += buf[yi as usize * w + xi as usize] as u32;
                        n += 1;
                    }
                }
            }
            out[y * w + x] = (sum / n) as u8;
        }
    }
    out
}

fn rotate(buf: &[u8], w: usize, h: usize, deg: f32) -> (Vec<u8>, usize, usize) {
    let a = deg.to_radians();
    let (sin, cos) = a.sin_cos();
    let (sw, sh) = (w as f32, h as f32);
    let nw = (sw * cos + sh * sin).ceil() as usize;
    let nh = (sh * cos + sw * sin).ceil() as usize;
    let mut out = vec![255u8; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let dx = x as f32 - nw as f32 / 2.0;
            let dy = y as f32 - nh as f32 / 2.0;
            let sx = (dx * cos + dy * sin + sw / 2.0).round() as isize;
            let sy = (-dx * sin + dy * cos + sh / 2.0).round() as isize;
            if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                out[y * nw + x] = buf[sy as usize * w + sx as usize];
            }
        }
    }
    (out, nw, nh)
}

fn fade(buf: &mut [u8], keep: f32) {
    for p in buf.iter_mut() {
        *p = 255u8.saturating_sub(((255 - *p) as f32 * keep).round() as u8);
    }
}

fn invert(buf: &mut [u8]) {
    for p in buf.iter_mut() {
        *p = 255 - *p;
    }
}

fn gradient(buf: &mut [u8], w: usize, h: usize, dark: u8) {
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / w.max(1) as f32;
            let bg = dark as f32 + (255.0 - dark as f32) * t;
            let i = y * w + x;
            buf[i] = buf[i].min(bg.round() as u8);
        }
    }
}

fn downscale(buf: &[u8], w: usize, h: usize, factor: usize) -> Vec<u8> {
    let (nw, nh) = (w / factor, h / factor);
    let mut small = vec![255u8; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    sum += buf[(y * factor + dy) * w + x * factor + dx] as u32;
                }
            }
            small[y * nw + x] = (sum / (factor * factor) as u32) as u8;
        }
    }
    small
}

fn font(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn run(ocr: &TemplateOcr, label: &str, buf: &[u8], w: usize, h: usize, expected: &str) {
    let gray = GrayImage {
        width: w,
        height: h,
        data: buf.to_vec(),
    };
    let r = ocr.recognize(&gray);
    let actual: String = r
        .lines
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let acc = accuracy(actual.trim(), expected);
    println!(
        "{:40} acc={:.3}  {:?}",
        label,
        acc,
        if acc < 0.999 { actual.as_str() } else { "" }
    );
}

static OCR: OnceLock<TemplateOcr> = OnceLock::new();

fn main() {
    let ocr = OCR.get_or_init(|| TemplateOcr::new().unwrap());

    let sans = font("/System/Library/Fonts/Helvetica.ttc").unwrap();
    let serif = font("/System/Library/Fonts/Supplemental/Georgia.ttf").unwrap();
    let mono = font("/System/Library/Fonts/Supplemental/Courier New.ttf").unwrap();

    let pangram = "The quick brown fox jumps over the 42 lazy dogs";

    // Font matrix @30px
    for (name, f) in [("sans", &sans), ("serif", &serif), ("mono", &mono)] {
        let (b, w, h) = render_lines(&[pangram], f, 30, 12).unwrap();
        run(ocr, &format!("{name} pangram 30px"), &b, w as usize, h as usize, pangram);
    }

    // Size ladder (sans)
    for px in [16u32, 20, 24, 30, 40, 54] {
        let (b, w, h) = render_lines(&[pangram], &sans, px, 12).unwrap();
        run(ocr, &format!("sans pangram {px}px"), &b, w as usize, h as usize, pangram);
    }

    // Content variety (sans 30px)
    for text in [
        "ALL CAPS HEADLINE 2026",
        "Order #1042: $3.50 (incl. 8% tax)",
        "Mixed Case Words With Numbers 99",
        "INV-2026-0342 XKCD-79 alpha",
        "e.g. i.e. etc. vs. Dr. Mr.",
    ] {
        let (b, w, h) = render_lines(&[text], &sans, 30, 12).unwrap();
        run(ocr, &format!("sans content: {text}"), &b, w as usize, h as usize, text);
    }

    // Multi-line paragraph
    let para = ["The quick brown fox", "jumps over the lazy", "dogs in the park."];
    let joined = para.join("\n");
    let (b, w, h) = render_lines(&para, &sans, 30, 24).unwrap();
    run(ocr, "sans paragraph 3-line", &b, w as usize, h as usize, &joined);

    let para2 = ["Short.", "A much longer second line of text here", "Mid-length line"];
    let joined2 = para2.join("\n");
    let (b, w, h) = render_lines(&para2, &sans, 30, 24).unwrap();
    run(ocr, "sans paragraph varied", &b, w as usize, h as usize, &joined2);

    // Degradations on the sans pangram @30
    let (b0, w0, h0) = render_lines(&[pangram], &sans, 30, 12).unwrap();
    let (w0, h0) = (w0 as usize, h0 as usize);

    let mut n1 = b0.clone();
    speckle(&mut n1, 0.01, 42);
    run(ocr, "speckle 1%", &n1, w0, h0, pangram);

    let mut n3 = b0.clone();
    speckle(&mut n3, 0.03, 42);
    run(ocr, "speckle 3%", &n3, w0, h0, pangram);

    let bl = box_blur3(&b0, w0, h0);
    run(ocr, "box blur 3x3", &bl, w0, h0, pangram);

    for deg in [2.0f32, 5.0] {
        let (r, rw, rh) = rotate(&b0, w0, h0, deg);
        run(ocr, &format!("skew {deg}deg"), &r, rw, rh, pangram);
    }

    let mut lc = b0.clone();
    fade(&mut lc, 0.5);
    run(ocr, "low contrast 50%", &lc, w0, h0, pangram);

    let mut inv = b0.clone();
    invert(&mut inv);
    run(ocr, "inverted", &inv, w0, h0, pangram);

    let mut gr = b0.clone();
    gradient(&mut gr, w0, h0, 170);
    run(ocr, "gradient bg 170", &gr, w0, h0, pangram);

    let ds = downscale(&b0, w0, h0, 2);
    run(ocr, "downscale 2x", &ds, w0 / 2, h0 / 2, pangram);
}
