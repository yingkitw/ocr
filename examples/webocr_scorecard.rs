//! Scorecard for the WebOCR template engine (`ocr::webocr`).
//!
//! Mirrors the sibling `../webocr` scorecard (`scripts/fixtures.ts` +
//! `scripts/score.ts`): the same plate catalogue and the same metric, so a
//! number here means the same thing as a number there.
//!
//! - **Metric** — character accuracy `1 - levenshtein / max(len(actual), len(expected))`
//!   over whitespace-collapsed strings (the industry-standard CER shape: a
//!   space inserted here and a line break there are the same kind of
//!   difference to a reader).
//! - **Faces group** — 7 faces × 5 sizes of one pangram. Measures the
//!   template library's coverage of the type it meets.
//! - **Conditions group** — the scans: light-on-dark, photocopy speckle,
//!   rotated 4°, faint ink, small dense, two columns.
//! - **Layouts group** — the reading order: two columns, a headline over a
//!   body, indented paragraphs, a table of rows, a caption between
//!   paragraphs, a wide left margin.
//!
//! Usage:
//! ```text
//! cargo run --release --example webocr_scorecard            # full card
//! cargo run --release --example webocr_scorecard -- faces   # one group
//! cargo run --release --example webocr_scorecard -- --min 0.95
//! ```
//! `--min` makes the example exit non-zero when the mean falls below the
//! gate, so it can be used as a regression check in CI.

use std::collections::HashMap;

use ocr::webocr::{GrayImage, TemplateOcr};
use rusttype::{Font, Scale};

// ── the metric ───────────────────────────────────────────────────────────────

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

/// Character accuracy: `1 - levenshtein / max(len)`.
fn accuracy(actual: &str, expected: &str) -> f32 {
    let m = levenshtein(actual, expected) as f32;
    let n = actual.chars().count().max(expected.chars().count()).max(1) as f32;
    1.0 - (m / n)
}

/// Collapse every whitespace run to a single space — the layout is not what
/// the accuracy metric is about.
fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── the plates ───────────────────────────────────────────────────────────────

const PANGRAM: &str = "The quick brown fox jumps over the 42 lazy dogs";
const SERIF: &str = "Pack my box with a dozen liquor jugs, OK?";

/// The seven faces the Latin library is built from (same set as WebOCR).
const FACES: &[(&str, &str)] = &[
    ("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"),
    (
        "Helvetica",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
    ("Georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf"),
    (
        "Times",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ),
    (
        "Courier",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ),
    ("Verdana", "/System/Library/Fonts/Supplemental/Verdana.ttf"),
    (
        "Trebuchet",
        "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    ),
];
const SIZES: &[u32] = &[16, 22, 28, 34, 40];

struct LineSpec {
    text: &'static str,
    face: &'static str,
    size: u32,
    x: Option<u32>,
    y: Option<u32>,
    /// Ink brightness (0 = black, 255 = white ink).
    ink: u8,
}

struct Plate {
    lines: Vec<LineSpec>,
    /// Background brightness.
    bg: u8,
    /// Rotation in degrees.
    rotate: Option<f32>,
    /// Number of speckle marks.
    speckle: u32,
}

impl Plate {
    fn new(lines: Vec<LineSpec>) -> Self {
        Self {
            lines,
            bg: 255,
            rotate: None,
            speckle: 0,
        }
    }
}

fn line(text: &'static str, face: &'static str, size: u32) -> LineSpec {
    LineSpec {
        text,
        face,
        size,
        x: None,
        y: None,
        ink: 17,
    }
}

/// Render a plate to a grayscale image.
fn render_plate(plate: &Plate) -> Option<GrayImage> {
    let mut fonts: HashMap<&str, Font<'static>> = HashMap::new();
    for (name, path) in FACES {
        if fonts.contains_key(*name) {
            continue;
        }
        let data = std::fs::read(path).ok()?;
        let font = Font::try_from_vec(data)?;
        fonts.insert(*name, font);
    }

    let content_w = plate
        .lines
        .iter()
        .map(|l| l.text.len() as f32 * l.size as f32 * 0.7)
        .fold(1180.0f32, f32::max)
        .ceil() as usize;
    let content_h = plate.lines.len() * 70 + 60;

    // Render the content, then rotate the whole bitmap into a larger canvas
    // (WebOCR rotates the canvas about the content's centre; the observable
    // result is the same: the text sits inside the bitmap, at an angle).
    let mut content = vec![plate.bg; content_w * content_h];
    let mut y = 54i32;
    for l in &plate.lines {
        let font = fonts.get(l.face)?;
        let scale = Scale::uniform(l.size as f32);
        let x = l.x.unwrap_or(40) as f32;
        let baseline = l.y.map(|v| v as f32).unwrap_or(y as f32);
        for g in font.layout(l.text, scale, rusttype::point(x, baseline)) {
            let Some(bb) = g.pixel_bounding_box() else {
                continue;
            };
            let x_off = bb.min.x as i32;
            let y_off = bb.min.y as i32;
            g.draw(|dx, dy, v| {
                let px = (x_off + dx as i32) as usize;
                let py = (y_off + dy as i32) as usize;
                if px < content_w && py < content_h {
                    let ink = (v * 255.0).round() as u8;
                    // Composite the coverage over the background.
                    let bg = content[py * content_w + px];
                    content[py * content_w + px] =
                        ((bg as u32 * (255 - ink as u32) + l.ink as u32 * ink as u32) / 255) as u8;
                }
            });
        }
        y += 70;
    }

    // Speckle — deterministic LCG, mirroring WebOCR's seed.
    if plate.speckle > 0 {
        let mut seed: u32 = 20261024;
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..plate.speckle {
            let x = (rand() * content_w as f32) as usize;
            let yy = (rand() * content_h as f32) as usize;
            let w = 1 + (rand() * 2.0) as usize;
            let h = 1 + (rand() * 2.0) as usize;
            let dark = rand() < 0.5;
            for dy in 0..h {
                for dx in 0..w {
                    let px = x + dx;
                    let py = yy + dy;
                    if px < content_w && py < content_h {
                        let c = &mut content[py * content_w + px];
                        *c = if dark { c.saturating_sub(90) } else { c.saturating_add(90) };
                    }
                }
            }
        }
    }

    // Rotate about the content's centre, into a grown canvas.
    let (width, height, data) = match plate.rotate {
        None => (content_w, content_h, content),
        Some(deg) => {
            let rad = deg.to_radians();
            let (sin, cos) = (rad.sin().abs(), rad.cos().abs());
            let w = (content_w as f32 * cos + content_h as f32 * sin).ceil() as usize;
            let h = (content_h as f32 * cos + content_w as f32 * sin).ceil() as usize;
            let mut out = vec![plate.bg; w * h];
            let ox = (w as f32 - content_w as f32) / 2.0;
            let oy = (h as f32 - content_h as f32) / 2.0;
            let cx = content_w as f32 / 2.0;
            let cy = content_h as f32 / 2.0;
            let (s, c) = (rad.sin(), rad.cos());
            for y in 0..content_h {
                for x in 0..content_w {
                    // Inverse-rotate the destination pixel into the source.
                    let dx = x as f32 - ox - cx;
                    let dy = y as f32 - oy - cy;
                    let sx = (dx * c + dy * s + cx).round() as isize;
                    let sy = (-dx * s + dy * c + cy).round() as isize;
                    if sx >= 0 && sy >= 0 && (sx as usize) < content_w && (sy as usize) < content_h {
                        out[y * w + x] = content[sy as usize * content_w + sx as usize];
                    }
                }
            }
            (w, h, out)
        }
    };

    Some(GrayImage {
        width,
        height,
        data,
    })
}

// ── the catalogue ────────────────────────────────────────────────────────────

struct Cell {
    group: &'static str,
    name: String,
    plate: Plate,
    expected: String,
}

fn catalogue() -> Vec<Cell> {
    let mut cells: Vec<Cell> = Vec::new();

    // Faces group: 7 faces × 5 sizes.
    for (face_name, _) in FACES {
        for &size in SIZES {
            cells.push(Cell {
                group: "faces",
                name: format!("{face_name} {size}px"),
                plate: Plate::new(vec![LineSpec {
                    text: PANGRAM,
                    face: face_name,
                    size,
                    x: None,
                    y: None,
                    ink: 17,
                }]),
                expected: PANGRAM.to_string(),
            });
        }
    }

    // Conditions group: the scans.
    let arial = "Arial";
    cells.push(Cell {
        group: "conditions",
        name: "plain".into(),
        plate: Plate::new(vec![line(PANGRAM, arial, 30)]),
        expected: PANGRAM.into(),
    });
    cells.push(Cell {
        group: "conditions",
        name: "light on dark".into(),
        plate: Plate {
            lines: vec![LineSpec {
                text: "Dark mode scan 42",
                face: arial,
                size: 34,
                x: None,
                y: None,
                ink: 242,
            }],
            bg: 20,
            rotate: None,
            speckle: 0,
        },
        expected: "Dark mode scan 42".into(),
    });
    cells.push(Cell {
        group: "conditions",
        name: "photocopy speckle".into(),
        plate: Plate {
            lines: vec![
                line(PANGRAM, "Arial", 30),
                line(SERIF, "Georgia", 30),
                line("raw pixels in, clean text out (v1.0).", "Times", 30),
                line("SEGMENT, NORMALIZE, MATCH -- then done.", "Courier", 30),
                line("OCR from scratch: no ML, no Tesseract.", "Verdana", 30),
            ],
            bg: 255,
            rotate: None,
            speckle: 2600,
        },
        expected: [
            PANGRAM,
            SERIF,
            "raw pixels in, clean text out (v1.0).",
            "SEGMENT, NORMALIZE, MATCH -- then done.",
            "OCR from scratch: no ML, no Tesseract.",
        ]
        .join(" "),
    });
    cells.push(Cell {
        group: "conditions",
        name: "rotated 4deg".into(),
        plate: Plate {
            lines: vec![line(PANGRAM, arial, 30)],
            bg: 255,
            rotate: Some(4.0),
            speckle: 0,
        },
        expected: PANGRAM.into(),
    });
    cells.push(Cell {
        group: "conditions",
        name: "speckle on sharp type".into(),
        plate: Plate {
            lines: vec![line(PANGRAM, arial, 30)],
            bg: 255,
            rotate: None,
            speckle: 900,
        },
        expected: PANGRAM.into(),
    });
    cells.push(Cell {
        group: "conditions",
        name: "faint ink".into(),
        plate: Plate {
            lines: vec![LineSpec {
                text: PANGRAM,
                face: arial,
                size: 30,
                x: None,
                y: None,
                ink: 182,
            }],
            bg: 244,
            rotate: None,
            speckle: 0,
        },
        expected: PANGRAM.into(),
    });
    let dense_line = "The quick brown fox jumps over the lazy dog while the pipeline measures every glyph.";
    cells.push(Cell {
        group: "conditions",
        name: "small dense".into(),
        plate: Plate::new(
            (0..12)
                .map(|_| line(dense_line, arial, 16))
                .collect(),
        ),
        // The plate holds twelve identical lines — the expectation is all
        // twelve, in order (WebOCR's default: the plate's lines joined).
        expected: vec![dense_line; 12].join(" "),
    });
    cells.push(Cell {
        group: "conditions",
        name: "two columns".into(),
        plate: Plate::new(vec![
            line("Column one reads first.", arial, 26),
            line("Blocks are cut where the", arial, 26),
            line("gutter is, not by a fixed", arial, 26),
            line("column count.", arial, 26),
            line("Column two follows.", arial, 26),
            line("Reading order is a", arial, 26),
            line("property of the block", arial, 26),
            line("tree, not of the lines.", arial, 26),
        ]),
        expected: "Column one reads first. Blocks are cut where the gutter is, not by a fixed column count. Column two follows. Reading order is a property of the block tree, not of the lines."
            .into(),
    });

    // Layouts group: the reading order.
    let l = |text: &'static str, face: &'static str, size: u32, x: u32, y: u32| LineSpec {
        text,
        face,
        size,
        x: Some(x),
        y: Some(y),
        ink: 17,
    };
    cells.push(Cell {
        group: "layouts",
        name: "two columns side by side".into(),
        plate: Plate::new(vec![
            l("The left column begins here.", arial, 28, 40, 60),
            l("It continues for a while.", arial, 28, 40, 100),
            l("And ends with this line.", arial, 28, 40, 140),
            l("The right column starts beside it.", arial, 28, 520, 60),
            l("Reading order should follow the", arial, 28, 520, 100),
            l("left column down first.", arial, 28, 520, 140),
        ]),
        expected: "The left column begins here. It continues for a while. And ends with this line. The right column starts beside it. Reading order should follow the left column down first."
            .into(),
    });
    cells.push(Cell {
        group: "layouts",
        name: "a headline over a body".into(),
        plate: Plate::new(vec![
            l("The Big Headline", arial, 44, 40, 70),
            l(
                "The body text under it is set in a smaller size.",
                arial,
                28,
                40,
                120,
            ),
            l("And it continues on the next line as usual.", arial, 28, 40, 160),
        ]),
        expected: "The Big Headline The body text under it is set in a smaller size. And it continues on the next line as usual."
            .into(),
    });
    cells.push(Cell {
        group: "layouts",
        name: "indented paragraphs".into(),
        plate: Plate::new(vec![
            l("The first paragraph starts at the margin.", arial, 28, 40, 60),
            l("The second paragraph is indented.", arial, 28, 90, 110),
            l("The third one is indented further.", arial, 28, 140, 160),
        ]),
        expected: "The first paragraph starts at the margin. The second paragraph is indented. The third one is indented further."
            .into(),
    });
    cells.push(Cell {
        group: "layouts",
        name: "a table of rows".into(),
        plate: Plate::new(vec![
            l("Name   Qty   Price", arial, 28, 40, 60),
            l("Apple   3   1.50", arial, 28, 40, 100),
            l("Pear    5   2.25", arial, 28, 40, 140),
            l("Plum   11   0.99", arial, 28, 40, 180),
        ]),
        expected: "Name Qty Price Apple 3 1.50 Pear 5 2.25 Plum 11 0.99".into(),
    });
    cells.push(Cell {
        group: "layouts",
        name: "a caption between the paragraphs".into(),
        plate: Plate::new(vec![
            l("The body text starts above.", arial, 28, 40, 60),
            l("Figure 1. A caption under a figure.", arial, 24, 90, 110),
            l("The body text resumes below the figure.", arial, 28, 40, 160),
        ]),
        expected: "The body text starts above. Figure 1. A caption under a figure. The body text resumes below the figure."
            .into(),
    });
    cells.push(Cell {
        group: "layouts",
        name: "a wide left margin".into(),
        plate: Plate::new(vec![
            l("Indented by a wide margin, as a scan", arial, 28, 260, 60),
            l("of a book page would be.", arial, 28, 260, 100),
        ]),
        expected: "Indented by a wide margin, as a scan of a book page would be.".into(),
    });

    cells
}

// ── the run ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let group_filter = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "all".to_string());
    let min: Option<f32> = args
        .iter()
        .position(|a| a == "--min")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());
    let show = args.iter().any(|a| a == "--show");

    let ocr = match TemplateOcr::new() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("template engine unavailable: {e}");
            std::process::exit(2);
        }
    };
    println!("library: {} templates\n", ocr.template_count());

    let cells = catalogue();
    let mut by_group: Vec<(&str, Vec<(String, f32)>)> = Vec::new();
    let mut all: Vec<(String, f32)> = Vec::new();

    for cell in cells {
        if group_filter != "all" && group_filter != cell.group {
            continue;
        }
        let Some(image) = render_plate(&cell.plate) else {
            eprintln!("skipping {}: render failed", cell.name);
            continue;
        };
        let result = ocr.recognize(&image);
        let actual = collapse(&result.text);
        let expected = collapse(&cell.expected);
        let acc = accuracy(&actual, &expected);
        if show {
            println!("  [{}] {}", cell.name, cell.group);
            println!("    expected: {expected}");
            println!("    actual  : {actual}");
        }
        all.push((format!("{} / {}", cell.group, cell.name), acc));
        match by_group.iter_mut().find(|(g, _)| *g == cell.group) {
            Some((_, list)) => list.push((cell.name.clone(), acc)),
            None => by_group.push((cell.group, vec![(cell.name.clone(), acc)])),
        }
    }

    for (group, list) in &by_group {
        println!("── {group} ──");
        for (name, acc) in list {
            let bar = "█".repeat((acc * 20.0).round() as usize);
            println!("  {:<28} {:>6.2}%  {bar}", name, acc * 100.0);
        }
        let mean = list.iter().map(|(_, a)| *a).sum::<f32>() / list.len() as f32;
        let perfect = list.iter().filter(|(_, a)| *a >= 0.9999).count();
        println!("  {:<28} {:>6.2}%  ({perfect}/{} perfect)\n", "MEAN", mean * 100.0, list.len());
    }

    let mean = all.iter().map(|(_, a)| *a).sum::<f32>() / all.len().max(1) as f32;
    println!("OVERALL: {:.2}% over {} cells", mean * 100.0, all.len());

    if let Some(gate) = min
        && mean < gate
    {
        eprintln!("\nFAIL: mean {:.4} below gate {:.4}", mean, gate);
        std::process::exit(1);
    }
}
