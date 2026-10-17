//! Geometry-aware plain-text assembly (inspired by WebOCR `layout` output).
//!
//! Rebuilds `TextResult.text` from line/word boxes: paragraph breaks from vertical
//! gap, leading indent from horizontal offset, and wide intra-line gaps as spaces.

use crate::core::text::{LineResult, TextResult, WordResult};

/// Reassemble plain text from structured lines (WebOCR-style layout retention).
pub fn reassemble_plain_text(result: &TextResult, cjk: bool) -> String {
    if result.lines.is_empty() {
        return result.text.clone();
    }

    let mut lines: Vec<&LineResult> = result.lines.iter().collect();
    lines.sort_by(|a, b| {
        a.bounding_box
            .top
            .cmp(&b.bounding_box.top)
            .then(a.bounding_box.left.cmp(&b.bounding_box.left))
    });

    let line_heights: Vec<f32> = lines
        .iter()
        .map(|l| (l.bounding_box.bottom.saturating_sub(l.bounding_box.top)) as f32)
        .filter(|h| *h > 0.0)
        .collect();
    let med_line_h = median(&line_heights).unwrap_or(12.0).max(1.0);

    let doc_left = lines
        .iter()
        .map(|l| line_start_x(l))
        .min()
        .unwrap_or(0);

    let mut out = String::new();
    let mut prev_bottom: Option<u32> = None;

    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            if let Some(prev_b) = prev_bottom {
                let gap = line.bounding_box.top.saturating_sub(prev_b) as f32;
                if gap > 1.75 * med_line_h {
                    out.push_str("\n\n");
                } else {
                    out.push('\n');
                }
            } else {
                out.push('\n');
            }
        }

        let pitch = line_pitch(line, med_line_h, cjk);
        let indent_cells = ((line_start_x(line).saturating_sub(doc_left)) as f32 / pitch)
            .round()
            .max(0.0) as usize;
        if indent_cells > 0 {
            if cjk {
                out.extend(std::iter::repeat('　').take(indent_cells));
            } else {
                out.extend(std::iter::repeat(' ').take(indent_cells));
            }
        }

        out.push_str(&words_with_gaps(&line.words, pitch, cjk));
        prev_bottom = Some(line.bounding_box.bottom);
    }

    out
}

fn line_start_x(line: &LineResult) -> u32 {
    if !line.words.is_empty() {
        line.words
            .iter()
            .map(|w| w.bounding_box.left)
            .min()
            .unwrap_or(line.bounding_box.left)
    } else {
        line.bounding_box.left
    }
}

fn line_pitch(line: &LineResult, med_line_h: f32, cjk: bool) -> f32 {
    let mut words: Vec<&WordResult> = line.words.iter().collect();
    words.sort_by_key(|w| w.bounding_box.left);
    let gaps: Vec<f32> = words
        .windows(2)
        .map(|pair| {
            pair[1]
                .bounding_box
                .left
                .saturating_sub(pair[0].bounding_box.right) as f32
        })
        .filter(|g| *g > 0.5)
        .collect();
    median(&gaps).unwrap_or(med_line_h * if cjk { 1.0 } else { 0.5 })
        .max(1.0)
}

fn words_with_gaps(words: &[WordResult], pitch: f32, cjk: bool) -> String {
    if words.is_empty() {
        return String::new();
    }
    let mut sorted: Vec<&WordResult> = words.iter().collect();
    sorted.sort_by_key(|w| w.bounding_box.left);

    let mut out = sorted[0].text.clone();
    for pair in sorted.windows(2) {
        let gap = pair[1]
            .bounding_box
            .left
            .saturating_sub(pair[0].bounding_box.right) as f32;
        let n = if cjk {
            0
        } else {
            (gap / pitch).round().max(1.0) as usize
        };
        if n > 0 {
            out.extend(std::iter::repeat(' ').take(n));
        }
        out.push_str(&pair[1].text);
    }
    out
}

fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v: Vec<f32> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

fn is_cjk_language(lang: &str) -> bool {
    matches!(
        lang.to_lowercase().as_str(),
        "zh" | "ja" | "ko" | "cjk" | "zh-cn" | "zh-tw"
    )
}

/// Whether layout reassembly should use CJK spacing rules for this result.
pub fn use_cjk_spacing(result: &TextResult) -> bool {
    if let Some(ref lang) = result.language {
        if is_cjk_language(lang) {
            return true;
        }
    }
    result.text.chars().any(|c| {
        let u = c as u32;
        (0x4E00..=0x9FFF).contains(&u)
            || (0x3040..=0x30FF).contains(&u)
            || (0xAC00..=0xD7AF).contains(&u)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::text::BoundingBox;

    fn bbox(left: u32, top: u32, right: u32, bottom: u32) -> BoundingBox {
        BoundingBox::new(left, top, right, bottom)
    }

    #[test]
    fn paragraph_gap_inserts_blank_line() {
        let mut result = TextResult::new(String::new(), 1.0, bbox(0, 0, 0, 0));
        result.lines.push({
            let mut line = crate::core::text::LineResult::new("Line one".into(), 1.0, bbox(10, 10, 90, 30));
            line.words.push(crate::core::text::WordResult::new(
                "Line".into(),
                1.0,
                bbox(10, 10, 40, 30),
            ));
            line.words.push(crate::core::text::WordResult::new(
                "one".into(),
                1.0,
                bbox(50, 10, 90, 30),
            ));
            line
        });
        result.lines.push({
            let line = crate::core::text::LineResult::new("Line two".into(), 1.0, bbox(10, 80, 90, 100));
            let mut line = line;
            line.words.push(crate::core::text::WordResult::new(
                "Line two".into(),
                1.0,
                bbox(10, 80, 90, 100),
            ));
            line
        });

        let text = reassemble_plain_text(&result, false);
        assert!(
            text.contains("\n\n"),
            "expected paragraph break, got {:?}",
            text
        );
    }

    #[test]
    fn indent_preserves_leading_offset() {
        let mut result = TextResult::new(String::new(), 1.0, bbox(0, 0, 0, 0));
        result.lines.push({
            let line = crate::core::text::LineResult::new("Left".into(), 1.0, bbox(10, 10, 50, 30));
            let mut line = line;
            line.words.push(crate::core::text::WordResult::new(
                "Left".into(),
                1.0,
                bbox(10, 10, 50, 30),
            ));
            line
        });
        result.lines.push({
            let line = crate::core::text::LineResult::new("Indented".into(), 1.0, bbox(60, 40, 140, 60));
            let mut line = line;
            line.words.push(crate::core::text::WordResult::new(
                "Indented".into(),
                1.0,
                bbox(60, 40, 140, 60),
            ));
            line
        });

        let text = reassemble_plain_text(&result, false);
        let second = text.lines().nth(1).unwrap_or("");
        assert!(
            second.starts_with(' '),
            "expected leading spaces on indented line, got {:?}",
            second
        );
    }
}
