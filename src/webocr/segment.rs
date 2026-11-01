//! Segmentation: components → glyph boxes → ordered lines and blocks
//! (WebOCR port of `webocr/src/ocr/segment.ts`).
//!
//! Includes the measured-in-webocr guards: valley splitting with arch
//! protection, stacked-fragment merging before AND after line grouping,
//! CJK stroke clustering with a width cap, row-segment/block detection via
//! union-find, and gap-cluster word-space detection (no fixed multiplier —
//! Courier's letter gaps exceed Georgia's word gaps at similar text heights).

use super::components::{connected_components, median};
use super::types::{BinaryImage, Comp, LineData};

/// Drop specks that cannot plausibly be glyphs (dust, sensor noise).
///
/// A speck is small in BOTH directions; a STEM is small in one. The "l" of a
/// 16px Arial is one pixel wide and fourteen tall — a width-only rule drops
/// it, and with it every "i", every "l" and every "t" of a 16px line
/// (WebOCR measured: "lazy" came back as "azy", "quick" as "qu ck", "while"
/// as "wh e"; the dense 16px cell read 87.1 %). A one-pixel-wide component
/// at least four pixels tall is a letter; a speck is not.
pub fn filter_noise(comps: &[Comp]) -> Vec<Comp> {
    comps
        .iter()
        .filter(|c| {
            c.area >= 3 && (c.width() >= 2 || c.height() >= 4) && c.height() >= 2
        })
        .copied()
        .collect()
}

/// Drop components that aren't text — page furniture like logos, horizontal
/// rules, and decorative blobs. Three shape heuristics, each firing on a
/// different non-text signature:
///
/// 1. **Solid blob**: density (area / bbox) > 0.85 AND min(w, h) > a
///    row-supplied scale floor. Real text comps at high density are limited
///    to thin stems (vertical bars, periods) whose narrow dimension is
///    tiny; filled logos and solid rectangles are square-ish and big.
/// 2. **Page-spanning rule**: density > 0.7 AND max(w, h) > 5·medH AND
///    aspect > 8 (15 in CJK). A horizontal rule across the page has density
///    ~1, a width much greater than 5× the text height, and a high aspect.
///    A thin letter `l` (2×22) has high aspect too, but maxSide=22 is well
///    below 5×medH — the rule fires on rules, not on letters.
/// 3. **Giant**: handled separately by [`filter_giant_in_row`], which scales
///    per row.
///
/// `med_h` is the median glyph height of the comps being filtered. It MUST
/// come from the comps themselves (the median of the heights), which is
/// robust to a single logo outlier.
pub fn filter_non_text(comps: &[Comp], med_h: Option<usize>, cjk: bool) -> Vec<Comp> {
    if comps.is_empty() {
        return comps.to_vec();
    }
    let reference = med_h.unwrap_or_else(|| {
        median(&comps.iter().map(|c| c.height()).collect::<Vec<_>>())
    });
    let reference = reference.max(1);
    let solid_floor = (4.0f32).max(0.35 * reference as f32);
    let rule_min_aspect = if cjk { 15.0 } else { 8.0 };
    let rule_min_max_side = 5.0 * reference as f32;
    comps
        .iter()
        .copied()
        .filter(|c| {
            let w = c.width() as f32;
            let h = c.height() as f32;
            let bbox = w * h;
            if bbox <= 0.0 {
                return false;
            }
            let density = c.area as f32 / bbox;
            let min_side = w.min(h);
            let max_side = w.max(h);
            if density > 0.85 && min_side > solid_floor {
                return false;
            }
            if density > 0.7
                && max_side > rule_min_max_side
                && max_side / min_side > rule_min_aspect
            {
                return false;
            }
            true
        })
        .collect()
}

/// Drop the "giant" components of a row — the ones too big to be text at
/// this row's scale (a logo, a decorative block).
///
/// The scale is the ROW's median height, not the page's: on a mixed-size
/// page (a 46px headline over 18px body text) the page median is the
/// body's, and the headline's glyphs then measured 2.5× the body and were
/// dropped as decoration. A row scales itself, so the headline survives.
///
/// A row with a single component keeps the page scale: that is the
/// lone-logo case, where the row's own median IS the logo and no
/// self-referential threshold would ever fire.
pub fn filter_giant_in_row(row: &[Comp], page_med_h: usize, cjk: bool) -> Vec<Comp> {
    let giant_max = if cjk { 3.0 } else { 2.5 };
    let scale = if row.len() < 2 {
        page_med_h
    } else {
        median(&row.iter().map(|c| c.height()).collect::<Vec<_>>()).max(1)
    };
    row.iter()
        .copied()
        .filter(|c| c.width().max(c.height()) as f32 <= giant_max * scale as f32)
        .collect()
}

fn column_ink_counts(bin: &BinaryImage, c: &Comp) -> (Vec<u32>, u32) {
    let mut counts = vec![0u32; c.width()];
    let mut peak = 0u32;
    for y in c.y0..c.y1 {
        for x in c.x0..c.x1 {
            if bin.get(x, y) == 1 {
                let i = x - c.x0;
                counts[i] += 1;
                if counts[i] > peak {
                    peak = counts[i];
                }
            }
        }
    }
    (counts, peak)
}

/// Options for [`split_candidates`] (WebOCR's options object).
#[derive(Debug, Clone, Copy)]
pub struct SplitOptions {
    /// Cap valley darkness relative to the peak column.
    pub max_ratio: f32,
    /// Reject any cut whose valley column carries ink in the top 40 % (an
    /// arch — "m", "W").
    pub check_bridge: bool,
    /// Minimum width of each half, as a fraction of `med_h`.
    pub min_part: f32,
    /// Pass-1 semantics: only the single lightest valley is eligible.
    pub strict: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            max_ratio: 0.45,
            check_bridge: true,
            min_part: 0.25,
            strict: false,
        }
    }
}

/// Candidate splits for a wide glyph — central-60 % valley columns, lightest
/// ink first.
///
/// The confidence-gated rescue pass in [`super::engine`] relaxes both the
/// ratio and the bridge check and scores several candidates: its real guard
/// is that both halves must outscore the fused match, so an off-seam local
/// minimum can't win just by being the lightest column.
pub fn split_candidates(
    bin: &BinaryImage,
    c: &Comp,
    med_h: usize,
    opts: SplitOptions,
) -> Vec<[Comp; 2]> {
    let w = c.width();
    let h = c.height();
    let (counts, peak) = column_ink_counts(bin, c);

    let mut order: Vec<usize> =
        ((w * 2 / 10)..((w * 8).div_ceil(10)).min(w)).collect();
    order.sort_by_key(|&i| counts[i]);

    // Pass-1 semantics: only the single lightest valley is eligible — if its
    // guards fail the glyph stays whole (the arch check must protect
    // "m"/"W"; falling through to the next valley would over-split them).
    let cols: &[usize] = if opts.strict {
        &order[..order.len().min(1)]
    } else {
        &order
    };
    let mut out = Vec::new();
    let top_end = c.y0 + h * 2 / 5;
    for &i in cols {
        if (counts[i] as f32) > opts.max_ratio * peak as f32 {
            break;
        }
        if opts.check_bridge {
            let vx = c.x0 + i;
            let bridged = (c.y0..top_end).any(|y| bin.get(vx, y) == 1);
            if bridged {
                continue;
            }
        }
        let right_w = w - i - 1;
        if (i as f32) < opts.min_part * med_h as f32
            || (right_w as f32) < opts.min_part * med_h as f32
        {
            continue;
        }
        out.push([
            Comp { x0: c.x0, y0: c.y0, x1: c.x0 + i + 1, y1: c.y1, area: 0 },
            Comp { x0: c.x0 + i + 1, y0: c.y0, x1: c.x1, y1: c.y1, area: 0 },
        ]);
    }
    out
}

/// Split glyphs that are wider than any single character — pairs that touch
/// ("rn", "wi") — at the darkest vertical valley. Guarded so wide-but-single
/// glyphs survive: "m" bridges its valleys with arches at the top, "W" peaks
/// in the middle, and dashes are too short — all are protected by explicit
/// checks.
pub fn split_touching(bin: &BinaryImage, comps: &[Comp], med_h: usize) -> Vec<Comp> {
    let mut out = Vec::new();
    let mut queue: Vec<Comp> = comps.to_vec();

    while let Some(c) = queue.pop() {
        let (w, h) = (c.width(), c.height());
        if (w as f32) < 1.4 * med_h as f32 || (h as f32) < 0.5 * med_h as f32 {
            out.push(c);
            continue;
        }
        let halves = split_candidates(bin, &c, med_h, SplitOptions { strict: true, ..Default::default() });
        match halves.into_iter().next() {
            Some([a, b]) => {
                queue.push(a);
                queue.push(b);
            }
            None => out.push(c),
        }
    }
    out
}

/// Merge vertically stacked fragments (i-dots, accents, "=", the two halves
/// of ";") into single glyph boxes.
///
/// Runs globally before line grouping — fragments are often vertically
/// disjoint, so lines alone can't bring them together — and again per line
/// with local metrics (WebOCR: merge pass 1 AND pass 2).
///
/// The gap budget is `0.6 · max(medH, largerHeight)` AND `2.4 · smallerHeight`
/// (both must hold). The second clause rejects the cross-line pairs on
/// mixed-size pages: a 60px headline next to a 12px body line has vgap 30 —
/// it fails 2.4·12 = 28.8. The cap *relaxes* for fragments of the SAME
/// height: those are the marks of ONE character (a colon's two dots — the
/// gap 8px against the cap's 2.4·3 = 7.2), not two lines.
pub fn merge_stacked(comps: &[Comp], med_h: usize) -> Vec<Comp> {
    let mut sorted: Vec<Comp> = comps.to_vec();
    sorted.sort_by_key(|c| c.x0);
    let mut out: Vec<Comp> = Vec::new();
    for c in sorted {
        if let Some(last) = out.last_mut() {
            // Saturating: horizontally disjoint boxes have a negative
            // overlap in the JS original, which usize cannot represent.
            let overlap = last.x1.min(c.x1).saturating_sub(last.x0.max(c.x0));
            let min_w = last.width().min(c.width());
            let vgap = last.y0.max(c.y0).saturating_sub(last.y1.min(c.y1));
            let h_last = last.height() as f32;
            let h_c = c.height() as f32;
            let upper = 0.6 * (med_h as f32).max(h_last).max(h_c);
            let same_height = (h_last - h_c).abs() <= 0.25 * h_last.max(h_c);
            let cap = if same_height {
                4.0 * h_last.max(h_c)
            } else {
                2.4 * h_last.min(h_c)
            };
            if min_w > 0
                && overlap as f32 >= 0.5 * min_w as f32
                && (vgap as f32) <= upper
                && (vgap as f32) <= cap
            {
                last.x0 = last.x0.min(c.x0);
                last.y0 = last.y0.min(c.y0);
                last.x1 = last.x1.max(c.x1);
                last.y1 = last.y1.max(c.y1);
                last.area += c.area;
                continue;
            }
        }
        out.push(c);
    }
    out
}

fn group_into_lines(comps: &[Comp]) -> Vec<Vec<Comp>> {
    let mut sorted: Vec<Comp> = comps.to_vec();
    sorted.sort_by_key(|c| c.y0);
    let mut lines: Vec<(Vec<Comp>, usize, usize)> = Vec::new(); // (comps, y0, y1)

    for c in sorted {
        let mut best: Option<usize> = None;
        let mut best_overlap = 0.0f32;
        for (idx, (_, ly0, ly1)) in lines.iter().enumerate() {
            let overlap = c.y1.min(*ly1) as i64 - c.y0.max(*ly0) as i64;
            let denom = (c.height().min(ly1 - ly0)) as i64;
            let ratio = if denom > 0 { overlap.max(0) as f32 / denom as f32 } else { 0.0 };
            if ratio > best_overlap {
                best_overlap = ratio;
                best = Some(idx);
            }
        }
        match best {
            Some(idx) if best_overlap >= 0.3 => {
                let (comps, y0, y1) = &mut lines[idx];
                comps.push(c);
                *y0 = (*y0).min(c.y0);
                *y1 = (*y1).max(c.y1);
            }
            _ => lines.push((vec![c], c.y0, c.y1)),
        }
    }
    lines.into_iter().map(|(comps, _, _)| comps).collect()
}

/// The seam a detached TOP MARK implies.
///
/// A lowercase "i" carries its dot as a separate component, and the dot sits
/// directly above the stem. When the dot is NOT over the middle of the glyph
/// box but near one end, the box is two characters: the wide one, and the
/// stem the dot belongs to. Georgia kerns "wi" so tightly that the "w"'s
/// last stroke touches the "i"'s stem — the pair arrives as ONE component,
/// and no vertical valley cut finds the seam (the "w"'s arm runs into the top
/// of the valley, so the arch guard rejects every cut and the fused glyph
/// matches "M"). The dot says exactly where the seam is.
///
/// Measured on Georgia 32px: the fused "wi" is one 32px-wide component, the
/// dot is a separate 4px component, and a cut at the dot's left edge leaves
/// the "w" intact.
///
/// Returns the two halves, or `None` when the box is one character.
pub fn dot_guided_split(comps: &[Comp], g: &Comp, med_h: usize) -> Option<[Comp; 2]> {
    let w = g.width();
    let h = g.height();
    if (w as f32) < 1.4 * med_h as f32 || (h as f32) < 0.5 * med_h as f32 {
        return None;
    }
    let inside: Vec<&Comp> = comps
        .iter()
        .filter(|c| c.x1 > g.x0 && c.x0 < g.x1 && c.y1 > g.y0 && c.y0 < g.y1)
        .collect();
    let center = (g.x0 + g.x1) as f32 / 2.0;
    for c in &inside {
        let cw = c.width();
        let ch = c.height();
        // The mark: a small component sitting ABOVE the body.
        if (cw as f32) > 0.35 * w as f32 || (ch as f32) > 0.4 * h as f32 {
            continue;
        }
        if c.y1 > g.y0 + h / 2 {
            continue;
        }
        let mark_center = (c.x0 + c.x1) as f32 / 2.0;
        // A mark over the middle of the box belongs to this character.
        if (mark_center - center).abs() < 0.25 * w as f32 {
            continue;
        }
        // The body is everything else in the box.
        let mut body_y0 = g.y1;
        let mut body_y1 = g.y0;
        for o in &inside {
            if std::ptr::eq(*o, *c) {
                continue;
            }
            body_y0 = body_y0.min(o.y0);
            body_y1 = body_y1.max(o.y1);
        }
        if body_y1 <= body_y0 {
            continue;
        }
        // The seam: the mark's near edge.
        let right = mark_center > center;
        let cut = if right { c.x0 } else { c.x1 };
        if cut <= g.x0 || cut >= g.x1 {
            continue;
        }
        let left = Comp {
            x0: g.x0,
            y0: if right { body_y0 } else { c.y0 },
            x1: cut,
            y1: if right { body_y1 } else { g.y1 },
            area: 0,
        };
        let stem = Comp {
            x0: cut,
            y0: if right { c.y0 } else { body_y0 },
            x1: g.x1,
            y1: if right { g.y1 } else { c.y1 },
            area: 0,
        };
        return Some([left, stem]);
    }
    None
}

/// Merge horizontally adjacent components while the union stays character-
/// sized. CJK glyphs are frequently multi-component (川, 二, 歩 all have
/// disconnected strokes), so in CJK mode a "glyph" is a cluster, not a
/// single component.
///
/// The width cap is what stops the merge from chaining across characters:
/// 川's three strokes fit in ~1em, but 川+も would be ~2em wide. The cap is
/// **1.1·medH**, not 1.25: at 32px a fullwidth character's ink spans ~1em,
/// and a cluster wider than that has almost certainly crossed into the next
/// character (WebOCR measured 懒惰 at tight spacing absorbing 惰's full-height
/// 忄 radical into 懒's cluster → a 35×30 box and two wrong reads).
///
/// The one legitimate exception is the kana dakuten/handakuten (ば's ゛,
/// ぱ's ゜), which sits at the TOP-RIGHT of the body and pushes the union to
/// ~1.15·medH. The marks are separable by **position**, not size: a dakuten
/// starts in the body's top quarter and ends clearly above the baseline,
/// while an absorbed radical is full-height and reaches the baseline. Such a
/// top mark is allowed the wider 1.3·medH cap.
pub fn cluster_components(comps: &[Comp], med_h: usize) -> Vec<Comp> {
    let max_width = 1.1 * med_h as f32;
    let mark_max_width = 1.3 * med_h as f32;
    let gap_limit = 0.35 * med_h as f32;
    let mut sorted: Vec<Comp> = comps.to_vec();
    sorted.sort_by_key(|c| c.x0);
    let mut out: Vec<Comp> = Vec::new();
    for c in sorted {
        if let Some(last) = out.last_mut() {
            let gap = c.x0 as f32 - last.x1 as f32;
            let merged_w = last.x1.max(c.x1) - last.x0;
            let last_h = last.height() as f32;
            // A top mark: starts in the top quarter, ends above the
            // baseline, and is at most half a character wide.
            let is_top_mark = c.y0 as f32 <= last.y0 as f32 + 0.25 * last_h
                && c.y1 as f32 <= last.y0 as f32 + 0.8 * last_h
                && (c.x1 - c.x0) as f32 <= 0.45 * med_h as f32;
            let cap = if is_top_mark { mark_max_width } else { max_width };
            if gap <= gap_limit && (merged_w as f32) <= cap {
                last.x0 = last.x0.min(c.x0);
                last.y0 = last.y0.min(c.y0);
                last.x1 = last.x1.max(c.x1);
                last.y1 = last.y1.max(c.y1);
                last.area += c.area;
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// Row → column segments. A row (all glyph boxes sharing a y-band across the
/// page) is split at horizontal gaps ≥ 2·medH — above every normal word gap
/// (~0.4–1.5·medH in real fonts), so single-column lines stay whole while
/// column gutters and deliberately wide spaces break the row into pieces.
/// Same-row pieces become sibling blocks; the engine rejoins them with
/// spacing rather than a break.
pub fn split_row_segments(row: &[Comp], med_h: usize) -> Vec<Vec<Comp>> {
    let mut sorted: Vec<Comp> = row.to_vec();
    sorted.sort_by_key(|c| c.x0);
    let mut segs: Vec<Vec<Comp>> = Vec::new();
    let mut cur: Vec<Comp> = Vec::new();
    let mut prev_x1: Option<usize> = None;
    for c in sorted {
        if let Some(px1) = prev_x1
            && !cur.is_empty()
            && c.x0 as i64 - px1 as i64 >= 2 * med_h as i64
        {
            segs.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        prev_x1 = Some(prev_x1.map_or(c.x1, |p| p.max(c.x1)));
    }
    if !cur.is_empty() {
        segs.push(cur);
    }
    segs
}

/// Text-block detection over row segments (WebOCR `detectBlocks`).
///
/// Segments union transitively when they sit close on both axes:
/// - same column — horizontal overlap + modest vertical gap (a column's
///   lines chain into one block)
/// - diagonal — a short indented line just right of the line above, close
///   on both axes without full overlap
///
/// The vertical gap is measured against the **row pitch** (the leading),
/// not the glyph height: a line's leading is ~2.9× the x-height, so a
/// criterion scaled by the glyph height never grouped a single pair of
/// lines — every line came out as its own block.
///
/// Side-by-side columns satisfy neither — they stay distinct blocks, so
/// line ordering can't interleave or concatenate them.
///
/// Returns each block as a list of indices into `segments`, in emit order.
pub fn detect_blocks(segments: &[Vec<Comp>], med_h: usize, row_pitch: usize) -> Vec<Vec<usize>> {
    let boxes: Vec<Comp> = segments
        .iter()
        .map(|list| {
            let mut b = list[0];
            for c in &list[1..] {
                b.x0 = b.x0.min(c.x0);
                b.y0 = b.y0.min(c.y0);
                b.x1 = b.x1.max(c.x1);
                b.y1 = b.y1.max(c.y1);
            }
            b
        })
        .collect();
    let n = boxes.len();

    // --- union-find over segments ---
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (boxes[i], boxes[j]);
            let x_overlap = a.x1.min(b.x1) as i64 - a.x0.max(b.x0) as i64;
            let x_gap = a.x0.max(b.x0) as i64 - a.x1.min(b.x1) as i64;
            let y_gap = a.y0.max(b.y0) as i64 - a.y1.min(b.y1) as i64;
            let min_w = a.width().min(b.width()) as i64;
            let mh = med_h as i64;
            let pitch = row_pitch as i64;
            if x_overlap >= 3 * min_w / 10 && y_gap <= 12 * pitch / 10 {
                let ra = find(&mut parent, i);
                let rb = find(&mut parent, j);
                if ra != rb {
                    parent[rb] = ra;
                }
                continue;
            }
            if x_gap <= mh && y_gap <= 12 * pitch / 10 {
                let ra = find(&mut parent, i);
                let rb = find(&mut parent, j);
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
    }

    // --- collect blocks (group segments by union-find root) ---
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (boxes[i].y0, boxes[i].x0));
    let mut root_group: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut blocks: Vec<(Vec<usize>, Comp)> = Vec::new(); // (segment idxs, union box)
    for &i in &order {
        let root = find(&mut parent, i);
        match root_group.get(&root) {
            Some(&gi) => {
                let (idxs, b) = &mut blocks[gi];
                idxs.push(i);
                b.x0 = b.x0.min(boxes[i].x0);
                b.y0 = b.y0.min(boxes[i].y0);
                b.x1 = b.x1.max(boxes[i].x1);
                b.y1 = b.y1.max(boxes[i].y1);
            }
            None => {
                root_group.insert(root, blocks.len());
                blocks.push((vec![i], boxes[i]));
            }
        }
    }

    // Segments within a block emit row-by-row, left-to-right inside a row.
    for (idxs, _) in &mut blocks {
        idxs.sort_by(|&i, &j| {
            let (a, c) = (boxes[i], boxes[j]);
            let y_overlap = a.y1.min(c.y1) as i64 - a.y0.max(c.y0) as i64;
            if y_overlap >= (5 * a.height().min(c.height()) / 10) as i64 {
                a.x0.cmp(&c.x0)
            } else {
                a.y0.cmp(&c.y0)
            }
        });
    }

    // Reading order: union blocks into column groups by x-range overlap (the
    // width-ratio guard keeps a spanning title/footer out of both columns),
    // emit groups top-to-bottom then left-to-right, blocks within a group
    // top-down.
    let bn = blocks.len();
    let mut gparent: Vec<usize> = (0..bn).collect();
    for i in 0..bn {
        for j in (i + 1)..bn {
            let a = blocks[i].1;
            let b = blocks[j].1;
            let x_overlap = a.x1.min(b.x1) as i64 - a.x0.max(b.x0) as i64;
            let min_w = a.width().min(b.width()) as i64;
            let max_w = a.width().max(b.width()) as i64;
            if x_overlap >= min_w / 2 && max_w <= 5 * min_w / 2 {
                let ra = find(&mut gparent, i);
                let rb = find(&mut gparent, j);
                if ra != rb {
                    gparent[rb] = ra;
                }
            }
        }
    }
    let mut group_order: Vec<usize> = (0..bn).collect();
    group_order.sort_by_key(|&i| (blocks[i].1.y0, blocks[i].1.x0));
    let mut seen_groups: Vec<usize> = Vec::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in group_order {
        let root = find(&mut gparent, i);
        if seen_groups.contains(&root) {
            continue;
        }
        seen_groups.push(root);
        let mut members: Vec<usize> = (0..bn)
            .filter(|&j| find(&mut gparent, j) == root)
            .collect();
        members.sort_by_key(|&j| (blocks[j].1.y0, blocks[j].1.x0));
        for m in members {
            out.push(blocks[m].0.clone());
        }
    }
    out
}

/// Components → lines of glyphs.
///
/// Segmentation that depends on glyph size — the kerned-pair cut, the
/// stacked-fragment merge, the CJK stroke clustering, and the column
/// gutter split — runs at **each row's own scale**, not one page-wide
/// median: a mixed-size page (headline + body) gets the wrong cut and merge
/// decisions when the page median is dominated by either extreme. A
/// page-wide scale is still needed for cross-row block detection (segments
/// union into text blocks), where it must be robust to single outlier rows
/// — the median of row medians gives that robustness.
///
/// The global merge before grouping is preserved: stacked fragments like
/// `;` (dot + tail) span visual rows so they never y-overlap with their
/// other half, and line grouping alone cannot bring them together. A
/// per-row pass then catches accents the global pass over- or under-merged
/// when one row is an outlier in size.
pub fn extract_lines(bin: &BinaryImage, comps: &[Comp], cjk: bool) -> Vec<LineData> {
    if comps.is_empty() {
        return Vec::new();
    }
    let global_med_h = median(&comps.iter().map(|c| c.height()).collect::<Vec<_>>()).max(1);
    let merged = merge_stacked(comps, global_med_h);

    // Row = every glyph sharing a y-band across the page (may hold two
    // columns).
    let mut segments: Vec<Vec<Comp>> = Vec::new();
    let mut row_med_hs: Vec<usize> = Vec::new();
    for mut raw in group_into_lines(&merged) {
        raw.sort_by_key(|c| c.x0);
        // The giant-component filter runs per row: a row scales itself, so a
        // headline over body text keeps its glyphs while a lone logo is
        // still judged against the page. Its only job is to make the row's
        // median robust — the giants themselves are still split below, so a
        // fused pair wider than 2.5·medH (a kerned "wn") is recovered
        // rather than dropped.
        let row = filter_giant_in_row(&raw, global_med_h, cjk);
        if row.is_empty() {
            continue;
        }
        let raw_med_h = median(&row.iter().map(|c| c.height()).collect::<Vec<_>>()).max(1);
        row_med_hs.push(raw_med_h);
        // Per-row kerned-pair split. Rows with different scales no longer
        // corrupt each other's 1.4·medH and 0.5·medH gates.
        let split = split_touching(bin, &raw, raw_med_h);
        let clustered = if cjk {
            // Clustering must scale by the CJK glyph size, not the raw row
            // median: a mixed-script line is full of Latin x-height comps
            // (~0.55·em) that drag the median down, and a 1.1·median cap then
            // refuses legitimate intra-character merges. The 90th-percentile
            // height tracks the hanzi/kana family and ignores the Latin dust.
            let mut heights: Vec<usize> = split.iter().map(|c| c.height()).collect();
            heights.sort_unstable();
            let idx = ((heights.len() as f64 * 0.9).floor() as usize).min(heights.len() - 1);
            cluster_components(&split, heights[idx].max(1))
        } else {
            split
        };
        let glyphs = merge_stacked(&clustered, raw_med_h);
        if glyphs.is_empty() {
            continue;
        }
        // The COLUMN split (a row may hold several columns) scales with the
        // row's LINE HEIGHT, not its median glyph height: the median is the
        // x-height, and in a monospace face a word space is a full advance
        // (2.2× the x-height in Courier) — so a 2×-median threshold split
        // ordinary word spaces into separate segments. The tallest glyph
        // (cap/ascender height) is the right scale for "is this a gutter".
        let line_height = glyphs.iter().map(|g| g.height()).max().unwrap_or(1);
        segments.extend(split_row_segments(&glyphs, line_height));
    }

    let cross_row_med_h = median(&row_med_hs).max(1);
    // The row pitch (the leading) — the scale for "these rows are the same
    // column". Taken from the vertical steps between the text rows.
    let mut row_tops: Vec<usize> = segments
        .iter()
        .map(|seg| seg.iter().map(|c| c.y0).min().unwrap_or(0))
        .collect();
    row_tops.sort_unstable();
    let steps: Vec<usize> = row_tops
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .filter(|d| *d > 0)
        .collect();
    let max_step = steps.iter().copied().max().unwrap_or(0);
    // Segments that share a row (the columns of a two-column page)
    // contribute a zero step, so only the real steps count.
    let real_steps: Vec<usize> = steps
        .iter()
        .copied()
        .filter(|d| (*d as f64) >= 0.25 * max_step as f64)
        .collect();
    let row_pitch = if real_steps.is_empty() {
        cross_row_med_h
    } else {
        median(&real_steps).max(1)
    };

    let mut out = Vec::new();
    for (b, block) in detect_blocks(&segments, cross_row_med_h, row_pitch)
        .into_iter()
        .enumerate()
    {
        for si in block {
            let glyphs = &segments[si];
            let med_h = median(&glyphs.iter().map(|g| g.height()).collect::<Vec<_>>()).max(1);
            let mut box_ = glyphs[0];
            for g in &glyphs[1..] {
                box_.x0 = box_.x0.min(g.x0);
                box_.y0 = box_.y0.min(g.y0);
                box_.x1 = box_.x1.max(g.x1);
                box_.y1 = box_.y1.max(g.y1);
            }
            out.push(LineData {
                box_,
                glyphs: glyphs.clone(),
                med_h,
                block: b,
            });
        }
    }
    out
}

/// Gap (px) beyond which two glyphs count as separate words. Scales with
/// text height so it works across font sizes. Fallback when a line has too
/// few gaps to cluster.
///
/// The absolute floor is 4px, not 6px: it is the fallback for a line whose
/// gaps do not cluster cleanly (the clustering needs a word gap to dominate
/// a letter gap by 1.35×, and at a small size the integer gaps never do).
/// The floor was calibrated on 30px+ type, where it never binds — at 16px
/// the word space is 4.4px, and a 6px floor swallowed every word space on
/// the line (WebOCR measured: the Georgia 16px cell read 76.6 %, and with a
/// 4px floor **93.6 %**; the Helvetica 16px cell 89.4 % → 95.7 %).
pub fn space_threshold(median_height: usize) -> f32 {
    (0.44 * median_height as f32).max(4.0)
}

/// Word-gap threshold from the line's actual gap distribution: letter gaps
/// and word gaps form two clusters, so we split at the biggest jump between
/// them.
///
/// A fixed multiplier can't work — Courier's letter gaps (7 px) exceed
/// Georgia's word gaps (7 px) even at similar text heights.
///
/// `line_height` (the tallest glyph in the line) scales the guard, not the
/// median: the median is the x-height, and in a monospace face a punctuation
/// advance (".", ",") lands between the letter gaps and the word gaps — a
/// Courier line has letter gaps 3-5 px, punctuation advances 7-14 px and
/// word gaps 19-27 px, so the only meaningful jump is 10→14 (ratio 1.4) and
/// the x-height guard (lo ≤ 0.55·medH = 6.6 px) rejected it, leaving the
/// fallback threshold — which then turned every punctuation advance into a
/// space ("(v1 . 0 ) ."). Scaled by the line height the guard lets that
/// jump through and lands the threshold between the punctuation advances and
/// the word gaps.
pub fn detect_space_threshold(gaps: &[usize], med_h: usize, line_height: usize) -> f32 {
    let fallback = space_threshold(med_h);
    if gaps.len() < 2 {
        return fallback;
    }
    let mut sorted: Vec<usize> = gaps.to_vec();
    sorted.sort_unstable();
    let min_word_gap = (0.3 * med_h as f32).max(5.5);
    let mut best_ratio = 1.35f32; // a word gap must clearly dominate a letter gap
    let mut best = fallback;
    for w in sorted.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if (hi as f32) < min_word_gap || (lo as f32) > 0.65 * line_height as f32 {
            continue;
        }
        let ratio = hi as f32 / lo.max(1) as f32;
        if ratio > best_ratio {
            best_ratio = ratio;
            best = (lo + hi) as f32 / 2.0;
        }
    }
    best.max(3.0)
}

/// Convenience: run the full segmentation stage on a grayscale image.
pub fn segment(gray: &super::types::GrayImage, cjk: bool) -> Vec<LineData> {
    let bin = super::binarize::binarize(gray).image;
    let comps = filter_noise(&connected_components(&bin));
    extract_lines(&bin, &comps, cjk)
}

/// Word splitter for any recognizer that already exposes per-character
/// text + x positions. The CRNN path's `char_x_centers` field feeds
/// this; the template engine's own engine does not need it (its
/// in-engine word grouping is spatial already).
///
/// Algorithm (WebOCR `detectSpaceThreshold`):
/// 1. Sort the per-glyph x-gaps; look for the biggest ratio jump
///    (≥1.35×) where the upper cluster clears `max(0.3·medH, 5.5)`
///    and the lower stays under `0.55·medH`.
///
/// When the recognizer doesn't supply a glyph height (the CRNN case
/// — `char_x_centers` are x-only), `med_h` is estimated from the
/// character advance distribution as a fallback so the threshold sits
/// above the largest inter-character gap in a single word.
///
/// Returns `(word, mean_char_confidence)` pairs.
pub fn split_words_by_gap_cluster(
    chars_and_confidences: &[(char, f32)],
    xs: &[f32],
) -> Vec<(String, f32)> {
    if chars_and_confidences.is_empty() || xs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(chars_and_confidences.len(), xs.len());

    let gaps: Vec<usize> = xs
        .windows(2)
        .map(|w| (w[1] - w[0]).max(0.0) as usize)
        .collect();
    if gaps.is_empty() {
        return chars_and_confidences
            .iter()
            .map(|&(c, conf)| (c.to_string(), conf))
            .collect();
    }

    // Look for the bimodal word/letter cluster split. The CRNN path
    // doesn't supply a glyph height, so we use a robust two-pass:
    // 1. Find the largest ratio jump in the sorted gap distribution.
    // 2. If no jump is found, assume the line is a single word (no
    //    reliable signal to split) — the WebOCR 0.44·medH fallback
    //    requires a known glyph height that we don't have here.
    let gap_limit = match find_biggest_gap_jump(&gaps) {
        Some((lo, hi)) => ((lo + hi) as f32 / 2.0).max(6.0),
        None => f32::INFINITY, // single word: don't split
    };

    let mut words = Vec::new();
    let mut current_chars = Vec::<char>::new();
    let mut current_confs = Vec::<f32>::new();
    for (i, &(c, conf)) in chars_and_confidences.iter().enumerate() {
        if c.is_whitespace() {
            if !current_chars.is_empty() {
                let mean = current_confs.iter().sum::<f32>() / current_confs.len() as f32;
                words.push((current_chars.iter().collect::<String>(), mean));
                current_chars.clear();
                current_confs.clear();
            }
            continue;
        }
        if i > 0 {
            let gap = (xs[i] - xs[i - 1]).max(0.0);
            if gap > gap_limit && !current_chars.is_empty() {
                let mean = current_confs.iter().sum::<f32>() / current_confs.len() as f32;
                words.push((current_chars.iter().collect::<String>(), mean));
                current_chars.clear();
                current_confs.clear();
            }
        }
        current_chars.push(c);
        current_confs.push(conf);
    }
    if !current_chars.is_empty() {
        let mean = current_confs.iter().sum::<f32>() / current_confs.len() as f32;
        words.push((current_chars.iter().collect::<String>(), mean));
    }
    words
}

/// Find the biggest ratio jump (≥1.35×) between adjacent sorted gaps.
/// Returns `Some((lo, hi))` for the two cluster endpoints that bracket
/// the natural word/letter split, or `None` when no jump qualifies
/// (single-cluster lines, i.e. single-word lines).
fn find_biggest_gap_jump(gaps: &[usize]) -> Option<(usize, usize)> {
    if gaps.len() < 4 {
        return None;
    }
    let mut sorted: Vec<usize> = gaps.to_vec();
    sorted.sort_unstable();
    let mut best: Option<(usize, usize, f32)> = None;
    for w in sorted.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if lo == 0 {
            continue;
        }
        let ratio = hi as f32 / lo as f32;
        if ratio >= 1.35 {
            match best {
                Some((_, _, r)) if r >= ratio => {}
                _ => best = Some((lo, hi, ratio)),
            }
        }
    }
    best.map(|(lo, hi, _)| (lo, hi))
}

#[cfg(test)]
mod tests_split {
    use super::split_words_by_gap_cluster;

    #[test]
    fn single_word_yields_one() {
        let chars: Vec<(char, f32)> = "Pack".chars().zip(std::iter::repeat(0.9f32)).collect();
        let xs = vec![10.0, 18.0, 26.0, 34.0];
        let out = split_words_by_gap_cluster(&chars, &xs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "Pack");
    }

    #[test]
    fn big_gap_splits_words() {
        // "Hi" at x=10, 18; "world" at x=80, 90, 100, 110, 120
        let chars: Vec<(char, f32)> = "Hiworld"
            .chars()
            .zip(std::iter::repeat(0.9f32))
            .collect();
        let xs = vec![10.0, 18.0, 80.0, 90.0, 100.0, 110.0, 120.0];
        let out = split_words_by_gap_cluster(&chars, &xs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "Hi");
        assert_eq!(out[1].0, "world");
    }

    #[test]
    fn small_gap_does_not_split() {
        // Tight kerning inside a single word
        let chars: Vec<(char, f32)> = "Pack"
            .chars()
            .zip(std::iter::repeat(0.9f32))
            .collect();
        let xs = vec![10.0, 14.0, 18.0, 22.0];
        let out = split_words_by_gap_cluster(&chars, &xs);
        assert_eq!(out.len(), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_img(w: usize, h: usize, ink: &[(usize, usize)]) -> BinaryImage {
        let mut data = vec![0u8; w * h];
        for &(x, y) in ink {
            data[y * w + x] = 1;
        }
        BinaryImage { width: w, height: h, data }
    }

    /// WebOCR measured-gap fixtures (scripts/measure-gaps.ts):
    /// Arial letters ≤5 px / words ≥9 px.
    #[test]
    fn gap_clusters_split_arial_style_gaps() {
        let gaps = [3usize, 4, 5, 4, 9, 10, 4]; // letters, word, letters
        let t = detect_space_threshold(&gaps, 21, 21);
        assert!((7.0..=9.0).contains(&t), "threshold {t} between 5 and 9");
    }

    /// Courier letters up to 7 px / words ≥21 px.
    #[test]
    fn gap_clusters_split_courier_style_gaps() {
        let gaps = [6usize, 7, 6, 7, 21, 6];
        let t = detect_space_threshold(&gaps, 21, 21);
        assert!(t > 7.0 && t < 21.0, "threshold {t} between 7 and 21");
    }

    /// Georgia bold words ≥7 px with letter gaps ≤5 — the 7 px word gap must
    /// be accepted (the historical 0.35·medH guard rejected it).
    #[test]
    fn small_word_gaps_are_not_rejected_by_the_guard() {
        let gaps = [3usize, 4, 5, 7, 4];
        let t = detect_space_threshold(&gaps, 21, 21);
        assert!(t >= 5.0 && t <= 7.0, "threshold {t} keeps the 7px word gap");
    }

    #[test]
    fn too_few_gaps_falls_back_to_scaled_threshold() {
        assert!((detect_space_threshold(&[4], 21, 21) - space_threshold(21)).abs() < 1e-6);
        assert!((space_threshold(30) - 13.2).abs() < 1e-4);
        // The floor is 4px, not 6px: at 16px the word space is 4.4px, and a
        // 6px floor swallowed every word space on the line (WebOCR measured
        // the Georgia 16px cell at 76.6 % with a 6px floor and 93.6 % with
        // a 4px one).
        assert!((space_threshold(5) - 4.0).abs() < 1e-6, "floors at 4");
    }

    /// "m" bridges its valleys with arches — the bridge guard must keep it
    /// whole (WebOCR regression: splitTouching arch protection).
    #[test]
    fn arch_guard_protects_m() {
        // A 30×20 "m"-like shape: three vertical stems joined by a top bar.
        let mut ink = Vec::new();
        for stem in [0usize, 13, 26] {
            for y in 4..20 {
                for x in stem..stem + 3 {
                    ink.push((x, y));
                }
            }
        }
        for x in 0..29 {
            ink.push((x, 4));
            ink.push((x, 5));
            ink.push((x, 6));
        }
        let bin = bin_img(30, 20, &ink);
        let med_h = 20;
        let comps = vec![Comp { x0: 0, y0: 4, x1: 29, y1: 20, area: ink.len() }];
        let out = split_touching(&bin, &comps, med_h);
        assert_eq!(out.len(), 1, "arched 'm' must not split");
    }

    /// A wide fused pair with clean valleys at baseline level splits apart.
    #[test]
    fn valley_split_separates_touching_pair() {
        // Two 8-wide stems touching through a 1px column with minimal ink.
        // medH 10 → the 17px pair clears the 1.4·medH split gate.
        let mut ink = Vec::new();
        for y in 0..10 {
            for x in 0..8 {
                ink.push((x, y));
            }
            for x in 9..17 {
                ink.push((x, y));
            }
        }
        ink.push((8, 9)); // single touching pixel at the bottom
        let bin = bin_img(20, 12, &ink);
        let comps = vec![Comp { x0: 0, y0: 0, x1: 17, y1: 10, area: ink.len() }];
        let out = split_touching(&bin, &comps, 10);
        assert_eq!(out.len(), 2, "fused pair splits at the light valley");
    }

    /// ";" = dot + tail with a vertical gap — merge_stacked must join them.
    #[test]
    fn stacked_fragments_merge_semicolon() {
        let med_h = 20;
        let dot = Comp { x0: 4, y0: 0, x1: 7, y1: 3, area: 9 };
        let tail = Comp { x0: 5, y0: 10, x1: 8, y1: 14, area: 6 };
        let merged = merge_stacked(&[dot, tail], med_h);
        assert_eq!(merged.len(), 1);
        assert_eq!((merged[0].y0, merged[0].y1), (0, 14));
    }

    /// CJK clustering: two strokes of one hanzi merge; two adjacent hanzi
    /// never chain (width cap 1.25·medH).
    #[test]
    fn cluster_components_width_cap_stops_chaining() {
        let med_h = 20; // em = 20, cap = 25
        let s1 = Comp { x0: 0, y0: 0, x1: 6, y1: 20, area: 1 };
        let s2 = Comp { x0: 9, y0: 0, x1: 15, y1: 20, area: 1 }; // gap 3 ≤ 7, union 15 ≤ 25
        let s3 = Comp { x0: 30, y0: 0, x1: 36, y1: 20, area: 1 }; // next glyph
        let out = cluster_components(&[s1, s2, s3], med_h);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].x0, out[0].x1), (0, 15));
    }

    /// The CJK width cap is 1.1·medH, not 1.25: a cluster wider than that
    /// has crossed into the next character (WebOCR measured 懒惰 at tight
    /// spacing absorbing 惰's radical → a 35×30 box and two wrong reads).
    #[test]
    fn cluster_components_refuses_a_cluster_wider_than_one_em() {
        let med_h = 20;
        // Union would be 26 wide — over the 1.1·20 = 22 cap.
        let s1 = Comp { x0: 0, y0: 0, x1: 12, y1: 20, area: 1 };
        let s2 = Comp { x0: 14, y0: 0, x1: 26, y1: 20, area: 1 };
        let out = cluster_components(&[s1, s2], med_h);
        assert_eq!(out.len(), 2, "a 26-wide cluster is two characters");
    }

    /// A dakuten sits at the TOP-RIGHT of its body and pushes the union past
    /// the width cap; it is allowed the wider 1.3·medH because it is
    /// separable by POSITION, not size.
    #[test]
    fn cluster_components_allows_a_top_mark_past_the_cap() {
        let med_h = 20;
        let body = Comp { x0: 0, y0: 0, x1: 12, y1: 20, area: 1 };
        // A small mark at the body's top-right.
        let mark = Comp { x0: 13, y0: 0, x1: 20, y1: 5, area: 1 };
        let out = cluster_components(&[body, mark], med_h);
        assert_eq!(out.len(), 1, "the top mark joins its body");
        assert_eq!((out[0].x0, out[0].x1), (0, 20));
    }

    /// A colon is ONE character: two equal-height dots with a gap. The
    /// same-height clause relaxes the gap cap for exactly this case (the
    /// gap 8px against the 2.4·3 = 7.2 cap).
    #[test]
    fn merge_stacked_keeps_a_colon_together() {
        let med_h = 20;
        let top = Comp { x0: 4, y0: 0, x1: 7, y1: 3, area: 9 };
        let bottom = Comp { x0: 4, y0: 11, x1: 7, y1: 14, area: 9 };
        let merged = merge_stacked(&[top, bottom], med_h);
        assert_eq!(merged.len(), 1, "a colon is one character");
        assert_eq!((merged[0].y0, merged[0].y1), (0, 14));
    }

    /// A page-spanning rule is not text: dense, very wide, and far larger
    /// than the line's median height.
    #[test]
    fn filter_non_text_drops_a_page_spanning_rule() {
        let letter = Comp { x0: 0, y0: 0, x1: 4, y1: 20, area: 40 };
        // A 200×3 solid rule across the page.
        let rule = Comp { x0: 0, y0: 0, x1: 200, y1: 3, area: 600 };
        let out = filter_non_text(&[letter, rule], Some(20), false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].x1, 4, "the letter survives, the rule does not");
    }

    /// A solid blob (a logo) is not text: its density is above 0.85 and
    /// both of its sides are large.
    #[test]
    fn filter_non_text_drops_a_solid_blob() {
        let letter = Comp { x0: 0, y0: 0, x1: 4, y1: 20, area: 40 };
        let blob = Comp { x0: 0, y0: 0, x1: 30, y1: 30, area: 900 };
        let out = filter_non_text(&[letter, blob], Some(20), false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].x1, 4);
    }

    /// A row scales itself: a headline over body text keeps its glyphs,
    /// while a lone logo is still judged against the page.
    #[test]
    fn filter_giant_in_row_scales_to_the_row() {
        let body = Comp { x0: 0, y0: 0, x1: 6, y1: 12, area: 20 };
        let headline = Comp { x0: 10, y0: 0, x1: 40, y1: 28, area: 200 };
        let logo = Comp { x0: 0, y0: 0, x1: 60, y1: 60, area: 3000 };
        // A row of body text plus a headline: the headline is 2.3× the
        // body's median, which is under the 2.5·medH giant limit.
        let row = filter_giant_in_row(&[body, headline], 12, false);
        assert_eq!(row.len(), 2, "the headline survives its own row");
        // A lone logo is judged against the page scale.
        let lone = filter_giant_in_row(&[logo], 12, false);
        assert!(lone.is_empty(), "a 60px logo is not text at a 12px scale");
    }

    /// A detached top mark near one end of a wide box says where the seam
    /// is: the Georgia "wi" that fuses into one component.
    #[test]
    fn dot_guided_split_finds_the_seam_of_a_kerned_pair() {
        let med_h = 20;
        // A wide fused body (the "w") plus a small dot at its top right.
        let body = Comp { x0: 0, y0: 0, x1: 30, y1: 20, area: 200 };
        let dot = Comp { x0: 26, y0: 0, x1: 30, y1: 4, area: 12 };
        let comps = vec![body, dot];
        let [left, stem] = dot_guided_split(&comps, &body, med_h).expect("a seam is found");
        assert_eq!(left.x1, 26, "the cut is at the mark's near edge");
        assert_eq!(stem.x0, 26);
        assert!(stem.x1 > left.x1);
    }

    /// A mark over the MIDDLE of the box belongs to that character — no
    /// seam there.
    #[test]
    fn dot_guided_split_ignores_a_centred_mark() {
        let med_h = 20;
        let body = Comp { x0: 0, y0: 0, x1: 30, y1: 20, area: 200 };
        let dot = Comp { x0: 13, y0: 0, x1: 17, y1: 4, area: 12 };
        assert!(dot_guided_split(&[body, dot], &body, med_h).is_none());
    }

    #[test]
    fn two_column_page_yields_two_blocks_in_reading_order() {        // Two columns of 6×8 glyph blobs. Row pitch 15 keeps the vertical
        // gap (7) above the 0.6·medH stacked-merge tolerance (4.8) but
        // under the 1.2·medH block-join limit (9.6) — so lines stay
        // separate while a column's rows still chain into one block.
        let mut ink = Vec::new();
        for row in 0..3usize {
            let y = row * 15;
            for x in 0..6 {
                for dy in 0..8 {
                    ink.push((x, y + dy)); // left column glyph
                    ink.push((50 + x, y + dy)); // right column glyph
                }
            }
        }
        let bin = bin_img(60, 38, &ink);
        let comps = filter_noise(&connected_components(&bin));
        let lines = extract_lines(&bin, &comps, false);
        assert_eq!(lines.len(), 6);
        // All rows of column 1 must precede all rows of column 2.
        let first_right = lines.iter().position(|l| l.glyphs[0].x0 >= 50).unwrap();
        assert!(lines[..first_right].iter().all(|l| l.glyphs[0].x0 < 50));
        // Left rows share a block; right rows share another.
        assert_ne!(lines[0].block, lines[first_right].block);
        assert_eq!(lines[0].block, lines[1].block);
    }
}
