//! Template matching: cosine similarity with geometry penalties (WebOCR
//! port of `webocr/src/ocr/classify.ts`).

/// A template glyph: normalized vector + precomputed geometry
/// (WebOCR `PreparedTemplate`).
#[derive(Debug, Clone)]
pub struct PreparedTemplate {
    pub char: char,
    /// Normalized `frame×frame` vector.
    pub vec: Vec<f32>,
    /// Precomputed L2 norm of `vec`.
    pub norm: f32,
    /// Glyph box width / height — aspect survives normalization and is a
    /// strong discriminator between look-alikes like "o"/"O" and "s"/"S".
    pub aspect: f32,
    /// Pooled low-res copy for coarse-to-fine shortlisting (CJK-sized sets).
    pub coarse: Option<Vec<f32>>,
    pub coarse_norm: f32,
    /// Glyph box height as a fraction of the raster em size.
    pub rel_h: f32,
    /// Ink top above the baseline, as a fraction of the raster em — the
    /// **case signal**.
    ///
    /// Own-height normalization scales every glyph into the same frame, so a
    /// lowercase "l" and an uppercase "I" (identical strokes in most sans
    /// fonts) become the same vector and the tie is broken by whatever the
    /// library order happens to be. The ink top keeps the distinction: an
    /// x-height letter tops out at the x-height, a capital at the cap
    /// height, a descender letter ("g") at the x-height again.
    pub top_rel: f32,
    /// The raster size (px) this glyph was drawn at — the phase rescue picks
    /// its variant pool by raster, and a 16px variant is no use for a 30px
    /// sample.
    pub raster: u32,
    /// The fractional x offset this glyph was drawn at (0 for the main
    /// ladder). A scan's glyphs sit at whatever fraction the text put them
    /// at, and the anti-aliasing of a one-pixel stem changes with it.
    pub phase: f32,
}

pub fn l2(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Average-pool a `frame×frame` vector down to `cells×cells`. Coarse vectors
/// are ~9× smaller, which makes scanning a 21K-glyph CJK library affordable.
pub fn pool_coarse(vec: &[f32], frame: usize, cells: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cells * cells];
    let step = frame as f32 / cells as f32;
    for cy in 0..cells {
        for cx in 0..cells {
            let y0 = (cy as f32 * step).floor() as usize;
            let y1 = (frame).min(((cy + 1) as f32 * step).floor() as usize).max(y0 + 1);
            let x0 = (cx as f32 * step).floor() as usize;
            let x1 = (frame).min(((cx + 1) as f32 * step).floor() as usize).max(x0 + 1);
            let mut sum = 0.0f32;
            let mut n = 0usize;
            for y in y0..y1.min(frame) {
                for x in x0..x1.min(frame) {
                    sum += vec[y * frame + x];
                    n += 1;
                }
            }
            out[cy * cells + cx] = if n > 0 { sum / n as f32 } else { 0.0 };
        }
    }
    out
}

/// Coarse-to-fine shortlist: rank templates by pooled low-res cosine and
/// keep the best `k`. Only meaningful when templates carry coarse vectors;
/// otherwise the full set is returned untouched.
///
/// With `shift > 0`, the pooled sample is also tried at ±shift pooled cells
/// (zero-padded) and each template keeps its best-shift score. Measured
/// necessity in WebOCR: small glyphs — punctuation, kana — sit 1 pooled
/// cell off their template's frame anchor (。 zero-shift coarse rank 20379
/// vs rank 1 with ±1). Ranking without the shift permanently excludes the
/// true char before the fine stage can see it; recall is a precondition for
/// every fine-stage signal.
pub fn shortlist<'a>(
    vec: &[f32],
    frame: usize,
    templates: &[&'a PreparedTemplate],
    k: usize,
    shift: isize,
) -> Vec<&'a PreparedTemplate> {
    let Some(first) = templates.first() else {
        return Vec::new();
    };
    let Some(coarse0) = &first.coarse else {
        return templates.to_vec();
    };
    if templates.len() <= k {
        return templates.to_vec();
    }
    let cells = (coarse0.len() as f32).sqrt().round() as usize;
    let base = pool_coarse(vec, frame, cells);

    // Zero-padded shift variants of the pooled sample.
    let mut variants: Vec<(Vec<f32>, f32)> = Vec::new();
    for dy in -shift..=shift {
        for dx in -shift..=shift {
            let mut s = vec![0.0f32; cells * cells];
            for y in 0..cells {
                let sy = y as isize - dy;
                if sy < 0 || sy >= cells as isize {
                    continue;
                }
                for x in 0..cells {
                    let sx = x as isize - dx;
                    if sx < 0 || sx >= cells as isize {
                        continue;
                    }
                    s[y * cells + x] = base[sy as usize * cells + sx as usize];
                }
            }
            let norm = l2(&s).max(1.0);
            variants.push((s, norm));
        }
    }

    let mut scored: Vec<(f32, &PreparedTemplate)> = templates
        .iter()
        .map(|t| {
            let c = t.coarse.as_deref().unwrap_or(&base);
            let cn = if t.coarse_norm > 0.0 { t.coarse_norm } else { 1.0 };
            let mut best = -1.0f32;
            for (v, vn) in &variants {
                let dot: f32 = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                let s = dot / (vn * cn);
                if s > best {
                    best = s;
                }
            }
            (best, *t)
        })
        .collect();
    // Top-k by score, descending.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(_, t)| t).collect()
}

/// Codepoint partition threshold (WebOCR `engine.ts` shortlist split).
/// Mirrors `super::templates::SCRIPT_PARTITION_CP`; duplicated here to
/// avoid a circular import.
const SCRIPT_PARTITION_CP: u32 = 0x2E80;

/// Partitioned shortlist for union (Latin + CJK) libraries: split the
/// template pool by codepoint, run the coarse shortlist on each subset
/// with a per-subset quota, and concat. This guarantees both script
/// families have recall — a single 32K-templated shortlist of 64 always
/// starves Latin against the CJK majority (WebOCR: 0–9 reach top-64
/// against 32K hanzi without this split).
pub fn shortlist_partitioned<'a>(
    vec: &[f32],
    frame: usize,
    templates: &[&'a PreparedTemplate],
    latin_k: usize,
    cjk_k: usize,
    shift: isize,
) -> Vec<&'a PreparedTemplate> {
    let (latin, cjk): (Vec<&'a PreparedTemplate>, Vec<&'a PreparedTemplate>) = templates
        .iter()
        .copied()
        .partition(|t| (t.char as u32) < SCRIPT_PARTITION_CP);
    let mut out = shortlist(vec, frame, &latin, latin_k, shift);
    out.extend(shortlist(vec, frame, &cjk, cjk_k, shift));
    out
}

/// One case reading of a glyph: the best lowercase and the best uppercase
/// candidate, with their scores and ink tops (WebOCR `Match.lower` /
/// `Match.upper`).
#[derive(Debug, Clone, Copy)]
pub struct CaseReading {
    pub char: char,
    pub score: f32,
    /// The template's ink top above the baseline, in em units.
    pub top_rel: f32,
}

/// Result of matching one glyph (WebOCR `Match`).
#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub char: char,
    /// Best penalized cosine similarity, 0–1.
    pub score: f32,
    /// Gap to the best *different* character — a confidence proxy.
    pub margin: f32,
    /// The best lowercase reading (if the pool had one).
    pub lower: Option<CaseReading>,
    /// The best uppercase reading (if the pool had one).
    pub upper: Option<CaseReading>,
}

/// Aspect-ratio penalty weight.
///
/// WebOCR uses 0.15. The Rust port originally carried 0.1 with its own
/// measurement ("λ=0.2 regressed sans 100→95.3"), but that measurement was
/// taken **without** the case channel: the aspect penalty was doing the
/// case discrimination's job (a lowercase "c" is wider than a capital "C"),
/// and weakening it let the capitals win. With the `top_rel` channel in
/// place the aspect penalty can go back to its measured 0.15.
pub const ASPECT_WEIGHT: f32 = 0.15;

/// Options for [`match_glyph`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchOptions {
    /// ±cell shift search (0 = off).
    pub shift: isize,
    /// Normalized frame side (must match the vector length).
    pub frame: usize,
    /// Sample glyph height as a fraction of its line's median height (em).
    pub rel_h: f32,
    /// em-relative-height penalty weight (0 = off; WebOCR uses 0.3 for CJK).
    pub rel_h_weight: f32,
    /// Sample ink top above the baseline, in em units — the case signal.
    pub top_rel: f32,
    /// Case-height penalty weight (0 = off; WebOCR uses 0.15 for Latin).
    pub top_rel_weight: f32,
    /// Skip templates whose BEST possible score (a perfect cosine minus the
    /// penalties) is below this. A candidate that cannot reach the bar
    /// cannot win, and skipping it costs a handful of arithmetic operations
    /// instead of a full-frame dot product.
    pub min_score: Option<f32>,
}

/// Can this template reach the bar at all?
///
/// The score is a cosine (at most 1) minus the channel penalties, so the
/// best score a template can produce is 1 minus its penalties. A template
/// whose best is below the bar cannot win, whatever its shape.
fn can_reach(
    t: &PreparedTemplate,
    aspect: f32,
    opts: &MatchOptions,
    bar: f32,
) -> bool {
    let mut penalty =
        ASPECT_WEIGHT * ((aspect.max(1e-6) / t.aspect.max(1e-6)).ln().abs());
    if opts.rel_h_weight > 0.0 && opts.rel_h > 0.0 && t.rel_h > 0.0 {
        penalty += opts.rel_h_weight * ((opts.rel_h / t.rel_h).ln().abs());
    }
    if opts.top_rel_weight > 0.0 && opts.top_rel > 0.0 && t.top_rel > 0.0 {
        penalty += opts.top_rel_weight * ((opts.top_rel / t.top_rel).ln().abs());
    }
    1.0 - penalty >= bar
}

/// Nearest-neighbour classification: cosine similarity over the normalized
/// bitmap, penalized by aspect-ratio mismatch. With `shift > 0`, each
/// template is matched at (2·shift+1)² offsets of the sample; the best wins.
///
/// Tolerant of the ~1-cell rasterization slack between sample and template —
/// critical for dense CJK where the raw-cosine margin is tiny (WebOCR:
/// ±1-cell shift lifted ja 79.3→86.2 %, zh 38.2→64.7 %, ko 72.7→100 % in a
/// single change). The optional relH channel penalizes em-relative-height
/// mismatches (a hanzi-scale glyph can never be a punctuation dot, however
/// blob-like both normalize).
///
/// The **topRel channel** is the case signal: it penalizes a mismatch
/// between the sample's ink top above the baseline and the template's, in em
/// units. Without it, own-height normalization makes a lowercase "c" and a
/// capital "C" nearly the same vector and the case is decided by library
/// order — the Rust port's scorecard read "quiCk brOwn dOgS" before the
/// channel landed.
pub fn match_glyph(
    vec: &[f32],
    aspect: f32,
    templates: &[&PreparedTemplate],
    opts: MatchOptions,
) -> Match {
    let frame = opts.frame;
    debug_assert!(frame > 0 && frame * frame == vec.len(), "frame must match vec length");

    let mut best_char = '?';
    let mut best = -1.0f32;
    let mut second = -1.0f32;
    // The best reading of each case — kept so a caller can break a case tie
    // with the context (see `engine::resolve_case`).
    let mut best_lower: Option<CaseReading> = None;
    let mut best_upper: Option<CaseReading> = None;

    for t in templates {
        if let Some(bar) = opts.min_score
            && !can_reach(t, aspect, &opts, bar)
        {
            continue;
        }
        let mut s = match_one_shift(vec, aspect, t, frame, 0, 0);
        if opts.shift > 0 {
            for dy in -opts.shift..=opts.shift {
                for dx in -opts.shift..=opts.shift {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let sc = match_one_shift(vec, aspect, t, frame, dx, dy);
                    if sc > s {
                        s = sc;
                    }
                }
            }
        }
        if opts.rel_h_weight > 0.0 && opts.rel_h > 0.0 && t.rel_h > 0.0 {
            s -= opts.rel_h_weight * ((opts.rel_h / t.rel_h).ln().abs());
        }
        if opts.top_rel_weight > 0.0 && opts.top_rel > 0.0 && t.top_rel > 0.0 {
            s -= opts.top_rel_weight * ((opts.top_rel / t.top_rel).ln().abs());
        }
        // ASCII case, by code point — the templates are matched in their
        // thousands, so this stays off the `char::is_lowercase` path.
        let code = t.char as u32;
        if (97..=122).contains(&code) {
            if best_lower.is_none_or(|b| s > b.score) {
                best_lower = Some(CaseReading {
                    char: t.char,
                    score: s,
                    top_rel: t.top_rel,
                });
            }
        } else if (65..=90).contains(&code) {
            if best_upper.is_none_or(|b| s > b.score) {
                best_upper = Some(CaseReading {
                    char: t.char,
                    score: s,
                    top_rel: t.top_rel,
                });
            }
        }
        if s > best {
            if t.char != best_char {
                second = best;
            }
            best = s;
            best_char = t.char;
        } else if s > second && t.char != best_char {
            second = s;
        }
    }
    Match {
        char: best_char,
        score: best.max(0.0),
        margin: (best - second).max(0.0),
        lower: best_lower,
        upper: best_upper,
    }
}

/// Cosine of the (shifted) sample against one template, minus the aspect
/// penalty. The per-shift L2 norm is recomputed because the clipped window
/// changes it.
fn match_one_shift(
    vec: &[f32],
    aspect: f32,
    t: &PreparedTemplate,
    frame: usize,
    dx: isize,
    dy: isize,
) -> f32 {
    let mut dot = 0.0f32;
    let mut wn = 0.0f32;
    for y in 0..frame {
        let sy = y as isize - dy;
        if sy < 0 || sy >= frame as isize {
            continue;
        }
        let yrow = y * frame;
        let syrow = sy as usize * frame;
        for x in 0..frame {
            let sx = x as isize - dx;
            if sx < 0 || sx >= frame as isize {
                continue;
            }
            let v = vec[syrow + sx as usize];
            if v > 0.0 {
                dot += v * t.vec[yrow + x];
                wn += v * v;
            }
        }
    }
    let vn = wn.sqrt().max(1.0);
    let penalty = ASPECT_WEIGHT * ((aspect.max(1e-6) / t.aspect.max(1e-6)).ln().abs());
    dot / (vn * t.norm).max(1e-6) - penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpl(ch: char, vec: Vec<f32>, aspect: f32) -> PreparedTemplate {
        PreparedTemplate {
            char: ch,
            norm: l2(&vec).max(1.0),
            vec,
            aspect,
            coarse: None,
            coarse_norm: 0.0,
            rel_h: 1.0,
            top_rel: 0.5,
            raster: 30,
            phase: 0.0,
        }
    }

    #[test]
    fn aspect_penalty_separates_round_look_alikes() {
        let frame = 4;
        // Identical square blob for both templates; only `aspect` differs.
        let make_blob = || {
            let mut v = vec![0.0f32; 16];
            for y in 1..3 {
                for x in 1..3 {
                    v[y * 4 + x] = 1.0;
                }
            }
            v
        };
        let o_t = tmpl('o', make_blob(), 1.0);
        let o_wide_t = tmpl('O', make_blob(), 2.0);
        let refs = [&o_t, &o_wide_t];
        let narrow = match_glyph(&make_blob(), 1.0, &refs, MatchOptions { frame, ..Default::default() });
        let wide = match_glyph(&make_blob(), 2.0, &refs, MatchOptions { frame, ..Default::default() });
        assert_eq!(narrow.char, 'o', "aspect-1.0 sample prefers the aspect-1.0 template");
        assert_eq!(wide.char, 'O', "aspect-2.0 sample prefers the aspect-2.0 template");
    }

    #[test]
    fn match_reports_margin_against_different_char() {
        let frame = 4;
        let a = tmpl('a', vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        let b = tmpl('b', vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        let pool = [a, b];
        let refs: Vec<&PreparedTemplate> = pool.iter().collect();
        let m = match_glyph(&vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0, &refs, MatchOptions { frame, ..Default::default() });
        assert_eq!(m.char, 'a');
        assert!(m.score > 0.5);
        assert!(m.margin > 0.0);
    }

    #[test]
    fn shift_search_recovers_offset_sample() {
        let frame = 8;
        let mut t_vec = vec![0.0f32; 64];
        for y in 2..6 {
            for x in 2..6 {
                t_vec[y * 8 + x] = 1.0;
            }
        }
        let t = tmpl('x', t_vec, 1.0);
        // Sample shifted +1 cell right/down.
        let mut sample = vec![0.0f32; 64];
        for y in 3..7 {
            for x in 3..7 {
                sample[y * 8 + x] = 1.0;
            }
        }
        let zero = match_glyph(&sample, 1.0, &[&t], MatchOptions { frame, shift: 0, ..Default::default() });
        let shifted = match_glyph(&sample, 1.0, &[&t], MatchOptions { frame, shift: 1, ..Default::default() });
        assert!(
            shifted.score > zero.score + 0.05,
            "shift search should beat zero-shift ({:.3} vs {:.3})",
            shifted.score,
            zero.score
        );
    }

    /// The **case channel**: own-height normalization makes a lowercase "c"
    /// and a capital "C" nearly the same vector, so without the ink-top
    /// channel the case is decided by library order and a sentence reads
    /// "quiCk brOwn dOgS".
    #[test]
    fn top_rel_channel_separates_case() {
        let frame = 8;
        // The same blob for both templates — only the ink top differs.
        let mut blob = vec![0.0f32; 64];
        for y in 2..6 {
            for x in 2..6 {
                blob[y * 8 + x] = 1.0;
            }
        }
        let mut lower = tmpl('c', blob.clone(), 1.0);
        lower.top_rel = 0.5; // an x-height letter
        let mut upper = tmpl('C', blob.clone(), 1.0);
        upper.top_rel = 0.72; // a capital
        let pool = [lower, upper];
        let refs: Vec<&PreparedTemplate> = pool.iter().collect();

        // A sample that sits at the x-height must read lowercase.
        let m = match_glyph(
            &blob,
            1.0,
            &refs,
            MatchOptions {
                frame,
                top_rel: 0.5,
                top_rel_weight: 0.15,
                ..Default::default()
            },
        );
        assert_eq!(m.char, 'c', "the ink-top channel picks the lowercase");

        // The same sample measured at cap height must read uppercase.
        let m = match_glyph(
            &blob,
            1.0,
            &refs,
            MatchOptions {
                frame,
                top_rel: 0.72,
                top_rel_weight: 0.15,
                ..Default::default()
            },
        );
        assert_eq!(m.char, 'C', "the ink-top channel picks the capital");
    }

    /// The case readings (best lowercase / best uppercase) travel with the
    /// match so the engine can break a tie with the word's context.
    #[test]
    fn match_reports_both_case_readings() {
        let frame = 4;
        let mut lower = tmpl('l', vec![1.0; 16], 0.2);
        lower.top_rel = 0.72;
        let mut upper = tmpl('I', vec![1.0; 16], 0.2);
        upper.top_rel = 0.72;
        let pool = [lower, upper];
        let refs: Vec<&PreparedTemplate> = pool.iter().collect();
        let m = match_glyph(
            &vec![1.0; 16],
            0.2,
            &refs,
            MatchOptions { frame, ..Default::default() },
        );
        assert!(m.lower.is_some() && m.upper.is_some());
        assert_eq!(m.lower.unwrap().char, 'l');
        assert_eq!(m.upper.unwrap().char, 'I');
    }

    /// A template whose best possible score cannot reach the bar is
    /// skipped — the merge pass scores a candidate pair against the union
    /// of their pools, and the pruning is what keeps that affordable.
    #[test]
    fn min_score_prunes_unreachable_templates() {
        let frame = 4;
        // A template with a wildly different aspect cannot reach a high bar.
        let far = tmpl('W', vec![1.0; 16], 4.0);
        let near = tmpl('w', vec![1.0; 16], 1.0);
        let pool = [far, near];
        let refs: Vec<&PreparedTemplate> = pool.iter().collect();
        let m = match_glyph(
            &vec![1.0; 16],
            1.0,
            &refs,
            MatchOptions {
                frame,
                min_score: Some(0.99),
                ..Default::default()
            },
        );
        assert_eq!(m.char, 'w', "the unreachable template never wins");
    }

    #[test]
    fn pool_coarse_averages_cells() {        let frame = 4;
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let c = pool_coarse(&v, frame, 2);
        assert_eq!(c.len(), 4);
        assert!((c[0] - (0.0 + 1.0 + 4.0 + 5.0) / 4.0).abs() < 1e-5);
        assert!((c[3] - (10.0 + 11.0 + 14.0 + 15.0) / 4.0).abs() < 1e-5);
    }

    #[test]
    fn shortlist_partitioned_guarantees_latin_recall_against_cjk_majority() {
        // WebOCR's stated motivation for the partition: a 32K-templated
        // shortlist of 64 always starves Latin (0–9 reach top-64 against
        // 32K hanzi). The partition splits at 0x2E80 and gives each side
        // its own quota, so Latin still has representation in the result.
        let frame = 4;
        let make = |ch: char, vec: Vec<f32>| PreparedTemplate {
            char: ch,
            norm: 1.0,
            vec,
            aspect: 1.0,
            coarse: Some(vec![0.0; 64]),
            coarse_norm: 1.0,
            rel_h: 1.0,
            top_rel: 0.5,
            raster: 30,
            phase: 0.0,
        };
        let latin = make('A', vec![1.0; 16]);
        // 100 CJK templates to simulate the union majority
        let cjk: Vec<PreparedTemplate> = (0..100)
            .map(|i| {
                let v = vec![if i % 7 == 0 { 0.0 } else { 0.1 }; 16];
                PreparedTemplate {
                    char: char::from_u32(0x4E00 + i).unwrap(),
                    norm: 1.0,
                    vec: v,
                    aspect: 1.0,
                    coarse: Some(vec![0.0; 64]),
                    coarse_norm: 1.0,
                    rel_h: 1.0,
                    top_rel: 0.9,
                    raster: 72,
                    phase: 0.0,
                }
            })
            .collect();
        let mut all: Vec<PreparedTemplate> = vec![latin];
        let mut cjk_only: Vec<PreparedTemplate> = cjk.clone();
        all.append(&mut cjk_only);
        let refs: Vec<&PreparedTemplate> = all.iter().collect();
        let sample = vec![1.0f32; 16]; // matches 'A' perfectly
        let out = shortlist_partitioned(&sample, frame, &refs, 4, 8, 0);
        let has_latin = out.iter().any(|t| t.char == 'A');
        let cjk_count = out.iter().filter(|t| (t.char as u32) >= SCRIPT_PARTITION_CP).count();
        assert!(has_latin, "Latin 'A' must survive the partition");
        assert!(cjk_count <= 8, "CJK quota is 8, got {cjk_count}");
        assert!(cjk_count > 0, "CJK must also have representation");
    }
}
