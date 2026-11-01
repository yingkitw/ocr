//! The template-OCR pipeline (WebOCR port of `webocr/src/ocr/engine.ts`).
//!
//! binarize → deskew → segment → recognize (normalize + cosine match, with
//! the case channel, the fragment merge, the constituent re-split and the
//! rescue split) → layout reassembly. Pure functions over pixel data — no
//! ML, no I/O. Post-hoc Latin confusion correction is NOT ported here: the
//! crate's `lang::confusion_correct` (same WebOCR design, same swap table)
//! already runs on the final text in `OcrEngine` post-processing.

use std::collections::HashMap;
use std::time::Instant;

use super::binarize::binarize;
use super::classify::{match_glyph, shortlist, CaseReading, Match, MatchOptions, PreparedTemplate};
use super::components::{connected_components, median, union_box};
use super::deskew::deskew;
use super::normalize::{blur3, glyph_scale, line_baseline, place_glyph, FRAME, SCALE_REF};
use super::segment::{
    detect_space_threshold, dot_guided_split, extract_lines, filter_non_text, filter_noise,
    merge_stacked, split_candidates, SplitOptions,
};
use super::types::{
    BinaryImage, CharBox, Comp, GrayImage, LineResult, TemplateOcrResult, Timing, WordSpan,
};

/// Tunables for one run (WebOCR `RunOptions`, Latin defaults).
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    /// Matches below this similarity become `?` (default 0.5).
    pub low_confidence: f32,
    /// Smoothing passes on normalized glyphs (default 1).
    pub blur_passes: usize,
    /// Normalized frame side (default 30).
    pub frame: usize,
    /// Run in CJK mode: enable CJK stroke clustering in line
    /// segmentation, use ideographic spaces (U+3000) for indents,
    /// and disable word-space insertion in zh/ja (Korean keeps spaces
    /// because it orthographically uses them).
    pub cjk: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            low_confidence: 0.5,
            blur_passes: 1,
            frame: FRAME,
            cjk: false,
        }
    }
}

/// How close two case readings must be to count as the same glyph.
///
/// Measured across the seven default faces: where a lowercase "l" and an
/// uppercase "I" are the SAME glyph the templates are bit-identical and the
/// scores differ by at most 0.0046 (relative 0.5 %); where they are
/// genuinely different (a serif foot on the "l") the gap is at least 0.02
/// (2.4 %). 1 % sits in the middle of that gap with room on both sides.
const CASE_TIE_EPS: f32 = 0.01;

/// How close two case readings' ink tops must be to count as the same glyph.
///
/// The case signal is the ink top above the baseline, in em units: an "l"
/// and an "I" are the same stem at the same height (they agree within a
/// tenth, even across faces), while an "s" and an "S" differ by a third of
/// an em (the x-height against the cap height). The tolerance sits between
/// the two.
const CASE_TOP_EPS: f32 = 0.15;

/// The case-height penalty weight (Latin only).
///
/// How hard a mismatch between the sample's ink top above the baseline and
/// the template's is punished. 0 disables the channel.
const TOP_REL_WEIGHT: f32 = 0.15;

/// The score below which a glyph is re-matched against the phase-variant
/// templates.
///
/// Measured on the Latin faces scorecard: 0.90–0.94 all read 97.39 % (a flat
/// optimum), 0.85 and 0.88 fall to 97.15 % — the rescue fires on glyphs that
/// were read correctly and a phase variant of the WRONG character wins.
const PHASE_RESCUE: f32 = 0.92;

/// How much better a phase variant must read before it replaces the main
/// match.
///
/// The variants are near-duplicates of the characters they mirror, so a
/// variant of the WRONG character can outscore the right reading by a hair —
/// and a hair is enough to lose a glyph. The rescue exists for the PHASE
/// MISMATCH, where the score jump is large, so a margin costs the rescue
/// nothing and keeps it from trading a right reading for a near-duplicate of
/// a wrong one.
const PHASE_MARGIN: f32 = 0.02;

/// The score below which a wide glyph's valley cuts are tried — the SPLIT
/// rescue.
///
/// The rescue used to share the "?" bar (0.5), which is a different question:
/// the bar decides what to PRINT, the rescue decides whether to LOOK AGAIN.
/// A glyph that reads 0.58 is not a "?" but it is not a character either.
const SPLIT_RESCUE: f32 = 0.95;

/// How much better the two halves must read than the fused glyph for the
/// split to be kept.
const SPLIT_MARGIN: f32 = 0.05;

/// The fragment merge thresholds (WebOCR `DEFAULT_MERGE_OPTIONS`).
///
/// A merge is only ATTEMPTED when one of the pair reads poorly (both
/// fragments of a broken glyph look like punctuation or a stray stem), and
/// it is only KEPT when the merged reading beats the weaker of the two by
/// the margin. That guard is what keeps "rn" from becoming "m".
const MERGE_FLOOR: f32 = 0.95;
const MERGE_MARGIN: f32 = 0.05;
/// The largest gap (× the median height) a merge may cross — a word space
/// is wider than this, so a merge never reaches across one.
const MERGE_MAX_GAP_RATIO: f32 = 0.3;
/// The widest merged box (× the median height) a merge may make — a single
/// character is narrower than two.
const MERGE_MAX_WIDTH_RATIO: f32 = 1.8;

/// One glyph of the line after the merge pass.
struct MergeUnit {
    box_: Comp,
    match_: Match,
    /// How many source glyphs this unit covers.
    span: usize,
}

/// One glyph's match result, with the spaces that precede it on the line.
struct Decision {
    parts: Vec<(Comp, Match)>,
    /// The whitespace between the previous glyph and this one.
    spaces: String,
}

/// A glyph's ink extent along the baseline: the x-range of the ink that sits
/// ON the line, ignoring the descenders that hang below it.
///
/// A Georgia "j" leans its tail three pixels to the left, and that overhang
/// closed the word space in front of it — "liquor jugs" came out as
/// "liquorjugs" (WebOCR measured: the gap was 5 px against a 7.5 px
/// threshold; with the baseline-anchored ink it is 9 px). A word space is a
/// gap in the x-height line, not in the descender's reach. A glyph with no
/// ink above the baseline (a comma) falls back to its own box.
fn baseline_extent(bin: &BinaryImage, g: &Comp, baseline: f32) -> (usize, usize) {
    let top = g.y1.min(baseline.max(0.0) as usize);
    let mut x0 = usize::MAX;
    let mut x1 = 0usize;
    for y in g.y0..top {
        for x in g.x0..g.x1 {
            if bin.get(x, y) == 1 {
                x0 = x0.min(x);
                x1 = x1.max(x);
            }
        }
    }
    if x0 == usize::MAX {
        (g.x0, g.x1)
    } else {
        (x0, x1)
    }
}

/// The font's cap-height ratio (the ink top of a capital, in em units),
/// measured from the templates themselves: the most common `top_rel` is the
/// capital height. It is what turns an all-caps line's tallest band back
/// into an em — a fixed 0.72 would be wrong for a pixel font, whose
/// capitals fill their em.
fn cap_ratio(templates: &[PreparedTemplate]) -> f32 {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for t in templates {
        if t.top_rel <= 0.0 {
            continue;
        }
        // Quantize to 1/20 em — the resolution of the measurement.
        *counts.entry((t.top_rel * 20.0).round() as u32).or_insert(0) += 1;
    }
    // The winner is the most common ink top. Ties are broken by the KEY,
    // not by the hash map's iteration order: `HashMap` is seeded per
    // process, so an order-dependent tie makes the whole reading
    // non-deterministic between runs.
    let mut best: Option<(u32, usize)> = None;
    for (key, count) in counts {
        match best {
            Some((bk, bc)) if bc > count || (bc == count && key > bk) => {}
            _ => best = Some((key, count)),
        }
    }
    best.map(|(k, _)| k as f32 / 20.0).unwrap_or(0.0)
}

/// The case decision for a glyph whose two case readings tie.
///
/// A lowercase "l" and an uppercase "I" are the SAME bare stroke at the SAME
/// height in a sans font: the templates are bit-identical, the scores tie
/// exactly, and the case channel (which reads the ink top) has nothing to
/// say — an "l" is ascender-height and an "I" is cap-height, and in Arial
/// those are the same number. So the shape cannot decide, and without a
/// decision EVERY lowercase "l" in a sans-serif scan is read as an "I".
///
/// The word decides. The letters around the glyph show its case: the word
/// around a bare stem in "lazy" is lowercase, the word around one in "IBM"
/// is uppercase. A word with no other letters to learn from (a lone "I")
/// falls back to the line's own case signal — a line that shows case
/// variation at all is mostly lowercase, because that is what running text
/// is.
fn resolve_case(decisions: &mut [Decision], mixed_line: bool) {
    // The glyphs whose case readings are the SAME glyph. Two conditions:
    //
    // - The two readings must be within CASE_TIE_EPS of each other in score —
    //   the shape does not prefer either.
    // - Their ink tops must AGREE (the templates' top_rel within
    //   CASE_TOP_EPS). This is the case signal itself: an "l" and an "I" are
    //   the same stem at the same height in a sans font, so their ink tops
    //   agree and the shape carries no case information at all. An "s" and an
    //   "S" differ in height (0.52 vs 0.72 em), so the shape DOES carry the
    //   case and the shape's answer stands.
    let mut readings: Vec<Option<(CaseReading, CaseReading)>> = vec![None; decisions.len()];
    let mut ambiguous = vec![false; decisions.len()];
    for (i, d) in decisions.iter().enumerate() {
        if d.parts.len() != 1 {
            continue;
        }
        let m = d.parts[0].1;
        let (Some(lower), Some(upper)) = (m.lower, m.upper) else {
            continue;
        };
        // The winner must be ONE of the two case readings. Otherwise the
        // glyph is not a case question at all: a hanzi that happens to match
        // an ASCII letter more closely than another ASCII letter must keep
        // its hanzi.
        if m.char != lower.char && m.char != upper.char {
            continue;
        }
        let gap = (lower.score - upper.score).abs();
        let scale = lower.score.max(upper.score).max(f32::MIN_POSITIVE);
        if gap > CASE_TIE_EPS * scale {
            continue;
        }
        if lower.top_rel <= 0.0 || upper.top_rel <= 0.0 {
            continue;
        }
        if (lower.top_rel - upper.top_rel).abs() > CASE_TOP_EPS * lower.top_rel.max(upper.top_rel) {
            continue;
        }
        ambiguous[i] = true;
        readings[i] = Some((lower, upper));
    }
    if !ambiguous.iter().any(|a| *a) {
        return;
    }

    // The words: runs of glyphs with no space between them.
    let mut words: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for i in 0..decisions.len() {
        if !decisions[i].spaces.is_empty() && !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        words.push(current);
    }

    for word in words {
        // The case the word's OTHER letters show. The ambiguous glyphs do not
        // vote — their own case is what is being decided.
        let mut lower_votes = 0usize;
        let mut upper_votes = 0usize;
        for &i in &word {
            if ambiguous[i] {
                continue;
            }
            let ch = decisions[i].parts[0].1.char;
            if ch.is_ascii_lowercase() {
                lower_votes += 1;
            } else if ch.is_ascii_uppercase() {
                upper_votes += 1;
            }
        }
        let decided = if lower_votes > upper_votes {
            Some(false) // want lowercase
        } else if upper_votes > lower_votes {
            Some(true) // want uppercase
        } else {
            None
        };
        for &i in &word {
            if !ambiguous[i] {
                continue;
            }
            let (lower, upper) = readings[i].expect("ambiguous implies a reading");
            // No other letter to learn from. A word of ONE letter is the
            // pronoun "I" (nothing else is a one-letter English word), so it
            // reads uppercase; a longer word falls back to the line's own
            // case signal.
            let want_upper = decided.unwrap_or(word.len() == 1 || !mixed_line);
            let pick = if want_upper { upper } else { lower };
            let current = &mut decisions[i].parts[0].1;
            if current.char == pick.char {
                continue;
            }
            // The two readings are the same glyph (they tie within the
            // ambiguity band), so the context decides and the wanted
            // reading's own score becomes the glyph's score.
            current.char = pick.char;
            current.score = pick.score;
        }
    }
}

/// True when the template pool holds at least one Latin codepoint.
fn has_latin_in(templates: &[PreparedTemplate]) -> bool {
    templates.iter().any(|t| (t.char as u32) < 0x2E80)
}

/// The whole pipeline over a grayscale image.
pub fn run_ocr(
    gray: &GrayImage,
    templates: &[PreparedTemplate],
    options: RunOptions,
) -> TemplateOcrResult {
    let started = Instant::now();
    let t0 = 0.0f64;

    // 1 · Binarize — Otsu threshold, polarity detection, halo bias.
    let raw_bin: BinaryImage = binarize(gray).image;
    let t1 = started.elapsed().as_secs_f64() * 1000.0;

    // 2 · Deskew — projection-profile skew search. The search runs on a
    // 600px-wide downsampled copy and the full rotation only fires when
    // the page is measurably skewed (var_best/var0 > 1.3), so a clean page
    // pays only the search.
    let skew = deskew(&raw_bin);
    let bin: BinaryImage = skew.bin.clone();
    let t_skew = started.elapsed().as_secs_f64() * 1000.0;

    // 3 · Segment — connected components → lines → glyphs. CJK mode
    // enables stroke clustering (multi-component hanzi); Latin mode
    // treats each component as a glyph.
    //
    // `filter_noise` drops dust/sensor specks; `filter_non_text` drops
    // logos, rules, and decorative blobs whose shape can't be a text glyph.
    // Order matters — the specks must be gone before measuring the median
    // height that the size rule uses.
    let noisy = filter_noise(&connected_components(&bin));
    let noisy_med_h = median(&noisy.iter().map(|c| c.height()).collect::<Vec<_>>()).max(1);
    let comps = filter_non_text(&noisy, Some(noisy_med_h), options.cjk);
    let lines = extract_lines(&bin, &comps, options.cjk);
    let t2 = started.elapsed().as_secs_f64() * 1000.0;

    // 4 · Recognize — normalize each glyph (own-height scale, aspect kept)
    // and match against the template library by cosine similarity.
    let frame = options.frame;

    // The template pools. The **phase variants** (the same characters drawn
    // at a fractional x offset) are a RESCUE, never a permanent addition to
    // the candidate pool: keeping them in the pool thins every margin (a
    // near-duplicate of the right character is a near-duplicate of the wrong
    // one too). The main match ignores them; a glyph that reads poorly is
    // re-matched against them alone.
    let main_refs: Vec<&PreparedTemplate> = templates.iter().collect();
    let mut variants_by_raster: HashMap<u32, Vec<&PreparedTemplate>> = HashMap::new();
    for t in templates.iter().filter(|t| t.phase != 0.0) {
        variants_by_raster.entry(t.raster).or_default().push(t);
    }
    // The rasters, sorted — the phase pool is picked by proximity to the
    // line's em, and a `HashMap`'s iteration order is seeded per process,
    // so an order-dependent tie would make the reading non-deterministic.
    let mut variant_rasters: Vec<u32> = variants_by_raster.keys().copied().collect();
    variant_rasters.sort_unstable();
    let shortlist_needed = main_refs.len() > 1500;
    // CJK libraries use shift search + relH penalty by default
    // (WebOCR `useShift = cjk ? 2 : 0`, `relHWeight = cjk ? 0.3 : 0`).
    let has_cjk_lib = main_refs.iter().any(|t| (t.char as u32) >= 0x2E80);
    let has_latin_lib = main_refs.iter().any(|t| (t.char as u32) < 0x2E80);
    let match_shift = if has_cjk_lib { 2isize } else { 0 };
    let match_rel_h_weight = if has_cjk_lib { 0.3f32 } else { 0.0 };
    // The case channel is Latin-only: a hanzi's ink top carries no case.
    let match_top_rel_weight = if has_cjk_lib && !has_latin_lib {
        0.0
    } else {
        TOP_REL_WEIGHT
    };

    // The Latin pools, resolved once — the Latin pool for the constituents
    // re-split, and the punctuation-only Latin subset for glyphs that are
    // too short to be a letter.
    let latin_refs: Vec<&PreparedTemplate> =
        main_refs.iter().copied().filter(|t| (t.char as u32) < 0x2E80).collect();
    // Letters and digits are never shorter than ~x-height; only punctuation
    // is. A sub-half-median glyph (。, ·) must not lose to a look-alike
    // letter ("o") — the Latin punct subset covers it instead.
    let latin_punct_refs: Vec<&PreparedTemplate> = latin_refs
        .iter()
        .copied()
        .filter(|t| !t.char.is_ascii_alphanumeric())
        .collect();
    let cjk_refs: Vec<&PreparedTemplate> =
        main_refs.iter().copied().filter(|t| (t.char as u32) >= 0x2E80).collect();

    // The font's cap-height ratio, measured from the templates.
    let cap_ratio = cap_ratio(templates);

    let mut results: Vec<LineResult> = Vec::new();
    let mut score_sum = 0.0f32;
    let mut char_count = 0usize;

    for line in &lines {
        if line.glyphs.is_empty() {
            continue;
        }
        // Baseline estimate: the median glyph bottom — most glyphs sit on it.
        let baseline = line_baseline(&line.glyphs);
        // The line's tallest glyph (cap/ascender height) — the scale the
        // word-gap guard uses, and the right reference for "is this a
        // gutter" (a monospace word space is 2.2× the x-height and would
        // otherwise split the line).
        let line_height = line.glyphs.iter().map(|g| g.height()).max().unwrap_or(1);

        // The gaps are measured between the glyphs' BASELINE-anchored ink: a
        // descender that leans into the next glyph must not close the word
        // space in front of it ("liquor jugs" → "liquorjugs").
        let extents: Vec<(usize, usize)> = line
            .glyphs
            .iter()
            .map(|g| baseline_extent(&bin, g, baseline))
            .collect();
        // Only positive gaps carry spacing information; a zero or negative
        // gap means the glyphs touch.
        let gaps: Vec<usize> = line
            .glyphs
            .windows(2)
            .enumerate()
            .map(|(i, _)| extents[i + 1].0.saturating_sub(extents[i].1))
            .filter(|g| *g > 0)
            .collect();
        // WebOCR `useSpaces`: zh/ja orthography has no inter-word spaces, so
        // we set gap_limit to Infinity and skip word insertion. Korean keeps
        // spaces, but for this port's CJK pass the library is the union
        // (HAN + KANA + HANGUL + ASCII), so the "CJK" mode here behaves like
        // the union's space policy.
        let gap_limit = if options.cjk {
            f32::INFINITY
        } else {
            detect_space_threshold(&gaps, line.med_h, line_height)
        };
        // The line's typical WORD gap — the unit for counting spaces in the
        // layout mode. The pitch (the median advance) is the wrong unit: in a
        // monospace face a word space is a full advance plus the
        // punctuation's own advance, so counting by pitch turned
        // "in, clean" into two spaces.
        let word_gaps: Vec<usize> = gaps.iter().copied().filter(|g| *g as f32 > gap_limit).collect();
        let word_gap = if word_gaps.is_empty() {
            0.0
        } else {
            median(&word_gaps) as f32
        };

        // The line's em — the scale the case channel compares against.
        //
        // The x-height band gives it when the line shows case variation (the
        // x-height is ≈0.5 em); otherwise the tallest band IS the cap height
        // and the font's own cap-height ratio (measured from the templates)
        // turns it back into an em.
        let tops: Vec<f32> = line
            .glyphs
            .iter()
            .map(|g| baseline - g.y0 as f32)
            .filter(|v| *v > 0.0)
            .collect();
        let mut em_estimate = 0.0f32;
        if match_top_rel_weight > 0.0 && tops.len() >= 3 {
            let mut sorted = tops.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // The mode: the band with the most members (±12 %).
            let mut best_band = (sorted[0], 0usize);
            for &h in &sorted {
                let count = sorted.iter().filter(|v| (**v - h).abs() <= 0.12 * h).count();
                if count > best_band.1 {
                    best_band = (h, count);
                }
            }
            let tallest = sorted[sorted.len() - 1];
            if best_band.1 >= 3 && best_band.0 < tallest * 0.92 {
                // The line has a lower band and an upper one: the modal band
                // is the x-height, the upper one the capitals.
                em_estimate = best_band.0 / 0.5;
                // The case channel is a DIFFERENCE, so it only makes sense on
                // a line that actually shows case variation: a uniform-height
                // line (all caps, or a pixel font with no x-height/cap
                // distinction) has no case signal, and a flat penalty there
                // would just push every score under the confidence bar.
                if tallest / best_band.0 < 1.15 {
                    em_estimate = 0.0;
                }
            }
        }
        let line_em = if options.cjk {
            line.med_h as f32
        } else if em_estimate > 0.0 {
            em_estimate
        } else {
            let t = tops.iter().copied().fold(0.0f32, f32::max);
            if t > 0.0 && cap_ratio > 0.0 {
                t / cap_ratio
            } else {
                line.med_h as f32
            }
        };

        // Normalize one glyph into its matching frame.
        let normalize_glyph = |g: &Comp| -> (Vec<f32>, f32, f32, f32) {
            let mut vec = place_glyph(&bin, g, glyph_scale(g, SCALE_REF), baseline, frame);
            for _ in 0..options.blur_passes {
                vec = blur3(&vec, frame);
            }
            let aspect = g.width() as f32 / g.height().max(1) as f32;
            let rel_h = g.height() as f32 / line.med_h.max(1) as f32;
            let top_rel = if line_em > 0.0 {
                (baseline - g.y0 as f32) / line_em
            } else {
                0.0
            };
            (vec, aspect, rel_h, top_rel)
        };

        // Match a glyph against an explicit candidate pool.
        let match_in_pool =
            |g: &Comp, pool: &[&PreparedTemplate], bar: Option<f32>| -> Match {
                let (vec, aspect, rel_h, top_rel) = normalize_glyph(g);
                match_glyph(
                    &vec,
                    aspect,
                    pool,
                    MatchOptions {
                        frame,
                        rel_h,
                        rel_h_weight: match_rel_h_weight,
                        top_rel,
                        top_rel_weight: match_top_rel_weight,
                        shift: match_shift,
                        min_score: bar,
                    },
                )
            };

        // The phase-variant pool for this line: the variants drawn at the
        // raster the line is at. A 16px variant is no use for a 30px sample.
        let rescue_pool = || -> Vec<&PreparedTemplate> {
            if variant_rasters.is_empty() {
                return Vec::new();
            }
            let mut best_raster = variant_rasters[0];
            let mut best_dist = f32::MAX;
            for raster in &variant_rasters {
                let d = (*raster as f32 - line_em).abs();
                if d < best_dist {
                    best_dist = d;
                    best_raster = *raster;
                }
            }
            variants_by_raster
                .get(&best_raster)
                .cloned()
                .unwrap_or_default()
        };

        // Match one glyph box — also used for rescue-split halves. Returns
        // the match and the candidate pool it was drawn from (the merge
        // pass needs the pool).
        let match_one = |g: &Comp| -> (Match, Vec<&PreparedTemplate>) {
            let (vec, aspect, rel_h, top_rel) = normalize_glyph(g);
            let (mut pool, mut m) = if shortlist_needed && has_cjk_lib && has_latin_lib {
                // Per-subset scoring: a union library carries two script
                // families whose match params differ — Latin templates
                // score cleanest the way Latin mode scores (no shift
                // search, no relH penalty: the sample relH is measured
                // against the line median while the template relH is
                // against its raster em, so the CJK-weighted penalty
                // systematically punishes x-height letters and flips
                // u/U). CJK candidates keep shift + relH.
                let lat_pool = shortlist(
                    &vec,
                    frame,
                    if rel_h < 0.42 {
                        &latin_punct_refs
                    } else {
                        &latin_refs
                    },
                    32,
                    1,
                );
                let cj_pool = shortlist(&vec, frame, &cjk_refs, 64, 1);
                let lat = match_glyph(
                    &vec,
                    aspect,
                    &lat_pool,
                    MatchOptions {
                        frame,
                        rel_h,
                        rel_h_weight: 0.0,
                        top_rel,
                        top_rel_weight: match_top_rel_weight,
                        shift: 0,
                        min_score: None,
                    },
                );
                let cj = match_glyph(
                    &vec,
                    aspect,
                    &cj_pool,
                    MatchOptions {
                        frame,
                        rel_h,
                        rel_h_weight: match_rel_h_weight,
                        top_rel,
                        top_rel_weight: 0.0,
                        shift: match_shift,
                        min_score: None,
                    },
                );
                let mut both = lat_pool;
                both.extend(cj_pool);
                let m = if lat.score > cj.score { lat } else { cj };
                (both, m)
            } else {
                let pool = if shortlist_needed {
                    shortlist(&vec, frame, &main_refs, 64, 1)
                } else {
                    main_refs.clone()
                };
                let m = match_glyph(
                    &vec,
                    aspect,
                    &pool,
                    MatchOptions {
                        frame,
                        rel_h,
                        rel_h_weight: match_rel_h_weight,
                        top_rel,
                        top_rel_weight: match_top_rel_weight,
                        shift: match_shift,
                        min_score: None,
                    },
                );
                (pool, m)
            };
            // The phase rescue: re-read a glyph that read poorly against
            // the phase-variant templates drawn at the line's raster. The
            // variant pool is narrowed to the character the main match chose
            // — a variant of a DIFFERENT character is a near-duplicate of
            // the wrong answer.
            if m.score < PHASE_RESCUE {
                let variants = rescue_pool();
                if !variants.is_empty() {
                    let same: Vec<&PreparedTemplate> =
                        variants.iter().copied().filter(|t| t.char == m.char).collect();
                    if !same.is_empty() {
                        let v = match_glyph(
                            &vec,
                            aspect,
                            &same,
                            MatchOptions {
                                frame,
                                rel_h,
                                rel_h_weight: match_rel_h_weight,
                                top_rel,
                                top_rel_weight: match_top_rel_weight,
                                shift: match_shift,
                                min_score: None,
                            },
                        );
                        if v.score > m.score + PHASE_MARGIN {
                            // The case readings stay the main match's: the
                            // phase moves both cases of a letter equally,
                            // and the case decision needs the pair the main
                            // pool produced.
                            m = Match {
                                char: v.char,
                                score: v.score,
                                margin: v.margin,
                                lower: m.lower,
                                upper: m.upper,
                            };
                        }
                    }
                }
            }
            let _ = &mut pool;
            (m, pool)
        };

        // Latin-only matcher for the constituents re-split below. The
        // pool is filtered to Latin templates (cp < 0x2E80), and the
        // relH penalty is disabled — a CJK glyph never matches here, so
        // the penalty just confuses close calls between letters.
        let match_latin = |g: &Comp| -> Match {
            let (vec, aspect, rel_h, top_rel) = normalize_glyph(g);
            let pool = if shortlist_needed && has_latin_lib {
                let sub = if rel_h < 0.42 {
                    &latin_punct_refs
                } else {
                    &latin_refs
                };
                shortlist(&vec, frame, sub, 32, 1)
            } else {
                latin_refs.clone()
            };
            match_glyph(
                &vec,
                aspect,
                &pool,
                MatchOptions {
                    frame,
                    rel_h,
                    rel_h_weight: 0.0,
                    top_rel,
                    top_rel_weight: match_top_rel_weight,
                    shift: 0,
                    min_score: None,
                },
            )
        };

        // Phase 1a · every glyph on its own.
        let reads: Vec<(Match, Vec<&PreparedTemplate>)> =
            line.glyphs.iter().map(|g| match_one(g)).collect();

        // Phase 1b · the fragment merge: a glyph whose ink broke apart is
        // put back together when the two halves read better as one
        // character. A template is built from a character rendered ALONE;
        // a scan's ink can break that character apart, and then each
        // fragment becomes a glyph of its own and loses against the whole.
        let mut units: Vec<MergeUnit> = Vec::new();
        let mut i = 0usize;
        while i < line.glyphs.len() {
            let mut box_ = line.glyphs[i];
            let mut m = reads[i].0;
            let mut pool = reads[i].1.clone();
            let mut span = 1usize;
            // Absorb the next fragment while it keeps paying: a "w" can
            // break into three strokes.
            loop {
                let Some((next_m, next_pool)) = reads.get(i + span) else {
                    break;
                };
                if m.score.min(next_m.score) >= MERGE_FLOOR {
                    break;
                }
                let next = line.glyphs[i + span];
                if (next.x0 as f64 - box_.x1 as f64) > MERGE_MAX_GAP_RATIO as f64 * line.med_h as f64 {
                    break;
                }
                let cand = Comp {
                    x0: box_.x0.min(next.x0),
                    y0: box_.y0.min(next.y0),
                    x1: box_.x1.max(next.x1),
                    y1: box_.y1.max(next.y1),
                    area: 0,
                };
                if (cand.x1 - cand.x0) as f64 > MERGE_MAX_WIDTH_RATIO as f64 * line.med_h as f64 {
                    break;
                }
                // The merged reading is scored against the UNION of the two
                // fragments' candidates, not against a fresh shortlist: the
                // coarse pre-filter is the expensive part of a match, and the
                // character a merged pair reads as is already a candidate
                // for at least one of the halves.
                let mut cand_pool = pool.clone();
                for t in next_pool {
                    if !cand_pool.iter().any(|x| std::ptr::eq(*x, *t)) {
                        cand_pool.push(t);
                    }
                }
                // The bar: the merged reading has to beat the weaker
                // fragment by the margin.
                let bar = m.score.min(next_m.score) + MERGE_MARGIN;
                let cand_m = match_in_pool(&cand, &cand_pool, Some(bar));
                if cand_m.score <= bar {
                    break;
                }
                box_ = cand;
                m = cand_m;
                pool = cand_pool;
                span += 1;
            }
            units.push(MergeUnit {
                box_,
                match_: m,
                span,
            });
            i += span;
        }

        // The gaps the spaces are measured on: the UNITS' ink, so a merged
        // fragment no longer leaves a hole where its own seam was.
        let unit_extents: Vec<(usize, usize)> = units
            .iter()
            .map(|u| baseline_extent(&bin, &u.box_, baseline))
            .collect();

        // Phase 1c · the per-unit decisions.
        let mut decisions: Vec<Decision> = Vec::with_capacity(units.len());
        for (gi, unit) in units.iter().enumerate() {
            let g = &unit.box_;
            // The whitespace before this glyph, on the same baseline-anchored
            // gap the threshold was measured on.
            let mut spaces = String::new();
            if gi > 0 {
                let gap = unit_extents[gi].0.saturating_sub(unit_extents[gi - 1].1) as f32;
                if gap > gap_limit {
                    // One space per word gap; a gap that is clearly wider than
                    // a word space (a column, a tab stop) expands by the
                    // extra word gaps it contains.
                    let extra = if word_gap > 0.0 {
                        ((gap - word_gap) / word_gap).round().max(0.0) as usize
                    } else {
                        0
                    };
                    for _ in 0..1 + extra {
                        spaces.push(' ');
                    }
                }
            }

            let match_ = unit.match_;
            let mut parts: Vec<(Comp, Match)> = vec![(*g, match_)];

            // Constituents re-split (webocr §3 "let the match decide"):
            // when a CJK-mode glyph matches poorly AND the library has
            // Latin templates, reconstruct the raw components inside the
            // box and re-evaluate them as Latin letters. Geometry alone
            // can't tell a detached-stroke hanzi (训's 讠+川) from a
            // fused Latin pair ("i"+"x") — both are tight unions under
            // the width cap. The match decides: keep the split only if
            // every part scores ≥ 0.72 against Latin AND the minimum
            // beats the fused score by ≥ 0.05. Real hanzi pass (whole
            // score high) or fail safely (a stroke like 弓 never clears
            // the Latin floor).
            if has_cjk_lib && has_latin_lib && match_.score < 0.75 {
                let inside: Vec<Comp> = comps
                    .iter()
                    .copied()
                    .filter(|c| {
                        c.x0 + 1 >= g.x0
                            && c.x1 <= g.x1 + 1
                            && c.y0 + 1 >= g.y0
                            && c.y1 <= g.y1 + 1
                    })
                    .collect();
                let pieces = merge_stacked(&inside, line.med_h);
                if pieces.len() >= 2 {
                    let ms: Vec<Match> = pieces.iter().map(|p| match_latin(p)).collect();
                    let min = ms
                        .iter()
                        .map(|m| m.score)
                        .fold(f32::INFINITY, f32::min);
                    if min >= 0.72 && min > match_.score + 0.05 {
                        parts = pieces.into_iter().zip(ms).collect();
                    }
                }
            }

            // The rescue split: a wide glyph that matched badly may be a
            // kerned pair the bridge guard spared in pass 1. Score several
            // valley cuts — the lightest column is not always the true seam
            // — and keep the split only if BOTH halves beat the fused score:
            // the score is the guard.
            //
            // A kerned pair can also fuse into a glyph that matches
            // CONFIDENTLY (Georgia's "wi" reads as "M"), so the rescue also
            // fires on a GEOMETRIC signal: a detached top mark near one end
            // of the box, which says where the seam is.
            if parts.len() == 1
                && g.width() as f32 >= 1.4 * line.med_h as f32
                && g.height() as f32 >= 0.5 * line.med_h as f32
            {
                let mut best_min = match_.score;
                let mut cands: Vec<[Comp; 2]> = Vec::new();
                if match_.score < SPLIT_RESCUE {
                    cands.extend(split_candidates(
                        &bin,
                        g,
                        line.med_h,
                        SplitOptions {
                            max_ratio: 0.7,
                            check_bridge: false,
                            min_part: 0.35,
                            strict: false,
                        },
                    ));
                }
                if let Some(marked) = dot_guided_split(&comps, g, line.med_h) {
                    cands.push(marked);
                }
                for halves in cands {
                    let m1 = match_one(&halves[0]).0;
                    let m2 = match_one(&halves[1]).0;
                    let m = m1.score.min(m2.score);
                    if m > best_min + SPLIT_MARGIN {
                        best_min = m;
                        parts = vec![(halves[0], m1), (halves[1], m2)];
                    }
                }
            }

            decisions.push(Decision { parts, spaces });
        }

        // Phase 2 · the case decision.
        resolve_case(&mut decisions, em_estimate > 0.0);

        // Phase 3 · emit the line.
        let mut text = String::new();
        let mut chars: Vec<CharBox> = Vec::new();
        let mut words: Vec<WordSpan> = Vec::new();
        for d in &decisions {
            // A word space (or the line's first glyph) opens a new word.
            if words.is_empty() || !d.spaces.is_empty() {
                words.push(WordSpan {
                    text: String::new(),
                    chars: Vec::new(),
                    score: 0.0,
                });
            }
            text.push_str(&d.spaces);
            for (box_, m) in &d.parts {
                let chosen = if m.score < options.low_confidence {
                    '?'
                } else {
                    m.char
                };
                let cb = CharBox {
                    x0: box_.x0,
                    y0: box_.y0,
                    x1: box_.x1,
                    y1: box_.y1,
                    char: chosen,
                    score: m.score,
                };
                if let Some(w) = words.last_mut() {
                    w.text.push(chosen);
                    w.chars.push(CharBox {
                        x0: cb.x0,
                        y0: cb.y0,
                        x1: cb.x1,
                        y1: cb.y1,
                        char: cb.char,
                        score: cb.score,
                    });
                }
                text.push(chosen);
                chars.push(cb);
                score_sum += m.score;
                char_count += 1;
            }
        }
        for w in &mut words {
            if !w.chars.is_empty() {
                w.score = w.chars.iter().map(|c| c.score).sum::<f32>() / w.chars.len() as f32;
            }
        }

        let box_ = union_box(
            &chars
                .iter()
                .map(|c| Comp {
                    x0: c.x0,
                    y0: c.y0,
                    x1: c.x1,
                    y1: c.y1,
                    area: 0,
                })
                .collect::<Vec<_>>(),
        );
        if let Some(b) = box_ {
            results.push(LineResult {
                x0: b.x0,
                y0: b.y0,
                x1: b.x1,
                y1: b.y1,
                text,
                chars,
                words,
                block: line.block,
            });
        }
    }
    let t3 = started.elapsed().as_secs_f64() * 1000.0;

    // 5 · Layout — reassemble line geometry into the text: a blank line
    // where the vertical gap clearly exceeds normal leading (a paragraph
    // break), leading whitespace where a line starts right of the document
    // edge, and column spacing where a sibling block continues the same row.
    let same_row_break = |i: usize| -> bool {
        let line = &results[i];
        let prev = &results[i - 1];
        line.block != prev.block
            && (line.y1.min(prev.y1) as i64 - line.y0.max(prev.y0) as i64)
                > (5 * (line.y1 - line.y0).min(prev.y1 - prev.y0) / 10) as i64
            && line.x0 > prev.x1
    };

    let mut text = String::new();
    if !results.is_empty() {
        let med_line_h = median(&results.iter().map(|l| l.height()).collect::<Vec<_>>()).max(1);
        let doc_left = results.iter().map(|l| l.x0).min().unwrap_or(0);
        for i in 0..results.len() {
            let line = &results[i];
            let row_continuation = i > 0 && same_row_break(i);
            if row_continuation {
                // Adjacent block on the same visual row (a side-by-side
                // region, not a paragraph break) — continue the line with
                // column spacing.
                let prev = &results[i - 1];
                let pitch = prev
                    .chars
                    .windows(2)
                    .map(|w| w[1].x0.saturating_sub(w[0].x0))
                    .collect::<Vec<_>>();
                let pitch = median(&pitch).max(med_line_h / 2).max(1) as f32;
                let n = (((line.x0 as i64 - prev.x1 as i64).max(0) as f32 / pitch).round()
                    as usize)
                    .max(2);
                for _ in 0..n {
                    text.push(' ');
                }
            } else if i > 0 {
                // A block boundary only earns a paragraph break when it
                // isn't a plain vertical continuation — same-column lines
                // split by tight merge thresholds still read as ordinary
                // lines; the gap threshold covers real paragraph spacing.
                let prev = &results[i - 1];
                let x_overlap = line.x1.min(prev.x1) as i64 - line.x0.max(prev.x0) as i64;
                let block_break = line.block != prev.block && x_overlap <= 0;
                if line.y0 as i64 - prev.y1 as i64 > (17 * med_line_h / 10) as i64
                    || block_break
                {
                    text.push_str("\n\n");
                } else {
                    text.push('\n');
                }
            }
            if !row_continuation {
                let advances: Vec<usize> = line
                    .chars
                    .windows(2)
                    .map(|w| w[1].x0.saturating_sub(w[0].x0))
                    .collect();
                let pitch = median(&advances).max(med_line_h / 2).max(1) as f32;
                let cells = ((line.x0 as i64 - doc_left as i64).max(0) as f32 / pitch).round()
                    as usize;
                // CJK would use an ideographic space here (\u{3000}) so 2-em
                // indents land right; the Latin cut emits ASCII spaces.
                for _ in 0..cells {
                    text.push(' ');
                }
            }
            text.push_str(&line.text);
        }
    }

    if text.is_empty() {
        // Flat assembly: same-row block pairs continue the line with a
        // space; everything else is \n.
        for i in 0..results.len() {
            if i > 0 {
                text.push(if same_row_break(i) { ' ' } else { '\n' });
            }
            text.push_str(&results[i].text);
        }
    }

    // The deskew moved the geometry: map every box back into the original
    // image's coordinates so the crate's API keeps its contract (boxes are
    // in image coordinates).
    if skew.angle != 0.0 {
        for line in &mut results {
            let (x0, y0, x1, y1) =
                skew.map_box_back(line.x0 as f64, line.y0 as f64, line.x1 as f64, line.y1 as f64);
            line.x0 = x0.max(0.0).round() as usize;
            line.y0 = y0.max(0.0).round() as usize;
            line.x1 = x1.max(0.0).round() as usize;
            line.y1 = y1.max(0.0).round() as usize;
            for c in &mut line.chars {
                let (cx0, cy0, cx1, cy1) =
                    skew.map_box_back(c.x0 as f64, c.y0 as f64, c.x1 as f64, c.y1 as f64);
                c.x0 = cx0.max(0.0).round() as usize;
                c.y0 = cy0.max(0.0).round() as usize;
                c.x1 = cx1.max(0.0).round() as usize;
                c.y1 = cy1.max(0.0).round() as usize;
            }
            for w in &mut line.words {
                for c in &mut w.chars {
                    let (cx0, cy0, cx1, cy1) =
                        skew.map_box_back(c.x0 as f64, c.y0 as f64, c.x1 as f64, c.y1 as f64);
                    c.x0 = cx0.max(0.0).round() as usize;
                    c.y0 = cy0.max(0.0).round() as usize;
                    c.x1 = cx1.max(0.0).round() as usize;
                    c.y1 = cy1.max(0.0).round() as usize;
                }
            }
        }
    }

    TemplateOcrResult {
        text,
        lines: results,
        width: gray.width,
        height: gray.height,
        char_count,
        avg_score: if char_count > 0 {
            score_sum / char_count as f32
        } else {
            0.0
        },
        script: if options.cjk { "cjk" } else { "latin" },
        timing: Timing {
            binarize_ms: t_skew - t0,
            segment_ms: t2 - t_skew,
            recognize_ms: t3 - t2,
            total_ms: t3,
        },
    }
}

/// Line helper used by the layout assembly above.
impl LineResult {
    fn height(&self) -> usize {
        self.y1 - self.y0
    }
}
