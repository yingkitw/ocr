# Tesseract 5.x Reverse-Engineering Study & Parity Matrix

Reverse-engineered from the Tesseract 5.5.3 source tree (`tesseract-main.zip`,
776 files, ~195k LOC of C++) plus the installed 5.5.3 binary/headers. This
document is the reference for making this crate capability-comparable with
Tesseract. Study date: 2026-10-14.

---

## 1. Tesseract architecture map

| Subsystem | LOC | Role | Our equivalent |
|---|---|---|---|
| `textord` | 42.9k | Layout analysis: line finding, tab stops, columns, stroke-width text/non-text classification, baselines, word spacing | `src/layout/` (CCL, line/column detectors, XY-cut, tables) |
| `ccutil` | 32.2k | Unicharset, params registry, tessdata manager, unichar compress (recoder) | `src/core/`, `src/lang/`, `src/recognition/model_format.rs` |
| `ccstruct` | 25.2k | Blobs, words, rows, pages, DENORM chains, Otsu | `src/core/{image,text,geometry}.rs` |
| `ccmain` | 20.2k | Pipeline control: SegmentPage/PSM dispatch, OSD, linerec, control flow | `src/core/engine.rs` |
| `training` | 18.6k | lstmtraining, combine_tessdata, text2image, legacy mf/cntraining | `src/training/`, `src/synthetic/`, `ocr train` |
| `lstm` | 18.2k | LSTM network + RecodeBeamSearch (CTC-variant beam with dict) | `src/recognition/{crnn,ctc_decoder}.rs` |
| `classify` | 17.0k | Legacy adaptive classifier (micro-features, inttemp/normproto) | *removed in v0.2 — no equivalent, intentional* |
| `wordrec` | 6.6k | Legacy chopper/segsearch + language model penalties | *no equivalent (legacy-only)* |
| `dict` | 5.0k | DAWG dictionaries consulted during beam decode | `src/lang/dictionary.rs` + `DictLmRescorer` (post-hoc) |
| `api` | 6.5k | BaseAPI, renderers (txt/hocr/alto/page/tsv/pdf/box/unlv) | `src/api/`, `src/core/output.rs`, `src/core/pdf.rs` |
| `arch` | 2.3k | SIMD/AVX/NEON dispatch | `src/utils/simd.rs`, `src/compute/` |
| `viewer` | 2.0k | ScrollView debug UI | *no equivalent* |

### Pipeline (Tesseract `--psm 3` default)

```
Threshold (Otsu | LeptonicaOtsu | Sauvola, DPI-clamped 70..2400)
→ LineFinder (rule/separator removal, Leptonica morphology)
→ ImageFind (photo mask)
→ connected components → resolution re-estimate (median_line_size × 10)
→ StrokeWidth grid (textline direction, noise, CJK rejoin)
→ vertical-text check → OSD (PSM 0/1/12) → CorrectOrientation (90° steps)
→ TabFind (tab vectors → fine deskew, capped at cos⁻¹(0.866) ≈ 30°)
→ MakeColumns (ColPartitionSet → column sets → sections)
→ TableFinder → TransformToBlocks → RotateAndReskewBlocks
→ TextordPage (rows via LMS gradient median, words, reading order)
→ per-word LSTM recognize (line crop +4px pad, 48px height, [-1,1] norm)
→ RecodeBeamSearch (60 heaps, dict-integrated)
→ post-process renderers
```

### Recognition constants (cheat sheet)

| Constant | Value | Source |
|---|---|---|
| Input line height | 48 px (`kMaxInputHeight`) | `lstm/input.cpp:28` |
| Crop padding | 4 px (`kImagePadding`) | `ccstruct/imagedata.h:39` |
| Pixel normalization | `(px − black)/contrast − 1` ∈ [−1, 1] | `lstm/networkio.cpp` |
| Beam widths per code position | `{5, 10, 16, 16, 16, 16, 16, 16, 16, 16}` | `lstm/recodebeam.cpp:31-33` |
| Heaps per timestep | 60 = 2 (dawg) × 3 (continuation) × 10 (code len) | `recodebeam.h:248` |
| Recoder max codes/char | 9 (`kMaxCodeLen`, Hangul/Han use 3) | `ccutil/unicharcompress.h:35` |
| `kMinCertainty` | −20.0 (ln prob floor) | `recodebeam.h:243` |
| `kDictRatio` (non-dict penalty) | ×2.25 | `lstmrecognizer.cpp:46` |
| `kCertOffset` (per char) | −0.085 | `lstmrecognizer.cpp:48` |
| Dict-entry gate | cert > −25/7 ≈ −3.571 | `ccmain/linerec.cpp:33-35` |
| Word certainty | `min(space_cert, choice_cert) × 7.0` | `linerec.cpp:290-299` |
| hOCR conf | `100 − lstm_rating_coefficient(5) × rating` | `api/hocrrenderer.cpp:419` |
| OSD blob filters | aspect ≤ 2.0, height ≥ 10 px, 50–250 blobs sampled | `ccmain/osdetect.cpp:41-42,270` |
| OSD margin guard | don't flip 180° if margin < 7.0 and non-CJK | `tesseractclass.cpp:401` |
| DPI band | 70 (`kMinCredibleResolution`) … 2400 | `publictypes.h:36-43` |
| Deskew cap | `kCosMaxSkewAngle = 0.866025` ≈ 30° | `textord/tabfind.cpp:60` |

### DAWG format (for a future port)

`SquishedDawg`: magic 42 (i16) → unicharset_size (i32) → num_edges (i32) →
`num_edges × u64` edge records. Edge = unichar_id (low bits) | 3 flag bits
(word-end / direction / marker) | next-node index (high bits). Types:
`PUNCTUATION, WORD, NUMBER, PATTERN`; permuter types distinguish
system/freq/user dawgs. Successor rules `kDawgSuccessors`: punc→{word,number},
word/number→punc. **The LSTM beam advances `DawgPositionVector`s inline**
(`Dict::def_letter_is_okay`), not post-hoc — this is the main accuracy gap
between our post-hoc `DictLmRescorer` and Tesseract.

---

## 2. PSM (Page Segmentation Modes) — full contract

`include/tesseract/publictypes.h:157-178`. Enum is ordered by decreasing
layout work so predicate *inequalities* work — we copied that design.

| # | Name | Layout? | Our variant | Our behavior |
|---|---|---|---|---|
| 0 | `osd_only` | none | `OsdOnly` | orientation+script report, no OCR output |
| 1 | `auto_osd` | full | `AutoOsd` | auto-rotate then auto pipeline; OSD in metadata |
| 2 | `auto_only` | full | `AutoOnly` | layout only, no recognition |
| 3 | `auto` | full | `Auto` | full pipeline (default, CLI and lib) |
| 4 | `single_column` | blocks only | `SingleColumn` | full page as one region |
| 5 | `single_block_vert_text` | minimal | `SingleBlockVertText` | rotate 90° CCW then single block |
| 6 | `single_block` | none | `SingleBlock` | full page as one region |
| 7 | `single_line` | none | `SingleLine` | whole image = one line |
| 8 | `single_word` | none | `SingleWord` | whole image = one word |
| 9 | `circle_word` | circle removal | `CircleWord` | remove enclosing ring, then single word |
| 10 | `single_char` | none | `SingleChar` | whole image = one character |
| 11 | `sparse_text` | sparse | `SparseText` | find as much text as possible, no order |
| 12 | `sparse_text_osd` | sparse | `SparseTextWithOsd` | sparse + OSD metadata |
| 13 | `raw_line` | none | `SingleLineRaw` | like 7, minus Tesseract-specific row hacks |

Both numeric (`--psm 11`) and symbolic (`--psm sparse_text`) spellings are
accepted, case-insensitive — mirroring Tesseract's `stringToPSM`.

**Deviation from Tesseract (documented):** our OSD derives orientation from
horizontal-projection variance across 4 rotations (`orientation_scores`) for
the axis plus vertical-mass asymmetry (ascender/caps mass vs descenders, with
an "upside-down is improbable" evidence guard) for the polarity, with a
normalized confidence (0–100); script comes from a fast single-block
recognition pass + `LanguageDetector`. Tesseract samples 50–250
blobs through its legacy adaptive classifier. Same contract, different
evidence; ours needs no separate `osd.traineddata`.

## 3. OEM

| # | Tesseract | Ours |
|---|---|---|
| 0 | Legacy engine | n/a — v0.2 removed pattern matching; `--engine pattern` maps to LSTM |
| 1 | LSTM only | LSTM only (the only engine) |
| 2 | Combined | n/a |
| 3 | Default (=1 when LSTM data present) | = LSTM |

## 4. Capability matrix

| Capability | Tesseract 5.5.3 | ocr crate (this repo) | Status |
|---|---|---|---|
| Recognition | LSTM + recodebeam, legacy engine | CRNN 5-layer CNN + 2×BiLSTM + CTC | ✅ comparable shape |
| Beam decode | dict-integrated, 60 heaps | beam(10) + n-best; **in-beam dict steering** (`beam_search_nbest_with_dictionary`, kDictRatio/kCertOffset scheme) + post-hoc `DictLmRescorer` | ✅ (2026-10-14) |
| Confidence | ln-prob certainties, scale 7 | `ConfidenceCalibrator` temperature softmax | ✅ |
| Thresholding | Otsu / LeptonicaOtsu / Sauvola | Otsu + adaptive (`image/thresholder.rs`) | ✅ (Sauvola optional TODO) |
| DPI handling | 70–2400 clamp, re-estimation | `--dpi` flag + super-resolution | ✅ |
| Layout | StrokeWidth/TabFind/ColPartition stack | CCL → line → XY-cut columns → ordering | ⚠️ simpler; tables/forms exceed Tesseract |
| PSM 0–13 | all 14 | all 14 (this change) | ✅ |
| OSD | per-blob classifier votes | projection scores + script detect | ✅ (different evidence) |
| Script/language | 163 trained langs | 8 scripts, 25+ dictionaries, per-script CRNN registry | ⚠️ coverage, not architecture |
| Whitelist/blacklist | whitelist/blacklist/unblacklist | whitelist (this change: CLI flag) | ⚠️ blacklist TODO |
| Output formats | txt hocr alto page tsv pdf box lstmbox wordstrbox unlv | text json hocr tsv csv alto box markdown structured-json rag page pdf | ✅ exceeds |
| Searchable PDF | CIDFont + ToUnicode CMap (full Unicode) | Helvetica WinAnsi (CJK skipped) | ⚠️ CJK text layer TODO |
| Training | lstmtraining (CTC) + legacy tools | pure-Rust CTC + full BPTT + Adam | ✅ |
| Multi-lang per page | per-word fallback across `-l a+b` | `--lang auto` post-detection | ⚠️ fallback TODO |
| Table detection | TableFinder (5.x) | `TableDetector` + span inference | ✅ |
| Form/KIE | none | checkbox/KV/underline fields | ✅ exceeds |
| Barcode/QR | none | none | = |
| Batch/API/server | ProcessPages, C API | async facade, REST (`web-api`), batch+progress | ✅ exceeds |

## 5. Prioritized roadmap (remaining gaps)

1. ~~**In-beam dictionary constraints**~~ — **DONE 2026-10-14**: `CtcDecoder::beam_search_nbest_with_dictionary` prices each frame `(certainty − 0.085) × 2.25` unless it extends the current word along a dictionary prefix/word (then ×1); per-path scoring (Tesseract `RecodeNode` style, not prefix-probability aggregates — see MEMORY.md §3 for why the aggregate mathematically cannot carry per-frame pricing); `Dictionary::word_prefix_set()` for O(1) prefix queries. Active whenever a dictionary is supplied (`--dict-correct`).
2. **CJK searchable PDF** — replace WinAnsi Helvetica text layer with an identity-H CIDFont + ToUnicode CMap (see `api/pdfrenderer.cpp` design comment) so Chinese/Japanese text becomes selectable.
3. **Blacklist + `user-words`/`user-patterns` files** — extend the whitelist mechanism; feed `DictLmRescorer` from user word lists.
4. **Per-word multi-language fallback** — `classify_word_and_language`-style retry: recognize word with primary model, retry with secondary when `WordsAcceptable` fails.
5. **Sauvola binarization option** (`thresholding_method=2` parity) — window = 0.33 × DPI, k = 0.34.
6. **Language-data scale** — per-language trained CRNN bundles + a `traineddata`-style multi-component model archive (see §DAWG/format notes).

## 6. Where the study came from

- Source: `~/Downloads/tesseract-main.zip` (5.5.3, extracted to a scratch dir
  during the 2026-10-14 session; key files cited by path above).
- Installed binary: `/opt/homebrew/bin/tesseract` 5.5.3 (leptonica 1.87,
  libarchive, NEON) with `tesseract-lang` (163 `traineddata` files) — useful
  for behavioral A/B tests (`phototest`-style fixtures).
- Method: full-source architecture study (three parallel subsystem passes);
  binary RE (REA) not required because complete source was available.
