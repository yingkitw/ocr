# Architecture

## Design Philosophy

This OCR system is built from first principles in Rust, informed by studying Tesseract, EasyOCR, PaddleOCR, and modern vision-language models. We prioritize:

1. **Pure Rust** — No Python or external OCR dependencies. Everything compiles with `cargo`.
2. **Incremental evolution** — Each phase delivers a working, testable pipeline.
3. **Separation of concerns** — Detection, recognition, and layout are distinct, swappable stages.
4. **Trainability** — Every recognition engine must be trainable on synthetic data before real data is required.

## High-Level Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────┐    ┌──────────┐
│ Image Input │ → │ Preprocessing │ → │ Layout Analysis │ → │ Recognition │ → │ Output   │
└─────────────┘    └──────────────┘    └────────────────┘    └─────────────┘    └──────────┘
                          │                    │
                   ┌──────┘                    └──────┐
                   ▼                                  ▼
            Binarization (Otsu,            Text region detection (CCL or CNN)
            Sauvola, Adaptive)             Line segmentation
            Noise reduction                Reading-order resolution
            Deskew / dewarp                Table / form / structure extraction
            Auto-rotate (0/90/180/270)     Word/char grouping
            Contrast enhancement
```

## Module Architecture

### API Layer (`src/api/`)
The public contract. `Ocr` is the main facade with async methods. `TextProcessor` provides filtering and cleaning.

```rust
pub struct Ocr {
    engine: RwLock<OcrEngine>,   // tokio RwLock; shared across requests
    config: OcrConfig,
}

impl Ocr {
    pub async fn recognize_text_from_file(&self, path: impl AsRef<Path>) -> Result<TextResult>;
    pub async fn analyze_layout(&self, data: &[u8], w: u32, h: u32) -> Result<LayoutResult>;
}
```

### Core Layer (`src/core/`)
Shared data structures and the central engine:

| Module | Purpose |
|--------|---------|
| `config.rs` | `OcrConfig` — top-level config with sub-configs for recognition, image processing, layout, language, performance |
| `engine.rs` | `OcrEngine` — state machine: init → load image → preprocess → segment → recognize → post-process |
| `geometry.rs` | `TBox`, `Polygon`, `Point2D` — layout geometry types |
| `layout.rs` | `LayoutResult`, `Block`, `TextRegion`, `Table`, `ReadingOrder` — layout analysis outputs |
| `output.rs` | Formatters: text, JSON, hOCR, TSV, CSV (RFC 4180 per-word rows), ALTO XML, box, Markdown, structured JSON, RAG chunks (`rag_chunks` + `format_rag_json`: one chunk per element with heading-path `section` context), PAGE XML |
| `pdf.rs` | Searchable PDF writer (feature `pdf`): zero-dependency — embeds image as JPEG XObject + invisible text layer (rendering mode 3) at word bounding boxes; built-in Helvetica + WinAnsiEncoding; multi-page support |
| `recognition.rs` | `RecognitionResult`, `CharacterRecognition`, `WordRecognition`, `LineRecognition` |
| `text.rs` | `TextResult` — final unified output with confidence, bounding boxes, lines, words |

### WebOCR Template Engine (`src/webocr/`)
A self-contained Rust port of the sibling `../webocr` template-matching engine. Mirrors WebOCR's module layout 1:1 so the two codebases stay cross-referenceable. Picked via `--engine template` (alongside the CRNN path). No ML model — templates are built at first recognition from platform fonts via rusttype.

| Module | Purpose |
|--------|---------|
| `types.rs` | `GrayImage`, `BinaryImage`, `Comp`, `LineData`, `CharBox`, `WordSpan`, `LineResult`, `TemplateOcrResult`, `Timing` |
| `binarize.rs` | ITU-R 601 luma, Otsu threshold, polarity detection (≥55 % dark → invert), halo bias for light-on-dark |
| `components.rs` | 8-connected components via iterative flood fill, `median`, `union_box` |
| `segment.rs` | `filter_noise`, `split_touching` (valley + arch guard), `merge_stacked` (pass 1 + 2), `group_into_lines`, `cluster_components` (CJK width cap 1.25·medH), `split_row_segments`, `detect_blocks` (union-find), `extract_lines`, `detect_space_threshold` (gap-cluster word spacing) |
| `normalize.rs` | `SCALE_REF=20`, `MAX_UPSCALE=2.2`, `FRAME=30`, `baseline_row_for`, `place_glyph` (area-averaged resample), `blur3` (separable 1-2-1) |
| `classify.rs` | `PreparedTemplate`, `l2`, `pool_coarse`, `shortlist` (with shift recall), `match_glyph` + `match_one_shift` (cosine + ASPECT_WEIGHT=0.1) |
| `templates.rs` | `TemplateRenderer` trait, `build_templates` (Latin per-char fan-out over faces × sizes), `make_template` (equal-footing normalization) |
| `font_renderer.rs` | `RustTypeRenderer` — loads platform font files (sans/serif/monospace/wide-sans, regular+bold) for the template library |
| `engine.rs` | `run_ocr` — orchestrates binarize → segment → recognize (rescue split, gap-cluster word spacing) → layout assembly |
| `mod.rs` | `TemplateOcr` facade (`new`/`with_renderer`/`recognize`/`template_count`) |

The WebOCR engine is **pure** (no I/O, no DOM, no ML) — same purity boundary as WebOCR's `src/ocr/`. The `TemplateRenderer` trait is the extension point for any font source (system fonts, custom training fonts, synthetic pixel fonts in tests).

### Image Pipeline (`src/image/`)

All preprocessing is deterministic and differentiable-friendly:

- `Processor` — high-level interface: `preprocess_for_ocr(image) -> OcrImage`
- `Pipeline` — composable chain of operations
- `Thresholder` — Otsu global, Sauvola local, adaptive mean/Gaussian
- `Enhancement` — CLAHE, unsharp mask, gamma correction, orientation detection, border removal, affine deskew
- `dewarp.rs` — `PerspectiveDewarp` (content-corner quad → rectangle) + `CurveRectifier` (quadratic baseline flatten)
- `super_resolution.rs` — `TextSuperResolution` Lanczos upscale for tiny / low-DPI text (noise-aware)
- `font_analysis.rs` — glyph-level font metrics used by preprocessing and synthetic generation
- `Quality` — blur detection, contrast score, resolution check

### Layout Analysis (`src/layout/`)

Inspired by Tesseract's layout analyzer but written in pure Rust:

- `union_find_ccl.rs` — Connected Component Labeling with union-find for character blob isolation
- `analyzer.rs` — Top-level layout analyzer: image → blocks → regions → lines
- `column_detector.rs` — XY-cut + whitespace analysis for multi-column documents
- `line_detector.rs` — Baseline detection and text line segmentation
- `text_ordering.rs` — Reading-order resolution (top-to-bottom, left-to-right, CJK vertical)
- `classifier.rs` — Region type classification (Heading, SubHeading, Body, ListItem, Caption, Footer, PageNumber, Header)
- `detection_cnn.rs` — Lightweight 3-layer CNN for text region heatmaps (conv1→conv2→conv3, sigmoid output)
- `detector.rs` — `TextDetector` trait (`CclDetector`, `CnnDetector`, `OrientedCclDetector`), `TableDetector` with span inference
- `form_extractor.rs` — Form field detection: checkboxes, key-value pairs, underline fields

### Recognition (`src/recognition/`)

The crate ships **two recognition engines** selectable via `--engine {pattern,lstm,template}`. The CRNN is the ML default; the WebOCR template engine is a no-ML alternative for headless / Latin-only use cases.

| Engine | Status | Architecture | Notes |
|--------|--------|-------------|-------|
| `CrnnModel` | **Working** | 5-layer CNN + 2-layer BiLSTM + CTC beam decode | Pure Rust ndarray. Model size < 5MB. Beam search + dict/LM rescoring + calibrated confidence. Selectable via `--engine lstm`; the legacy `--engine pattern` value is accepted and silently mapped to LSTM (`src/cli/commands/helpers.rs`). |
| `webocr::TemplateOcr` | **Working** | Otsu + polarity + halo bias → 8-CC → valley split + stacked merge + CJK cluster + gap-cluster word spacing → own-height normalization → cosine + aspect + relH penalties + shift search | Self-contained Rust port of the sibling `../webocr` engine (see `src/webocr/`). No model file — templates built at first recognition from platform fonts via rusttype. Latin charset. Selectable via `--engine template`. WebOCR-measured accuracy: sans 100 %, monospace 100 %, serif ≈97.8 %, dark/inverted 100 %. |
| `ScriptModelRegistry` | **Working** | HashMap of per-script CRNN models | Routes to Latin/CJK/Arabic/Cyrillic/etc. vocabularies. |
| `ModelManager` | **Working** | Runtime model switching (`engine.rs`) | Loads/swaps CRNN models by config. |

Supporting modules:

- `model_format.rs` — `OCR1` binary format (magic + version 1) for model bundles and checkpoints
- `loader.rs` — loads the bundled `models/crnn-en.bin` (3.07 MB, embedded via `include_bytes!`) on first recognition (cached in `OcrEngine` via `OnceLock<CrnnModel>`), plus checkpoint I/O
- `confidence.rs` — `ConfidenceCalibrator` (temperature-scaled CTC softmax → per-char/word scores)
- `ctc_decoder.rs` — greedy + beam search with n-best output for dict/LM rescoring
- `backward.rs` — full BPTT gradients through CNN + BiLSTM + FC (`ParamGrads`, consumed by `CrnnTrainer::train_batch_full`)
- `font_attributes.rs` — `FontAttributeDetector`: bold (stroke thickness), italic (slant angle), monospace (width CV)

**Current state:**
1. **Two engines** selectable via `--engine {pattern,lstm,template}`. CRNN is the ML default; WebOCR template engine is the no-ML alternative.
2. Training: pure-Rust CTC loss + full BPTT in `CrnnTrainer`; bundled weights are not yet fully trained (see `TODO.md`)
3. ONNX weight *import* is implemented behind the `onnx` feature (`OnnxLoader`); an ONNX *inference* backend remains in the backlog
4. WebOCR template engine: 31 unit tests + 2 round-trip integration tests; CJK union charset is future work

### Language Support (`src/lang/`)

- `dictionary.rs` — Edit-distance spell correction with per-language dictionaries for 25+ languages; also used for CTC beam rescoring
- `ngram.rs` — Character/word n-gram LM for beam hypothesis rescoring
- `detector.rs` — Script + dictionary-hit language identification (`LanguageDetector`, `--lang auto`)
- `cjk.rs` — CJK character segmentation and vertical text handling
- `unicode.rs` — Unicode block classification for script routing (Latin, CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari)

### Synthetic Data (`src/synthetic/`)

- `generator.rs` — `TextLineGenerator`: TTF font rendering + bitmap fallback, batch generation
- `distortion.rs` — Rotation, blur, noise, contrast, shear
- `bitmap_font.rs` — 5×7 bitmap glyph definitions for ASCII
- `multi_script.rs` — `ScriptLineGenerator` + `ScriptCharPool`: generate text in CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari
- `document_generator.rs` — `DocumentGenerator`: synthetic multi-column documents with ground-truth bounding boxes
- `benchmark.rs` — CER/WER metrics, clean/mild/heavy test sets

### Training (`src/training/`)

Only the CRNN trainer is wired today; the broader training stack
(generic data loaders, augmentation pipeline, checkpoint manager,
loss zoo) was speculative and has been removed until a second concrete
consumer arrives.

- `crnn_trainer.rs` — `CrnnTrainer`:
  - `train_batch_full` — full BPTT through CNN + 2× BiLSTM + FC with Adam (`FullAdamState`), the real training path (also used by `examples/train_pure_rust.rs`)
  - `train_batch_ctc` — FC-only CTC training (CNN/BiLSTM frozen), kept as a fast smoke path
  - `save_checkpoint` / `load_checkpoint` persist the model in the `OCR1` format
- `ctc.rs` — pure-Rust CTC loss: forward-backward algorithm + gradient (`ctc_loss_and_grad`)

### ONNX Import (`src/onnx/`)

- `mod.rs` — `OnnxLoader`: parse ONNX files via `onnx-rs`, extract weight tensors as ndarrays, inspect node types (Conv, Gemm, LSTM); enabled via `onnx` feature flag

## Data Flow (Detailed)

```
1. CLI parses args → builds OcrConfig
2. Ocr::with_config(config) → creates OcrEngine
3. OcrEngine::initialize():
   a. Prepare the bundled CRNN model (`models/crnn-en.bin` via `include_bytes!`, loaded lazily on first recognition via `OnceLock`)
   b. Load language dictionaries
   c. Warm up image preprocessing pipeline
4. Image loaded via image::open() → converted to OcrImage (Grayscale, 300 DPI)
5. PSM dispatch (Tesseract `--psm` parity, `process_image_internal`):
   PSM 0 short-circuits to orientation+script detection; PSM 1/12 detect
   orientation on the raw input and attach OSD metadata; PSM 5 rotates the
   page 90° anticlockwise; PSM 9 removes an enclosing ring; PSM 2 returns
   after layout without recognition (see TESSERACT_PARITY.md §2)
6. Preprocessing (if enabled):
   a. Auto-rotate via projection-variance axis detection + vertical-mass polarity (180° guard)
   b. Deskew (projection-variance angle search) + perspective dewarp
   c. Quality gate (optional auto-sharpen / contrast boost from quality metrics)
   d. Super-resolution for tiny/low-DPI text (noise-aware)
   e. Binarization (Sauvola for variable backgrounds)
   f. Noise reduction (median filter)
   g. Border removal
7. Layout analysis:
   a. TextDetector (CCL or CNN) → text regions
   b. Blob grouping → text lines (baseline clustering)
   c. Line grouping → columns (whitespace valleys)
   d. Table detection (grid line scan + span inference)
   e. Form field extraction (checkboxes, key-value pairs)
   f. Reading order resolution
8. Recognition (per text line):
   a. Curved-line rectification on region crops (if enabled)
   b. Normalize line height
   c. Script detection → route to appropriate CRNN vocabulary
   d. CRNN: CNN features → BiLSTM → CTC beam decode (+ dict/LM rescoring)
   e. Build WordRecognition → LineRecognition → TextResult
9. Post-processing:
   a. Document structure classification (headings, lists, paragraphs)
   b. Dictionary correction (if enabled)
   c. Confidence score aggregation
10. Output formatting:
   a. Plain text, JSON, hOCR, TSV, CSV, ALTO XML, PAGE XML, box, Markdown, structured JSON, RAG JSON, searchable PDF (feature `pdf`)
   b. Write to stdout or file
```

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Pure Rust, no Tesseract dep | Learn from scratch, full control, easier deployment |
| ndarray instead of burn/candle | Burn/candle are immature; ndarray gives full control for educational purposes |
| Manual LSTM/CNN implementations | Pure Rust control; slower than BLAS but no C dependencies |
| Pattern matching as default (configurable) | Obsolete in v0.2: the CRNN is the only engine; the legacy `--engine pattern` flag is accepted and mapped to LSTM. The WebOCR template engine (`src/webocr/`) was added in 2026-10-18 as a no-ML alternative; select with `--engine template`. |
| Synthetic data first | Removes dependency on collecting labeled real-world data before training |
| Two-stage pipeline (detect + recognize) | Proven architecture (Tesseract, EasyOCR, PaddleOCR), easier to debug and optimize per-stage |
| Per-script CRNN models | Vocab size varies wildly (95 ASCII vs 3000+ CJK); separate models keep each lightweight |
