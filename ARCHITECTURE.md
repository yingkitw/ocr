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
    engine: OcrEngine,
    config: OcrConfig,
}

impl Ocr {
    pub async fn recognize_text_from_file(&self, path: &str) -> Result<TextResult>;
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
| `output.rs` | Formatters: text, JSON, hOCR, TSV, ALTO XML, box, PDF, Markdown, structured JSON |
| `recognition.rs` | `RecognitionResult`, `CharacterRecognition`, `WordRecognition`, `LineRecognition` |
| `text.rs` | `TextResult` — final unified output with confidence, bounding boxes, lines, words |

### Image Pipeline (`src/image/`)

All preprocessing is deterministic and differentiable-friendly:

- `Processor` — high-level interface: `preprocess_for_ocr(image) -> OcrImage`
- `Pipeline` — composable chain of operations
- `Thresholder` — Otsu global, Sauvola local, adaptive mean/Gaussian
- `Enhancement` — CLAHE, unsharp mask, gamma correction, orientation detection, border removal, affine deskew
- `dewarp.rs` — `PerspectiveDewarp` (content-corner quad → rectangle) + `CurveRectifier` (quadratic baseline flatten)
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

Multiple engines behind a unified trait:

| Engine | Status | Architecture | Notes |
|--------|--------|-------------|-------|
| `PatternModel` | **Working** | Bitmap template matching (L1 distance) | 37 glyphs, 5×7 bitmaps. Baseline. |
| `BasicOcrEngine` | **Working** | Tesseract-inspired: CCL → lines → pattern match | More robust segmentation. Trainable via `TemplateTrainer`. |
| `CrnnModel` | **Working** | 5-layer CNN + 2-layer BiLSTM + CTC beam decode | Pure Rust ndarray. Model size < 5MB. Beam search + dict/LM rescoring + calibrated confidence. Selectable via `--engine lstm`. |
| `ScriptModelRegistry` | **Working** | HashMap of per-script CRNN models | Routes to Latin/CJK/Arabic/Cyrillic/etc. vocabularies. |
| `LstmModel` | Working (internal) | Manual LSTM cell (ndarray) | Used inside `CrnnModel` BiLSTM layers. |
| `TransformerModel` | Stub | Self-attention encoder (ndarray) | Future work. |
| `HybridModel` | Stub | Ensemble wrapper | Future work. |

**Evolution completed:**
1. CRNN (CNN feature extractor + BiLSTM + CTC decoder) implemented and wired
2. Trained templates from synthetic renders via `TemplateTrainer`
3. ONNX runtime integration remains in backlog

### Language Support (`src/lang/`)

- `dictionary.rs` — Edit-distance spell correction with per-language dictionaries for 25+ languages; also used for CTC beam rescoring
- `ngram.rs` — Character/word n-gram LM for beam hypothesis rescoring
- `detector.rs` — N-gram based language identification
- `cjk.rs` — CJK character segmentation and vertical text handling
- `unicode.rs` — Unicode block classification for script routing (Latin, CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari)

### Synthetic Data (`src/synthetic/`)

- `generator.rs` — `TextLineGenerator`: TTF font rendering + bitmap fallback, batch generation
- `distortion.rs` — Rotation, blur, noise, contrast, shear
- `bitmap_font.rs` — 5×7 bitmap glyph definitions for ASCII
- `template_trainer.rs` — `TemplateTrainer`: crop glyphs from renders, average across fonts, build binary templates
- `multi_script.rs` — `ScriptLineGenerator` + `ScriptCharPool`: generate text in CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari
- `document_generator.rs` — `DocumentGenerator`: synthetic multi-column documents with ground-truth bounding boxes
- `benchmark.rs` — CER/WER metrics, clean/mild/heavy test sets

### Training (`src/training/`)

Infrastructure for learning models from data:

- `data.rs` — Dataset loading, batching
- `augmentation.rs` — Image distortions: blur, noise, rotation, scaling, perspective
- `losses.rs` — CTC loss, cross-entropy
- `optimizers.rs` — SGD, Adam (manual implementation with ndarray)
- `metrics.rs` — Character Error Rate (CER), Word Error Rate (WER)
- `checkpoint.rs` — Save/load model weights
- `crnn_trainer.rs` — `CrnnTrainer`: synthetic batch training, checkpoint saving/loading
- `quantization.rs` — `QuantizedTensor`, `quantize_array2`, `quantized_matmul`: INT8 symmetric quantization for edge deployment

### ONNX Import (`src/onnx/`)

- `mod.rs` — `OnnxLoader`: parse ONNX files via `onnx-rs`, extract weight tensors as ndarrays, inspect node types (Conv, Gemm, LSTM); enabled via `onnx` feature flag

### Recognition (`src/recognition/`)

- `font_attributes.rs` — `FontAttributeDetector`: bold (stroke thickness), italic (slant angle), monospace (width CV)

## Data Flow (Detailed)

```
1. CLI parses args → builds OcrConfig
2. Ocr::with_config(config) → creates OcrEngine
3. OcrEngine::initialize():
   a. Load recognition engine (PatternModel or CRNN via config)
   b. Load language dictionaries
   c. Warm up image preprocessing pipeline
4. Image loaded via image::open() → converted to OcrImage (Grayscale, 300 DPI)
5. Preprocessing (if enabled):
   a. Auto-rotate via projection-variance orientation detection
   b. Deskew (projection-variance angle search) + perspective dewarp
   c. Binarization (Sauvola for variable backgrounds)
   d. Noise reduction (median filter)
   e. Border removal
6. Layout analysis:
   a. TextDetector (CCL or CNN) → text regions
   b. Blob grouping → text lines (baseline clustering)
   c. Line grouping → columns (whitespace valleys)
   d. Table detection (grid line scan + span inference)
   e. Form field extraction (checkboxes, key-value pairs)
   f. Reading order resolution
7. Recognition (per text line):
   a. Curved-line rectification on region crops (if enabled)
   b. Normalize line height
   c. Script detection → route to appropriate engine
   d. PatternModel: L1 similarity against templates
   e. CRNN: CNN features → BiLSTM → CTC beam decode (+ dict/LM rescoring)
   f. Build WordRecognition → LineRecognition → TextResult
8. Post-processing:
   a. Document structure classification (headings, lists, paragraphs)
   b. Dictionary correction (if enabled)
   c. Confidence score aggregation
9. Output formatting:
   a. Plain text, JSON, hOCR, TSV, ALTO XML, box, PDF, Markdown, structured JSON
   b. Write to stdout or file
```

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Pure Rust, no Tesseract dep | Learn from scratch, full control, easier deployment |
| ndarray instead of burn/candle | Burn/candle are immature; ndarray gives full control for educational purposes |
| Manual LSTM/CNN implementations | Pure Rust control; slower than BLAS but no C dependencies |
| Pattern matching as default (configurable) | Works today as baseline; CRNN selectable via `--engine lstm` |
| Synthetic data first | Removes dependency on collecting labeled real-world data before training |
| Two-stage pipeline (detect + recognize) | Proven architecture (Tesseract, EasyOCR, PaddleOCR), easier to debug and optimize per-stage |
| Per-script CRNN models | Vocab size varies wildly (95 ASCII vs 3000+ CJK); separate models keep each lightweight |
