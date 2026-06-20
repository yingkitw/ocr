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
            Binarization (Otsu,            Text region detection
            Sauvola, Adaptive)             Line segmentation
            Noise reduction                Reading-order resolution
            Deskew / dewarp                Word/char grouping
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
| `layout.rs` | `LayoutResult`, `Block`, `TextRegion`, `ReadingOrder` — layout analysis outputs |
| `output.rs` | Formatters: text, JSON, hOCR, TSV, ALTO XML, box, PDF |
| `recognition.rs` | `RecognitionResult`, `CharacterRecognition`, `WordRecognition`, `LineRecognition` |
| `text.rs` | `TextResult` — final unified output with confidence, bounding boxes, lines, words |

### Image Pipeline (`src/image/`)

All preprocessing is deterministic and differentiable-friendly (so we can eventually backprop through it if needed):

- `Processor` — high-level interface: `preprocess_for_ocr(image) -> OcrImage`
- `Pipeline` — composable chain of operations
- `Thresholder` — Otsu global, Sauvola local, adaptive mean/Gaussian
- `Enhancement` — CLAHE, unsharp mask, gamma correction
- `Quality` — blur detection, contrast score, resolution check

### Layout Analysis (`src/layout/`)

Inspired by Tesseract's layout analyzer but written in pure Rust:

- `union_find_ccl.rs` — Connected Component Labeling with union-find for character blob isolation
- `analyzer.rs` — Top-level layout analyzer: image → blocks → regions → lines
- `column_detector.rs` — XY-cut + whitespace analysis for multi-column documents
- `line_detector.rs` — Baseline detection and text line segmentation
- `text_ordering.rs` — Reading-order resolution (top-to-bottom, left-to-right, CJK vertical)
- `classifier.rs` — Region type classification (heading, body, caption, footer, page-number)

### Recognition (`src/recognition/`)

Multiple engines behind a unified trait. Currently only **PatternModel** is fully functional.

| Engine | Status | Architecture | Notes |
|--------|--------|-------------|-------|
| `PatternModel` | **Working** | Bitmap template matching (L1 distance) | 37 glyphs, 5×7 bitmaps. Baseline. |
| `BasicOcrEngine` | **Working** | Tesseract-inspired: CCL → lines → pattern match | More robust segmentation than raw PatternModel |
| `LstmModel` | Stub | Manual LSTM cell (ndarray) | Needs training pipeline + weights |
| `CnnModel` | Stub | Conv layers (ndarray) | Needs training pipeline + weights |
| `TransformerModel` | Stub | Self-attention encoder (ndarray) | Needs training pipeline + weights |
| `HybridModel` | Stub | Ensemble wrapper | Needs working sub-models first |

**Planned evolution:**
1. Replace stubs with a real **CRNN** (CNN feature extractor + BiLSTM + CTC decoder)
2. Train on synthetic rendered-font data
3. Add optional **ONNX runtime** integration for importing pre-trained models (PaddleOCR, etc.)

### Language Support (`src/lang/`)

- `dictionary.rs` — Edit-distance spell correction with per-language dictionaries (en, fr, de, es, it, pt, ru)
- `detector.rs` — N-gram based language identification
- `cjk.rs` — CJK character segmentation and vertical text handling
- `unicode.rs` — Unicode block classification for script routing

### Training (`src/training/`)

Infrastructure for learning models from data:

- `data.rs` — Dataset loading, batching
- `augmentation.rs` — Image distortions: blur, noise, rotation, scaling, perspective
- `losses.rs` — CTC loss, cross-entropy
- `optimizers.rs` — SGD, Adam (manual implementation with ndarray)
- `metrics.rs` — Character Error Rate (CER), Word Error Rate (WER)
- `checkpoint.rs` — Save/load model weights

## Data Flow (Detailed)

```
1. CLI parses args → builds OcrConfig
2. Ocr::with_config(config) → creates OcrEngine
3. OcrEngine::initialize():
   a. Load recognition engine (currently PatternModel)
   b. Load language dictionaries
   c. Warm up image preprocessing pipeline
4. Image loaded via image::open() → converted to OcrImage (Grayscale, 300 DPI)
5. Preprocessing (if enabled):
   a. Deskew (Hough transform for angle estimation)
   b. Binarization (Sauvola for variable backgrounds)
   c. Noise reduction (median filter)
6. Layout analysis:
   a. Union-Find CCL → character blobs
   b. Blob grouping → text lines (baseline clustering)
   c. Line grouping → columns (whitespace valleys)
   d. Reading order resolution
7. Recognition (per text line):
   a. Normalize line height
   b. Segment into character candidates (or use CTC for no segmentation)
   c. PatternModel: compute L1 similarity against templates
   d. Build WordRecognition → LineRecognition → TextResult
8. Post-processing:
   a. Dictionary correction (if enabled)
   b. Confidence score aggregation
9. Output formatting:
   a. Plain text, JSON, hOCR, TSV, ALTO XML, box, or PDF
   b. Write to stdout or file
```

## Evolution Plan

### Phase 0 (Current)
Pattern matching baseline. Everything wired together and tested.

### Phase 1 — Synthetic Data Engine
Add a `synthetic/` module that renders text with random fonts, backgrounds, and distortions. Use it to:
- Generate 10k+ training samples
- Benchmark the pattern matcher (establish baseline CER/WER)
- Bootstrap the CRNN with pre-trained weights

### Phase 2 — Detection (CRAFT/DBNet-lite)
Implement a lightweight detection CNN in pure Rust (ndarray) or via ONNX. This replaces the CCL-based text finding for better robustness on complex layouts and scene text.

### Phase 3 — CRNN Recognition
Implement a real CRNN:
- 5-layer CNN (VGG-style) for feature extraction
- 2-layer BiLSTM for sequence modeling
- CTC decoder (greedy + beam search)
- Train end-to-end on synthetic data, fine-tune on real data

### Phase 4 — Scale & Optimize
- ONNX runtime for running imported PaddleOCR/EasyOCR models
- SIMD/AVX2 optimization for CNN convolutions
- GPU backend (CUDA/OpenCL) via cudarc/ocl
- 80+ language support via unicode routing

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Pure Rust, no Tesseract dep | Learn from scratch, full control, easier deployment |
| ndarray instead of burn/candle | Burn/candle are immature; ndarray gives full control for educational purposes |
| Manual LSTM/CNN/Transformer stubs | We will fill them with learned weights once the training pipeline is ready |
| Pattern matching as default | It works today and gives us a testable baseline to measure improvement against |
| Synthetic data first | Removes dependency on collecting labeled real-world data before training |
| Two-stage pipeline (detect + recognize) | Proven architecture (Tesseract, EasyOCR, PaddleOCR), easier to debug and optimize per-stage |
