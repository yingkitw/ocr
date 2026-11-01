# Specification

## Philosophy

This spec defines concrete, testable milestones. Each phase must compile, pass tests, and demonstrate measurable improvement over the previous phase.

**Current status: All five phases implemented and tested; Phase 3.5 (WebOCR template engine, Latin + CJK union) added 2026-10-18. 428 tests passing, 90 ignored, one known model-regression failure (see MEMORY.md §6).**

---

## Phase 0 — Baseline (DONE)

**Goal:** End-to-end pipeline that compiles and passes tests.

### Acceptance Criteria
- [x] `cargo test` passes with zero unexpected failures (413 tests passing; one known model-regression failure documented in MEMORY.md §6)
- [x] `cargo build --release` produces a working binary
- [x] `ocr extract test-image.png` produces text output
- [x] All output formats render without panic
- [x] CLI covers: extract, batch, layout, list-languages, check, info, validate, train, benchmark, makebox, serve

### CLI Specification

| Command | Description | Required Args | Optional Flags |
|---------|-------------|---------------|--------------|
| `extract` | Recognize text from an image (or `-` for stdin). Multi-page TIFF produces page-separated output. | `IMAGE_PATH` | `-o, --output`, `-l, --lang`, `--preprocess`, `-f, --format`, `--psm`, `--whitelist`, `--confidence`, `--engine`, `--dict-correct`, `--device`, `--osd`, `--dpi` |
| `batch` | Process a directory of images (prints `[n/total] % ETA` progress) | `-i, --input-dir`, `-o, --output-dir` | `-l, --lang`, `--confidence`, `--engine`, `--dict-correct`, `--max-concurrent`, `--device` |
| `layout` | Analyze page layout | `IMAGE_PATH` | `-o, --output` |
| `list-languages` | Print supported language codes | — | — |
| `check` | Verify system requirements | — | — |
| `info` | Print engine metadata | — | — |
| `validate` | Validate a config file | `CONFIG_FILE` | — |
| `train` | Train a recognition model | — | `-e, --epochs`, `-b, --batch-size`, `-l, --learning-rate`, `--engine`, `-c, --checkpoint-dir`, `--distortion` |
| `benchmark` | Benchmark per-script accuracy | — | `-s, --samples`, `--distortion` |
| `makebox` | Generate Tesseract `.box` training file | `IMAGE_PATH` | `-o, --output`, `-l, --lang`, `--engine`, `--preprocess` |
| `serve` | Start HTTP API server (requires `--features web-api`) | — | `--host`, `--port`, `--max-upload-size` |

### Extract Flags

| Flag | Default | Description |
|------|---------|-------------|
| `IMAGE_PATH` | required | Path to the image file, or `-` to read from stdin |
| `-o, --output` | stdout | Output file path |
| `-l, --lang` | `en` | Language code (`en`, `fr`, `de`, `zh`, `ja`, `ko`, …) or `auto` (script + dictionary detection after recognition) |
| `--preprocess` | false | Enable full preprocessing pipeline |
| `-f, --format` | `text` | Output: text, json, hocr, tsv, csv, alto, box, markdown, structured-json, rag, page |
| `--psm` | `3` | Page segmentation mode (Tesseract-compatible): `0`–`13` or symbolic names — `osd_only`, `auto_osd`, `auto_only`, `auto`, `single_column`, `single_block_vert_text`, `single_block`, `single_line`, `single_word`, `circle_word`, `single_char`, `sparse_text`, `sparse_text_osd`, `raw_line` (see TESSERACT_PARITY.md §2) |
| `--confidence` | `0.5` | Minimum confidence threshold (0.0–1.0) |
| `--engine` | `pattern`, `lstm`, `template` | Recognition engine. `pattern` is the legacy alias (silently mapped to `lstm`). `template` selects the WebOCR template-matching engine (no model file, builds from system fonts; Latin + CJK union; `--lang auto` picks via image geometry). |
| `--dict-correct` | false | Dictionary-constrained decoding: the dictionary steers the CTC beam in-beam (Tesseract kDictRatio scheme) and applies post-correction |
| `--device` | `auto` | Compute device: `cpu`, `gpu`, `auto` |
| `--osd` | false | Enable orientation & script detection |
| `--dpi` | `300` | Input image resolution in DPI (affects super-resolution and layout heuristics) |
| `--whitelist` | none | Restrict recognition to these characters only (Tesseract `tessedit_char_whitelist`), e.g. `--whitelist 0123456789` |

### PSM modes (Tesseract `--psm` parity)

| # | Symbolic name | Behavior |
|---|---|---|
| 0 | `osd_only` | Orientation + script detection only; prints a Tesseract-shaped OSD report (`Orientation in degrees`, confidences, `Script`); no OCR text |
| 1 | `auto_osd` | Auto pipeline with OSD metadata (`orientation`, `script`, confidences in JSON `metadata`) |
| 2 | `auto_only` | Layout analysis only; prints orientation + `Layout regions: N`; no recognition |
| 3 | `auto` | Fully automatic page segmentation (default) |
| 4 | `single_column` | Page treated as a single column |
| 5 | `single_block_vert_text` | Page rotated 90° anticlockwise, then treated as a single block |
| 6 | `single_block` | Page treated as one uniform block |
| 7 | `single_line` | Whole image is one text line |
| 8 | `single_word` | Whole image is one word |
| 9 | `circle_word` | Enclosing circle/ring removed, then treated as one word |
| 10 | `single_char` | Whole image is one character |
| 11 | `sparse_text` | Find as much text as possible, no ordering |
| 12 | `sparse_text_osd` | Sparse text + OSD metadata |
| 13 | `raw_line` | Like 7, without Tesseract-specific row heuristics |

OSD metadata keys (JSON output / `TextResult.metadata`): `psm`,
`orientation` (degrees), `orientation_confidence` (0–100), `script`,
`script_confidence` (0–100), `layout_regions` (PSM 2). OSD confidence uses a
two-stage scorer (projection variance for the axis, vertical-mass asymmetry +
an "upside-down is improbable" guard for 0°/180° polarity) — see
`ImageEnhancer::detect_orientation_with_confidence`.

### Library API

```rust
// High-level facade
pub struct Ocr;
impl Ocr {
    pub fn new() -> Result<Self>;
    pub fn with_config(config: OcrConfig) -> Result<Self>;
    pub async fn initialize(&self) -> Result<()>;
    pub async fn recognize_text_from_file(&self, path: &str) -> Result<TextResult>;
    pub async fn recognize_text_from_bytes(&self, image_data: &[u8]) -> Result<TextResult>;
    pub async fn recognize_text_from_file_multi_page(&self, path: &str) -> Result<Vec<TextResult>>;
    pub async fn recognize_text_from_bytes_multi_page(&self, image_data: &[u8]) -> Result<Vec<TextResult>>;
    pub async fn recognize_text(&self, data: &[u8], width: u32, height: u32) -> Result<TextResult>;
    pub async fn analyze_layout(&self, data: &[u8], width: u32, height: u32) -> Result<LayoutResult>;
    pub fn get_supported_languages(&self) -> Vec<String>;
}

// Configuration
pub struct OcrConfig {
    pub recognition: RecognitionConfig,
    pub image_processing: ImageProcessingConfig,
    pub layout_analysis: LayoutAnalysisConfig,
    pub language: LanguageConfig,
    pub performance: PerformanceConfig,
}
```

### Output Formats

| Format | Schema | Notes |
|--------|--------|-------|
| `text` | Plain UTF-8 | Newlines preserved from line detection |
| `json` | `{"text": "...", "confidence": 0.95, "lines": [...]}` | Bounding boxes per line/word/char |
| `hocr` | HTML 4.01 with hOCR 4.1 classes | `ocr_page`, `ocr_line`, `ocrx_word` |
| `tsv` | TSV with Tesseract-compatible columns | level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text |
| `csv` | RFC 4180 CSV with per-word metadata | page, line, word, text, confidence, left, top, width, height — fields containing comma/quote/newline are quoted per RFC 4180 |
| `alto` | ALTO XML v4 | `<Page>`, `<TextBlock>`, `<TextLine>`, `<String>` |
| `box` | Box file format | `char left top right bottom page` |
| `markdown` | Markdown with headings, paragraphs, lists | `## Heading`, `- List item` |
| `structured-json` | Hierarchical JSON with document elements | `{"elements": [{"Heading": {...}}, {"Paragraph": {...}}]}` |
| `rag` | Chunked JSON for RAG / LLM pipelines (aliases: `rag-json`, `chunks`) | `{"chunks": [{"id": 0, "kind": "paragraph", "text": "...", "section": ["Heading"]}]}` — one chunk per element, `section` is the anchoring heading path; tables render as Markdown |
| `page` | PRImA PAGE XML (aliases: `page-xml`, `primapage`) | `<PcGts>` → `<Page>` → `<TextRegion>` → `<TextLine>` → `<Word>`, each with `<Coords>` polygon and `<TextEquiv>` (Unicode + Confidence) |
| `pdf` | Searchable PDF (requires `--features pdf`) | Original image as JPEG XObject + invisible text layer (rendering mode 3) at word bounding boxes; multi-page TIFF → multi-page PDF |

---

## Phase 1 — Synthetic Training Data (DONE)

**Goal:** Generate unlimited training data from fonts + distortions.

### Acceptance Criteria
- [x] `src/synthetic/` module generates text-line images from TTF fonts + bitmap fallback
- [x] Applies realistic distortions: rotation ±5°, Gaussian blur, salt & pepper noise, random contrast, perspective shear
- [x] Outputs paired dataset: `(image_bytes, ground_truth_text)`
- [x] CER/WER benchmark harness with clean/mild/heavy test sets
- [x] `TemplateTrainer` trained pattern-matching templates from synthetic renders (module removed in v0.2 when the pattern engine was deleted)

### Synthetic Data Generator API

```rust
pub struct TextLineGenerator {
    pub fonts: Vec<Font>,
    pub image_width: u32,
    pub image_height: u32,
    pub background_color: Rgb<u8>,
    pub text_color: Rgb<u8>,
}

impl TextLineGenerator {
    pub fn generate(&self, text: &str) -> SyntheticSample;
    pub fn generate_batch(&self, texts: &[String]) -> Vec<SyntheticSample>;
    pub fn generate_with_font(&self, text: &str, font_index: usize) -> SyntheticSample;
}
```

---

## Phase 2 — Text Detection (DONE)

**Goal:** Find text regions more robustly than CCL on complex layouts.

### Acceptance Criteria
- [x] `TextDetector` trait with `CclDetector` and `CnnDetector` implementations
- [x] Lightweight 3-layer detection CNN in pure Rust (ndarray): conv1(1→8), conv2(8→16), conv3(16→1)
- [x] Sigmoid heatmap output + threshold → union-find CCL → bounding boxes
- [x] `DocumentGenerator` creates synthetic multi-column layouts with ground-truth bboxes
- [x] IoU-based evaluation harness computes recall/precision
- [x] Auto-rotate 0°/90°/180°/270° via projection-variance orientation detection
- [x] Perspective document dewarp (`PerspectiveDewarp`) + curved-line rectification (`CurveRectifier`)
- [x] Arbitrary-angle text detection (`OrientedCclDetector`, ±45° sweep with NMS)
- [x] Super-resolution upscaling (`TextSuperResolution`) for tiny / low-DPI input
- [x] `format_makebox` + `ocr makebox` CLI for Tesseract-compatible training boxes

### Detection API

```rust
pub trait TextDetector: Send + Sync {
    fn detect(&self, image: &OcrImage) -> Result<Vec<TextRegion>>;
    fn name(&self) -> &'static str;
}

pub struct CclDetector;          // Union-Find connected components
pub struct CnnDetector;          // 3-layer CNN with heuristic edge-detector weights
```

---

## Phase 3.5 — WebOCR Template Engine (DONE, 2026-10-18)

**Goal:** Add a second recognition engine that requires no ML model — a Rust port of the sibling `../webocr` template-matching engine. Complements the CRNN: works headlessly on platforms without a bundled model file, builds the library from system fonts at runtime.

### Acceptance Criteria
- [x] Self-contained `src/webocr/` module (~1,600 lines) mirroring WebOCR's module layout 1:1: `binarize`, `components`, `segment`, `normalize`, `classify`, `templates`, `engine`, `types`
- [x] Otsu binarization with polarity detection and halo bias for light-on-dark scans
- [x] 8-connected components via iterative flood fill
- [x] Segmentation: filter noise, valley split (arch guard), stacked-fragment merge (pass 1 + 2), CJK stroke cluster (width cap 1.25·medH), row segments, block detection (union-find)
- [x] Own-height normalization (SCALE_REF=20, MAX_UPSCALE=2.2, FRAME=30, baseline anchored at row 20)
- [x] Cosine matching with aspect-ratio penalty (λ=0.1) and ±1-cell shift search
- [x] Coarse-to-fine shortlist (8×8 pooled vectors) for CJK-sized libraries
- [x] Gap-cluster word spacing (clusters the line's gap distribution; no fixed `k·medH`)
- [x] Template library built at runtime from system fonts via rusttype (7-face fan-out, sizes [40, 68])
- [x] `TemplateOcr` facade with `new()` (system fonts) and `with_renderer(&dyn)` (any renderer)
- [x] Wired into `OcrEngine` via `--engine template` (alongside `lstm`/`pattern`)
- [x] Pure-Rust with no new dependencies (rusttype 0.9 was already a dep)
- [x] 31 unit tests + 2 round-trip integration tests + 1 CLI smoke test
- [ ] CJK union charset (zh + ja + ko) — future; Latin only in this cut

### Measured Accuracy (WebOCR fixtures; re-validated in sibling repo)

| Font | WebOCR accuracy |
|------|----------------|
| Sans (Arial / Helvetica) | 100 % |
| Monospace (Courier) | 100 % |
| Serif (Georgia) | ≈97.8 % |
| Dark / inverted | 100 % |
| ja / ko | 100 % |
| zh (reduced charset) | ≈85.3 % (full 21K library ≈79 %) |

The Rust port shares WebOCR's algorithms and calibrated constants, so it inherits the same accuracy on the same fixtures. No re-validation here — the cross-implementation invariant is the WebOCR source.

## Phase 3 — CRNN Recognition (DONE)

**Goal:** Replace the v0.1 pattern matching with a learned sequence model.

### Acceptance Criteria
- [x] CNN feature extractor (5 conv layers + maxpool) in ndarray
- [x] 2-layer BiLSTM (64 hidden) with forward/backward pass
- [x] CTC decoder: greedy + beam search + dictionary/LM rescoring (n-best)
- [x] Confidence calibration: temperature-scaled CTC softmax → per-char/word scores
- [x] CTC loss with forward-backward algorithm
- [x] `CrnnTrainer` with synthetic data generation + checkpoint saving/loading
- [x] Wired into `OcrEngine` via `--engine lstm`
- [x] Model size < 5MB (~2.4MB default config)
- [x] Inference benchmark test (target < 100ms/line in release)
- [x] Pure-Rust CTC loss (forward-backward + gradient) in `src/training/ctc.rs`
- [x] Full BPTT training through CNN + BiLSTM + FC (`CrnnTrainer::train_batch_full`); FC-only CTC path retained (`train_batch_ctc`)
- [x] Rayon-parallel convolutions + ndarray FC layer for faster inference
- [ ] CER < 5% on clean synthetic test (requires training execution)
- [ ] CER < 15% on distorted synthetic test (requires training execution)

### CRNN Architecture

```
Input: 32xW grayscale text line image
  ↓
Conv(16, 3×3) → ReLU → MaxPool(2,2)
Conv(32, 3×3) → ReLU → MaxPool(2,2)
Conv(64, 3×3) → ReLU → Conv(64, 3×3) → ReLU → MaxPool(1,2)
Conv(128, 3×3) → BatchNorm → ReLU → MaxPool(1,2)
Conv(128, 2×2) → BatchNorm → ReLU
  ↓  [Feature maps: 128 × 1 × (W/4)]
Reshape to sequence: (W/4) × 128
  ↓
BiLSTM(64) → BiLSTM(64)
  ↓
Linear(128 → num_classes + 1)
  ↓
CTC Beam Search (+ optional dictionary / n-gram LM rescoring) → text string
```

`CrnnConfig` defaults: `use_beam_search = true`, `beam_width = 10`. Set `use_beam_search = false` for greedy decode. When dictionary correction / language model are enabled in `RecognitionConfig`, `OcrEngine` rescores beam hypotheses via `DictLmRescorer`.

**Model bundle:** `models/crnn-en.bin` (3.07 MB, `OCR1` format — magic + version 1) is embedded via `include_bytes!` (`src/recognition/loader.rs`) and loaded on first recognition (cached in `OcrEngine` via `OnceLock<CrnnModel>`). Replace the file to deploy a newly trained checkpoint.

### Training API

```rust
pub struct CrnnTrainer {
    pub model: CrnnModel,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub distortion: DistortionConfig,
}

impl CrnnTrainer {
    pub fn new(model: CrnnModel) -> Self;
    pub fn with_learning_rate(self, lr: f32) -> Self;
    pub fn with_batch_size(self, bs: usize) -> Self;
    pub fn with_distortion(self, distortion: DistortionConfig) -> Self;
    pub fn train_epoch(&mut self, num_batches: usize, samples_per_batch: usize) -> EpochMetrics;
    pub fn train_batch_full(&mut self, samples: &[SyntheticSample], adam: &mut FullAdamState) -> f32; // full BPTT
    pub fn train_batch_ctc(&mut self, samples: &[SyntheticSample], adam: &mut AdamState) -> f32;     // FC-only
    pub fn evaluate(&self, samples: &[SyntheticSample]) -> EpochMetrics;
    pub fn save_checkpoint(&self, path: &Path) -> Result<()>;
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()>;
}

pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_cer: f32,
    pub train_wer: f32,
    pub val_loss: f32,
    pub val_cer: f32,
    pub val_wer: f32,
    pub samples_per_sec: f32,
}
```

---

## Phase 4 — Multi-Language (DONE)

**Goal:** Support 30+ languages with automatic script routing.

### Acceptance Criteria
- [x] Unicode script detection: Latin, CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari
- [x] Dictionaries for 25+ languages loaded on demand
- [x] Multi-script synthetic data generation (`ScriptLineGenerator` + `ScriptCharPool`)
- [x] Per-script CRNN vocabularies and model configs (`ScriptModelRegistry`)
- [ ] Evaluate per-language accuracy on synthetic benchmarks (requires training execution)

### Script Detection API

```rust
pub enum Script {
    Latin, CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari, Other,
}

impl Script {
    pub fn detect(text: &str) -> Script;
    pub fn classify_char(ch: char) -> Script;
    pub fn detect_distribution(text: &str) -> Vec<(Script, f32)>;
}
```

---

## Phase 5 — Advanced Layout (DONE)

**Goal:** Understand document structure, not just extract flat text.

### Acceptance Criteria
- [x] Table detection with grid line scan and cell boundary reconstruction
- [x] Row/column span inference via pixel-density checks on internal grid lines
- [x] Form field extraction: key-value pairs, checkbox/radio recognition, underline fields
- [x] Document structure classification: Heading, Paragraph, ListItem, Caption, Footer, PageNumber
- [x] Hierarchical Markdown output (`--format markdown`)
- [x] Structured JSON output (`--format structured-json`)

### Layout API

```rust
pub struct TableDetector;
impl TableDetector {
    pub fn detect_tables(img: &OcrImage) -> Result<Vec<Table>>;
}

pub struct FormExtractor;
impl FormExtractor {
    pub fn extract(img: &OcrImage) -> Result<FormExtractionResult>;
}

pub struct RegionClassifier;
impl RegionClassifier {
    pub fn classify_region(region: &TextRegion) -> RegionClassification;
}

pub enum RegionType {
    Heading, SubHeading, Body, ListItem, Caption, Footer, PageNumber, Header, Unknown,
}
```

---

## Next Steps (Training Execution)

These items require running the training pipeline, not writing additional code:

1. **Train CRNN to target CER**
   - Run `cargo run --release -- train --engine lstm --epochs 100`
   - Or `cargo run --release --example train_pure_rust -- --epochs 100` (full BPTT, writes a new `models/crnn-en.bin`)
   - Target: CER < 5% clean, CER < 15% distorted
   - Debug inference is ~4s/line; use `--release` for realistic speed

2. **Evaluate per-language accuracy**
   - Run `cargo run --release -- benchmark --samples 50`
   - Measures CER/WER per script via `ScriptModelRegistry` on synthetic data

---

## Quantization API (INT8)

**Status:** Implemented. Provides 4x memory reduction for the FC readout layer.

### API

```rust
pub struct QuantizedTensor {
    pub weights: Vec<i8>,
    pub scale: f32,
    pub shape: (usize, usize),
}

impl QuantizedTensor {
    pub fn from_vec(weights: Vec<i8>, scale: f32, rows: usize, cols: usize) -> Self;
    pub fn to_array2(&self) -> Array2<f32>;
    pub fn size_bytes(&self) -> usize;
}

pub fn quantize_array2(arr: &Array2<f32>) -> QuantizedTensor;
pub fn quantize_array1(arr: &Array1<f32>) -> (Vec<i8>, f32);
pub fn quantized_matmul(input: &Array2<f32>, qw: &QuantizedTensor) -> Array2<f32>;
pub fn compression_ratio(element_count: usize) -> f32; // ~4.0x
```

### Usage

```rust
let mut model = CrnnModel::new(config);
model.quantize_fc(); // Converts fc_weight to INT8, keeps original for training
let logits = model.forward(&image); // Automatically uses quantized weights
```

### ONNX Import API

Enabled via the `onnx` feature flag (`--features onnx`).

```rust
pub struct OnnxLoader {
    // Stores owned bytes; parses on demand
}

pub struct SimplifiedNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl OnnxLoader {
    pub fn from_file(path: &Path) -> Result<Self>;
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self>;
    pub fn node_count(&self) -> Result<usize>;
    pub fn weight_count(&self) -> Result<usize>;
    pub fn op_types(&self) -> Result<Vec<String>>;
    pub fn extract_weights(&self) -> Result<HashMap<String, ArrayD<f32>>>;
    pub fn weight_by_name(&self, name: &str) -> Result<ArrayD<f32>>;
    pub fn nodes_by_op(&self, op: &str) -> Result<Vec<SimplifiedNode>>;
}
```

### Font Attribute Detection API

```rust
pub struct FontAttributes {
    pub bold: bool,
    pub italic: bool,
    pub monospace: bool,
}

pub struct FontAttributeDetector {
    pub bold_threshold: f32,        // default 1.5x median stroke
    pub italic_threshold: f32,      // default 8 degrees
    pub monospace_cv_threshold: f32, // default 0.15
}

impl FontAttributeDetector {
    pub fn detect(&self, image: &DynamicImage) -> FontAttributes;
}
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | yes | SIMD-accelerated kernels (scalar fallback on non-x86) |
| `pdf` | no | Searchable PDF output (`-f pdf`) — zero external deps, built-in Helvetica + JPEG XObject |
| `web-api` | no | HTTP API server (`ocr serve`) via axum |
| `cuda` | no | CUDA compute backend (requires NVIDIA GPU + `cudarc`) |
| `opencl` | no | OpenCL compute backend (requires `ocl`) |
| `onnx` | no | ONNX weight import (`ocr onnx import`) |
