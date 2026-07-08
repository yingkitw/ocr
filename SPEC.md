# Specification

## Philosophy

This spec defines concrete, testable milestones. Each phase must compile, pass tests, and demonstrate measurable improvement over the previous phase.

**Current status: All five phases implemented and tested. 373+ tests passing, 0 failures.**

---

## Phase 0 — Baseline (DONE)

**Goal:** End-to-end pipeline that compiles and passes tests.

### Acceptance Criteria
- [x] `cargo test` passes with zero failures (373+ tests)
- [x] `cargo build --release` produces a working binary
- [x] `ocr extract test-image.png` produces text output
- [x] All output formats render without panic
- [x] CLI covers: extract, batch, layout, list-languages, check, info, validate, train, benchmark

### CLI Specification

| Command | Description | Required Args | Optional Flags |
|---------|-------------|---------------|--------------|
| `extract` | Recognize text from an image | `IMAGE_PATH` | `-o, --output`, `-l, --lang`, `--preprocess`, `-f, --format`, `--psm`, `--confidence`, `--engine`, `--dict-correct`, `--osd` |
| `batch` | Process a directory of images | `-i, --input-dir`, `-o, --output-dir` | `-l, --lang`, `--confidence`, `--engine`, `--dict-correct` |
| `layout` | Analyze page layout | `IMAGE_PATH` | `-o, --output` |
| `list-languages` | Print supported language codes | — | — |
| `check` | Verify system requirements | — | — |
| `info` | Print engine metadata | — | — |
| `validate` | Validate a config file | `CONFIG_FILE` | — |
| `train` | Train a recognition model | — | `-e, --epochs`, `-b, --batch-size`, `-l, --learning-rate`, `--engine`, `-c, --checkpoint-dir`, `--distortion` |
| `benchmark` | Benchmark per-script accuracy | — | `-s, --samples`, `--distortion` |

### Extract Flags

| Flag | Default | Description |
|------|---------|-------------|
| `IMAGE_PATH` | required | Path to the image file |
| `-o, --output` | stdout | Output file path |
| `-l, --lang` | `en` | Language code (en, fr, de, es, it, pt, ru, zh, ja, ko, ...) |
| `--preprocess` | false | Enable full preprocessing pipeline |
| `-f, --format` | `text` | Output: text, json, hocr, tsv, alto, box, pdf, markdown, structured-json |
| `--psm` | `3` | Page segmentation mode (Tesseract-compatible) |
| `--confidence` | `0.5` | Minimum confidence threshold (0.0–1.0) |
| `--engine` | `pattern` | Recognition engine: pattern, lstm, hybrid |
| `--dict-correct` | false | Enable dictionary post-correction |
| `--osd` | false | Enable orientation & script detection |

### Library API

```rust
// High-level facade
pub struct Ocr;
impl Ocr {
    pub fn new() -> Result<Self>;
    pub fn with_config(config: OcrConfig) -> Result<Self>;
    pub async fn initialize(&self) -> Result<()>;
    pub async fn recognize_text_from_file(&self, path: &str) -> Result<TextResult>;
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
| `alto` | ALTO XML v4 | `<Page>`, `<TextBlock>`, `<TextLine>`, `<String>` |
| `box` | Box file format | `char left top right bottom page` |
| `pdf` | Searchable PDF with invisible text overlay | Original image + invisible text layer |
| `markdown` | Markdown with headings, paragraphs, lists | `## Heading`, `- List item` |
| `structured-json` | Hierarchical JSON with document elements | `{"elements": [{"Heading": {...}}, {"Paragraph": {...}}]}` |

---

## Phase 1 — Synthetic Training Data (DONE)

**Goal:** Generate unlimited training data from fonts + distortions.

### Acceptance Criteria
- [x] `src/synthetic/` module generates text-line images from TTF fonts + bitmap fallback
- [x] Applies realistic distortions: rotation ±5°, Gaussian blur, salt & pepper noise, random contrast, perspective shear
- [x] Outputs paired dataset: `(image_bytes, ground_truth_text)`
- [x] CER/WER benchmark harness with clean/mild/heavy test sets
- [x] `TemplateTrainer` trains pattern-matching templates from synthetic renders and evaluates vs baseline

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

pub struct TemplateTrainer {
    pub fn train_templates(&self, chars: &[char]) -> HashMap<char, TrainedTemplate>;
    pub fn train_ascii(&self) -> HashMap<char, TrainedTemplate>;
    pub fn evaluate_templates(...) -> (trained_correct, baseline_correct, total);
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

## Phase 3 — CRNN Recognition (DONE)

**Goal:** Replace pattern matching with a learned sequence model.

### Acceptance Criteria
- [x] CNN feature extractor (5 conv layers + maxpool) in ndarray
- [x] 2-layer BiLSTM (64 hidden) with forward/backward pass
- [x] CTC decoder: greedy + beam search
- [x] CTC loss with forward-backward algorithm
- [x] `CrnnTrainer` with synthetic data generation + checkpoint saving/loading
- [x] Wired into `OcrEngine` via `--engine lstm`
- [x] Model size < 5MB (~2.4MB default config)
- [x] Inference benchmark test (target < 100ms/line in release)
- [x] FC-layer backprop via frame-level cross-entropy (CNN/BiLSTM frozen)
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
CTC Greedy Decode → text string
```

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
    pub fn evaluate(&self, samples: &[SyntheticSample]) -> EpochMetrics;
    pub fn save_checkpoint(&self, path: &Path) -> Result<()>;
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
- [x] Searchable PDF generation with invisible text overlay

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
