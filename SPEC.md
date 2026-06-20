# Specification

## Philosophy

This spec defines concrete, testable milestones. Each phase must compile, pass tests, and demonstrate measurable improvement over the previous phase.

---

## Phase 0 — Baseline (DONE)

**Goal:** End-to-end pipeline that compiles and passes 250+ tests.

### Acceptance Criteria
- [x] `cargo test` passes with zero failures
- [x] `cargo build --release` produces a working binary
- [x] `ocr extract test-image.png` produces text output
- [x] All output formats (text, JSON, hOCR, TSV, ALTO, box) render without panic
- [x] CLI covers: extract, batch, layout, list-languages, check, info, validate

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

### Extract Flags

| Flag | Default | Description |
|------|---------|-------------|
| `IMAGE_PATH` | required | Path to the image file |
| `-o, --output` | stdout | Output file path |
| `-l, --lang` | `en` | Language code (en, fr, de, es, it, pt, ru, zh, ja, ko, ...) |
| `--preprocess` | false | Enable full preprocessing pipeline |
| `-f, --format` | `text` | Output: text, json, hocr, tsv, alto, box |
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

---

## Phase 1 — Synthetic Training Data

**Goal:** Generate unlimited training data from fonts + distortions.

### Acceptance Criteria
- [ ] `src/synthetic/` module generates 1000+ unique text-line images per second
- [ ] Supports 10+ system fonts (monospace, serif, sans-serif)
- [ ] Applies realistic distortions: rotation ±5°, Gaussian blur, salt & pepper noise, random contrast, perspective shear
- [ ] Outputs paired dataset: `(image_bytes, ground_truth_text)`
- [ ] CER benchmark script runs against pattern matcher and reports baseline score

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
    pub fn generate(&self, text: &str) -> DynamicImage;
    pub fn generate_batch(&self, texts: &[&str]) -> Vec<(DynamicImage, String)>;
    pub fn apply_distortion(&self, image: &mut DynamicImage, distortion: Distortion);
}
```

---

## Phase 2 — Text Detection

**Goal:** Find text regions more robustly than CCL on complex layouts.

### Acceptance Criteria
- [ ] Detection recall ≥ 85% on synthetic multi-column documents
- [ ] Detection precision ≥ 90% on synthetic multi-column documents
- [ ] Works on rotated text (0°, 90°, 180°, 270°)
- [ ] Falls back to CCL when detection model is not loaded

### Detection API

```rust
pub trait TextDetector {
    fn detect(&self, image: &OcrImage) -> Result<Vec<TextRegion>>;
}

pub struct CclDetector;          // Current approach
pub struct CraftLiteDetector;    // CNN-based character regions
pub struct DbnetLiteDetector;    // Differentiable binarization
```

---

## Phase 3 — CRNN Recognition

**Goal:** Replace pattern matching with a learned sequence model.

### Acceptance Criteria
- [ ] CER < 5% on clean synthetic test set (vs pattern matcher's current ~15%)
- [ ] CER < 15% on distorted synthetic test set
- [ ] Inference time < 100ms per text line on a single CPU core
- [ ] Model size < 5MB (lightweight for deployment)
- [ ] Can be serialized/deserialized for distribution

### CRNN Architecture

```
Input: 32xW grayscale text line image
  ↓
Conv(64, 3×3) → ReLU → MaxPool(2,2)
Conv(128, 3×3) → ReLU → MaxPool(2,2)
Conv(256, 3×3) → ReLU → Conv(256, 3×3) → ReLU → MaxPool(1,2)
Conv(512, 3×3) → BatchNorm → ReLU → MaxPool(1,2)
Conv(512, 2×2) → BatchNorm → ReLU
  ↓  [Feature maps: 512 × 1 × (W/4)]
Reshape to sequence: (W/4) × 512
  ↓
BiLSTM(256) → BiLSTM(256)
  ↓
Linear(256 → num_classes + 1)
  ↓
CTC Greedy Decode → text string
```

### Training API

```rust
pub struct CrnnTrainer {
    model: CrnnModel,
    optimizer: Adam,
    loss_fn: CtcLoss,
}

impl CrnnTrainer {
    pub fn train_epoch(&mut self, dataset: &SyntheticDataset) -> Metrics;
    pub fn evaluate(&self, dataset: &SyntheticDataset) -> Metrics;
    pub fn save_checkpoint(&self, path: &Path) -> Result<()>;
}
```

---

## Phase 4 — Multi-Language

**Goal:** Support 30+ languages with automatic script routing.

### Acceptance Criteria
- [ ] Unicode script detection (Latin, CJK, Arabic, Cyrillic, etc.)
- [ ] Language-specific preprocessing (CJK vertical text, Arabic RTL, etc.)
- [ ] Per-language dictionaries loaded on demand
- [ ] OCR accuracy ≥ 90% on printed text for supported languages

---

## Phase 5 — Advanced Layout

**Goal:** Understand document structure, not just extract flat text.

### Acceptance Criteria
- [ ] Table detection with row/column reconstruction
- [ ] Form field extraction (key-value pairs)
- [ ] Heading/paragraph/list classification
- [ ] Output as structured markdown or JSON with hierarchy preserved
