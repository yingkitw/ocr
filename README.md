# OCR — Pure Rust Optical Character Recognition

A from-scratch OCR system built in Rust, informed by decades of OCR research and modern deep-learning pipelines. The goal is a clean, extensible, pure-Rust toolchain that can rival Tesseract on CPU and approach EasyOCR/PaddleOCR accuracy.

## Current State

**Compiles and passes 410+ tests.** The full pipeline works end-to-end for printed text on clean images, with a modular architecture supporting both classical and learned recognition:

- **Image preprocessing**: deskew, perspective dewarp, curved-line rectification, super-resolution for tiny/low-DPI text, quality gate (auto-sharpen/contrast), binarization (Otsu, Sauvola), noise reduction, contrast enhancement, auto-rotate (0°/90°/180°/270°)
- **Layout analysis**: Union-Find CCL, multi-angle oriented detection (±45°), column/line detection, reading-order resolution, table detection, form field extraction
- **Text detection**: `TextDetector` trait with CCL and lightweight CNN implementations
- **Recognition**: Pattern-matching engine (trainable from synthetic fonts) + CRNN (CNN + BiLSTM + CTC beam search with dictionary/LM rescoring and calibrated confidence) selectable via `--engine`
- **Multi-language**: Unicode script detection (Latin, CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari), dictionaries for 25+ languages, `--lang auto`, per-script CRNN vocabularies
- **Post-processing**: Dictionary-based spell correction, document structure classification (headings, lists, paragraphs), hierarchical Markdown/JSON output
- **Output formats**: text, JSON, hOCR, TSV, ALTO XML, box files, **searchable PDF** (feature `pdf-output`), **Markdown**, **structured JSON**
- **CLI**: `extract`, `batch` (with progress/ETA), `layout`, `list-languages`, `check`, `info`, `validate`, `train`, `benchmark`, `makebox`
- **Training**: `CrnnTrainer` with FC-layer backprop on synthetic data + checkpoint saving
- **Benchmarking**: Per-script CER/WER evaluation across all supported scripts
- **Font attributes**: Bold/italic/monospace detection from stroke analysis (`FontAttributeDetector`)
- **Quantization**: INT8 per-tensor symmetric quantization for FC layer (4x memory reduction)
- **ONNX import** (feature `onnx`): Load pre-trained PaddleOCR/EasyOCR weights via `OnnxLoader`
- **Web API** (feature `web-api`): HTTP server with multipart upload
- **PDF input** (feature `pdf`)

## What We Learned from Popular OCR Offerings

| Engine | Detection | Recognition | Languages | Strengths |
|--------|-----------|-------------|-----------|-----------|
| **Tesseract** | Connected components + LSTM line finder | Conv + BiLSTM + CTC (VGSL) | 100+ | Mature, tiny binary, great for scanned docs |
| **EasyOCR** | CRAFT (CNN) | CRNN (ResNet + BiLSTM + CTC) | 80+ | Simple PyTorch API, good scene text |
| **PaddleOCR** | DB (differentiable binarization) | SVTR / CRNN | 80+ | Best speed/accuracy trade-off, ultra-light models |
| **TrOCR** | — | Transformer encoder-decoder | 100+ | Excellent on printed text, no detection stage |
| **dots.ocr / VLMs** | End-to-end vision-language | 3B-param multimodal | 100+ | Structured output (tables, headings) but needs GPU |

### Key Architectural Lessons

1. **Two-stage pipeline** (detection → recognition) dominates for documents. End-to-end VLMs are promising but overkill for pure OCR.
2. **CTC decoders** remove the need for per-character segmentation in sequence models.
3. **Synthetic training data** (rendered fonts + realistic distortions) lets you bootstrap accuracy before collecting real data.
4. **Image preprocessing matters**: deskew, binarization, and contrast enhancement can improve accuracy 10–20%.
5. **Post-processing dictionaries / language models** close the gap between raw recognition and human-level accuracy.
6. **CPU deployment** still matters: Tesseract remains ubiquitous because it runs everywhere without GPU dependencies.

## Roadmap

### Phase 0 — Baseline (DONE)
- [x] End-to-end pipeline: image → preprocess → layout → recognize → output
- [x] Pattern-matching recognition engine for Latin glyphs
- [x] Layout analysis with CCL, column detection, reading order
- [x] Dictionary-based post-correction
- [x] CLI with extract, batch, layout, list-languages
- [x] 408+ tests passing

### Phase 1 — Synthetic Training Infrastructure (DONE)
- [x] Synthetic text-image generator with TTF font rendering and bitmap fallback
- [x] Distortion pipeline (rotation, blur, noise, contrast, shear)
- [x] CER/WER benchmark harness with clean/mild/heavy test sets
- [x] Train pattern-matching templates from synthetic renders (`TemplateTrainer`)

### Phase 2 — Robust Text Detection (DONE)
- [x] `TextDetector` trait with `CclDetector` and `CnnDetector`
- [x] Lightweight 3-layer detection CNN in pure Rust (ndarray)
- [x] Synthetic document generator + IoU-based evaluation harness
- [x] Auto-rotate 0°/90°/180°/270° via projection-variance orientation detection

### Phase 3 — Learned Recognition (CRNN) (DONE)
- [x] CNN feature extractor (5 conv layers + maxpool)
- [x] 2-layer bidirectional LSTM (64 hidden)
- [x] CTC decoder (greedy + beam search + dictionary/LM rescoring)
- [x] Calibrated per-character / word confidence (`ConfidenceCalibrator`)
- [x] Perspective dewarp + curved-line rectification
- [x] Arbitrary-angle text detection (`OrientedCclDetector`)
- [x] Super-resolution upscaling for tiny / low-DPI text
- [x] `ocr makebox` — Tesseract-compatible training `.box` export
- [x] CTC loss with forward-backward algorithm
- [x] `CrnnTrainer` with synthetic data + checkpoint saving
- [x] Wired into `OcrEngine` via `--engine lstm`
- [x] Model size < 5MB (~2.4MB default config)

### Phase 4 — Multi-Language Scale (DONE)
- [x] Unicode script detection and routing (8 scripts)
- [x] Dictionaries for 25+ languages
- [x] Multi-script synthetic data generation (`ScriptLineGenerator`)
- [x] Per-script CRNN models (`ScriptModelRegistry`)

### Phase 5 — Advanced Layout & Structure (DONE)
- [x] Table detection with grid line scan and cell reconstruction
- [x] Row/column span inference via pixel-density checks
- [x] Form field extraction (checkboxes, key-value pairs, underline fields)
- [x] Document structure classification (headings, paragraphs, lists, captions)
- [x] Hierarchical Markdown and structured JSON output
- [x] Searchable PDF generation with invisible text overlay

### Next Steps (requires training execution)
- [x] Training CLI wired up (`ocr train --engine lstm --epochs 10`)
- [x] Benchmark CLI wired up (`ocr benchmark --samples 10`)
- [x] CRNN inference optimized (rayon-parallel convolutions, ndarray FC layer)
- [ ] Train CRNN to CER < 5% on clean synthetic test
- [ ] Train CRNN to CER < 15% on distorted synthetic test
- [ ] Evaluate per-language accuracy on synthetic benchmarks

## Installation

```bash
# Default build (pattern matching engine)
cargo build --release

# Searchable PDF output (`--format pdf`)
cargo build --release --features pdf-output

# With all features
cargo build --release --all-features

# Run tests
cargo test
```

## Usage

```bash
# Recognize text from an image
ocr extract document.png

# JSON output with bounding boxes and confidence
ocr extract document.png -f json

# Markdown output with document structure
ocr extract document.png -f markdown

# Structured JSON with headings, paragraphs, lists
ocr extract document.png -f structured-json

# Enable preprocessing pipeline
ocr extract document.png --preprocess

# Use CRNN engine
ocr extract document.png --engine lstm

# Batch process a directory
ocr batch -i ./images -o ./results --lang en

# List supported languages
ocr list-languages

# Analyze image layout
ocr layout document.png -o layout.json

# Train CRNN on synthetic data
ocr train --engine lstm --epochs 10 --batch-size 8 --learning-rate 0.001

# Benchmark per-script recognition accuracy
ocr benchmark --samples 10 --distortion clean

# Generate Tesseract-style training .box file from a real image
ocr makebox scan.png -o train/page001
```

### Library

```rust
use ocr::api::Ocr;
use ocr::core::config::OcrConfig;

let config = OcrConfig::default();
let ocr = Ocr::with_config(config)?;
ocr.initialize().await?;

let result = ocr.recognize_text_from_file("document.png").await?;
println!("{}", result.text);
```

## Project Structure

```
src/
├── api/          High-level OCR API (Ocr, TextProcessor)
├── cli/          CLI argument parsing with clap
├── core/         Core OCR engine (config, engine, geometry, layout, output, text)
├── image/        Image preprocessing pipeline (binarization, enhancement, quality)
├── lang/         Language support (CJK, detection, dictionary, N-gram, unicode)
├── layout/       Layout analysis (column/line detection, text ordering, CCL, detection CNN, table/form extractors, classifier)
├── recognition/  Recognition models (pattern matching, CRNN, CTC decoder, LSTM)
├── synthetic/    Synthetic data generation (fonts, distortion, multi-script, document layouts, template training)
├── training/     Model training pipeline (data, losses, metrics, optimizers, CRNN trainer)
└── utils/        Shared utilities (async, error, hash, math, SIMD, time)
```

## License

Apache-2.0
