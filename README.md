# OCR — Pure Rust Optical Character Recognition

A from-scratch OCR system built in Rust, informed by decades of OCR research and modern deep-learning pipelines. The goal is a clean, extensible, pure-Rust toolchain that can eventually rival Tesseract on CPU and approach EasyOCR/PaddleOCR accuracy with optional GPU acceleration.

## Current State

**Compiles and passes 289+ tests.** The baseline pipeline works end-to-end for printed Latin text on clean images:

- **Image preprocessing**: binarization (Otsu, Sauvola), noise reduction, deskew, contrast enhancement
- **Layout analysis**: Union-Find CCL, column/line detection, reading-order resolution
- **Recognition**: Pattern-matching engine with built-in glyph templates (37 characters in 5×7 bitmaps)
- **Post-processing**: Dictionary-based spell correction for English, French, Spanish, German, Italian, Portuguese, Russian
- **Output**: text, JSON, hOCR, TSV, ALTO XML, box files
- **CLI**: `extract`, `batch`, `layout`, `list-languages`, `check`, `info`, `validate`
- **Web API** (feature `web-api`): HTTP server with multipart upload

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
- [x] 289+ tests passing

### Phase 1 — Synthetic Training Infrastructure
- [ ] Synthetic text-image generator (render fonts + realistic distortions)
- [ ] Train pattern-matching templates from synthetic data
- [ ] Automated accuracy benchmark on synthetic test set

### Phase 2 — Robust Text Detection
- [ ] Implement CRAFT-style character-region awareness (lightweight CNN)
- [ ] DBNet-inspired differentiable binarization for text regions
- [ ] Evaluate detection recall/precision on ICDAR-style benchmarks

### Phase 3 — Learned Recognition (CRNN)
- [ ] Implement CNN feature extractor (VGG-like or lightweight ResNet)
- [ ] BiLSTM sequence modeling layer
- [ ] CTC decoder (greedy + beam search)
- [ ] Train end-to-end on synthetic data
- [ ] Replace pattern matching as default engine

### Phase 4 — Scale Languages
- [ ] Expand synthetic data to CJK scripts
- [ ] Unicode script detection and routing
- [ ] Multi-script line recognition

### Phase 5 — Advanced Layout & Structure
- [ ] Table detection and reconstruction
- [ ] Form field extraction
- [ ] Heading/paragraph/figure classification

## Installation

```bash
# Default build (pattern matching engine)
cargo build --release

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

# Enable preprocessing pipeline
ocr extract document.png --preprocess

# Batch process a directory
ocr batch -i ./images -o ./results --lang en

# List supported languages
ocr list-languages

# Analyze image layout
ocr layout document.png -o layout.json
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
├── layout/       Layout analysis (column/line detection, text ordering, CCL)
├── recognition/  Recognition models (pattern, LSTM, CNN, transformer stubs)
├── training/     Model training pipeline (data, losses, metrics, optimizers)
└── utils/        Shared utilities (async, error, hash, math, SIMD, time)
```

## License

Apache-2.0
