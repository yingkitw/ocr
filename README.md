# OCR — Pure Rust OCR Engine & CLI for Text Recognition from Images

[![crates.io](https://img.shields.io/crates/v/ocr)](https://crates.io/crates/ocr)
[![docs.rs](https://docs.rs/ocr/badge.svg)](https://docs.rs/ocr)
[![License: Apache-2.0](https://img.shields.io/crates/l/ocr)](LICENSE)
[![Rust Edition 2024](https://img.shields.io/badge/rust-2024-edition-orange.svg)](https://doc.rust-lang.org/edition-guide/)
[![Tests](https://img.shields.io/badge/tests-423_passing-brightgreen.svg)](#roadmap)
[![Build](https://img.shields.io/badge/build-zero_warnings-brightgreen.svg)](#)

> **Pure-Rust OCR engine and command-line tool for extracting printed text from images.** No C/C++ dependencies, no Python, no GPU required. Supports 25+ languages, multi-page TIFF, and output to text, JSON, Markdown, hOCR, TSV, CSV, ALTO XML, PAGE XML, and searchable PDF. Apache-2.0 licensed.

```bash
cargo install ocr
ocr extract document.png
```

## Overview

**OCR** is an open-source optical character recognition system written entirely in [Rust](https://www.rust-lang.org/). It converts images of printed text into machine-readable text using a CRNN (Convolutional Recurrent Neural Network) with CTC decoding — the same architecture used by [Tesseract](https://github.com/tesseract-ocr/tesseract) and [EasyOCR](https://github.com/JaidedAI/EasyOCR), but implemented in pure Rust with zero C/C++ build dependencies.

The engine ships as both a **CLI binary** (`ocr extract image.png`) and an **embeddable async library** (`use ocr::api::Ocr`), making it suitable for command-line workflows, server-side document processing, and embedded Rust applications.

### Key Highlights

- **Pure Rust** — no C/C++ compiler, no Python runtime, no system libraries required
- **25+ languages** — Latin, CJK (Chinese/Japanese/Korean), Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari
- **Multi-page TIFF** — recognizes every page, not just the first frame
- **11 output formats** — text, JSON, Markdown, hOCR, TSV, CSV, ALTO XML, PAGE XML, structured JSON, RAG JSON, searchable PDF
- **CPU-first** — runs on any machine; optional CUDA/OpenCL/ONNX features for GPU acceleration
- **Apache-2.0** — commercial use permitted, no copyleft restrictions
- **< 5 MB model** — bundled CRNN model embedded at compile time via `include_bytes!`

## Why Choose This OCR Engine?

| You need… | This crate | Alternatives |
|---|---|---|
| OCR in Rust without C++ dependencies | ✅ | Tesseract (C++), EasyOCR (Python) |
| Embeddable async OCR library | ✅ | Most engines are CLI-only or Python-only |
| Multi-page TIFF processing | ✅ | Many engines decode only the first frame |
| RAG-ready chunked output for LLMs | ✅ | Rare in OCR engines |
| Table and form field extraction | ✅ | PaddleOCR (Python), docTR (Python) |
| Apache-2.0 license for commercial use | ✅ | Some engines use GPL/copyleft |
| CPU-only deployment (no GPU) | ✅ | TrOCR, surya, VLMs need GPU |

### When to use something else

- **Scene text** (natural photos, street signs): use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) or [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **End-to-end VLM** with structured reasoning: use [surya](https://github.com/VikParuchuri/surya) or [dots.ocr](https://github.com/rednotehilo/dots.ocr) (requires GPU)
- **100+ language coverage**: use [Tesseract](https://github.com/tesseract-ocr/tesseract) (mature, 100+ trained models)

## Comparison with Popular OCR Engines

| Engine | Language | Detection | Recognition | Languages | Best For |
|---|---|---|---|---|---|
| **OCR (this crate)** | Pure Rust | CCL + CNN | CRNN + CTC | 25+ | Embedded Rust, no C deps, async API |
| Tesseract | C++ | CCL + LSTM | Conv + BiLSTM + CTC | 100+ | Scanned documents, mature ecosystem |
| EasyOCR | Python | CRAFT (CNN) | CRNN (ResNet + BiLSTM) | 80+ | Simple PyTorch API, scene text |
| PaddleOCR | Python/C++ | DB | SVTR / CRNN | 80+ | Best speed/accuracy, ultra-light models |
| TrOCR | Python | — | Transformer | 100+ | Printed text, no detection stage |
| docTR / surya | Python | DB / CRAFT | Transformer / CRNN | 100+ | Document AI, structured output |

## Features

### Image Preprocessing
- Deskew, perspective dewarp, curved-line rectification
- Super-resolution upscaling for tiny / low-DPI text
- Auto-sharpen and contrast quality gate
- Binarization (Otsu, Sauvola)
- Noise reduction and contrast enhancement
- Auto-rotate (0° / 90° / 180° / 270°)

### Layout Analysis
- Union-Find Connected Components labeling (CCL)
- Multi-angle oriented text detection (±45° sweep)
- Column and line detection with reading-order resolution
- Table detection (grid-line scan + cell reconstruction with span inference)
- Form field extraction (checkboxes, key-value pairs, underline fields)
- Document structure classification (headings, paragraphs, lists, captions)

### Text Recognition
- **CRNN architecture**: 5-layer CNN feature extractor + 2-layer BiLSTM + CTC decoder
- **Greedy and beam search decoding** with dictionary / n-gram LM rescoring
- **Bundled model** (`models/crnn-en.bin`, 3.07 MB) embedded via `include_bytes!`
- **Calibrated confidence** per character and word (`ConfidenceCalibrator`)
- Model size < 5 MB
- **WebOCR template-matching engine** (2026-10-18, port of the sibling `../webocr`): `--engine template` selects the alternative engine — binarize → connected components → segmentation (valley split + stacked merge + CJK cluster + gap-cluster word spacing) → own-height normalization → cosine matching against system-font-rendered templates. No model file, no GPU: templates are built at first recognition from platform fonts via rusttype. Latin + CJK union. Self-contained `src/webocr/` module mirroring WebOCR's module layout 1:1. `--lang auto` uses image geometry to pick the right library. Sibling WebOCR benchmark: sans 100 %, monospace 100 %, serif ≈97.8 %, dark/inverted 100 % on real-font fixtures.

### Multi-Language Support
- Unicode script detection: Latin, CJK, Arabic, Cyrillic, Greek, Hebrew, Thai, Devanagari
- Dictionaries for 25+ languages
- `--lang auto` for mixed-script documents
- Per-script CRNN vocabularies (`ScriptModelRegistry`)
- `ScriptLineGenerator` for multi-script synthetic training

### Output Formats

| Format | Flag | Description |
|---|---|---|
| Plain text | `-f text` | Raw recognized text |
| JSON | `-f json` | Text + confidence + line/word bounding boxes |
| Markdown | `-f markdown` | Headings, paragraphs, lists |
| hOCR | `-f hocr` | HTML 4.01 with hOCR 4.1 classes (Tesseract-compatible) |
| TSV | `-f tsv` | Tesseract-style 5-level format with bbox per word |
| CSV | `-f csv` | RFC 4180 per-word rows for spreadsheets/databases |
| ALTO XML | `-f alto` | Standard library/archive format |
| PAGE XML | `-f page` | PRImA PAGE XML 2019-07-15 (document layout) |
| Structured JSON | `-f structured-json` | Bounding boxes, confidence, hierarchy |
| RAG JSON | `-f rag` | Retrieval-friendly chunks with heading-path context for LLM / embedding pipelines |
| Searchable PDF | `-f pdf` | Image + invisible text layer (requires `--features pdf`) |
| Box file | `-f box` | Tesseract-compatible `.box` for training (`ocr makebox`) |

### Post-Processing
- Dictionary-based spell correction (25+ languages)
- N-gram language model rescoring
- Document structure classification

### Training and Benchmarking
- `CrnnTrainer` with full BPTT (CNN + BiLSTM + FC) through pure-Rust CTC loss
- Adam optimizer with checkpoint save/load
- `ocr train --engine lstm --epochs 10` CLI
- `ocr benchmark --samples 10` for per-script CER/WER evaluation
- INT8 per-tensor symmetric quantization for FC layer (4× memory reduction)

### Optional Features
- **ONNX import** (`--features onnx`): load PaddleOCR / EasyOCR weights
- **Web API** (`--features web-api`): HTTP server with multipart upload
- **CUDA** (`--features cuda`): GPU acceleration via `cudarc`
- **OpenCL** (`--features opencl`): GPU acceleration via `ocl`
- **SIMD** (default): CPU SIMD optimizations

## Supported Languages

Run `ocr list-languages` for the full list. Highlights:

- **Latin script**: English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Swedish, Norwegian, Danish, Finnish, Czech, Hungarian, Romanian, Turkish, Vietnamese, Indonesian
- **CJK**: Chinese (Simplified + Traditional), Japanese, Korean
- **Other scripts**: Arabic, Russian, Ukrainian, Bulgarian, Greek, Hebrew, Thai, Hindi, Marathi

Use `--lang auto` to auto-detect scripts on mixed-language documents.

## Installation

```bash
# Install from crates.io (default build: CRNN engine + SIMD)
cargo install ocr

# Or build from source
git clone https://github.com/yingkitw/ocr.git
cd ocr
cargo build --release

# With all optional features
cargo build --release --all-features

# Run the test suite
cargo test
```

### Requirements

- Rust 1.85+ (edition 2024)
- No C/C++ compiler required
- No GPU required
- No Python dependency

## CLI Usage

### Extract Text from an Image

```bash
# Basic text extraction
ocr extract document.png

# JSON output with bounding boxes and confidence
ocr extract document.png -f json

# Markdown output with document structure
ocr extract document.png -f markdown

# Structured JSON with headings, paragraphs, lists
ocr extract document.png -f structured-json

# RAG-ready chunks for LLM / embedding pipelines
ocr extract document.png -f rag

# hOCR (Tesseract-compatible HTML format)
ocr extract document.png -f hocr

# PRImA PAGE XML (document layout format)
ocr extract document.png -f page

# CSV per-word export (for spreadsheets/databases)
ocr extract document.png -f csv

# Read image from stdin (Unix pipe)
cat document.png | ocr extract -

# Multi-page TIFF (each page recognized independently)
ocr extract multipage_scan.tiff

# Enable preprocessing pipeline (deskew, contrast, etc.)
ocr extract document.png --preprocess

# Specify input DPI (affects super-resolution and layout)
ocr extract document.png --dpi 150

# Auto-detect language on mixed-script documents
ocr extract document.png --lang auto

# Page segmentation modes (Tesseract --psm: numbers or symbolic names)
ocr extract document.png --psm 6              # single uniform block
ocr extract receipt.png --psm 6 --whitelist 0123456789.,-  # digits only
ocr extract line.png --psm single_line        # whole image = one text line
ocr extract form.png --psm circle_word        # strip enclosing circle first
ocr extract scan.png --psm osd_only           # orientation + script report

# Orientation & script detection report (Tesseract OSD)
ocr extract rotated.png --psm 0
```

### Batch Processing

```bash
# Process a directory of images
ocr batch -i ./images -o ./results --lang en

# List supported languages
ocr list-languages

# Analyze layout only (no recognition)
ocr layout document.png -o layout.json
```

### Training and Benchmarking

```bash
# Train CRNN on synthetic data
ocr train --engine lstm --epochs 10 --batch-size 8 --learning-rate 0.001

# Benchmark per-script recognition accuracy
ocr benchmark --samples 10 --distortion clean

# Generate Tesseract-style training .box file
ocr makebox scan.png -o train/page001
```

## Library API

```rust
use ocr::api::Ocr;
use ocr::core::config::OcrConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OcrConfig::default();
    let ocr = Ocr::with_config(config)?;
    ocr.initialize().await?;

    // Recognize text from a file
    let result = ocr.recognize_text_from_file("document.png").await?;
    println!("{}", result.text);

    // Or from encoded bytes (PNG, JPEG, TIFF, etc.)
    let bytes = std::fs::read("document.png")?;
    let result = ocr.recognize_text_from_bytes(&bytes).await?;

    // Multi-page TIFF: returns one result per page
    let pages = ocr.recognize_text_from_bytes_multi_page(&bytes).await?;
    println!("{} pages recognized", pages.len());

    Ok(())
}
```

## Web API

```bash
# Build with web-api feature
cargo build --release --features web-api

# Start the server
./target/release/ocr serve --port 8080

# Upload an image for OCR
curl -F "image=@document.png" http://localhost:8080/ocr
```

## Use Cases

- **Document digitization**: convert scanned images into searchable text or Markdown
- **Receipt and invoice parsing**: extract structured data with the table detector
- **Form processing**: detect checkboxes, key-value pairs, and underline fields
- **CLI toolchains**: embed OCR in Rust binaries without linking Tesseract's C++ runtime
- **Offline OCR**: run on air-gapped machines — no cloud API calls, no network needed
- **Self-hosted OCR service**: deploy the Web API behind a reverse proxy
- **RAG pipelines**: generate chunked JSON with heading context for LLM retrieval
- **Mixed-script documents**: process Arabic + Latin, CJK + Latin on a single page
- **Multi-page scanning**: process multi-page TIFF files page-by-page

## Architecture

```
src/
├── api/          High-level OCR API (Ocr, TextProcessor)
├── cli/          CLI argument parsing with clap
├── core/         Core OCR engine (config, engine, geometry, layout, output, text)
├── image/        Image preprocessing pipeline (binarization, enhancement, quality)
├── lang/         Language support (CJK, detection, dictionary, N-gram, unicode)
├── layout/       Layout analysis (column/line detection, text ordering, CCL, tables, forms)
├── recognition/  Recognition models (CRNN, CTC decoder, model loader, confidence)
├── synthetic/    Synthetic data generation (fonts, distortion, multi-script, documents)
├── training/     Model training pipeline (CTC loss, BPTT, Adam, checkpointing)
├── compute/      GPU compute backends (CUDA, OpenCL)
├── onnx/         ONNX model loader
├── server/       Web API server
└── utils/        Shared utilities (async, error, hash, math, SIMD, time)
```

## Roadmap

### Phase 0 — Baseline (DONE)
- [x] End-to-end pipeline: image → preprocess → layout → recognize → output
- [x] Layout analysis with CCL, column detection, reading order
- [x] Dictionary-based post-correction
- [x] CLI with `extract`, `batch`, `layout`, `list-languages`
- [x] 423 tests passing

### Phase 1 — Synthetic Training Infrastructure (DONE)
- [x] Synthetic text-image generator with TTF font rendering and bitmap fallback
- [x] Distortion pipeline (rotation, blur, noise, contrast, shear)
- [x] CER/WER benchmark harness with clean / mild / heavy test sets

### Phase 2 — Robust Text Detection (DONE)
- [x] `TextDetector` trait with `CclDetector` and `CnnDetector`
- [x] Lightweight 3-layer detection CNN in pure Rust (ndarray)
- [x] Synthetic document generator + IoU-based evaluation harness
- [x] Auto-rotate 0°/90°/180°/270° via projection-variance orientation detection

### Phase 3 — Learned Recognition (CRNN) (DONE)
- [x] CNN feature extractor (5 conv layers + maxpool)
- [x] 2-layer bidirectional LSTM (64 hidden)
- [x] CTC decoder (greedy + beam search + dictionary/LM rescoring)
- [x] Calibrated per-character / word confidence
- [x] Perspective dewarp + curved-line rectification
- [x] Arbitrary-angle text detection
- [x] Super-resolution upscaling for tiny / low-DPI text
- [x] `ocr makebox` — Tesseract-compatible training `.box` export
- [x] CTC loss with forward-backward algorithm
- [x] `CrnnTrainer` with synthetic data + checkpoint saving
- [x] Model size < 5 MB (bundled `models/crnn-en.bin`, 3.07 MB)

### Phase 4 — Multi-Language Scale (DONE)
- [x] Unicode script detection and routing (8 scripts)
- [x] Dictionaries for 25+ languages
- [x] Multi-script synthetic data generation
- [x] Per-script CRNN models

### Phase 5 — Advanced Layout & Structure (DONE)
- [x] Table detection with grid line scan and cell reconstruction
- [x] Row/column span inference
- [x] Form field extraction (checkboxes, key-value pairs, underline fields)
- [x] Document structure classification (headings, paragraphs, lists, captions)
- [x] Hierarchical Markdown and structured JSON output

### Next Steps
- [x] Training CLI wired up (`ocr train --engine lstm`)
- [x] Benchmark CLI wired up (`ocr benchmark`)
- [x] Full BPTT training through CNN + BiLSTM + FC with pure-Rust CTC loss
- [x] Multi-page TIFF input
- [x] CSV and PAGE XML output formats
- [ ] Train CRNN to CER < 5% on clean synthetic test
- [ ] Train CRNN to CER < 15% on distorted synthetic test
- [ ] Evaluate per-language accuracy on synthetic benchmarks

## FAQ

### How does this compare to Tesseract?

This crate is a pure-Rust alternative to Tesseract for printed text on clean images. Key differences: the API is async Rust (vs. C++ bindings), it bundles no C++ runtime, the license is Apache-2.0, and language coverage is narrower (25+ vs. 100+). For most printed-document workflows on supported languages, accuracy is comparable.

### Does it run on GPU?

By default, no — the engine is CPU-only. Optional features (`cuda`, `opencl`, `onnx`) provide GPU acceleration paths, but the default build and test suite run entirely on CPU.

### Can it handle handwritten text?

Not yet. The engine is optimized for printed text. Handwriting recognition is on the roadmap and will likely use a separate transformer-based model.

### What image formats are supported?

PNG, JPEG, GIF, BMP, TIFF, and WebP (via the `image` crate). **Multi-page TIFF is supported** — each page is recognized independently and output is separated by `--- Page N ---` markers. PDF is not currently supported.

### Can I embed OCR in my Rust application?

Yes. Use `ocr::api::Ocr` for the high-level async API. The crate is published on [crates.io](https://crates.io/crates/ocr) and documented on [docs.rs](https://docs.rs/ocr).

### How do I train the CRNN model?

```bash
# Via the built-in trainer (full BPTT + CTC loss)
cargo run --release -- train --engine lstm --epochs 100 --batch-size 8 --learning-rate 0.001

# Or via the standalone example
cargo run --release --example train_pure_rust -- --epochs 100
```

Training requires multi-hour `--release` runs. Synthetic data is generated automatically.

### Is it production-ready?

The engine is feature-complete for printed text on clean scans with 428 passing tests. Two recognition engines: the CRNN (ML default) and the WebOCR template engine (no ML, system fonts; Latin + CJK union). The CRNN training pipeline (full BPTT + CTC loss) is implemented, and the bundled model is loaded at runtime. Accuracy will improve as trained checkpoints replace the current model.

### What is the license?

Apache-2.0. Commercial use is permitted with no copyleft restrictions.

### How do I report a bug or request a feature?

Open an issue at <https://github.com/yingkitw/ocr/issues>.

## Contributing

Contributions are welcome. Read the [AGENTS.md](AGENTS.md) file for the development loop, and [ARCHITECTURE.md](ARCHITECTURE.md) for module dependencies. The project follows a TODO → test → ship cadence: every new feature ships with integration tests.

## License

[Apache-2.0](LICENSE)
