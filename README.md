# ocr

A pure Rust CLI OCR tool for printed text extraction with multiple recognition engines (pattern matching, LSTM, CNN, transformer) and full layout analysis.

## Features

- **Pure Rust** — No Tesseract or external OCR dependencies
- **Multiple engines** — Pattern matching, LSTM, CNN, transformer, hybrid, ViT
- **Layout analysis** — Column detection, text ordering, line segmentation via Union-Find CCL
- **Image preprocessing** — Binarization (Otsu, Sauvola), noise reduction, deskew, contrast enhancement
- **Language support** — English, CJK (Chinese/Japanese/Korean), N-gram language detection
- **CLI** — Extract, batch, layout analysis, config validation
- **Output formats** — Plain text, JSON, hOCR, TSV
- **SIMD acceleration** — Optional SIMD-optimized routines

## Installation

```bash
cargo build --release
```

## Usage

```bash
# Recognize text from an image
ocr extract document.png

# JSON output with word-level detail
ocr extract document.png -f json

# Enable preprocessing
ocr extract document.png --preprocess

# Batch process a directory
ocr batch -i ./images -o ./results

# List supported languages
ocr list-languages

# Analyze image layout
ocr layout document.png
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
├── recognition/  Recognition models (LSTM, CNN, transformer, ViT, pattern, hybrid)
├── training/     Model training pipeline (data, losses, metrics, optimizers)
└── utils/        Shared utilities (async, error, hash, math, SIMD, time)
```

## License

Apache-2.0
