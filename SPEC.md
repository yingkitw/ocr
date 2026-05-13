# Specification

## CLI Interface

### Commands

| Command         | Description |
|-----------------|-------------|
| `extract`       | Recognize text from an image |
| `batch`         | Batch process multiple images |
| `layout`        | Analyze image layout |
| `list-languages`| List supported languages |
| `check`         | Check system requirements |
| `info`          | Display engine information |
| `validate`      | Validate a configuration file |

### Extract Options

| Flag | Default | Description |
|------|---------|-------------|
| `IMAGE_PATH` | required | Path to the image file |
| `-o, --output` | stdout | Output file path |
| `-l, --lang` | `eng` | Language code |
| `--preprocess` | false | Enable image preprocessing |
| `-f, --format` | `text` | Output format: text, json, hocr, tsv |
| `--psm` | `3` | Page segmentation mode (Tesseract compatible) |
| `--confidence` | `0.5` | Confidence threshold (0.0 to 1.0) |

## Library API

### Ocr

| Method | Description |
|--------|-------------|
| `new()` | Create with default config |
| `with_config(OcrConfig)` | Create with custom config |
| `initialize()` | Initialize the OCR engine |
| `recognize_text_from_file(path)` | Recognize text from an image file |
| `recognize_text(image_data, width, height)` | Recognize text from raw image data |
| `analyze_layout(image_data, width, height)` | Analyze image layout |
| `get_supported_languages()` | List supported languages |
| `get_metadata()` | Get engine metadata |

### OcrConfig

Sections: `recognition`, `image_processing`, `layout_analysis`, `language`, `performance`, `debug`

### Recognition Engines

- `PatternMatching` — Default, bitmap template matching
- `LSTM` — LSTM neural network
- `CNN` — Convolutional neural network
- `Transformer` — Transformer model
- `Hybrid` — Ensemble of multiple engines
- `EndToEnd` — Integrated detection and recognition
- `ViT` — Vision transformer

## Output Formats

| Format | Description |
|--------|-------------|
| `text` | Plain text recognition result |
| `json` | Structured JSON with bounding boxes, confidence |
| `hocr` | HTML-based OCR format |
| `tsv` | Tab-separated values with character-level detail |
