# Architecture

## Overview

The OCR system is organized as a layered Rust library with a CLI frontend. The pipeline flows:

```
Image Input → Preprocessing → Layout Analysis → Recognition → Post-processing → Output
```

## Layers

### API Layer (`src/api/`)
High-level entry point. `Ocr` struct provides the public API with async methods for text recognition and layout analysis. `TextProcessor` handles filtering and text cleaning.

### Core Layer (`src/core/`)
Contains the central `OcrEngine` and all shared data structures:
- `OcrConfig` — Full configuration (recognition, image processing, layout, language, performance)
- `OcrEngine` — State machine managing initialization, image processing, recognition, and layout
- `TextResult` / `RecognitionResult` — Output data with confidence scores, bounding boxes
- `Output` formatters — Plain text, JSON, hOCR, TSV
- `Geometry` types — Points, rectangles, polygons for layout

### Image Pipeline (`src/image/`)
Modular preprocessing:
- `Processor` — Binarization, noise reduction, deskew, enhancement
- `Pipeline` — Configurable processing chain
- `Thresholder` — Otsu, Sauvola, adaptive methods
- `Quality` — Image quality assessment
- `Enhancement` — Contrast adjustment, sharpening

### Language Support (`src/lang/`)
- CJK segmentation and detection
- N-gram based language identification
- Unicode character set handling
- Dictionary-based correction

### Layout Analysis (`src/layout/`)
Text segmentation and structure detection:
- Union-Find Connected Component Labeling for character isolation
- Column and line detection
- Text ordering and classification
- Feature extraction for text lines

### Recognition (`src/recognition/`)
Multiple engine implementations sharing a common trait:
- `PatternModel` — Bitmap template matching (baseline)
- `LSTM` / `LSTM_Model` — Sequence-based recognition
- `CNN_Model` / `ViT_Model` — Vision transformer
- `TransformerModel` — Full transformer
- `HybridModel` — Ensembled engines
- `EndToEndModel` — Integrated detection + recognition
- CTC decoder for sequence decoding

### Training (`src/training/`)
Model training infrastructure:
- Data loading and augmentation
- Loss functions, metrics
- Optimizers (SGD, Adam)
- Checkpoint management

### CLI (`src/cli/`)
Clap-based argument parsing with commands: extract, batch, layout, list-languages, check, info, validate.

## Data Flow

```
1. User invokes CLI → cli/mod.rs parses args
2. main.rs builds OcrConfig, creates Ocr
3. Ocr initializes the OcrEngine
4. Image loaded from disk → image pipeline processes it
5. Layout analyzer segments image → text regions → lines → characters
6. Recognition engine matches characters → builds TextResult
7. Formatter converts to requested output format
8. Result written to stdout or file
```
