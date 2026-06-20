# TODO

## Phase 0 — Baseline (DONE)

Establish a working, tested pipeline before adding new capabilities.

- [x] Fix compilation errors (removed conflicting `domain/` module, fixed CLI helpers, layout, pdf)
- [x] `cargo test` passes with 289+ tests
- [x] End-to-end pipeline: image → preprocess → layout → recognize → output
- [x] Pattern-matching recognition engine with 37 glyph templates
- [x] Layout analysis: Union-Find CCL, column/line detection, reading order
- [x] Image preprocessing: deskew, binarization (Otsu, Sauvola), noise reduction, contrast enhancement
- [x] Output formats: text, JSON, hOCR, TSV, ALTO XML, box files
- [x] Dictionary post-correction for 7 languages (en, fr, de, es, it, pt, ru)
- [x] CLI: extract, batch, layout, list-languages, check, info, validate
- [x] Web API server (feature `web-api`)
- [x] PDF input support (feature `pdf`)

## Phase 1 — Synthetic Training Infrastructure

**Goal:** Generate unlimited labeled training data from fonts + distortions.

- [ ] Create `src/synthetic/` module for text-line image generation
  - [ ] Render text with system fonts (monospace, serif, sans-serif, 10+ fonts)
  - [ ] Configurable image dimensions, colors, padding
  - [ ] Batch generation API
- [ ] Implement distortion pipeline
  - [ ] Rotation ±5°
  - [ ] Gaussian blur, salt & pepper noise
  - [ ] Random contrast/brightness adjustment
  - [ ] Perspective shear
- [ ] Build benchmark harness
  - [ ] Generate 10k synthetic test samples
  - [ ] Run pattern matcher, report CER/WER baseline
  - [ ] Save results to `benchmarks/` for regression tracking
- [ ] Train pattern-matching templates from synthetic data
  - [ ] Extract normalized character images from rendered lines
  - [ ] Build per-font templates, average them
  - [ ] Evaluate accuracy improvement over hand-coded 5×7 bitmaps

## Phase 2 — Robust Text Detection

**Goal:** Replace pure CCL with learned detection for complex layouts and scene text.

- [ ] Implement lightweight detection CNN in pure Rust (ndarray)
  - [ ] 3-layer U-Net style architecture for text region heatmap
  - [ ] Sigmoid output for text probability per pixel
  - [ ] Post-process: threshold → contour → bounding boxes
- [ ] Integrate detection into pipeline
  - [ ] `TextDetector` trait with `CclDetector` and `CnnDetector` implementations
  - [ ] Configurable via `OcrConfig`
  - [ ] Fallback to CCL when CNN weights not loaded
- [ ] Evaluate on synthetic multi-column documents
  - [ ] Recall ≥ 85%, Precision ≥ 90%
  - [ ] Handle rotated text (0°, 90°, 180°, 270°)

## Phase 3 — CRNN Recognition

**Goal:** Replace pattern matching with a trainable sequence model.

- [ ] Implement CNN feature extractor (ndarray)
  - [ ] 5-layer VGG-style conv stack
  - [ ] MaxPooling, ReLU activations
  - [ ] Output: 512-channel feature sequence
- [ ] Implement BiLSTM sequence model (ndarray)
  - [ ] 2-layer bidirectional LSTM (256 hidden units)
  - [ ] Forward/backward pass with gate mechanics
- [ ] Implement CTC decoder
  - [ ] Greedy decoding (collapse repeats, remove blanks)
  - [ ] Beam search decoding (optional, width 5–10)
- [ ] Implement CTC loss function
  - [ ] Forward-backward algorithm for gradient computation
- [ ] Wire training pipeline
  - [ ] `CrnnTrainer` with Adam optimizer
  - [ ] Batch training on synthetic data
  - [ ] Checkpoint saving/loading
- [ ] Acceptance criteria
  - [ ] CER < 5% on clean synthetic test
  - [ ] CER < 15% on distorted synthetic test
  - [ ] Inference < 100ms/line on single CPU core
  - [ ] Model size < 5MB

## Phase 4 — Multi-Language Scale

**Goal:** Support 30+ languages with automatic script detection.

- [ ] Expand synthetic data to CJK, Arabic, Cyrillic scripts
  - [ ] CJK: system font rendering, vertical text layout
  - [ ] Arabic: RTL handling, diacritic support
  - [ ] Cyrillic: extended character sets
- [ ] Unicode script detection
  - [ ] Script classifier based on unicode blocks
  - [ ] Route to language-specific CRNN model
- [ ] Expand dictionaries
  - [ ] Add 20+ language dictionaries
  - [ ] On-demand loading to keep memory low
- [ ] Evaluate per-language accuracy on synthetic benchmarks

## Phase 5 — Advanced Layout & Structure

**Goal:** Understand document structure, not just flat text.

- [ ] Table detection
  - [ ] Grid line detection (Hough transform for horizontal/vertical lines)
  - [ ] Cell boundary reconstruction
  - [ ] Row/column span inference
- [ ] Form field extraction
  - [ ] Key-value pair detection
  - [ ] Checkbox/radio button recognition
- [ ] Document structure classification
  - [ ] Heading, paragraph, list, figure, caption classification
  - [ ] Hierarchical JSON/markdown output
- [ ] Searchable PDF generation
  - [ ] Overlay invisible text layer on original image coordinates
  - [ ] Proper font encoding for Unicode

## Backlog / Ideas

- ONNX runtime integration for importing PaddleOCR/EasyOCR pre-trained models
- GPU acceleration for CNN inference (CUDA via `cudarc`, OpenCL via `ocl`)
- SIMD optimization for convolutions (AVX2, NEON)
- Quantization (INT8) for edge deployment
- Font attribute detection (bold, italic, monospace) from stroke analysis
- Handwriting recognition (separate model, likely transformer-based)
