# TODO

## Completed

All five phases of the OCR roadmap are implemented and tested (408+ tests passing).

- **Phase 0 — Baseline**: End-to-end pipeline, pattern matching, layout analysis, CLI, Web API, PDF input
- **Phase 1 — Synthetic Training**: Font rendering, distortion pipeline, CER/WER benchmarks, `TemplateTrainer` for pattern-matching templates
- **Phase 2 — Text Detection**: `TextDetector` trait, CCL + CNN detectors, `DocumentGenerator` + IoU evaluation, auto-rotate 0/90/180/270
- **Phase 3 — CRNN Recognition**: 5-layer CNN extractor, 2-layer BiLSTM, CTC decoder/loss, `CrnnTrainer`, checkpointing, `--engine lstm`, model < 5MB
- **Phase 4 — Multi-Language**: Script detection (8 scripts), 25+ dictionaries, `ScriptLineGenerator`, `ScriptModelRegistry` for per-script CRNN
- **Phase 5 — Layout & Structure**: Table detection with span inference, form field extraction, document classification, Markdown/structured JSON output, searchable PDF
- **Hygiene pass**: fixed PDF command wiring redundancy (`handle_pdf_extraction` dead code), synced doc drift (test counts 357→373, CLI version now derived from `Cargo.toml`), and drove the whole crate to a **zero-warning build** (267 → 0 across `cargo build`, tests, and examples): annotated experimental/unwired model modules + SIMD fallbacks, removed dead imports/fields, fixed a `CheckpointInfo` visibility bug, and cleaned redundant match arms in script detection

## Next Steps

These are **blocked on execution or hardware**, not additional code:

- [x] **Training CLI wired up** (`ocr train --engine lstm --epochs 10 --batch-size 8`)
- [x] **Benchmark CLI wired up** (`ocr benchmark --samples 10`)
- [x] **FC-layer backprop implemented** in `CrnnTrainer::train_batch` (CNN/BiLSTM frozen)
- [x] **CRNN inference optimized** (rayon-parallel convolutions, ndarray FC layer)
- [ ] **Train CRNN to target accuracy** — *blocked: requires multi-hour `--release` training run*
  - CER < 5% on clean synthetic test
  - CER < 15% on distorted synthetic test
  - Run `cargo run --release -- train --epochs 100` (debug is ~4s/forward, use --release)
- [ ] **Evaluate per-language accuracy** — *blocked: requires trained checkpoints*
  - Run `cargo run --release -- benchmark --samples 50` to measure per-script CER/WER

## Backlog / Ideas

- [x] **Font attribute detection** (`FontAttributeDetector`: bold via stroke thickness, italic via slant, monospace via width CV)
- [x] **INT8 quantization** (`QuantizedTensor`, `quantize_array2`, `quantized_matmul`: 4x memory reduction for FC layer)
- [x] **ONNX weight loader** (`OnnxLoader` with `onnx-rs`: parse ONNX, extract weights as ndarrays, `load_crnn_weights` maps Conv/Gemm/LSTM into `CrnnModel` by graph topology)
- [~] **GPU acceleration** (`ComputeBackend` wired into `CnnFeatureExtractor`; OpenCL kernels implemented; CUDA path still falls back to CPU — *blocked: needs NVIDIA hardware to validate real `cudarc` kernels*)
- [ ] Handwriting recognition (separate model, likely transformer-based)
- [x] **Zero-warning build** — annotated stub/future-work modules (`end_to_end`, `vit`, `cnn`, `hybrid`, `transformer` models, SIMD scalar fallbacks), removed dead imports/fields/functions, fixed `CheckpointInfo` visibility, cleaned script-detection match arms

## Brainstorming (competitive intelligence)

Gaps observed vs. Tesseract / PaddleOCR / EasyOCR / RapidOCR / docTR / surya. Prioritized by accuracy/UX impact, lowest-cost first:

- [x] **Wire batch concurrency** — `--max-concurrent` CLI flag exists but was unused; `handle_batch` now processes images concurrently via `tokio::spawn` + a `Semaphore` (recognition holds only a read-lock, so tasks run in parallel). Verified with a CLI integration test.
- [x] **CTC beam search + dictionary/LM rescoring** — CRNN inference defaults to beam search (`CrnnConfig::use_beam_search`); `CtcDecoder::beam_search_nbest` + `DictLmRescorer` rescore hypotheses with dictionary hits and n-gram LM; wired in `OcrEngine::recognize_with_crnn` when `--dict-correct` / language-model flags are on
- [x] **Confidence calibration** — `ConfidenceCalibrator` (temperature-scaled softmax) extracts per-char/word confidence from CTC logits; `CrnnModel::recognize_detailed` + engine fill `CharacterRecognition` / `WordRecognition` instead of a hardcoded 0.7
- [x] **Curved-line / perspective dewarp** — `PerspectiveDewarp` (content-corner quad → rectangle) + `CurveRectifier` (quadratic baseline flatten); wired into preprocess + per-region recognition via `enable_perspective_dewarp` / `enable_curve_rectification`
- [x] **Arbitrary-angle text detection** — `OrientedCclDetector` sweeps ±45° (default 15° steps), maps boxes back, NMS; deskew expanded to ±15°; region crops rotated upright via `rotation_deg` (opt-in via `enable_arbitrary_angle_detection`)
- [x] **Super-resolution upscaling for tiny text** — `TextSuperResolution` (Lanczos + stroke/DPI/line-height heuristics, noise-aware skip); page preprocess + short-crop upscale; `enable_super_resolution` / `target_dpi`
- [x] **makebox-style box export** — `format_makebox` (Tesseract bottom-left `.box`) + `ocr makebox IMAGE [-o base]` CLI for training-data generation from real images
- [ ] **Publish to crates.io** — package is docs.rs-ready (`documentation = "https://docs.rs/ocr"`) but unpublished; a published crate is table-stakes for adoption vs. Tesseract bindings.
