# TODO

## Completed

All five phases of the OCR roadmap are implemented and tested (357+ tests passing).

- **Phase 0 — Baseline**: End-to-end pipeline, pattern matching, layout analysis, CLI, Web API, PDF input
- **Phase 1 — Synthetic Training**: Font rendering, distortion pipeline, CER/WER benchmarks, `TemplateTrainer` for pattern-matching templates
- **Phase 2 — Text Detection**: `TextDetector` trait, CCL + CNN detectors, `DocumentGenerator` + IoU evaluation, auto-rotate 0/90/180/270
- **Phase 3 — CRNN Recognition**: 5-layer CNN extractor, 2-layer BiLSTM, CTC decoder/loss, `CrnnTrainer`, checkpointing, `--engine lstm`, model < 5MB
- **Phase 4 — Multi-Language**: Script detection (8 scripts), 25+ dictionaries, `ScriptLineGenerator`, `ScriptModelRegistry` for per-script CRNN
- **Phase 5 — Layout & Structure**: Table detection with span inference, form field extraction, document classification, Markdown/structured JSON output, searchable PDF

## Next Steps

These require **training execution**, not additional code:

- [ ] **Train CRNN to target accuracy**
  - CER < 5% on clean synthetic test
  - CER < 15% on distorted synthetic test
  - Use `cargo run --release -- train --engine crnn --epochs 100` (or equivalent CLI)
- [ ] **Evaluate per-language accuracy**
  - Generate synthetic test sets per script via `ScriptLineGenerator`
  - Run `CrnnModel` with `ScriptModelRegistry` and measure CER/WER

## Backlog / Ideas

- ONNX runtime integration for importing PaddleOCR/EasyOCR pre-trained models
- GPU acceleration for CNN inference (CUDA via `cudarc`, OpenCL via `ocl`)
- SIMD optimization for convolutions (AVX2, NEON)
- Quantization (INT8) for edge deployment
- Font attribute detection (bold, italic, monospace) from stroke analysis
- Handwriting recognition (separate model, likely transformer-based)
