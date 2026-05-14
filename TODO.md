# TODO

## Near-term

- [ ] Improve layout analysis for complex multi-column documents
- [ ] Add CUDA/OpenCL backend support for neural network models

## Completed

- [x] Add PDF input support — `ocr extract file.pdf` extracts embedded images via `pdf` crate
- [x] Add web API mode for HTTP-based OCR — `ocr serve` with axum, multipart upload, /health, /languages, /recognize
- [x] Enable and test SIMD acceleration (`simd` feature flag) — NEON on aarch64, SSE4.1/AVX2 on x86_64
- [x] Benchmark and profile recognition performance — Profiler wired into engine stages, synthetic benchmarks
- [x] Wire up LSTM/CNN/transformer recognition engines as alternatives via `--engine` flag
- [x] Add CJK language codes (zh, ja, ko) to CLI `--lang` and `list-languages`
- [x] Complete image preprocessing pipeline (deskew, binarization, contrast, noise reduction)
- [x] Add dictionary-based post-correction via `--dict-correct` flag
- [x] Comprehensive CLI test suite (14 tests covering engine, lang, dict, format options)
- [x] Fix clippy ambiguous glob re-export warnings
- [x] Add `ocr_demo` example with full pipeline demo
- [x] Merge miniocr library into main crate
- [x] CLI with extract, batch, layout, list-languages, check, info, validate commands
- [x] Pattern matching recognition engine (default)
- [x] Image preprocessing pipeline
- [x] Multiple output formats (text, json, hocr, tsv)
- [x] Layout analysis (column/line detection, text ordering)
- [x] Language detection (N-gram based)
- [x] CJK character segmentation
- [x] Round-trip tests and snapshot tests
- [x] 235+ tests across all modules
