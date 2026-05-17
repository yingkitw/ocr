# TODO

## Near-term

- [x] Add CUDA/OpenCL backend support for neural network models

### Tesseract-like Capabilities

Implement features comparable to Tesseract OCR:

- [ ] **ALTO XML output format** — Standard XML for digital library interchange (Library of Congress standard)
- [ ] **OSD (Orientation & Script Detection)** — Auto-detect text rotation angle (0/90/180/270°) and dominant script before recognition
- [ ] **Font attribute detection** — Detect bold, italic, monospace from character stroke width/slant analysis; populate WordFlag::Bold/Italic
- [ ] **Searchable PDF output** — Generate PDF with invisible text layer overlaid on original image (`ocr extract input.png --format pdf`)
- [ ] **Vertical text support** — CJK vertical text layout detection and proper top-to-bottom reading order
- [ ] **Fix hOCR output** — Remove duplicate title attributes; ensure valid hOCR 4.1 spec compliance
- [ ] **Box file generation** — Generate `.box` files for model training from ground-truth + image pairs
- [ ] **Word-level and char-level confidence in TSV** — Match Tesseract's `level`, `page_num`, `block_num`, `par_num`, `line_num`, `word_num`, `left`, `top`, `width`, `height`, `conf`, `text` columns

## Completed

- [x] Add A-Z, 0-9, and common symbols to test font renderer (`core/engine.rs` glyph_rows) — 37 glyphs in 5x7 bitmap
- [x] Expand supported languages from 10 to 30 codes (added nl, pl, sv, da, fi, no, tr, el, hi, th, vi, ar, he, id, ms, uk, cs, hu, ro, bg)
- [x] Update CLI `--lang` help text in `cli/mod.rs` (Extract + Batch)
- [x] Add per-language dictionaries: French, Spanish, German, Italian, Portuguese, Russian (`lang/dictionary.rs`)
- [x] Wire language-aware dictionary selection into engine post-processing (`DictionaryHandler::new_for_language`)

- [x] Improve layout analysis for complex multi-column documents — recursive XY-cut, region classifier (heading/body/caption/footer/page-number)
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
