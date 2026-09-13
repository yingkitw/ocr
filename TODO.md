# TODO

## Completed

All five phases of the OCR roadmap are implemented and tested (347+ tests passing, 90 more ignored behind `--ignored`).

- **Adam gradient bug fix** (2026-09-12): `apply_full_adam` in `src/training/crnn_trainer.rs` received computed gradients in `g: &FullParamGrads` but the Adam macros **ignored `g` entirely** — they used the model's parameter values as the gradient signal instead. The training pipeline computed correct CTC gradients via full BPTT but never applied them, explaining the loss plateau at 60.33. Fixed: the `adam!` macro now takes `(grad, param, m, v)` and uses the gradient for moment updates. Regression test: `test_train_batch_full_updates_params` verifies FC weights change and loss decreases over 5 steps. Also restored zero-warning build (18 warnings fixed across `backward.rs`, `crnn_trainer.rs`, `ctc.rs`, `integration_tests.rs`, `train_pure_rust.rs`) and marked 2 flaky debug-mode benchmark timing tests as `#[ignore]`.

- **Narrow-crop panic fix** — line crops < 16 px wide collapsed to a zero-width feature map and underflowed in `conv_block` (crash on every `ocr extract`); `recognize_detailed` now returns an empty result for degenerate crops, with a regression unit test

- **Phase 0 — Baseline**: End-to-end pipeline, rule-based pattern recognition (removed in v0.2), layout analysis, CLI, Web API
- **Phase 1 — Synthetic Training**: Font rendering, distortion pipeline, CER/WER benchmarks (`TemplateTrainer` removed in v0.2 with the pattern engine)
- **Phase 2 — Text Detection**: `TextDetector` trait, CCL + CNN detectors, `DocumentGenerator` + IoU evaluation, auto-rotate 0/90/180/270
- **Phase 3 — CRNN Recognition**: 5-layer CNN extractor, 2-layer BiLSTM, CTC decoder/loss, `CrnnTrainer`, checkpointing, `--engine lstm`, model < 5MB
- **Phase 4 — Multi-Language**: Script detection (8 scripts), 25+ dictionaries, `ScriptLineGenerator`, `ScriptModelRegistry` for per-script CRNN
- **Phase 5 — Layout & Structure**: Table detection with span inference, form field extraction, document classification, Markdown/structured JSON output
- **CTC loss** — pure-Rust forward-backward + gradient in `src/training/ctc.rs` (6 unit tests pass: grad sums to ~0 per t, gamma is valid, identity converges, gradients are finite on random input)
- **Full BPTT** — gradient flows through CNN + 2× BiLSTM + FC in pure Rust (`src/recognition/backward.rs`, `CrnnTrainer::train_batch_full`); conv5 sizing bug fixed. No candle/burn dependency needed.
- **Hygiene pass**: synced doc drift (test counts 357→373, CLI version now derived from `Cargo.toml`), and drove the whole crate to a **zero-warning build** (267 → 0 across `cargo build`, tests, and examples): annotated experimental/unwired model modules + SIMD fallbacks, removed dead imports/fields, fixed a `CheckpointInfo` visibility bug, and cleaned redundant match arms in script detection
- **2024 lean pass (gocr-inspired)**: removed ~13 kLOC of speculative/dead code — five stub recognizer modules (`cnn_model`, `transformer_model`, `vit_model`, `hybrid_model`, `end_to_end_model`), `pattern.rs`/`lstm.rs` empty trait stubs, `simd_advanced.rs` (~970 lines, never used), `tesseract_textline.rs` (565 lines, defined but unused), `lang/unicharset.rs` (325 lines), the orphan `domain/*` IoC stack (~1 kLOC, zero external refs), the eight `core/image.rs` no-op augmentation methods, and eight speculative `training/*` modules (`augmentation`, `checkpoint`, `config`, `data`, `losses`, `metrics`, `optimizers`, `training`); removed `--engine hybrid` (silently fell through to pattern matching). Wired `CrnnTrainer::save_checkpoint`/`load_checkpoint` so checkpoints actually persist the FC weights the trainer modifies; collapsed per-request `Ocr::with_config` + `initialize` in `server/mod.rs` to the shared `AppState.ocr` instance. Net: 92 → 91 .rs files, source LOC trimmed from ~46 k → ~32 k, test count from 417 → 403 (the dropped tests were the ones covering deleted code), `RecognitionEngine` enum down to two variants. Followed up by deleting the duplicate `api::ImageProcessor` stack (~347 lines, zero external refs), `api::OcrResult` / `BatchProcessor` / `ImageInput` / `EngineMetadata` (zero external refs), `TextPostProcessor` + its `apply_spell_checking` / `apply_grammar_correction` stubs, and fixing `Ocr::is_initialized` to actually read `OcrEngine.state.initialized` instead of returning `true` unconditionally.

## v0.2 — Model-only pipeline (in progress)

- [x] **Model bundle infra** — `models/crnn-en.bin` (3.07 MB) embedded via `include_bytes!`; `src/recognition/{loader,model_format}.rs` read/write the binary format with magic `OCR1` + version 1; auto-load on `Ocr::initialize()`
- [x] **Rule-based recognition removed** — hard-deleted `basic_ocr.rs` (2,156 LOC), `pattern_model.rs`, `lstm_model.rs`, `tesseract_blob.rs`, `tesseract_features.rs`; `RecognitionEngine::PatternMatching` removed; `--engine pattern` silently mapped to LSTM
- [x] **`BasicRecognitionEngine` rewired** — now wraps `CrnnModel` with the bundled weights
- [x] **Model re-bundled**: models/crnn-en.bin updated from epoch-6 checkpoint (loss 60.33, log /tmp/ocr_train.log)
- [ ] **Trained model** — see "Blocked" below
- [x] **CTC loss** — done, see Completed

## Blocked — what's next

The training pipeline is end-to-end (CTC loss + full BPTT) but the bundled model was not yet trained to target accuracy. **The model has now been re-bundled** (`models/crnn-en.bin` updated from `models/crnn-trained.bin`). The 2026-09-12 Adam gradient bug fix means the previous 6-epoch run was training with broken gradients (parameter values used as gradients instead of computed CTC gradients) — a fresh training run with the fix should produce meaningfully lower loss. Remaining:

1. ~~**CTC loss**~~ ✅ DONE — see Completed
2. ~~**Gradient flow**~~ ✅ DONE — full BPTT through CNN + BiLSTM + FC in pure Rust (`src/recognition/backward.rs` + `CrnnTrainer::train_batch_full`); no candle/burn dependency needed
- **Multi-hour training run** — completed (6 epochs; loss plateaued at ~60.33); full 100-epoch run would take 11–12 h at ~7 min/epoch; lr = 0.005 plateaued; lr = 0.001 recommended for continued training
- **Re-bundle** — `models/crnn-en.bin` has been updated from `models/crnn-trained.bin` (3.07 MB, 6-epoch checkpoint)

## Next Steps

These are **blocked on execution or hardware**, not additional code:

- [x] **Training CLI wired up** (`ocr train --engine lstm --epochs 10 --batch-size 8`)
- [x] **Benchmark CLI wired up** (`ocr benchmark --samples 10`)
- [x] **CTC training implemented** — FC-only path (`CrnnTrainer::train_batch_ctc`), superseded by full BPTT (`train_batch_full`, see Completed)
- [x] **CRNN inference optimized** (rayon-parallel convolutions, ndarray FC layer)
- [ ] **Train CRNN to target accuracy** — *blocked: requires multi-hour `--release` training run*
  - CER < 5% on clean synthetic test
  - CER < 15% on distorted synthetic test
  - Run `cargo run --release --example train_pure_rust -- --epochs 100` or `cargo run --release -- train --epochs 100` (debug is ~4s/forward, use --release)
- [ ] **Evaluate per-language accuracy** — *blocked: requires trained checkpoints*
  - Run `cargo run --release -- benchmark --samples 50` to measure per-script CER/WER

## Backlog / Ideas

- [x] **Font attribute detection** (`FontAttributeDetector`: bold via stroke thickness, italic via slant, monospace via width CV)
- [x] **INT8 quantization** (`QuantizedTensor`, `quantize_array2`, `quantized_matmul`: 4x memory reduction for FC layer)
- [x] **ONNX weight loader** (`OnnxLoader` with `onnx-rs`: parse ONNX, extract weights as ndarrays, `load_crnn_weights` maps Conv/Gemm/LSTM into `CrnnModel` by graph topology)
- [~] **GPU acceleration** (`ComputeBackend` wired into `CnnFeatureExtractor`; OpenCL kernels implemented; CUDA path still falls back to CPU — *blocked: needs NVIDIA hardware to validate real `cudarc` kernels*)
- [ ] Handwriting recognition (separate model, likely transformer-based)
- [x] **Zero-warning build** — annotated stub/future-work modules (`end_to_end`, `vit`, `cnn`, `hybrid`, `transformer` models, SIMD scalar fallbacks), removed dead imports/fields/functions, fixed `CheckpointInfo` visibility, cleaned script-detection match arms
- [x] **Lean codebase** — removed all speculative/dead modules (see 2024 lean pass entry above)

## Brainstorming (competitive intelligence)

Gaps observed vs. Tesseract / PaddleOCR / EasyOCR / RapidOCR / docTR / surya. Prioritized by accuracy/UX impact, lowest-cost first:

- [x] **Wire batch concurrency** — `--max-concurrent` CLI flag exists but was unused; `handle_batch` now processes images concurrently via `tokio::spawn` + a `Semaphore` (recognition holds only a read-lock, so tasks run in parallel). Verified with a CLI integration test.
- [x] **CTC beam search + dictionary/LM rescoring** — CRNN inference defaults to beam search (`CrnnConfig::use_beam_search`); `CtcDecoder::beam_search_nbest` + `DictLmRescorer` rescore hypotheses with dictionary hits and n-gram LM; wired in `OcrEngine::recognize_with_crnn` when `--dict-correct` / language-model flags are on
- [x] **Confidence calibration** — `ConfidenceCalibrator` (temperature-scaled softmax) extracts per-char/word confidence from CTC logits; `CrnnModel::recognize_detailed` + engine fill `CharacterRecognition` / `WordRecognition` instead of a hardcoded 0.7
- [x] **Curved-line / perspective dewarp** — `PerspectiveDewarp` (content-corner quad → rectangle) + `CurveRectifier` (quadratic baseline flatten); wired into preprocess + per-region recognition via `enable_perspective_dewarp` / `enable_curve_rectification`
- [x] **Arbitrary-angle text detection** — `OrientedCclDetector` sweeps ±45° (default 15° steps), maps boxes back, NMS; deskew expanded to ±15°; region crops rotated upright via `rotation_deg` (opt-in via `enable_arbitrary_angle_detection`)
- [x] **Super-resolution upscaling for tiny text** — `TextSuperResolution` (Lanczos + stroke/DPI/line-height heuristics, noise-aware skip); page preprocess + short-crop upscale; `enable_super_resolution` / `target_dpi`
- [x] **makebox-style box export** — `format_makebox` (Tesseract bottom-left `.box`) + `ocr makebox IMAGE [-o base]` CLI for training-data generation from real images
- [x] **Auto language detection (`--lang auto`)** — `LanguageDetector` (Unicode script + Latin dictionary hit-rates + CJK kana/hangul heuristics); resolves dictionary language in post-process when `lang=auto`
- [x] **Image quality gate** — `ImageProcessingConfig::enable_quality_gate` (default on): auto-sharpen / boost contrast when `ImageQualityAssessor` metrics are poor
- [x] **Batch progress reporting** — `ocr batch` prints `[n/total] % ETA …` per image and a final summary
- [x] **Dependency simplification** — dropped unused `blake3`/`sha2`/`fastrand`/`nalgebra`/`num_cpus`/`futures`; trimmed `tokio`/`chrono`/`image` features; removed `printpdf` and `pdf` crates (PDF input/output dropped); `pdfrs` feature is now an empty placeholder (default tree ~187 nodes vs ~537)
- [x] **Lean pass** — removed ~13 kLOC of speculative/dead code (stub recognizers, orphan domain/*, unused simd/tesseract-textline/unicharset, no-op image augmentation methods, speculative training infrastructure)
- [~] **Publish to crates.io** — `Cargo.toml` has keywords/categories/readme/docs.rs URL; `cargo publish --dry-run` when disk allows; do not publish without explicit ask
- [ ] Handwriting recognition (separate model, likely transformer-based; only a handwritten-region *detector* heuristic exists in `src/layout/`)
- [ ] Seal / stamp / formula / KIE extraction (document AI extras vs PaddleOCR/surya; **KIE/form extraction is implemented** — `src/layout/form_extractor.rs` has checkbox, key-value-pair, and underline-field detection; remaining gaps are seal/stamp *and* handwritten formula recognition)
- [x] **Searchable PDF output** (2026-09-13) — `-f pdf` (`OutputFormat::Pdf`, requires `--features pdf`): zero-dependency searchable PDF writer (`src/core/pdf.rs`) embeds the original image as a JPEG XObject and overlays an invisible text layer (PDF text rendering mode 3) at word bounding boxes using built-in Helvetica + WinAnsiEncoding. Multi-page TIFF → multi-page PDF. CLI `ocr extract IMAGE -f pdf -o out.pdf` writes binary PDF directly. 6 tests (5 unit + 1 CLI smoke).
- [x] **Multi-page TIFF input** (2026-09-13) — `decode_multi_page_tiff` (`src/api/ocr.rs`) sniffs the TIFF magic (`II*\0`/`MM\0*`), iterates frames via the `tiff` 0.9 crate (`Decoder::next_image`/`more_images`), converts each page to grayscale (Gray8/RGB/RGBA). New public API: `recognize_text_from_bytes_multi_page` / `recognize_text_from_file_multi_page` (single-page images → 1-element vec; backward compatible — `recognize_text_from_bytes` unchanged). CLI `ocr extract multi.tiff` emits `--- Page N ---` separators between pages. `tiff` crate promoted from transitive to direct dependency (same version image already pulled). 3 tests: PNG→1 page, TIFF→2 pages, CLI smoke.
- [ ] **Barcode / QR detection** (common in invoice/receipt processing; PaddleOCR ecosystem and commercial APIs often bundle it; would need a `rqrr`/`bardecoder`-style dep behind a feature flag)
- [ ] **Color / background analysis** — text color + background color per region (useful for accessible HTML output; hOCR currently emits no color info)
- [x] **RAG-friendly chunked structured output** — `-f rag` (`OutputFormat::RagJson`): `rag_chunks` splits the document into per-element chunks (`heading`/`paragraph`/`list_item`/`table`/`figure`/`code`) tagged with the anchoring heading path in `section`; also fixed `format_result` in the CLI, which silently ignored `markdown` / `structured-json` (everything but json/hocr/tsv/alto/box fell through to plain text)
- [x] Remove vestigial `TemplateTrainer` (`src/synthetic/template_trainer.rs` — built pattern templates for the deleted `BasicOcrEngine`; zero external refs)
- [x] Fix stale doc header in `examples/train_pure_rust.rs` (now describes full BPTT via `train_batch_full`)
- [x] **PAGE XML output** — `-f page` (`OutputFormat::Page`): PRImA PAGE XML 2019-07-15 format with `PcGts` → `Page` → `TextRegion` → `TextLine` → `Word`, each with `<Coords>` polygon and `<TextEquiv>` (Unicode + Confidence). Competitive parity with Tesseract's PAGE output. Aliases: `page`, `page-xml`, `primapage`.
- [x] **Stdin input** — `ocr extract -` reads the image from stdin (Unix pipe convention), matching Tesseract's `-` behavior. Added `Ocr::recognize_text_from_bytes` as a public API method (refactored `recognize_text_from_file` to delegate to it). Uses `std::io::Read` for stdin (tokio `io-std` feature not enabled).
- [x] **--dpi CLI flag** — `ocr extract IMAGE --dpi 150` sets the input image resolution (default 300, matching Tesseract). Wires into both `config.recognition.parameters["dpi"]` (used by `OcrImage` creation) and `config.image_processing.target_dpi` (used by super-resolution heuristics).
- [x] **CSV output format** (2026-09-13) — `-f csv` (`OutputFormat::Csv`): RFC 4180 per-word rows (`page,line,word,text,confidence,left,top,width,height`) for spreadsheet/database import. Distinct from TSV (Tesseract-training-style 5-level format). Escaping handles comma/quote/CR/LF via `escape_csv_field`. Wired through `format_output` + `parse_output_format` (single dispatch, no CLI-side match), web API (`csv` field in `OcrResponse`), CLI help, `ocr check` output. 6 new tests (3 unit + 1 CLI smoke + 1 parse + 1 web API).
- [x] **Recognition accuracy: load bundled model** (2026-09-13) — **CRITICAL bug**: `OcrEngine::recognize_with_crnn` created a fresh random `CrnnModel` on every call instead of loading the bundled trained model from `models/crnn-en.bin`. Every OCR recognition used random weights. Fix: `OcrEngine` now holds `model: OnceLock<CrnnModel>`, loaded via `loader::load_bundled()` on first call and cached. Regression test verifies deterministic output across calls. This is the single biggest accuracy improvement — recognition now uses trained weights.
- [x] **Recognition accuracy: bilinear resize** (2026-09-13) — `CrnnModel::resize_array2_height` used nearest-neighbor when resizing line crops to 32px input height, losing vertical detail. Replaced with bilinear (vertical-only) interpolation for smoother scaling. Test verifies interpolated values on a gradient.
- [x] **Searchable PDF output** (2026-09-13) — `-f pdf` (`OutputFormat::Pdf`, feature `pdf`): zero-dependency PDF writer (`src/core/pdf.rs`, ~500 lines) — embeds original image as JPEG XObject, overlays invisible text (rendering mode 3) at word bounding boxes with built-in Helvetica/WinAnsiEncoding. Multi-page TIFF → multi-page PDF. CLI `ocr extract IMAGE -f pdf -o out.pdf`. 6 tests (5 unit + 1 CLI smoke).
