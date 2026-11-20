# Test Suite for OCR

This directory contains comprehensive tests for the OCR library.

## Test Files

### `basic_tests.rs`
Basic functionality tests for OCR initialization, configuration, and metadata.

### `integration_tests.rs`
Integration tests for OCR pipeline components including CJK language detection and model creation.

### `snapshot_tests.rs`
Snapshot tests using `insta` to ensure consistent output over time. Snapshots live in `snapshots/`; review changes with `cargo insta review` (or inspect the `.snap` diffs).

### `cli_tests.rs`
CLI smoke tests via `assert_cmd` + `predicates`: subcommand dispatch, flags, output formats, batch progress.

### `round_trip_test.rs`
Comprehensive round-trip tests: text → synthetic image → OCR → text (per-character, words, multi-line, scale/spacing variations, inverted images, result structure, reproducibility). Images are generated in-process — no fixtures required.

### `ocr_test.rs`
Simple OCR test with a sample image.

### `test_images_test.rs`
**Comprehensive tests against all generated test images** with varying complexity:
- Simple text recognition
- Multi-line text
- Mixed case text
- Numbers and special characters
- Size variations (small/large text)
- Background variations (dark, noisy, low contrast)
- Layout complexity (columns, tables, dense text)
- Special cases (rotated text, mixed languages)
- Batch processing
- Confidence scores
- Word and character level results
- Bounding boxes

> **Note:** `test_simple_text` currently fails — the bundled CRNN model is not
> trained to target accuracy, so its output does not match the fixture's
> expected text. See `TODO.md` → "Known test failures".

### `docs_sync_test.rs`
Guards against documentation drift — asserts the root `*.md` files describe the code as it is:

- every `src/` submodule is listed in the README / ARCHITECTURE module documentation
- every `src/webocr/*.rs` file appears in the ARCHITECTURE module table
- no doc invokes a CLI subcommand that does not exist (catches phantom commands)
- the README's `/recognize` endpoint matches the routes actually mounted in `src/server/mod.rs`
- the test count is identical in README (badge + roadmap), SPEC, TODO, MEMORY and AGENTS

These fail the build when a feature lands without updating the docs. Run with `cargo test --test docs_sync_test`.

### `binarization_test.rs`
Binarization-method tests (Tesseract `thresholding_method` parity): the `BinarizationMethod` → `ThresholdMethod` mapping, layout analysis under every method, the legacy `_with(Otsu)` equivalence, and the uneven-background fixture where global Otsu loses text that Sauvola recovers.

### `psm_osd_tests.rs`
Integration tests for Tesseract `--psm` parity and orientation/script detection (OSD): all 14 page-segmentation modes, OSD metadata, and the `--whitelist` filter.

### `webocr_round_trip.rs`
Round-trip tests for the WebOCR template engine (`--engine template`): system-font renders → template OCR → Levenshtein accuracy gate, plus script auto-detection dispatch (Latin vs CJK). Assertions collect `line.text` (the space-aware channel) — the `chars` vector carries glyphs only and drops word gaps by design.

### `round_trip_matrix.rs`
Expanded round-trip matrix for the template engine (2026-11-05): rendered text → template OCR → Levenshtein gates across a fixture matrix —

- **Fonts**: sans / serif / monospace system faces (first hit from per-platform candidate lists)
- **Sizes**: 16–54 px ladder bracketing the template rasters
- **Content**: ALL CAPS, punctuation/symbols, mixed case, hyphenated codes, abbreviations
- **Layout**: multi-line paragraphs with varied line lengths (line-count asserts)
- **Degradations**: deterministic xorshift speckle, 3×3 box blur, 2°/5° skew, faded ink, inverted polarity, uneven background gradient, 2× downscale
- **Invariants**: clean ≥ noisy, word count preserved, deterministic output, result structure sane
- **Pipeline**: `OcrEngine` + `RecognitionEngine::Template` end-to-end (async), plus a regression pin that dictionary correction (default off) does not mangle good output

All distortions are deterministic (seeded xorshift, closed-form warps) — no `rand`, so fixtures are reproducible. Gates sit one notch under measured accuracy in the house convention. Skips gracefully when no system fonts are present.

### `test_images_benchmark.rs`
Performance benchmark tests (marked with `#[ignore]`):
- Run with: `cargo test -- --ignored`
- Measures OCR processing time for different image types
- Useful for tracking performance improvements

### `test_web_api.rs`
Web API tests (multipart upload, endpoints). Requires the `web-api` feature:
```bash
cargo test --features web-api --test test_web_api
```

## Running Tests

### Run all tests:
```bash
cargo test
```

### Run specific test file:
```bash
cargo test --test test_images_test
```

### Run with output:
```bash
cargo test --test test_images_test -- --nocapture
```

### Run benchmarks:
```bash
cargo test --test test_images_benchmark -- --ignored
```

### Run a specific test:
```bash
cargo test --test test_images_test test_simple_text
```

## Test Images

`test_images_test.rs` and `test_images_benchmark.rs` expect images under `test_images/` (repo root). This directory **is committed** — the PNGs ship with the repo, so the tests run out of the box. To regenerate them:

```bash
cargo run --example generate_test_images
```

A second, smaller set lives in `test-images/` (hyphen), generated by `examples/create_test_images.rs`; it is used by the examples rather than by the test suite.

Tests skip gracefully when the images are missing. Other examples use pre-generated images from `examples/images/`.

## Ignored tests

`tests/round_trip_test.rs` (88 tests) is fully `#[ignore]`d — it was marked ignored while the bundled model was an untrained placeholder and has not been re-enabled since. Run it explicitly with:

```bash
cargo test --test round_trip_test -- --ignored
```

`tests/test_images_benchmark.rs` also marks 2 timing-sensitive tests `#[ignore]` (see "Run benchmarks" below).

## Test Structure

Tests are designed to:
1. **Verify API structure** - Ensure all methods return expected types
2. **Check data validity** - Verify confidence scores, bounding boxes, etc.
3. **Handle missing images gracefully** - Skip tests if images don't exist
4. **Be lenient on OCR results** - Recognition depends on the bundled model's training state, so structural tests assert shape and invariants rather than perfect recognition; round-trip tests assert actual recognition on synthetic inputs

## Adding New Tests

When adding new test images:
1. Add a generator to `examples/generate_test_images.rs` (or reuse `examples/images/`)
2. Add a corresponding test in `test_images_test.rs`
3. Update this README if needed
