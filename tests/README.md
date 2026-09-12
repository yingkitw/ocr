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

`test_images_test.rs` and `test_images_benchmark.rs` expect images under `test_images/` (repo root). This directory is not committed — generate the images first:

```bash
cargo run --example generate_test_images
```

Tests skip gracefully when the images are missing. Other examples use pre-generated images from `examples/images/`.

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
