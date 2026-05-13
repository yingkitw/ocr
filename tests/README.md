# Test Suite for OCR

This directory contains comprehensive tests for the OCR library.

## Test Files

### `basic_tests.rs`
Basic functionality tests for OCR initialization, configuration, and metadata.

### `integration_tests.rs`
Integration tests for OCR pipeline components including CJK language detection and model creation.

### `snapshot_tests.rs`
Snapshot tests using `insta` to ensure consistent output over time.

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

Tests use images from `../test_images/` directory. See `../test_images/README.md` for details.

## Test Structure

Tests are designed to:
1. **Verify API structure** - Ensure all methods return expected types
2. **Check data validity** - Verify confidence scores, bounding boxes, etc.
3. **Handle missing images gracefully** - Skip tests if images don't exist
4. **Be lenient on OCR results** - Since OCR may not be fully implemented, tests verify structure rather than perfect recognition

## Adding New Tests

When adding new test images:
1. Add the image to `test_images/` directory
2. Add a corresponding test in `test_images_test.rs`
3. Update this README if needed

