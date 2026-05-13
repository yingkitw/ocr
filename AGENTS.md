# AGENTS.md

This file contains guidelines and commands for agentic coding agents working on this OCR CLI project.

## Build Commands

### Standard Build
```bash
cargo build
```

### Release Build
```bash
cargo build --release
```

### Check compilation without building
```bash
cargo check
```

## Test Commands

### Run all tests
```bash
cargo test
```

### Run a single test
```bash
cargo test test_name
```

### Run tests for a specific module
```bash
cargo test module_name
```

### Run tests with output
```bash
cargo test -- --nocapture
```

### Run integration tests
```bash
cargo test --test cli_tests
```

## Linting and Formatting

### Format code
```bash
cargo fmt
```

### Check code formatting
```bash
cargo fmt --check
```

### Run clippy lints
```bash
cargo clippy
```

### Run clippy with all features
```bash
cargo clippy --all-features
```

## Project Structure

```
src/
├── main.rs          # CLI entry point and command handling
├── lib.rs           # Library entry point re-exporting modules
├── cli/mod.rs       # CLI argument parsing with clap
├── api/             # High-level OCR API (MiniOcr, TextProcessor, config)
│   ├── mod.rs
│   ├── config.rs
│   ├── error.rs
│   ├── image.rs
│   ├── ocr.rs
│   └── text.rs
├── core/            # Core OCR engine and data structures
│   ├── mod.rs
│   ├── config.rs
│   ├── engine.rs
│   ├── geometry.rs
│   ├── image.rs
│   ├── layout.rs
│   ├── output.rs
│   ├── recognition.rs
│   └── text.rs
├── image/           # Image preprocessing pipeline
│   ├── mod.rs
│   ├── enhancement.rs
│   ├── pipeline.rs
│   ├── processor.rs
│   ├── quality.rs
│   └── thresholder.rs
├── lang/            # Language support and detection
│   ├── mod.rs
│   ├── cjk.rs
│   ├── detector.rs
│   ├── dictionary.rs
│   ├── ngram.rs
│   ├── unicharset.rs
│   └── unicode.rs
├── layout/          # Layout analysis and text segmentation
│   ├── mod.rs
│   ├── analyzer.rs
│   ├── classifier.rs
│   ├── column_detector.rs
│   ├── detector.rs
│   ├── line_detector.rs
│   ├── text_line_features.rs
│   ├── text_ordering.rs
│   └── union_find_ccl.rs
├── recognition/     # Character recognition models
│   ├── mod.rs
│   ├── basic_ocr.rs
│   ├── cnn_model.rs
│   ├── ctc_decoder.rs
│   ├── end_to_end_model.rs
│   ├── engine.rs
│   ├── hybrid_model.rs
│   ├── lstm.rs
│   ├── lstm_model.rs
│   ├── pattern.rs
│   ├── pattern_model.rs
│   ├── tesseract_blob.rs
│   ├── tesseract_features.rs
│   ├── tesseract_textline.rs
│   ├── transformer_model.rs
│   └── vit_model.rs
├── training/        # Model training pipeline
│   ├── mod.rs
│   ├── augmentation.rs
│   ├── checkpoint.rs
│   ├── config.rs
│   ├── data.rs
│   ├── losses.rs
│   ├── metrics.rs
│   ├── optimizers.rs
│   └── training.rs
└── utils/           # Shared utilities
    ├── mod.rs
    ├── async_utils.rs
    ├── error.rs
    ├── hash.rs
    ├── math.rs
    ├── simd.rs
    ├── simd_advanced.rs
    └── time.rs
```

## Code Style Guidelines

### Imports
- Group imports in this order:
  1. Standard library imports (`std::*`)
  2. External crate imports (alphabetical)
  3. Local module imports (`crate::*`)
- Use `use` statements at the top of the file
- Prefer fully qualified paths for external crates when there are name conflicts

### Error Handling
- Use `anyhow::Result<T>` for most functions
- Use `anyhow::anyhow!()` to create error messages
- Use `thiserror` for custom error types (currently not used in project)
- Return early with `?` operator for error propagation
- Use descriptive error messages with context

### Naming Conventions
- `snake_case` for variables and functions
- `PascalCase` for types, structs, and enums
- `SCREAMING_SNAKE_CASE` for constants
- Use descriptive names that convey purpose
- Prefix boolean variables with `is_`, `has_`, `can_`, etc.

### Functions
- Keep functions focused and small (ideally < 50 lines)
- Use descriptive function names that indicate what they do
- Document public functions with `///` doc comments
- Use `impl` blocks to group related methods
- Prefer builder pattern for complex object construction

### Structs and Enums
- Use `#[derive(Debug)]` for most public types
- Use `#[derive(Clone)]` when cloning is reasonable
- Use `#[derive(Serialize, Deserialize)]` for data structures
- Mark fields as `pub` when they need to be accessed directly
- Consider using newtype pattern for type safety

### Module Organization
- Use `mod.rs` files for submodules
- Keep related functionality together
- Re-export public items from `mod.rs` when needed
- Use `#[cfg(test)]` for test modules

### Testing
- Write unit tests in `tests` modules within each source file
- Write integration tests in the `tests/` directory
- Use descriptive test names
- Test both success and error cases
- Use helper functions for common test setup

### Constants and Magic Numbers
- Define constants for magic numbers and repeated values
- Use `const` for compile-time constants
- Group related constants together
- Use descriptive names for constants

### Documentation
- Add module-level documentation explaining purpose
- Document public APIs with examples
- Use `///` for item documentation
- Use `//` for implementation comments
- Include parameter descriptions in doc comments

### Logging
- Use the `log` crate with appropriate levels:
  - `error!`: For unrecoverable errors
  - `warn!`: For concerning but recoverable issues
  - `info!`: For important operational information
  - `debug!`: For detailed debugging information
  - `trace!`: For very detailed tracing

### Performance Considerations
- Use `rayon` for parallel processing when appropriate
- Consider using `Cow<str>` for string handling to avoid allocations
- Profile before optimizing
- Use iterators and functional style where appropriate

## Testing Guidelines

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies when needed
- Test edge cases and error conditions
- Keep tests fast and focused

### Integration Tests
- Test the complete CLI workflow
- Test with real image files
- Verify output formats (text and JSON)
- Test error handling for invalid inputs

### Test Organization
- Group related tests with `mod test_blocks`
- Use helper functions for common test setup
- Create test data programmatically when possible
- Clean up test files after use

## Development Workflow

1. Run `cargo check` to verify compilation
2. Run `cargo fmt` to format code
3. Run `cargo clippy` to check for issues
4. Run `cargo test` to verify tests pass
5. Run tests for specific modules you've changed
6. Create meaningful commit messages

## Common Patterns

### Result Handling
```rust
fn example_function() -> Result<String> {
    let value = some_operation()?;
    let processed = process_value(value)?;
    Ok(processed)
}
```

### Image Processing
```rust
pub struct ImageProcessor {
    pub image: DynamicImage,
}

impl ImageProcessor {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let img = image::open(path)?;
        Ok(Self { image: img })
    }
}
```

### CLI Structure
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Extract { /* fields */ },
    ListLanguages,
    Check,
}
```

## Dependencies

### Core Dependencies
- `clap`: CLI argument parsing
- `image`: Image processing
- `anyhow`: Error handling
- `thiserror`: Custom error types
- `log`: Logging facade
- `env_logger`: Logger implementation
- `serde`: Serialization
- `rayon`: Parallel processing

### Dev Dependencies
- `tempfile`: Temporary files for testing
- `assert_cmd`: Command testing
- `predicates`: Test assertions