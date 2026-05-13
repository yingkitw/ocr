# ocr

A minimalist OCR library for Rust with CLI and MCP server interfaces. Recognition is **implemented in Rust** (bitmap font templates, segmentation, and matching)—**no Tesseract** or other external OCR engine.

The MCP server is built on [rmcp](https://github.com/modelcontextprotocol/rust-sdk).

## Features

- **No native OCR dependencies** - Pure Rust pipeline (binarization, line/character segmentation, template matching)
- **Library** - `OcrEngine` API with structured results (text, words, confidence, bounding boxes)
- **CLI** - Text and JSON output; optional preprocessing
- **MCP server** - Model Context Protocol server for assistant integration (stdio transport)
- **Preprocessing** - Optional grayscale + threshold preprocessing
- **English-focused** - Bitmap glyph set suited to Latin letters, digits, and common punctuation

## Installation

```bash
cargo build --release
```

Binaries (from this crate):

| Binary   | Purpose                          |
|----------|----------------------------------|
| `ocr`    | CLI tool                         |
| `ocr-mcp`| MCP server (stdio transport)     |

## Usage

### Library

Crate name: **`ocr`** (import as `ocr`, not `ocr_rs`).

```rust
use ocr::OcrEngine;
use std::path::Path;

let engine = OcrEngine::new()
    .language("eng")
    .preprocessing(true);

let result = engine.recognize_file(Path::new("document.png"))?;
println!("{}", result.text);
println!("Confidence: {:.1}%", result.confidence);
```

### CLI

```bash
# Plain text to stdout
ocr document.png

# JSON with word-level detail
ocr document.png -f json

# Enable preprocessing
ocr document.png --preprocess
```

### MCP server

Add to your MCP client configuration (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "ocr": {
      "command": "ocr-mcp",
      "args": []
    }
  }
}
```

Exposed tools:

| Tool         | Description                                |
|--------------|--------------------------------------------|
| `ocr_image`  | OCR on an image file (path)                |
| `ocr_base64` | OCR on base64-encoded image data           |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for design details and [SPEC.md](SPEC.md) for the API specification.

## License

Apache-2.0
