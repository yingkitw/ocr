# TODO

## Near-term

- [ ] Enable and test SIMD acceleration (`simd` feature flag)
- [ ] Wire up LSTM/CNN/transformer recognition engines as usable alternatives to pattern matching
- [ ] Add CJK language support to the CLI language selector
- [ ] Complete image preprocessing pipeline integration (deskew, adaptive binarization)
- [ ] Add dictionary-based post-correction for recognition results
- [ ] Benchmark and profile recognition performance

## Medium-term

- [ ] Implement training pipeline CLI commands
- [ ] Add CUDA/OpenCL backend support for neural network models
- [ ] Add web API mode for HTTP-based OCR
- [ ] Improve layout analysis for complex multi-column documents
- [ ] Add PDF input support

## Completed

- [x] CLI with extract, batch, layout, list-languages, check, info, validate commands
- [x] Pattern matching recognition engine (default)
- [x] Image preprocessing (grayscale, thresholding, noise removal)
- [x] Multiple output formats (text, json, hocr, tsv)
- [x] Layout analysis (column/line detection, text ordering)
- [x] Language detection (N-gram based)
- [x] CJK character segmentation
- [x] Round-trip tests and snapshot tests
- [x] Comprehensive test suite (228+ tests)
