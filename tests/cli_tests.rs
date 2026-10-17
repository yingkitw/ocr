use assert_cmd::prelude::*;
use image::{ImageBuffer, Luma};
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

fn create_test_image_with_rectangles() -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".png").unwrap();
    let mut img = ImageBuffer::from_pixel(200, 50, Luma([255u8]));

    for y in 10..40 {
        for x in 10..20 {
            img.put_pixel(x, y, Luma([0u8]));
        }
    }
    for y in 10..40 {
        for x in 25..35 {
            img.put_pixel(x, y, Luma([0u8]));
        }
    }

    img.write_to(&mut file, image::ImageFormat::Png).unwrap();
    file.flush().unwrap();
    file.reopen().unwrap();
    file
}

#[test]
fn test_cli_check_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("check");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Checking system requirements"))
        .stdout(predicates::str::contains(
            "✓ Pattern matching engine initialized",
        ));
}

#[test]
fn test_cli_list_languages_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("list-languages");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Supported languages"))
        .stdout(predicates::str::contains("en"))
        .stdout(predicates::str::contains("zh"))
        .stdout(predicates::str::contains("ja"))
        .stdout(predicates::str::contains("ko"))
        .stdout(predicates::str::contains("fr"))
        .stdout(predicates::str::contains("de"))
        .stdout(predicates::str::contains("es"))
        .stdout(predicates::str::contains("nl"))
        .stdout(predicates::str::contains("ru"));
}

#[test]
fn test_cli_extract_with_nonexistent_file() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", "nonexistent.png"]);
    cmd.assert().failure();
}

#[test]
fn test_cli_extract_with_unsupported_format() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", "test.xyz"]);
    cmd.assert().failure();
}

#[test]
fn test_cli_extract_with_valid_image() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap()]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_preprocessing() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--preprocess"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_lstm_engine() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--engine", "lstm"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_dict_correct() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--dict-correct"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_cjk_lang() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--lang", "zh"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_french_lang() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--lang", "fr"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_german_lang() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--lang", "de"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_spanish_lang_and_dict_correct() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args([
        "extract",
        file.path().to_str().unwrap(),
        "--lang",
        "es",
        "--dict-correct",
    ]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_device_cpu() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "--device", "cpu"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_json_format() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "-f", "json"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_rag_format() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "-f", "rag"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("\"chunks\""));
}

#[test]
fn test_cli_extract_with_page_format() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "-f", "page"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("<PcGts"))
        .stdout(predicates::str::contains("primaresearch.org/PAGE"));
}

#[test]
fn test_cli_extract_with_csv_format() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", file.path().to_str().unwrap(), "-f", "csv"]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains(
            "page,line,word,text,confidence,left,top,width,height",
        ));
}

#[test]
fn test_cli_extract_from_stdin() {
    // Create a test image and read its bytes
    let file = create_test_image_with_rectangles();
    let image_bytes = std::fs::read(file.path()).unwrap();

    let mut cmd = assert_cmd::Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", "-"]);
    cmd.write_stdin(image_bytes);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_from_stdin_empty() {
    let mut cmd = assert_cmd::Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", "-"]);
    cmd.write_stdin(b"");
    cmd.assert().failure();
}

#[test]
fn test_cli_extract_with_all_options() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args([
        "extract",
        file.path().to_str().unwrap(),
        "--lang",
        "en",
        "--preprocess",
        "--engine",
        "pattern",
        "--dict-correct",
        "--confidence",
        "0.3",
        "--psm",
        "6",
        "-f",
        "json",
    ]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_dpi() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args([
        "extract",
        file.path().to_str().unwrap(),
        "--dpi",
        "150",
    ]);
    cmd.assert().success();
}

#[test]
fn test_cli_info_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("info");
    cmd.assert().success();
}

#[test]
fn test_cli_validate_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["validate", "nonexistent.json"]);
    cmd.assert().failure();
}

#[test]
fn test_cli_batch_processes_all_images() {
    use std::fs;
    use tempfile::TempDir;

    let input_dir = TempDir::new().unwrap();
    let output_dir = TempDir::new().unwrap();

    // Write three small images into the input directory.
    for name in &["a.png", "b.png", "c.png"] {
        let path = input_dir.path().join(name);
        let img: ImageBuffer<Luma<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(200, 50, Luma([255u8]));
        img.save(&path).unwrap();
    }

    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args([
        "batch",
        "-i",
        input_dir.path().to_str().unwrap(),
        "-o",
        output_dir.path().to_str().unwrap(),
        "--max-concurrent",
        "2",
    ]);
    let assert = cmd.assert().success();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr);
    assert!(
        stderr.contains("Batch complete: 3/3"),
        "expected batch progress summary, got: {}",
        stderr
    );
    assert!(
        stderr.contains("[3/3]"),
        "expected per-image progress lines, got: {}",
        stderr
    );

    // Every input image must produce a .txt output, regardless of concurrency.
    for stem in &["a", "b", "c"] {
        let out = output_dir.path().join(format!("{}.txt", stem));
        assert!(fs::metadata(&out).map(|m| m.is_file()).unwrap_or(false),
                "missing batch output for {}", stem);
    }
}

#[test]
fn test_cli_makebox_writes_box_file() {
    use tempfile::TempDir;

    let file = create_test_image_with_rectangles();
    let out_dir = TempDir::new().unwrap();
    let out_base = out_dir.path().join("train");

    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args([
        "makebox",
        file.path().to_str().unwrap(),
        "-o",
        out_base.to_str().unwrap(),
    ]);
    cmd.assert().success();

    let box_path = out_dir.path().join("train.box");
    assert!(box_path.exists(), "expected {}", box_path.display());
}

#[test]
fn test_cli_extract_multi_page_tiff() {
    use std::io::Cursor;

    // Create a 2-page TIFF
    let page1 = ImageBuffer::from_pixel(200, 50, Luma([255u8]));
    let page2 = ImageBuffer::from_pixel(200, 50, Luma([255u8]));

    let mut tiff_file = NamedTempFile::with_suffix(".tiff").unwrap();
    let mut tiff_bytes = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut tiff_bytes).unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(
                page1.width(),
                page1.height(),
                page1.as_raw(),
            )
            .unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(
                page2.width(),
                page2.height(),
                page2.as_raw(),
            )
            .unwrap();
    }
    tiff_file
        .write_all(&tiff_bytes.into_inner())
        .unwrap();
    tiff_file.flush().unwrap();

    // Extract should succeed and output page separators for multi-page TIFF
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(["extract", tiff_file.path().to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("--- Page 1 ---"))
        .stdout(predicates::str::contains("--- Page 2 ---"));
}

#[cfg(feature = "pdf")]
#[test]
fn test_cli_extract_with_pdf_format() {
    let file = create_test_image_with_rectangles();
    let pdf_output = tempfile::NamedTempFile::new().unwrap();
    let pdf_path = pdf_output.path().to_str().unwrap().to_string();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args([
        "extract",
        file.path().to_str().unwrap(),
        "-f",
        "pdf",
        "-o",
        &pdf_path,
    ]);
    cmd.assert().success();
    let pdf_bytes = std::fs::read(&pdf_path).unwrap();
    assert!(pdf_bytes.starts_with(b"%PDF-1.4"), "PDF should start with header");
    assert!(pdf_bytes.ends_with(b"%%EOF\n") || pdf_bytes.ends_with(b"%%EOF"), "PDF should end with EOF marker");
    assert!(pdf_bytes.windows(8).any(|w| w == b"/XObject"), "PDF should contain XObject");
}

// ---- Tesseract-parity: --psm (symbolic + numeric) and --whitelist ----

#[test]
fn test_cli_psm_osd_only_outputs_orientation_report() {
    let img = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("extract")
        .arg(img.path())
        .arg("--psm")
        .arg("osd_only");
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Orientation in degrees:"))
        .stdout(predicates::str::contains("Script:"));
}

#[test]
fn test_cli_psm_numeric_still_works() {
    let img = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("extract").arg(img.path()).arg("--psm").arg("6");
    cmd.assert().success();
}

#[test]
fn test_cli_psm_invalid_name_fails_with_helpful_error() {
    let img = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("extract").arg(img.path()).arg("--psm").arg("bogus");
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Invalid PSM"));
}

#[test]
fn test_cli_psm_auto_only_layout_report() {
    let img = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("extract").arg(img.path()).arg("--psm").arg("2");
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Layout regions:"));
}

#[test]
fn test_cli_psm_circle_word_symbolic() {
    let img = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("extract")
        .arg(img.path())
        .arg("--psm")
        .arg("circle_word");
    cmd.assert().success();
}

#[test]
fn test_cli_whitelist_flag_accepted() {
    let img = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("extract")
        .arg(img.path())
        .arg("--whitelist")
        .arg("0123456789")
        .arg("--psm")
        .arg("single_block");
    cmd.assert().success();
}
