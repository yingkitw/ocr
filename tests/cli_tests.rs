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
    cmd.args(&["extract", "nonexistent.png"]);
    cmd.assert().failure();
}

#[test]
fn test_cli_extract_with_unsupported_format() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", "test.xyz"]);
    cmd.assert().failure();
}

#[test]
fn test_cli_extract_with_valid_image() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap()]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_preprocessing() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--preprocess"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_lstm_engine() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--engine", "lstm"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_hybrid_engine() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&[
        "extract",
        file.path().to_str().unwrap(),
        "--engine",
        "hybrid",
    ]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_dict_correct() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--dict-correct"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_cjk_lang() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--lang", "zh"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_french_lang() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--lang", "fr"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_german_lang() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--lang", "de"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_spanish_lang_and_dict_correct() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&[
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
    cmd.args(&["extract", file.path().to_str().unwrap(), "--device", "cpu"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_json_format() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "-f", "json"]);
    cmd.assert().success();
}

#[test]
fn test_cli_extract_with_all_options() {
    let file = create_test_image_with_rectangles();
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&[
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
fn test_cli_info_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("info");
    cmd.assert().success();
}

#[test]
fn test_cli_validate_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["validate", "nonexistent.json"]);
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
    cmd.args(&[
        "batch",
        "-i",
        input_dir.path().to_str().unwrap(),
        "-o",
        output_dir.path().to_str().unwrap(),
        "--max-concurrent",
        "2",
    ]);
    cmd.assert().success();

    // Every input image must produce a .txt output, regardless of concurrency.
    for stem in &["a", "b", "c"] {
        let out = output_dir.path().join(format!("{}.txt", stem));
        assert!(fs::metadata(&out).map(|m| m.is_file()).unwrap_or(false),
                "missing batch output for {}", stem);
    }
}
