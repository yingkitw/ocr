use assert_cmd::prelude::*;
use image::{ImageBuffer, Luma};
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

// Helper function to create a simple test image with text
fn create_test_image_with_text(_text: &str, width: u32, height: u32) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".png").unwrap();

    // Create a blank white image
    let img = ImageBuffer::from_pixel(width, height, Luma([255u8]));

    // Save the image
    img.write_to(&mut file, image::ImageFormat::Png).unwrap();
    file.flush().unwrap();
    file.reopen().unwrap();

    file
}

// Helper function to create a simple test image with black rectangles (simulating text)
fn create_test_image_with_rectangles() -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".png").unwrap();

    // Create a blank white image
    let mut img = ImageBuffer::from_pixel(200, 50, Luma([255u8]));

    // Add some black rectangles to simulate text characters
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

    // Save the image
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
            "✓ OCR engine initialized successfully",
        ));
}

#[test]
fn test_cli_list_languages_command() {
    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.arg("list-languages");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Supported languages"))
        .stdout(predicates::str::contains("en"));
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

    cmd.assert()
        .failure();
}

#[test]
fn test_cli_extract_with_unsupported_language() {
    let file = create_test_image_with_text("test", 100, 50);

    let mut cmd = Command::cargo_bin("ocr").unwrap();
    cmd.args(&["extract", file.path().to_str().unwrap(), "--lang", "fra"]);

    cmd.assert().success();
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
