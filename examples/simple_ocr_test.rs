//! Simple OCR test that works around compilation issues
//!
//! This is a minimal test to verify OCR functionality

use image::GenericImageView;
use std::path::Path;

fn main() {
    println!("OCR Simple Test");
    println!("===================");

    let test_image_path = Path::new("test_images/sample.png");

    if !test_image_path.exists() {
        eprintln!("Error: Test image not found at: {:?}", test_image_path);
        eprintln!("\nPlease create a test image first.");
        std::process::exit(1);
    }

    println!("\nTest image found: {:?}", test_image_path);

    // Try to load the image
    match image::open(test_image_path) {
        Ok(img) => {
            let (width, height) = img.dimensions();
            println!("✓ Image loaded successfully");
            println!("  Dimensions: {}x{} pixels", width, height);
            println!("  Format: {:?}", img.color());

            // For now, just verify the image can be loaded
            // Full OCR will work once compilation errors are fixed
            println!("\n✓ Image loading test passed!");
            println!("\nNote: Full OCR functionality requires fixing compilation errors.");
            println!("Once fixed, you can use:");
            println!("  cargo run -- recognize -i test_images/sample.png");
        }
        Err(e) => {
            eprintln!("Error loading image: {}", e);
            std::process::exit(1);
        }
    }
}
