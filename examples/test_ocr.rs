//! Simple OCR test example
//!
//! Run with: cargo run --example test_ocr -- test_images/sample.png

use clap::Parser;
use ocr::api::MiniOcr;
use ocr::utils::Result;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Input image file
    #[arg(default_value = "test_images/sample.png")]
    input: PathBuf,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.debug {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    println!("MiniOCR Test");
    println!("============");
    println!("Input image: {}", args.input.display());

    // Check if file exists
    if !args.input.exists() {
        eprintln!("Error: Image file not found: {}", args.input.display());
        eprintln!("\nPlease provide a valid image file path.");
        eprintln!("Example: cargo run --example test_ocr -- path/to/image.png");
        std::process::exit(1);
    }

    println!("\nInitializing OCR engine...");

    // Create OCR instance
    let mut ocr = match MiniOcr::new() {
        Ok(ocr) => ocr,
        Err(e) => {
            eprintln!("Error creating OCR engine: {}", e);
            std::process::exit(1);
        }
    };

    // Initialize the engine
    match ocr.initialize().await {
        Ok(_) => println!("OCR engine initialized successfully"),
        Err(e) => {
            eprintln!("Error initializing OCR engine: {}", e);
            std::process::exit(1);
        }
    }

    println!("\nProcessing image...");

    // Recognize text
    let start = std::time::Instant::now();
    let result = match ocr.recognize_text_from_file(&args.input).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error recognizing text: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("--------");
    println!("Processing time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("\nRecognized text:");
    println!("{}", "=".repeat(50));
    println!("{}", result.text);
    println!("{}", "=".repeat(50));

    if !result.words.is_empty() {
        println!("\nWord details:");
        for (i, word) in result.words.iter().enumerate().take(10) {
            println!(
                "  {}: '{}' (confidence: {:.2}%)",
                i + 1,
                word.text,
                word.confidence * 100.0
            );
        }
        if result.words.len() > 10 {
            println!("  ... and {} more words", result.words.len() - 10);
        }
    }

    println!("\nTest completed successfully!");
    Ok(())
}
