use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod commands;

#[derive(Parser)]
#[command(name = "ocr")]
#[command(about = "A pure Rust CLI OCR tool for printed text extraction")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Recognize text from an image file
    Extract {
        /// Path to the image file
        #[arg(value_name = "IMAGE_PATH")]
        image_path: PathBuf,

        /// Output file path (optional, prints to stdout if not specified)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Language code: en, zh, ja, ko, fr, de, es, … or `auto` to detect from text
        #[arg(short, long, default_value = "en")]
        lang: String,

        /// Preprocess image for better OCR results
        #[arg(long)]
        preprocess: bool,

        /// Output format: text, json, hocr, tsv, alto, xml
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Page segmentation mode (Tesseract PSM): 3=Auto, 4=SingleColumn, 6=SingleBlock, 11=SparseText, 13=SingleLine
        #[arg(long, default_value = "3")]
        psm: u8,

        /// Confidence threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        confidence: f32,

        /// Recognition engine: pattern, lstm, hybrid
        #[arg(long, default_value = "pattern")]
        engine: String,

        /// Enable dictionary-based post-correction
        #[arg(long)]
        dict_correct: bool,

        /// Compute device: cpu, gpu, auto (default: auto)
        #[arg(long, default_value = "auto")]
        device: String,

        /// Enable orientation and script detection (OSD)
        #[arg(long)]
        osd: bool,
    },

    /// Batch process multiple images from a directory
    Batch {
        /// Input directory containing images
        #[arg(short, long, value_name = "DIR")]
        input_dir: PathBuf,

        /// Output directory for results
        #[arg(short, long, value_name = "DIR")]
        output_dir: PathBuf,

        /// Language code: en, zh, ja, ko, … or `auto` to detect from text
        #[arg(short, long, default_value = "en")]
        lang: String,

        /// Confidence threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        confidence: f32,

        /// Maximum concurrent images
        #[arg(long, default_value = "4")]
        max_concurrent: usize,

        /// Recognition engine: pattern, lstm, hybrid
        #[arg(long, default_value = "pattern")]
        engine: String,

        /// Enable dictionary-based post-correction
        #[arg(long)]
        dict_correct: bool,

        /// Compute device: cpu, gpu, auto (default: auto)
        #[arg(long, default_value = "auto")]
        device: String,
    },

    /// Analyze image layout
    Layout {
        /// Path to the image file
        #[arg(value_name = "IMAGE_PATH")]
        image_path: PathBuf,

        /// Output file for layout JSON (optional, prints to stdout if not specified)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },

    /// List supported languages
    ListLanguages,

    /// Check system requirements
    Check,

    /// Display engine information
    Info,

    /// Validate a configuration file
    Validate {
        /// Configuration file to validate
        #[arg(value_name = "FILE")]
        config_file: PathBuf,
    },

    /// Train a recognition model on synthetic data
    Train {
        /// Number of training epochs
        #[arg(short, long, default_value_t = 10)]
        epochs: usize,

        /// Batch size per training step
        #[arg(short, long, default_value_t = 8)]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.001")]
        learning_rate: f32,

        /// Recognition engine to train: lstm
        #[arg(long, default_value = "lstm")]
        engine: String,

        /// Directory to save model checkpoints
        #[arg(short, long, value_name = "DIR")]
        checkpoint_dir: Option<PathBuf>,

        /// Distortion level: clean, mild, heavy
        #[arg(long, default_value = "mild")]
        distortion: String,
    },

    /// Benchmark per-script recognition accuracy on synthetic data
    Benchmark {
        /// Number of synthetic samples per script
        #[arg(short, long, default_value_t = 10)]
        samples: usize,

        /// Distortion level: clean, mild, heavy
        #[arg(long, default_value = "clean")]
        distortion: String,
    },

    /// Generate a Tesseract-style .box file from an image (training makebox)
    Makebox {
        /// Path to the image file
        #[arg(value_name = "IMAGE_PATH")]
        image_path: PathBuf,

        /// Output base path (writes `<base>.box`; default: image path without extension)
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Language code: en, zh, … or `auto`
        #[arg(short, long, default_value = "en")]
        lang: String,

        /// Recognition engine: pattern, lstm, hybrid
        #[arg(long, default_value = "pattern")]
        engine: String,

        /// Preprocess image before recognition
        #[arg(long)]
        preprocess: bool,
    },

    /// Start HTTP API server for OCR
    #[cfg(feature = "web-api")]
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Maximum upload size in MB
        #[arg(long, default_value_t = 20)]
        max_upload_size: usize,
    },
}

pub fn parse() -> Cli {
    Cli::parse()
}
