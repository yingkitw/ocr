//! Output Formatting Domain Module
//!
//! Provides output formatting capabilities for OCR results.
//! Supports multiple formats including plain text, JSON, hOCR, TSV, ALTO, and PDF.

pub use service::OutputFormattingService;

mod service;

/// Output formatting domain error types
#[derive(Debug, thiserror::Error)]
pub enum OutputFormattingError {
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    
    #[error("Failed to format output: {0}")]
    FormattingFailed(String),
    
    #[error("Failed to write output: {0}")]
    WriteFailed(String),
    
    #[error("PDF generation failed: {0}")]
    PdfGenerationFailed(String),
}

/// Supported output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    PlainText,
    Json,
    Hocr,
    Tsv,
    Alto,
    Box,
    Pdf,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Result<Self, OutputFormattingError> {
        match s.to_lowercase().as_str() {
            "text" | "txt" => Ok(OutputFormat::PlainText),
            "json" => Ok(OutputFormat::Json),
            "hocr" | "html" => Ok(OutputFormat::Hocr),
            "tsv" => Ok(OutputFormat::Tsv),
            "alto" | "xml" => Ok(OutputFormat::Alto),
            "box" => Ok(OutputFormat::Box),
            "pdf" => Ok(OutputFormat::Pdf),
            _ => Err(OutputFormattingError::UnsupportedFormat(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            OutputFormat::PlainText => "text",
            OutputFormat::Json => "json",
            OutputFormat::Hocr => "hocr",
            OutputFormat::Tsv => "tsv",
            OutputFormat::Alto => "alto",
            OutputFormat::Box => "box",
            OutputFormat::Pdf => "pdf",
        }
    }
}
