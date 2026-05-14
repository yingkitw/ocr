//! HTTP API server for OCR
//!
//! Provides REST endpoints for text recognition from uploaded images.
//! Enabled via the `web-api` feature flag.

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;

use crate::api::Ocr;
use crate::core::output::{format_hocr, format_tsv, to_json_output};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_upload_size_mb: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_upload_size_mb: 20,
        }
    }
}

/// Shared application state
struct AppState {
    ocr: Arc<Ocr>,
}

/// OCR request parameters (via query string or form field)
#[derive(Debug, Deserialize)]
struct OcrParams {
    #[serde(default = "default_lang")]
    lang: String,
    #[serde(default = "default_format")]
    format: String,
    #[serde(default)]
    preprocess: bool,
    #[serde(default = "default_engine")]
    engine: String,
}

fn default_lang() -> String {
    "en".to_string()
}
fn default_format() -> String {
    "text".to_string()
}
fn default_engine() -> String {
    "pattern".to_string()
}

/// OCR response
#[derive(Debug, Serialize)]
struct OcrResponse {
    success: bool,
    text: String,
    confidence: f64,
    language: String,
    words_count: usize,
    lines_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hocr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tsv: Option<String>,
    processing_time_ms: u64,
}

/// Error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    success: bool,
    error: String,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    supported_languages: Vec<String>,
}

/// Start the HTTP server
pub async fn run_server(config: ServerConfig) -> crate::utils::Result<()> {
    let ocr = Ocr::new()?;
    ocr.initialize().await?;

    let state = Arc::new(AppState {
        ocr: Arc::new(ocr),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/languages", get(list_languages))
        .route("/recognize", post(recognize_text))
        .layer(cors)
        .layer(RequestBodyLimitLayer::new(
            config.max_upload_size_mb * 1024 * 1024,
        ))
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting OCR API server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// GET /health
async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let metadata = state.ocr.get_metadata();
    Json(HealthResponse {
        status: "ok".to_string(),
        version: metadata.version.clone(),
        supported_languages: metadata.supported_languages.clone(),
    })
}

/// GET /languages
async fn list_languages(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let metadata = state.ocr.get_metadata();
    Json(serde_json::json!({
        "languages": metadata.supported_languages.clone(),
    }))
}

/// POST /recognize
///
/// Accepts multipart form data with:
/// - `image` (required): the image file (png, jpg, tiff, bmp, webp)
/// - `lang` (optional): language code, default "en"
/// - `format` (optional): output format "text", "json", "hocr", "tsv", default "text"
/// - `preprocess` (optional): "true"/"false", default "false"
/// - `engine` (optional): "pattern", "lstm", "hybrid", default "pattern"
async fn recognize_text(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> std::result::Result<Json<OcrResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let mut image_data: Option<Vec<u8>> = None;
    let mut params = OcrParams {
        lang: "en".to_string(),
        format: "text".to_string(),
        preprocess: false,
        engine: "pattern".to_string(),
    };

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "image" => {
                image_data = Some(field.bytes().await.unwrap_or_default().to_vec());
            }
            "lang" => {
                params.lang = field.text().await.unwrap_or_else(|_| "en".to_string());
            }
            "format" => {
                params.format = field.text().await.unwrap_or_else(|_| "text".to_string());
            }
            "preprocess" => {
                let val = field.text().await.unwrap_or_else(|_| "false".to_string());
                params.preprocess = val == "true";
            }
            "engine" => {
                params.engine = field.text().await.unwrap_or_else(|_| "pattern".to_string());
            }
            _ => {}
        }
    }

    let image_data = match image_data {
        Some(data) if !data.is_empty() => data,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    success: false,
                    error: "No image file provided. Use field name 'image'.".to_string(),
                }),
            ));
        }
    };

    let img = match image::load_from_memory(&image_data) {
        Ok(img) => img,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    success: false,
                    error: format!("Invalid image: {}", e),
                }),
            ));
        }
    };

    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let raw_pixels: Vec<u8> = gray.pixels().map(|p| p[0]).collect();

    let mut config = crate::core::config::OcrConfig::default();
    config.recognition.language = params.lang.clone();
    config.recognition.engine = match params.engine.as_str() {
        "lstm" => crate::core::config::RecognitionEngine::LSTM,
        "hybrid" => crate::core::config::RecognitionEngine::Hybrid,
        _ => crate::core::config::RecognitionEngine::PatternMatching,
    };
    config.image_processing.enable_preprocessing = params.preprocess;
    config.image_processing.enable_binarization = params.preprocess;
    config.image_processing.enable_noise_reduction = params.preprocess;
    config.image_processing.enable_contrast_enhancement = params.preprocess;
    config.image_processing.enable_deskewing = params.preprocess;

    let ocr = match crate::api::Ocr::with_config(config) {
        Ok(o) => o,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: format!("Failed to create OCR engine: {}", e),
                }),
            ));
        }
    };

    if let Err(e) = ocr.initialize().await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                success: false,
                error: format!("Failed to initialize OCR engine: {}", e),
            }),
        ));
    }

    let result = match ocr.recognize_text(&raw_pixels, width, height).await {
        Ok(r) => r,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: format!("OCR failed: {}", e),
                }),
            ));
        }
    };

    let elapsed = start.elapsed().as_millis() as u64;
    let words_count = result.words.len();
    let lines_count = result.lines.len();

    let (text, json_output, hocr, tsv) = match params.format.to_lowercase().as_str() {
        "json" => {
            let json_val = to_json_output(&result);
            (
                result.text.clone(),
                Some(serde_json::to_value(&json_val).unwrap_or_default()),
                None,
                None,
            )
        }
        "hocr" | "html" => {
            let hocr_str = format_hocr(&result).unwrap_or_default();
            (result.text.clone(), None, Some(hocr_str), None)
        }
        "tsv" => {
            let tsv_str = format_tsv(&result).unwrap_or_default();
            (result.text.clone(), None, None, Some(tsv_str))
        }
        _ => (result.text.clone(), None, None, None),
    };

    Ok(Json(OcrResponse {
        success: true,
        text,
        confidence: result.confidence as f64,
        language: params.lang,
        words_count,
        lines_count,
        json_output,
        hocr,
        tsv,
        processing_time_ms: elapsed,
    }))
}
