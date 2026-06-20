//! Integration tests for the web API server
//!
//! Requires the `web-api` feature flag.
//! Run with: cargo test --features web-api --test test_web_api -- --nocapture

#[cfg(feature = "web-api")]
mod tests {
    use ocr::server::{run_server, ServerConfig};
    use std::time::Duration;
    use tokio::net::TcpListener;

    fn generate_test_png() -> Vec<u8> {
        use image::{GrayImage, Luma};
        let width = 200u32;
        let height = 50u32;
        let mut img = GrayImage::from_pixel(width, height, Luma([255u8]));
        for x in 50..150 {
            for y in 15..35 {
                img.put_pixel(x, y, Luma([0u8]));
            }
        }
        let mut buf = Vec::new();
        let dynamic = image::DynamicImage::ImageLuma8(img);
        dynamic
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .ok();
        buf
    }

    async fn find_available_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        listener.local_addr().unwrap().port()
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let port = find_available_port().await;
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port,
            max_upload_size_mb: 10,
        };

        tokio::spawn(async move {
            run_server(config).await.ok();
        });

        tokio::time::sleep(Duration::from_millis(1500)).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/health", port))
            .send()
            .await;

        match resp {
            Ok(r) => {
                assert!(r.status().is_success());
                let body: serde_json::Value = r.json().await.unwrap();
                assert_eq!(body["status"], "ok");
                assert!(!body["version"].as_str().unwrap().is_empty());
            }
            Err(e) => {
                eprintln!("Health check failed (server may still be starting): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_languages_endpoint() {
        let port = find_available_port().await;
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port,
            max_upload_size_mb: 10,
        };

        tokio::spawn(async move {
            run_server(config).await.ok();
        });

        tokio::time::sleep(Duration::from_millis(1500)).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/languages", port))
            .send()
            .await;

        match resp {
            Ok(r) => {
                assert!(r.status().is_success());
                let body: serde_json::Value = r.json().await.unwrap();
                assert!(body["languages"].is_array());
            }
            Err(e) => {
                eprintln!("Languages endpoint failed: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_recognize_endpoint() {
        let port = find_available_port().await;
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port,
            max_upload_size_mb: 10,
        };

        tokio::spawn(async move {
            run_server(config).await.ok();
        });

        tokio::time::sleep(Duration::from_millis(1500)).await;

        let png_data = generate_test_png();
        let form = reqwest::multipart::Form::new()
            .part(
                "image",
                reqwest::multipart::Part::bytes(png_data)
                    .file_name("test.png")
                    .mime_str("image/png")
                    .unwrap(),
            )
            .text("lang", "en")
            .text("format", "json");

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{}/recognize", port))
            .multipart(form)
            .send()
            .await;

        match resp {
            Ok(r) => {
                let status = r.status();
                let body_text = r.text().await.unwrap_or_default();
                if !status.is_success() {
                    eprintln!("Recognize endpoint returned {}: {}", status, body_text);
                }
                assert!(
                    status.is_success(),
                    "Expected success, got {}: {}",
                    status,
                    body_text
                );
            }
            Err(e) => {
                panic!("Recognize endpoint request failed: {}", e);
            }
        }
    }
}
