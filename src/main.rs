mod cli;

use anyhow::Result;
use cli::{commands::*, parse, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = parse();

    match cli.command {
        Commands::Extract {
            image_path,
            output,
            lang,
            preprocess,
            format,
            psm,
            confidence,
            engine,
            dict_correct,
            device,
            osd,
        } => {
            handle_extract(
                image_path,
                output,
                &lang,
                preprocess,
                &format,
                psm,
                confidence,
                &engine,
                dict_correct,
                &device,
                osd,
            )
            .await?;
        }
        Commands::Batch {
            input_dir,
            output_dir,
            lang,
            confidence,
            max_concurrent,
            engine,
            dict_correct,
            device,
        } => {
            handle_batch(
                input_dir,
                output_dir,
                &lang,
                confidence,
                max_concurrent,
                &engine,
                dict_correct,
                device.as_str(),
            )
            .await?;
        }
        Commands::Layout { image_path, output } => {
            handle_layout(image_path, output).await?;
        }
        Commands::ListLanguages => {
            handle_list_languages().await?;
        }
        Commands::Check => {
            handle_check().await?;
        }
        Commands::Info => {
            handle_info().await?;
        }
        Commands::Validate { config_file } => {
            handle_validate(config_file).await?;
        }
        Commands::Train {
            epochs,
            batch_size,
            learning_rate,
            engine,
            checkpoint_dir,
            distortion,
        } => {
            handle_train(epochs, batch_size, learning_rate, engine, checkpoint_dir, distortion)
                .await?;
        }
        Commands::Benchmark { samples, distortion } => {
            handle_benchmark(samples, distortion).await?;
        }
        #[cfg(feature = "web-api")]
        Commands::Serve {
            host,
            port,
            max_upload_size,
        } => {
            use ocr::server::{run_server, ServerConfig};
            let config = ServerConfig {
                host,
                port,
                max_upload_size_mb: max_upload_size,
            };
            run_server(config).await?;
        }
    }

    Ok(())
}
