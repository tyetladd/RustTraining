use anyhow::Result;
use clap::Parser;
use mogilefs_rs::config;

#[derive(Parser, Debug)]
#[command(
    name = "mogilefsd",
    about = "Single-binary MogileFS tracker + storage server"
)]
struct Args {
    /// Path to config file (TOML)
    #[arg(short, long, default_value = "mogilefsd.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let cfg = config::Config::load(&args.config)?;
    let _handle = mogilefs_rs::spawn(cfg).await?;

    // Run forever; ^C / SIGTERM ends the process (and thus all spawned tasks).
    std::future::pending::<()>().await;
    Ok(())
}
