use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Native CUDA PP-OCRv6 server: pages, lines, and glyphs"
)]
pub(crate) struct Cli {
    /// Compiled model bundle directory.
    #[arg(long, default_value = "models")]
    pub(crate) model_dir: PathBuf,
    /// HTTP bind address.
    #[arg(long, default_value = "127.0.0.1:8184")]
    pub(crate) listen: SocketAddr,
    /// Maximum decoded image data per request, in MiB.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..=256))]
    pub(crate) max_request_mib: u32,
    /// Maximum time waiting for an inference slot.
    #[arg(long, default_value_t = 30000, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) queue_timeout_ms: u64,
}

fn native_library() -> PathBuf {
    std::env::current_exe()
        .expect("executable path")
        .parent()
        .expect("binary directory")
        .join("../lib/libppocrv6_native.so")
}

// Fixed, validated PP-OCRv6 medium deployment profile. These are model/runtime
// settings, not independent server switches.
pub(crate) struct Settings {
    pub(crate) lines_max_images: usize,
    pub(crate) lines_max_total_bytes: usize,
    pub(crate) lines_max_total_pixels: u64,
    pub(crate) native_lib: PathBuf,
    pub(crate) engine: String,
    pub(crate) weight: String,
    pub(crate) bias: String,
    pub(crate) characters: String,
    pub(crate) glyph_character_policy: String,
    pub(crate) glyph_score_mode: String,
    pub(crate) width: i32,
    pub(crate) batch_size: i32,
    pub(crate) max_width: i32,
    pub(crate) max_batch_size: i32,
    pub(crate) max_images: usize,
    pub(crate) max_request_bytes: usize,
    pub(crate) max_image_bytes: usize,
    pub(crate) max_image_pixels: u64,
    pub(crate) vocab_size: i32,
    pub(crate) hidden_size: i32,
    pub(crate) vocab_tile_size: i32,
    pub(crate) blank_id: i32,
    pub(crate) warmup_runs: i32,
    pub(crate) worker_permits: usize,
    pub(crate) queue_timeout_ms: u64,
    pub(crate) ocr_det_engine: String,
    pub(crate) ocr_det_default_height: i32,
    pub(crate) ocr_det_default_width: i32,
    pub(crate) ocr_det_max_height: i32,
    pub(crate) ocr_det_max_width: i32,
    pub(crate) ocr_det_warmup_runs: i32,
    pub(crate) ocr_rec_engine: String,
    pub(crate) ocr_rec_default_width: i32,
    pub(crate) ocr_rec_batch_size: i32,
    pub(crate) ocr_rec_max_batch_size: i32,
    pub(crate) ocr_rec_max_width: i32,
    pub(crate) ocr_rec_warmup_runs: i32,
    pub(crate) ocr_rec_buckets: String,
    pub(crate) ocr_det_limit_side_len: u32,
    pub(crate) ocr_det_limit_type: String,
    pub(crate) ocr_det_max_side_limit: u32,
    pub(crate) ocr_det_thresh: f32,
    pub(crate) ocr_det_box_thresh: f32,
    pub(crate) ocr_det_unclip_ratio: f32,
    pub(crate) ocr_det_max_candidates: i32,
    pub(crate) ocr_det_min_size: i32,
    pub(crate) ocr_max_image_bytes: usize,
    pub(crate) ocr_max_image_pixels: u64,
    pub(crate) listen: SocketAddr,
}
impl Settings {
    pub(crate) fn from_cli(cli: Cli) -> Self {
        let path = |name: &str| cli.model_dir.join(name).to_string_lossy().into_owned();
        Self {
            lines_max_images: 128,
            lines_max_total_bytes: cli.max_request_mib as usize * 1024 * 1024,
            lines_max_total_pixels: 16_000_000,
            native_lib: native_library(),
            engine: path("recognition"),
            weight: path("classifier/weight.fp16.bin"),
            bias: path("classifier/bias.fp16.bin"),
            characters: path("classifier/characters.txt"),
            glyph_character_policy: "cjk_focus_fallback".into(),
            glyph_score_mode: "accepted".into(),
            width: 80,
            batch_size: 128,
            max_width: 80,
            max_batch_size: 128,
            max_images: 1024,
            max_request_bytes: cli.max_request_mib as usize * 1024 * 1024,
            max_image_bytes: 1024 * 1024,
            max_image_pixels: 1024 * 1024,
            vocab_size: 18710,
            hidden_size: 192,
            vocab_tile_size: 512,
            blank_id: 0,
            warmup_runs: 1,
            worker_permits: 1,
            queue_timeout_ms: cli.queue_timeout_ms,
            ocr_det_engine: path("detection"),
            ocr_det_default_height: 1280,
            ocr_det_default_width: 992,
            ocr_det_max_height: 1280,
            ocr_det_max_width: 1280,
            ocr_det_warmup_runs: 0,
            ocr_rec_engine: path("recognition"),
            ocr_rec_default_width: 3200,
            ocr_rec_batch_size: 8,
            ocr_rec_max_batch_size: 8,
            ocr_rec_max_width: 3200,
            ocr_rec_warmup_runs: 0,
            ocr_rec_buckets: "384,3200".into(),
            ocr_det_limit_side_len: 1280,
            ocr_det_limit_type: "max".into(),
            ocr_det_max_side_limit: 4000,
            ocr_det_thresh: 0.2,
            ocr_det_box_thresh: 0.45,
            ocr_det_unclip_ratio: 1.4,
            ocr_det_max_candidates: 3000,
            ocr_det_min_size: 3,
            ocr_max_image_bytes: cli.max_request_mib as usize * 1024 * 1024,
            ocr_max_image_pixels: 16_000_000,
            listen: cli.listen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paths_follow_one_bundle_and_ipv6_is_supported() {
        let settings = Settings::from_cli(
            Cli::try_parse_from(["server", "--model-dir", "/bundle", "--listen", "[::1]:8184"])
                .unwrap(),
        );
        assert_eq!(settings.engine, "/bundle/recognition");
        assert_eq!(settings.ocr_rec_engine, settings.engine);
        assert_eq!(settings.ocr_det_engine, "/bundle/detection");
        assert_eq!(settings.listen.to_string(), "[::1]:8184");
        assert_eq!(settings.max_batch_size, 128);
    }
    #[test]
    fn rejects_invalid_limits_and_removed_backend_flags() {
        for args in [
            vec!["server", "--max-request-mib", "0"],
            vec!["server", "--max-request-mib", "257"],
            vec!["server", "--queue-timeout-ms", "0"],
            vec!["server", "--engine", "legacy.trt"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
    }
}
