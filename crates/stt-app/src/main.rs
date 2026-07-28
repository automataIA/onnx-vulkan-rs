//! STT host app: wav → nemo128 (log-mel) → encoder → TDT greedy → text.
//!
//! Usage: `stt-app <file.wav> [model_dir]`
//! ORT loaded at runtime (`load-dynamic`); default: workspace third_party/
//! override with `ORT_DYLIB_PATH`.

mod audio;
mod plugin;
mod tdt;
mod vocab;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

fn default_ort_dylib() -> PathBuf {
    let name = if cfg!(windows) {
        "win-x64/lib/onnxruntime.dll"
    } else {
        "linux-x64/lib/libonnxruntime.so"
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/onnxruntime")
        .join(name)
}

fn main() -> Result<()> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let wav_path = PathBuf::from(args.next().context("uso: stt-app <file.wav> [model_dir]")?);
    let model_dir = PathBuf::from(args.next().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/parakeet-tdt-0.6b-v3-onnx")
            .to_string_lossy()
            .into_owned()
    }));

    let dylib = std::env::var("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_ort_dylib());
    ort::init_from(dylib.to_string_lossy().as_ref()).commit()?;

    let vocab = vocab::Vocab::load(&model_dir.join("vocab.txt"))?;
    let samples = audio::read_wav_mono_16k(&wav_path)?;
    log::info!(
        "wav: {} campioni ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Vulkan EP plugin: active if present next to the executable (or VULKAN_EP_PATH),
    // disable with STT_NO_VULKAN=1. Only encoder uses EP.
    let vulkan_ep = if std::env::var_os("STT_NO_VULKAN").is_some() {
        None
    } else {
        let path = std::env::var("VULKAN_EP_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let lib = if cfg!(windows) {
                    "onnxruntime_ep_vulkan.dll"
                } else {
                    "libonnxruntime_ep_vulkan.so"
                };
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.join(lib)))
                    .unwrap_or_else(|| PathBuf::from(lib))
            });
        path.exists().then_some(path)
    };

    let mut features_session =
        Session::builder()?.commit_from_file(model_dir.join("nemo128.onnx"))?;
    let mut encoder_builder = Session::builder()?;
    if let Some(path) = &vulkan_ep {
        plugin::register(path)?;
        let n = plugin::append_to_session(&mut encoder_builder)?;
        log::info!(
            "VulkanEP registrato da {}: {n} device aggiunti alla sessione encoder",
            path.display()
        );
    } else {
        log::info!("VulkanEP non attivo (plugin assente o STT_NO_VULKAN)");
    }
    let mut encoder =
        encoder_builder.commit_from_file(model_dir.join("encoder-model.int8.onnx"))?;
    let mut decoder =
        Session::builder()?.commit_from_file(model_dir.join("decoder_joint-model.int8.onnx"))?;

    // 1) log-mel
    let n = samples.len();
    let feat_outputs = features_session.run(ort::inputs![
        "waveforms" => Tensor::from_array(([1usize, n], samples))?,
        "waveforms_lens" => Tensor::from_array(([1usize], vec![n as i64]))?,
    ])?;
    let (feat_shape, features) = feat_outputs["features"].try_extract_tensor::<f32>()?;
    let (_, feat_lens) = feat_outputs["features_lens"].try_extract_tensor::<i64>()?;
    let feat_shape: Vec<usize> = feat_shape.iter().map(|&d| d as usize).collect();
    log::info!("features: {feat_shape:?}, len={}", feat_lens[0]);

    // 2) encoder — `STT_BENCH=N` runs N iterations (warm-up + steady-state) for a
    // valid perf measurement: first run includes one-time pipeline compilation
    // pipeline (WGSL→SPIR-V); the steady-state ms are the subsequent iterations.
    let bench = std::env::var("STT_BENCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    // warm-up run (output discarded: holds &mut borrow of session)
    for it in 0..bench - 1 {
        let enc_start = std::time::Instant::now();
        let out = encoder.run(ort::inputs![
            "audio_signal" => Tensor::from_array((feat_shape.clone(), features.to_vec()))?,
            "length" => Tensor::from_array(([1usize], vec![feat_lens[0]]))?,
        ])?;
        log::info!(
            "encoder iter {}/{bench}: {:.1} ms",
            it + 1,
            enc_start.elapsed().as_secs_f64() * 1000.0
        );
        drop(out);
    }
    // final run (output retained for decoding)
    let enc_start = std::time::Instant::now();
    let enc_outputs = encoder.run(ort::inputs![
        "audio_signal" => Tensor::from_array((feat_shape.clone(), features.to_vec()))?,
        "length" => Tensor::from_array(([1usize], vec![feat_lens[0]]))?,
    ])?;
    log::info!(
        "encoder iter {bench}/{bench}: {:.1} ms",
        enc_start.elapsed().as_secs_f64() * 1000.0
    );
    let (enc_shape, encodings) = enc_outputs["outputs"].try_extract_tensor::<f32>()?;
    let (_, enc_lens) = enc_outputs["encoded_lengths"].try_extract_tensor::<i64>()?;
    // shape [1, 1024, T]
    let t_total = enc_shape[2] as usize;
    let t_len = (enc_lens[0] as usize).min(t_total);
    log::info!("encoder out: {enc_shape:?}, t_len={t_len}");

    // 3) TDT greedy decoding
    let encodings = encodings.to_vec();
    let mut tdt = tdt::TdtDecoder::new(&mut decoder, vocab.size(), vocab.blank_idx);
    let token_ids = tdt.greedy_decode(&encodings, t_len)?;

    // 4) text
    println!("{}", vocab.decode(&token_ids));

    // plugin unregistration: only after session destruction
    // (outputs hold session borrow: release first)
    drop(feat_outputs);
    drop(enc_outputs);
    drop(features_session);
    drop(encoder);
    drop(decoder);
    if vulkan_ep.is_some() {
        plugin::unregister()?;
    }
    Ok(())
}
