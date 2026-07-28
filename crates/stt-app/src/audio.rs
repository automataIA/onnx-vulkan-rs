//! WAV reading: 16 kHz required; stereo averaged to mono; samples normalized to f32.

use anyhow::{Context, Result, bail};
use std::path::Path;

pub const SAMPLE_RATE: u32 = 16_000;

pub fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening wav {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "sample rate {} not supported (expected {SAMPLE_RATE} Hz)",
            spec.sample_rate
        );
    }
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    if channels == 1 {
        Ok(samples)
    } else {
        Ok(samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect())
    }
}
