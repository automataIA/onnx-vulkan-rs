//! An image-text chain (SAM 3: vision → text → detector) in one process, run
//! two ways: with every handoff carried through host memory, and with the
//! encoders' outputs kept in VRAM ([`Run::take`], [`Session::run_with_device`]).
//!
//! ```text
//! cargo run --release -p onnx-vulkan --example sam3_chain -- \
//!     VISION.onnx TEXT.onnx DETECTOR.onnx CASE [--iters N] [--dump DIR] [--device-outputs NAME]
//! ```
//!
//! `CASE/vision/input_*.pb` are the vision inputs, `CASE/prompts/*/input_*.pb`
//! each prompt's text inputs (directories in name order; files bound to the
//! graph's inputs by position when they carry no name). The detector reads, by
//! name, the vision outputs, the text outputs and the text inputs it declares
//! as inputs (the attention mask).
//!
//! The vision encoder runs once per image, then the text encoder and the
//! detector once per prompt. Each iteration runs both ways, in alternating
//! order, and their detector outputs must agree bit for bit. Timed: the first
//! prompt (vision + text + detector, outputs read on the host) and each later
//! prompt (text + detector). As in `model-runner`, host outputs are read into
//! tensors kept across iterations, and the host copies a caller of
//! [`Session::run`] makes of its inputs (the image features for every prompt,
//! in the host chain) are not timed. The first
//! iteration compiles pipelines and is not reported. `--dump` writes the
//! resident detector outputs of the last iteration as `.npy`.
//!
//! `--device-outputs NAME` times the detector the way a framework returning
//! device tensors is timed: every output but `NAME` stays in VRAM, and reading
//! `NAME` on the host (pick a tiny one) is the wait for the GPU, as
//! `torch.cuda.synchronize()` is. Only `NAME` is then compared between the two
//! ways.

use onnx_vulkan::{DeviceValue, ElementType, HostTensor, Run, Session};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Error = Box<dyn std::error::Error>;

/// Reads `dir/input_*.pb`, naming unnamed tensors after the session's inputs.
fn read_inputs(dir: &Path, session: &Session) -> Result<Vec<(String, HostTensor)>, Error> {
    let mut inputs = Vec::new();
    loop {
        let path = dir.join(format!("input_{}.pb", inputs.len()));
        if !path.exists() {
            break;
        }
        let (name, tensor) = onnx_vulkan_frontend::read_tensor_proto(&std::fs::read(&path)?)?;
        let name = if name.is_empty() {
            session
                .inputs()
                .get(inputs.len())
                .ok_or_else(|| format!("{}: no input at this position", path.display()))?
                .name
                .clone()
        } else {
            name
        };
        inputs.push((
            name,
            HostTensor::new(tensor.dtype, tensor.shape, tensor.data),
        ));
    }
    if inputs.is_empty() {
        return Err(format!("{}: no input_*.pb", dir.display()).into());
    }
    Ok(inputs)
}

/// Names of `from`'s outputs that `to` reads.
fn handoff(from: &Session, to: &Session) -> Vec<String> {
    from.outputs()
        .iter()
        .filter(|o| to.inputs().iter().any(|i| i.name == o.name))
        .map(|o| o.name.clone())
        .collect()
}

/// Host tensors kept across iterations for one way of running the chain, as an
/// application reading every frame would keep them: a fresh 67 MB output pays
/// its page faults on every run (see `model-runner`'s standalone loop).
#[derive(Default)]
struct Buffers {
    vision: Vec<HostTensor>,
    text: Vec<HostTensor>,
    /// Per prompt, the detector's outputs.
    detector: Vec<Vec<HostTensor>>,
}

fn empty(count: usize) -> Vec<HostTensor> {
    vec![HostTensor::new(0, Vec::new(), Vec::new()); count]
}

/// Ends an encoder's run: in `host`, its outputs the detector reads, or, when
/// resident, those outputs taken in VRAM.
fn hand_over(
    mut run: Run<'_>,
    names: &[String],
    resident: bool,
    host: &mut Vec<HostTensor>,
) -> Result<Vec<(String, DeviceValue)>, Error> {
    let mut device = Vec::new();
    if resident {
        for name in names {
            device.push((name.clone(), run.take(name)?));
        }
    } else {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        if host.len() != names.len() {
            *host = empty(names.len());
        }
        run.get_many_into(&refs, host)?;
    }
    run.finish();
    Ok(device)
}

struct Chain {
    vision: Session,
    text: Session,
    detector: Session,
    vision_inputs: Vec<(String, HostTensor)>,
    prompts: Vec<(String, Vec<(String, HostTensor)>)>,
}

/// One image through the chain: per prompt, the elapsed ms; the detector's
/// outputs are left in `buffers.detector`.
fn image(
    chain: &Chain,
    resident: bool,
    sync: Option<&str>,
    buffers: &mut Buffers,
) -> Result<Vec<f64>, Error> {
    let to_detector = |from: &Session| handoff(from, &chain.detector);
    let (from_vision, from_text) = (to_detector(&chain.vision), to_detector(&chain.text));
    let outputs: Vec<String> = chain
        .detector
        .outputs()
        .iter()
        .map(|o| o.name.clone())
        .collect();
    let outputs: Vec<&str> = outputs.iter().map(String::as_str).collect();
    buffers
        .detector
        .resize_with(chain.prompts.len(), || empty(outputs.len()));

    // the caller's copy of its own inputs, outside the timer as in model-runner
    let vision_inputs = chain.vision_inputs.clone();
    let start = Instant::now();
    let vision = chain.vision.run(vision_inputs)?;
    let image = hand_over(vision, &from_vision, resident, &mut buffers.vision)?;
    let mut vision_ms = start.elapsed().as_secs_f64() * 1e3;

    let mut times = Vec::new();
    for (prompt, (_, text_inputs)) in chain.prompts.iter().enumerate() {
        // untimed: `run` takes its host inputs by value, so a host chain must
        // copy the image features for every prompt
        let mut host: Vec<(String, HostTensor)> = text_inputs
            .iter()
            .filter(|(name, _)| chain.detector.inputs().iter().any(|i| &i.name == name))
            .cloned()
            .collect();
        if !resident {
            host.extend(
                from_vision
                    .iter()
                    .cloned()
                    .zip(buffers.vision.iter().cloned()),
            );
        }
        let text_inputs = text_inputs.clone();

        let start = Instant::now();
        let text = chain.text.run(text_inputs)?;
        let words = hand_over(text, &from_text, resident, &mut buffers.text)?;
        if !resident {
            host.extend(from_text.iter().cloned().zip(buffers.text.iter().cloned()));
        }
        let device: Vec<(&str, &DeviceValue)> = image
            .iter()
            .chain(&words)
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        let mut run = chain.detector.run_with_device(host, &device)?;
        match sync {
            Some(sync) => {
                // kept alive past the wait, as the caller's device tensors are
                let mut kept = Vec::new();
                for name in outputs.iter().filter(|name| **name != sync) {
                    kept.push(run.take(name)?);
                }
                run.get_many_into(&[sync], &mut buffers.detector[prompt][..1])?;
                drop(kept);
            }
            None => run.get_many_into(&outputs, &mut buffers.detector[prompt])?,
        }
        run.finish();
        times.push(start.elapsed().as_secs_f64() * 1e3 + vision_ms);
        vision_ms = 0.0;
    }
    Ok(times)
}

fn write_npy(path: &Path, tensor: &HostTensor) -> Result<(), Error> {
    let descr = match tensor.dtype {
        d if d == ElementType::Float32 as i32 => "<f4",
        d if d == ElementType::Uint8 as i32 => "|u1",
        d => return Err(format!("{}: dtype {d} is neither f32 nor u8", path.display()).into()),
    };
    let dims: Vec<String> = tensor.shape.iter().map(i64::to_string).collect();
    let shape = match dims.len() {
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&tensor.data);
    std::fs::write(path, bytes)?;
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let (mut iters, mut dump, mut sync) = (11, None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--iters" => iters = it.next().ok_or("--iters N")?.parse()?,
            "--dump" => dump = Some(PathBuf::from(it.next().ok_or("--dump DIR")?)),
            "--device-outputs" => sync = Some(it.next().ok_or("--device-outputs NAME")?.clone()),
            _ => positional.push(arg.clone()),
        }
    }
    let [vision, text, detector, case] = positional.as_slice() else {
        return Err(
            "usage: sam3_chain VISION TEXT DETECTOR CASE [--iters N] [--dump DIR] \
                    [--device-outputs NAME]"
                .into(),
        );
    };
    if sync.is_some() && dump.is_some() {
        return Err("--dump needs every output on the host: drop --device-outputs".into());
    }
    if iters < 2 {
        return Err("--iters must be at least 2: the first one is not reported".into());
    }
    let case = PathBuf::from(case);
    let (vision, text, detector) = (
        Session::load(vision)?,
        Session::load(text)?,
        Session::load(detector)?,
    );
    let vision_inputs = read_inputs(&case.join("vision"), &vision)?;
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(case.join("prompts"))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    dirs.sort();
    let mut prompts = Vec::new();
    for dir in dirs {
        let name = dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        prompts.push((name, read_inputs(&dir, &text)?));
    }
    println!(
        "vision → detector: {:?}; text → detector: {:?}; {} prompts",
        handoff(&vision, &detector),
        handoff(&text, &detector),
        prompts.len()
    );
    let chain = Chain {
        vision,
        text,
        detector,
        vision_inputs,
        prompts,
    };

    // [mode][iteration] → per-prompt ms
    let mut times: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
    let mut buffers: [Buffers; 2] = Default::default();
    for iteration in 0..iters {
        let order = if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for resident in order {
            let ms = image(
                &chain,
                resident,
                sync.as_deref(),
                &mut buffers[resident as usize],
            )?;
            if iteration > 0 {
                times[resident as usize].push(ms);
            }
        }
        let [staged, resident] = &buffers;
        for (prompt, (s, r)) in staged.detector.iter().zip(&resident.detector).enumerate() {
            if s.iter()
                .zip(r)
                .any(|(s, r)| s.shape != r.shape || s.data != r.data)
            {
                return Err(format!(
                    "iteration {iteration}, prompt {}: resident outputs differ from staged",
                    chain.prompts[prompt].0
                )
                .into());
            }
        }
    }
    println!(
        "resident == staged, bit for bit: {iters} iterations × {} prompts",
        chain.prompts.len()
    );
    if let Some(sync) = &sync {
        println!("detector outputs left in VRAM; only {sync} read on the host");
    }
    for (mode, name) in [(0, "staged"), (1, "resident")] {
        let mut first: Vec<f64> = times[mode].iter().map(|t| t[0]).collect();
        let mut later: Vec<f64> = times[mode].iter().flat_map(|t| t[1..].to_vec()).collect();
        let mut total: Vec<f64> = times[mode].iter().map(|t| t.iter().sum()).collect();
        println!(
            "{name:>8}: first prompt {:.1} ms, later prompt {:.1} ms, image ({} prompts) {:.1} ms \
             (medians of {} iterations)",
            median(&mut first),
            if later.is_empty() {
                f64::NAN
            } else {
                median(&mut later)
            },
            chain.prompts.len(),
            median(&mut total),
            times[mode].len()
        );
    }
    if let Some(dir) = dump {
        let names: Vec<&str> = chain
            .detector
            .outputs()
            .iter()
            .map(|o| o.name.as_str())
            .collect();
        for ((prompt, _), outputs) in chain.prompts.iter().zip(&buffers[1].detector) {
            let dir = dir.join(prompt);
            std::fs::create_dir_all(&dir)?;
            for (name, tensor) in names.iter().zip(outputs) {
                write_npy(&dir.join(format!("{name}.npy")), tensor)?;
            }
        }
        println!("resident outputs: {}", dir.display());
    }
    Ok(())
}
