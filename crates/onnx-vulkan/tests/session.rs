//! The public API on real models, exercised the way a caller would.
//!
//! Every test skips when its model is not staged: `scripts/fetch-models.sh`
//! downloads hundreds of megabytes, and a checkout without them must still be
//! able to run `cargo test`. A skip prints why, so a green run that tested
//! nothing does not read as a green run that tested something.

use onnx_vulkan::{Dim, ElementType, HostTensor, Session};
use std::path::PathBuf;

/// A model path resolved against the workspace root: `cargo test` runs with the
/// **package** directory as its working directory, so a repo-relative path
/// would silently miss and every test would skip.
fn staged(relative: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let path = root.join(relative);
    if path.exists() {
        return Some(path);
    }
    eprintln!("skipped: {} is not staged", path.display());
    None
}

/// Builds every input from the session's own metadata, symbols pinned to 1.
fn inputs(session: &Session) -> Vec<(String, HostTensor)> {
    session
        .inputs()
        .iter()
        .map(|info| {
            let shape: Vec<i64> = info
                .shape
                .as_ref()
                .expect("input shape")
                .iter()
                .map(|d| match d {
                    Dim::Fixed(n) => *n,
                    Dim::Symbol(_) | Dim::Unknown => 1,
                })
                .collect();
            let count: usize = shape.iter().product::<i64>().max(0) as usize;
            let tensor = match info.dtype {
                Some(ElementType::Float32) => HostTensor::from_f32(shape, &vec![0.05; count]),
                Some(ElementType::Int64) | Some(ElementType::Int32) => {
                    HostTensor::from_i64(shape, &vec![0; count])
                }
                other => panic!("input '{}': dtype {other:?}", info.name),
            };
            (info.name.clone(), tensor)
        })
        .collect()
}

#[test]
fn loads_and_runs_a_model_from_a_path() {
    let Some(path) = staged("models/zoo/mobilenetv2/mobilenetv2-12.onnx") else {
        return;
    };
    let session = Session::load(path).expect("load");

    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.outputs().len(), 1);
    assert_eq!(session.inputs()[0].dtype, Some(ElementType::Float32));
    // the exported shape is symbolic in its first dimension, and the API says so
    // instead of pretending it is 1
    assert!(matches!(
        session.inputs()[0].shape.as_ref().expect("shape")[0],
        Dim::Symbol(_)
    ));

    let bound = inputs(&session);
    let run = session.run(bound).expect("run");
    let output = run.get("output").expect("output");
    assert_eq!(output.shape, [1, 1000]);
    assert_eq!(output.dtype, ElementType::Float32 as i32);
    assert!(output.to_f32().expect("f32").iter().all(|v| v.is_finite()));
    run.finish();
}

/// Weights and pipelines outlive a run: this is what makes a session worth
/// keeping instead of reloading the file.
#[test]
fn a_second_run_reuses_the_sessions_resources() {
    let Some(path) = staged("models/zoo/yolov8/yolov8n.onnx") else {
        return;
    };
    let session = Session::load(path).expect("load");
    let bound = inputs(&session);

    let first = session.run(bound.clone()).expect("first run");
    let a = first.get("output0").expect("output").to_f32().expect("f32");
    first.finish();

    let second = session.run(bound).expect("second run");
    let b = second
        .get("output0")
        .expect("output")
        .to_f32()
        .expect("f32");
    second.finish();

    // same inputs, same session: bit-identical, and no shader recompiled
    assert_eq!(a, b);
}

/// All-or-nothing, and the error names what is missing rather than the first
/// thing that happened to be missing.
#[test]
fn an_unsupported_model_is_refused_with_the_whole_list() {
    let Some(path) = staged("models/zoo/golden/FasterRCNN-12/FasterRCNN-12.onnx") else {
        return;
    };
    match Session::load(path) {
        Err(onnx_vulkan::Error::Unsupported(message)) => {
            assert!(
                message.contains("types are not implemented"),
                "the error must enumerate: {message}"
            );
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("a model with uncovered ops was accepted"),
    }
}

#[test]
fn an_unknown_input_name_is_refused_before_the_gpu_sees_it() {
    let Some(path) = staged("models/zoo/mobilenetv2/mobilenetv2-12.onnx") else {
        return;
    };
    let session = Session::load(path).expect("load");
    let bogus = [("not_an_input", HostTensor::from_f32(vec![1], &[0.0]))];
    assert!(matches!(
        session.run(bogus),
        Err(onnx_vulkan::Error::NoSuchValue(_))
    ));
    // and a missing required input is the same class of mistake
    let empty: [(&str, HostTensor); 0] = [];
    assert!(matches!(
        session.run(empty),
        Err(onnx_vulkan::Error::NoSuchValue(_))
    ));
}
