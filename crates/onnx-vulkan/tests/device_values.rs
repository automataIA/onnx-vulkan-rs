//! Outputs kept in VRAM and bound to the next run, against the same chain
//! carried through host memory.
//!
//! `models/affine.onnx` is `y = x · scale + shift` over `[1, 8192]` f32, with
//! `scale = (i mod 7 + 1) / 3` and `shift` evenly spaced in `[-1, 1]`; written
//! with `onnx.helper` (opset 17). Two sessions of it form a chain whose second
//! stage reads the first one's output, which is the shape of an image encoder
//! feeding a decoder.

use onnx_vulkan::{ElementType, HostTensor, Session};
use std::path::PathBuf;

fn affine() -> Session {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/models/affine.onnx");
    Session::load(path).expect("load affine.onnx")
}

fn input() -> HostTensor {
    let x: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.37).sin() * 4.0).collect();
    HostTensor::from_f32(vec![1, 8192], &x)
}

#[test]
fn a_resident_output_feeds_another_session_like_its_host_copy() {
    let (first, second) = (affine(), affine());

    let run = first.run([("x", input())]).expect("first, host");
    let y = run.get("y").expect("y");
    run.finish();
    let run = second.run([("x", y)]).expect("second, host");
    let staged = run.get("y").expect("staged");
    run.finish();

    let mut run = first.run([("x", input())]).expect("first, resident");
    let y = run.take("y").expect("take y");
    assert_eq!(y.shape(), [1, 8192]);
    assert_eq!(y.dtype(), Some(ElementType::Float32));
    // taken, so no longer the run's to read
    assert!(run.get("y").is_err());
    run.finish();
    // borrowed, not consumed: the same features serve every prompt
    for _ in 0..2 {
        let none: [(&str, HostTensor); 0] = [];
        let run = second
            .run_with_device(none, &[("x", &y)])
            .expect("second, resident");
        let resident = run.get("y").expect("resident");
        run.finish();
        assert_eq!(resident.shape, staged.shape);
        assert_eq!(
            resident.data, staged.data,
            "bit-exact against the host chain"
        );
    }
}

#[test]
fn names_are_checked_on_both_sides_of_the_handoff() {
    let session = affine();
    let mut run = session.run([("x", input())]).expect("run");
    assert!(run.take("t").is_err(), "an intermediate is not an output");
    let y = run.take("y").expect("take y");
    run.finish();

    assert!(
        session
            .run_with_device([("x", input())], &[("z", &y)])
            .is_err(),
        "an input the graph does not have"
    );
    let none: [(&str, HostTensor); 0] = [];
    assert!(
        session.run_with_device(none, &[]).is_err(),
        "a required input supplied by neither side"
    );
}
