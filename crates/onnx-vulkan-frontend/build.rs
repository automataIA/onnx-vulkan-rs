//! Generates the ONNX protobuf types from the vendored schema.
//!
//! `protox` compiles the `.proto` in pure Rust: no `protoc` binary to install
//! or ship, which is the whole point of the standalone frontend. The schema
//! lives in `proto/onnx.proto` (ONNX 1.22, Apache-2.0) instead of being fetched
//! at build time, so a build is reproducible offline.

fn main() {
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    let descriptors =
        protox::compile(["proto/onnx.proto"], ["proto"]).expect("compilazione di proto/onnx.proto");
    prost_build::compile_fds(descriptors).expect("generazione dei tipi prost");
}
