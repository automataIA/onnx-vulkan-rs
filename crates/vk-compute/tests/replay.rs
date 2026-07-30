//! Capturing a stream and issuing it again.
//!
//! What a replay has to be worth: the same dispatches on the same memory,
//! without the caller deciding them a second time. So the test drives the
//! second step through `replay` alone — it never calls `stream_dispatch` again
//! — and changes both the inputs in VRAM and a push constant, because those are
//! exactly the two things a decode step changes between tokens.

use vk_compute::{StreamOp, VkContext, compile_wgsl};

fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn from_bytes<T: Copy>(data: &[u8]) -> Vec<T> {
    assert_eq!(data.len() % std::mem::size_of::<T>(), 0);
    let n = data.len() / std::mem::size_of::<T>();
    let mut out = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr().cast::<T>(), out.as_mut_ptr(), n);
        out.set_len(n);
    }
    out
}

/// `out[i] = a[i] * scale`, over the first `n` elements. `n` stands in for the
/// cache length: a scalar that grows with the step.
const SCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { n: u32, scale: f32 }
var<immediate> pc: Push;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < pc.n) {
        out[gid.x] = a[gid.x] * pc.scale;
    }
}
"#;

fn push_bytes(n: u32, scale: f32) -> Vec<u8> {
    let mut bytes = n.to_le_bytes().to_vec();
    bytes.extend_from_slice(&scale.to_le_bytes());
    bytes
}

#[test]
fn a_replayed_step_is_the_step_it_captured() {
    let ctx = VkContext::new().expect("Vulkan context");
    let n = 256usize;
    let spirv = compile_wgsl(SCALE_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 2, 8).unwrap();
    let size = (n * 4) as u64;
    let buf_a = ctx.create_storage_buffer(size).unwrap();
    let buf_mid = ctx.create_storage_buffer(size).unwrap();
    let buf_out = ctx.create_storage_buffer(size).unwrap();
    let groups = [(n as u32).div_ceil(64), 1, 1];

    // Step one, captured: an upload, then two chained dispatches. Chained on
    // purpose — the second reads what the first wrote, so a replay that lost
    // the barriers would race and the result would not be stable.
    let first: Vec<f32> = (0..n).map(|i| i as f32).collect();
    ctx.begin_capture();
    ctx.stream_upload(&buf_a, as_bytes(&first)).unwrap();
    ctx.stream_dispatch(
        &pipeline,
        &[&buf_a, &buf_mid],
        &push_bytes(n as u32, 2.0),
        groups,
    )
    .unwrap();
    ctx.stream_dispatch(
        &pipeline,
        &[&buf_mid, &buf_out],
        &push_bytes(n as u32, 3.0),
        groups,
    )
    .unwrap();
    let mut plan = ctx.end_capture().expect("a capture was open");
    ctx.flush().unwrap();
    let captured: Vec<f32> = from_bytes(&ctx.download(&buf_out).unwrap());
    for (i, value) in captured.iter().enumerate() {
        assert_eq!(*value, i as f32 * 6.0, "captured step at {i}");
    }

    assert_eq!(plan.len(), 3, "one upload and two dispatches");
    assert!(matches!(plan[0], StreamOp::Upload(_)));

    // Step two, replayed: different bytes uploaded, a different scale, and half
    // the elements — the plan is data, so the caller patches it in place.
    let second: Vec<f32> = (0..n).map(|i| 1000.0 - i as f32).collect();
    let half = n / 2;
    match &mut plan[0] {
        StreamOp::Upload(upload) => upload.bytes = as_bytes(&second).to_vec(),
        _ => panic!("the first op is not the upload"),
    }
    match &mut plan[2] {
        StreamOp::Dispatch(dispatch) => dispatch.push = push_bytes(half as u32, 5.0),
        _ => panic!("the third op is not a dispatch"),
    }
    ctx.replay(&plan).unwrap();
    ctx.flush().unwrap();
    let replayed: Vec<f32> = from_bytes(&ctx.download(&buf_out).unwrap());
    for (i, value) in replayed.iter().enumerate().take(half) {
        assert_eq!(*value, (1000.0 - i as f32) * 10.0, "replayed step at {i}");
    }
    // past the shortened dispatch the previous step's output is still there,
    // which is what says the grid came from the patched op and not the capture
    for (i, value) in replayed.iter().enumerate().skip(half) {
        assert_eq!(*value, i as f32 * 6.0, "outside the replayed range at {i}");
    }

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_mid);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

/// A capture must not change what the captured step does: it is recording, not
/// deferring. Two identical runs, one captured and one not, produce the same
/// bytes and the same number of dispatches.
#[test]
fn capturing_a_step_does_not_change_it() {
    let ctx = VkContext::new().expect("Vulkan context");
    let n = 64usize;
    let spirv = compile_wgsl(SCALE_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 2, 8).unwrap();
    let size = (n * 4) as u64;
    let buf_a = ctx.create_storage_buffer(size).unwrap();
    let buf_out = ctx.create_storage_buffer(size).unwrap();
    let values: Vec<f32> = (0..n).map(|i| i as f32 + 0.25).collect();
    ctx.upload(&buf_a, as_bytes(&values)).unwrap();
    let groups = [(n as u32).div_ceil(64), 1, 1];

    let run = || {
        ctx.stream_dispatch(
            &pipeline,
            &[&buf_a, &buf_out],
            &push_bytes(n as u32, 7.0),
            groups,
        )
        .unwrap();
        ctx.flush().unwrap();
        from_bytes::<f32>(&ctx.download(&buf_out).unwrap())
    };
    let plain = run();
    ctx.begin_capture();
    let while_capturing = run();
    let plan = ctx.end_capture().expect("a capture was open");

    assert_eq!(plain, while_capturing);
    assert_eq!(plan.len(), 1);
    assert!(ctx.end_capture().is_none(), "the capture is closed");

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

/// `same_shape` is what a plan is validated with: two steps that differ only in
/// their scalars replay the same way, two that differ in structure do not.
#[test]
fn the_shape_of_an_op_ignores_what_a_step_changes() {
    let ctx = VkContext::new().expect("Vulkan context");
    let n = 64usize;
    let spirv = compile_wgsl(SCALE_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 2, 8).unwrap();
    let size = (n * 4) as u64;
    let buf_a = ctx.create_storage_buffer(size).unwrap();
    let buf_b = ctx.create_storage_buffer(size).unwrap();
    let buf_out = ctx.create_storage_buffer(size).unwrap();
    let groups = [1, 1, 1];

    let capture = |inputs: [&vk_compute::GpuBuffer; 2], scale: f32, groups: [u32; 3]| {
        ctx.begin_capture();
        ctx.stream_dispatch(&pipeline, &inputs, &push_bytes(n as u32, scale), groups)
            .unwrap();
        ctx.end_capture().expect("a capture was open")
    };
    let base = capture([&buf_a, &buf_out], 1.0, groups);
    let other_scale = capture([&buf_a, &buf_out], 2.0, groups);
    let other_buffer = capture([&buf_b, &buf_out], 1.0, groups);
    let other_grid = capture([&buf_a, &buf_out], 1.0, [2, 1, 1]);

    assert!(
        base[0].same_shape(&other_scale[0]),
        "a push constant is what a step changes"
    );
    assert!(
        !base[0].same_shape(&other_buffer[0]),
        "a different buffer is a different plan"
    );
    assert!(
        !base[0].same_shape(&other_grid[0]),
        "a different grid is a different plan"
    );

    ctx.flush().unwrap();
    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}
