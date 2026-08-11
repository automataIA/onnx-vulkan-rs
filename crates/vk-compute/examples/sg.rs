//! Does a WGSL subgroup butterfly reduction survive naga 30 and the driver?
//!
//! `enable subgroups;` is **rejected** by naga 30 ("this enable-extension
//! specifies standard functionality which is not yet implemented in Naga",
//! wgpu#5555) — but the builtins themselves compile without it, because
//! `compile_wgsl` validates under `Capabilities::all()`. This checks the rest of
//! the chain: SPIR-V 1.3 emits `OpGroupNonUniformShuffleDown`, the device
//! accepts the module, and 4 contiguous lanes really do sum to lane 0.
//!
//! Run: `cargo run --release -p vk-compute --example sg`

use vk_compute::{VkContext, compile_wgsl};

const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) tid: u32) {
    var acc = f32(tid);
    acc = acc + subgroupShuffleDown(acc, 2u);
    acc = acc + subgroupShuffleDown(acc, 1u);
    if (tid % 4u == 0u) { out[tid / 4u] = acc; }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    println!("subgroup_size = {}", ctx.subgroup_size);

    let spv = compile_wgsl(SRC)?;
    println!("SPIR-V: {} words", spv.len());

    let pipe = ctx.create_pipeline(&spv, 1, 0)?;
    let out = ctx.create_storage_buffer(4 * 16)?;
    ctx.stream_dispatch(&pipe, &[&out], &[], [1, 1, 1])?;
    ctx.flush()?;

    let got: Vec<f32> = ctx
        .stream_download(&out, 4 * 16)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // group g holds tids 4g..4g+3, so the sum is 16g + 6
    let want: Vec<f32> = (0..16).map(|g| (16 * g + 6) as f32).collect();
    println!("got  {:?}", &got[..8]);
    println!("want {:?}", &want[..8]);
    println!(
        "{}",
        if got == want {
            "OK — the butterfly reduces 4 contiguous lanes"
        } else {
            "MISMATCH"
        }
    );
    Ok(())
}
