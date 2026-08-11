//! What does one `flush()` cost when there is nothing to compute?
//!
//! The Pop!_OS matrix (`runs/popos-3`) found GPU compute unchanged against the
//! Windows baseline on every model, while the wall grew in proportion to the
//! **flush count** — ~0.5 ms per submit+fence, inferred from two models (57 and
//! 77 flushes). Inferred, not measured: that number came from differencing two
//! whole-model walls on two different operating systems. This measures the round
//! trip directly, on one host, with no model around it.
//!
//! Regimes, each isolating one term:
//!
//! * `empty` — `flush()` with nothing recorded. Returns before any Vulkan call,
//!   so this is the Rust-side floor and should be ~0.
//! * `copy4` — a 4-byte GPU→GPU copy: the smallest thing that forces a real
//!   `vkQueueSubmit` + fence wait through our own `flush()`.
//! * `dispatch` — one workgroup of a shader that writes one `u32`. Adds the
//!   pipeline bind and a descriptor set to the same round trip.
//! * `batch32` — 32 of those dispatches inside a **single** flush. If the cost
//!   is per-submit this stays near `dispatch`; if it were per-dispatch it would
//!   grow ~32×.
//! * `raw` — the same submit + fence wait with everything of ours stripped: a
//!   pre-recorded empty command buffer and one reused fence.
//! * `raw+fence`, `raw+cmd` — `raw` with one of the two per-flush allocations
//!   `flush()` actually makes added back, to name the gap instead of guessing.
//!
//! Measured on an RTX 4070 / driver 595.84 (2026-08-03), in ms:
//!
//! ```text
//!               NVIDIA   lavapipe
//!   copy4        0.496      0.011
//!   batch32      0.530      0.460   (32× the work, +0.034 on NVIDIA)
//!   raw          0.136      0.011
//!   raw+fence    0.494      0.012
//!   raw+cmd      0.137      0.013
//! ```
//!
//! So the ~0.5 ms is **not** one thing. 0.136 ms is the driver's round trip, and
//! **0.358 ms is `vkCreateFence` + `vkDestroyFence`**, which `flush()` does once
//! per submit and which costs nothing on lavapipe. Command buffer churn is free
//! on both. It is also not a power state: sampled during the run the card sits
//! at P0 / 2475 MHz and the number does not move.
//!
//! Run: `cargo run --release -p vk-compute --example flush_cost`

use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main() { out[0] = out[0] + 1u; }
"#;

const WARMUP: usize = 32;
const ITERS: usize = 512;

/// Median, min and p95 of one regime, in milliseconds.
fn report(label: &str, mut us: Vec<f64>) {
    us.sort_by(f64::total_cmp);
    let at = |q: f64| us[((us.len() - 1) as f64 * q) as usize] / 1000.0;
    println!(
        "{label:>9}  median {:.3} ms   min {:.3}   p95 {:.3}",
        at(0.5),
        at(0.0),
        at(0.95)
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    println!("device: {}\n", ctx.device_name);

    let pipe = ctx.create_pipeline(&compile_wgsl(SRC)?, 1, 0)?;
    let a = ctx.create_storage_buffer(256)?;
    let b = ctx.create_storage_buffer(256)?;

    // one closure per regime, each recording the work that precedes one flush
    let regimes: [(&str, &dyn Fn() -> anyhow::Result<()>); 4] = [
        ("empty", &|| Ok(())),
        ("copy4", &|| ctx.stream_copy(&a, &b, 4)),
        ("dispatch", &|| {
            ctx.stream_dispatch(&pipe, &[&a], &[], [1, 1, 1])
        }),
        ("batch32", &|| {
            for _ in 0..32 {
                ctx.stream_dispatch(&pipe, &[&a], &[], [1, 1, 1])?;
            }
            Ok(())
        }),
    ];

    for (label, record) in regimes {
        for _ in 0..WARMUP {
            record()?;
            ctx.flush()?;
        }
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            record()?;
            // only the flush is timed: recording is measured separately by the
            // profiler's `recording_ms`, and is not what the wall grew by
            let t = Instant::now();
            ctx.flush()?;
            samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        report(label, samples);
    }

    raw_round_trip(&ctx)?;
    Ok(())
}

/// The same round trip with everything of ours removed: one command buffer
/// recorded **empty** and re-submitted, one fence reset instead of created and
/// destroyed, no descriptor reset, no staging pool. Whatever is left here is the
/// driver's, not the stream's — which is the difference between a constant to
/// live with and a cost to engineer away.
fn raw_round_trip(ctx: &VkContext) -> Result<(), Box<dyn std::error::Error>> {
    use ash::vk;
    let d = &ctx.device;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    // ...and then the two per-flush allocations `flush()` does make, added back
    // one at a time, so the 0.36 ms gap has a name rather than a suspect
    for (label, fence_each, cmd_each) in [
        ("raw", false, false),
        ("raw+fence", true, false),
        ("raw+cmd", false, true),
    ] {
        let mut samples = Vec::with_capacity(ITERS);
        unsafe {
            let mut cmds = [d.allocate_command_buffers(&alloc_info)?[0]];
            d.begin_command_buffer(cmds[0], &vk::CommandBufferBeginInfo::default())?;
            d.end_command_buffer(cmds[0])?;
            let mut fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;

            for i in 0..WARMUP + ITERS {
                let t = Instant::now();
                if cmd_each {
                    cmds[0] = d.allocate_command_buffers(&alloc_info)?[0];
                    d.begin_command_buffer(cmds[0], &vk::CommandBufferBeginInfo::default())?;
                    d.end_command_buffer(cmds[0])?;
                }
                if fence_each {
                    fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
                }
                let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                d.queue_submit(ctx.queue, &submit, fence)?;
                d.wait_for_fences(&[fence], true, u64::MAX)?;
                if fence_each {
                    d.destroy_fence(fence, None);
                } else {
                    d.reset_fences(&[fence])?;
                }
                if cmd_each {
                    d.free_command_buffers(ctx.command_pool, &cmds);
                }
                if i >= WARMUP {
                    samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
                }
            }

            if !fence_each {
                d.destroy_fence(fence, None);
            }
            if !cmd_each {
                d.free_command_buffers(ctx.command_pool, &cmds);
            }
        }
        report(label, samples);
    }
    Ok(())
}
