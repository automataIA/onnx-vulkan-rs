//! A long stream is submitted in chunks while it is recorded.
//!
//! What the chunks must not change: a dispatch still sees what every earlier
//! one wrote, also across chunk boundaries, and a buffer handed to
//! `defer_destroy` while a submitted chunk may still read it lives until the
//! flush. A chain of dependent dispatches many times longer than a chunk makes
//! any lost ordering show up as a wrong count.

use vk_compute::{VkContext, compile_wgsl};

fn as_bytes(data: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn to_u32(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `out[i] = a[i] + 1`
const INCREMENT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

struct Push { n: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < pc.n) {
        out[gid.x] = a[gid.x] + 1u;
    }
}
"#;

#[test]
fn a_chain_across_many_chunks_counts_every_step() {
    let ctx = VkContext::new().expect("Vulkan context");
    let pipeline = ctx
        .create_pipeline(&compile_wgsl(INCREMENT_WGSL).expect("WGSL"), 2, 4)
        .unwrap();
    let n = 4096u32;
    let size = u64::from(n) * 4;
    let groups = [n.div_ceil(64), 1, 1];
    let push = n.to_le_bytes();
    let start: Vec<u32> = (0..n).map(|i| i * 7).collect();

    // twice: the second run reuses the fences the first one retired
    for run in 0..2u32 {
        let mut ping = ctx.create_storage_buffer(size).unwrap();
        let mut pong = ctx.create_storage_buffer(size).unwrap();
        ctx.stream_upload(&ping, as_bytes(&start)).unwrap();
        let steps = 1000u32;
        for step in 0..steps {
            ctx.stream_dispatch(&pipeline, &[&ping, &pong], &push, groups)
                .unwrap();
            std::mem::swap(&mut ping, &mut pong);
            if step == 300 {
                // the end of a run: the tail goes out and chunks start small
                ctx.stream_break().unwrap();
            }
            if step == 600 {
                // `pong` was read by the last dispatch, likely already
                // submitted: it moves to the graveyard and a fresh buffer
                // carries on, refilled from the GPU by a copy
                let fresh = ctx.create_storage_buffer(size).unwrap();
                ctx.stream_copy(&ping, &fresh, size).unwrap();
                ctx.defer_destroy(std::mem::replace(&mut ping, fresh));
            }
        }
        let got = to_u32(&ctx.stream_download(&ping, size as usize).unwrap());
        let wrong = got
            .iter()
            .zip(&start)
            .filter(|&(&g, &s)| g != s + steps)
            .count();
        assert_eq!(wrong, 0, "run {run}: {wrong} of {n} values wrong");
        ctx.destroy_buffer(ping);
        ctx.destroy_buffer(pong);
    }
    ctx.destroy_pipeline(pipeline);
}
