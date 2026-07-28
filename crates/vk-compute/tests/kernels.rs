//! End-to-end test on GPU (or lavapipe): upload → dispatch → readback → CPU comparison.

use vk_compute::{VkContext, compile_wgsl};

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

const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push { n: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < pc.n) {
        out[gid.x] = a[gid.x] + b[gid.x];
    }
}
"#;

#[test]
fn add_f32() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let n = 4099usize; // not a multiple of the workgroup: verifies the bounds check

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let spirv = compile_wgsl(ADD_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 3, 4).unwrap();

    let size = (n * 4) as u64;
    let buf_a = ctx.create_storage_buffer(size).unwrap();
    let buf_b = ctx.create_storage_buffer(size).unwrap();
    let buf_out = ctx.create_storage_buffer(size).unwrap();
    ctx.upload(&buf_a, as_bytes(&a)).unwrap();
    ctx.upload(&buf_b, as_bytes(&b)).unwrap();

    let groups = [(n as u32).div_ceil(256), 1, 1];
    ctx.dispatch(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        &(n as u32).to_le_bytes(),
        groups,
    )
    .unwrap();

    let result: Vec<f32> = from_bytes(&ctx.download(&buf_out).unwrap());
    assert_eq!(result, expected);

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

/// ONNX-style MatMulInteger: dynamic A u8 [M,K], constant B u8 [K,N],
/// scalar zero points, i32 [M,N] output. B packed by column:
/// b_packed[col][i] = 4 bytes along K.
const MATMUL_U8_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;   // [M, K/4] packed
@group(0) @binding(1) var<storage, read> b: array<u32>;   // [N, K/4] packed (B trasposta)
@group(0) @binding(2) var<storage, read_write> out: array<i32>; // [M, N]

struct Push { m: u32, k: u32, n: u32, a_zp: i32, b_zp: i32 }
var<immediate> pc: Push;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= pc.n || row >= pc.m) {
        return;
    }
    let k4 = pc.k / 4u;
    var acc: i32 = 0;
    for (var i = 0u; i < k4; i = i + 1u) {
        let av = vec4<i32>(unpack4xU8(a[row * k4 + i])) - vec4<i32>(pc.a_zp);
        let bv = vec4<i32>(unpack4xU8(b[col * k4 + i])) - vec4<i32>(pc.b_zp);
        acc = acc + dot(av, bv);
    }
    out[row * pc.n + col] = acc;
}
"#;

#[test]
fn matmul_integer_u8() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let (m, k, n) = (17usize, 64usize, 23usize); // K is a multiple of 4
    let a_zp: u8 = 121;
    let b_zp: u8 = 130;

    // deterministic pseudo-random data
    let a: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    let b: Vec<u8> = (0..k * n).map(|i| ((i * 53 + 7) % 249) as u8).collect();

    // CPU reference
    let mut expected = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for i in 0..k {
                acc +=
                    (a[row * k + i] as i32 - a_zp as i32) * (b[i * n + col] as i32 - b_zp as i32);
            }
            expected[row * n + col] = acc;
        }
    }

    // packing: A row-major along K; B transposed [N][K] then packed along K
    let a_packed: Vec<u32> = a
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut b_t = vec![0u8; n * k];
    for i in 0..k {
        for col in 0..n {
            b_t[col * k + i] = b[i * n + col];
        }
    }
    let b_packed: Vec<u32> = b_t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let spirv = compile_wgsl(MATMUL_U8_WGSL).expect("WGSL compilation");
    let pipeline = ctx.create_pipeline(&spirv, 3, 20).unwrap();

    let buf_a = ctx
        .create_storage_buffer((a_packed.len() * 4) as u64)
        .unwrap();
    let buf_b = ctx
        .create_storage_buffer((b_packed.len() * 4) as u64)
        .unwrap();
    let buf_out = ctx.create_storage_buffer((m * n * 4) as u64).unwrap();
    ctx.upload(&buf_a, as_bytes(&a_packed)).unwrap();
    ctx.upload(&buf_b, as_bytes(&b_packed)).unwrap();

    let mut push = Vec::with_capacity(20);
    for v in [m as u32, k as u32, n as u32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    for v in [a_zp as i32, b_zp as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let groups = [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1];
    ctx.dispatch(&pipeline, &[&buf_a, &buf_b, &buf_out], &push, groups)
        .unwrap();

    let result: Vec<i32> = from_bytes(&ctx.download(&buf_out).unwrap());
    assert_eq!(result, expected);

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

/// Variant with `dot4U8Packed` + zero-point correction:
/// Σ(aᵢ−az)(bᵢ−bz) = Σaᵢbᵢ − az·Σbᵢ − bz·Σaᵢ + 4·az·bz  (per block of 4)
/// naga polyfills the builtin if the SPIR-V target < 1.6, so the path
/// stays portable; on drivers with VK_KHR_shader_integer_dot_product it
/// becomes a native instruction.
const MATMUL_U8_DOT4_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

struct Push { m: u32, k: u32, n: u32, a_zp: i32, b_zp: i32 }
var<immediate> pc: Push;

fn sum4_u8(v: u32) -> i32 {
    let u = unpack4xU8(v);
    return i32(u.x + u.y + u.z + u.w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= pc.n || row >= pc.m) {
        return;
    }
    let k4 = pc.k / 4u;
    var acc: i32 = 0;
    for (var i = 0u; i < k4; i = i + 1u) {
        let av = a[row * k4 + i];
        let bv = b[col * k4 + i];
        acc = acc + i32(dot4U8Packed(av, bv))
            - pc.a_zp * sum4_u8(bv)
            - pc.b_zp * sum4_u8(av)
            + 4 * pc.a_zp * pc.b_zp;
    }
    out[row * pc.n + col] = acc;
}
"#;

#[test]
fn matmul_integer_u8_dot4() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let (m, k, n) = (17usize, 64usize, 23usize);
    let a_zp: u8 = 121;
    let b_zp: u8 = 130;

    let a: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    let b: Vec<u8> = (0..k * n).map(|i| ((i * 53 + 7) % 249) as u8).collect();

    let mut expected = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for i in 0..k {
                acc +=
                    (a[row * k + i] as i32 - a_zp as i32) * (b[i * n + col] as i32 - b_zp as i32);
            }
            expected[row * n + col] = acc;
        }
    }

    let a_packed: Vec<u32> = a
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut b_t = vec![0u8; n * k];
    for i in 0..k {
        for col in 0..n {
            b_t[col * k + i] = b[i * n + col];
        }
    }
    let b_packed: Vec<u32> = b_t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let spirv = compile_wgsl(MATMUL_U8_DOT4_WGSL).expect("WGSL dot4 compilation");
    let pipeline = ctx.create_pipeline(&spirv, 3, 20).unwrap();

    let buf_a = ctx
        .create_storage_buffer((a_packed.len() * 4) as u64)
        .unwrap();
    let buf_b = ctx
        .create_storage_buffer((b_packed.len() * 4) as u64)
        .unwrap();
    let buf_out = ctx.create_storage_buffer((m * n * 4) as u64).unwrap();
    ctx.upload(&buf_a, as_bytes(&a_packed)).unwrap();
    ctx.upload(&buf_b, as_bytes(&b_packed)).unwrap();

    let mut push = Vec::with_capacity(20);
    for v in [m as u32, k as u32, n as u32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    for v in [a_zp as i32, b_zp as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let groups = [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1];
    ctx.dispatch(&pipeline, &[&buf_a, &buf_b, &buf_out], &push, groups)
        .unwrap();

    let result: Vec<i32> = from_bytes(&ctx.download(&buf_out).unwrap());
    assert_eq!(result, expected);

    ctx.destroy_buffer(buf_a);
    ctx.destroy_buffer(buf_b);
    ctx.destroy_buffer(buf_out);
    ctx.destroy_pipeline(pipeline);
}

#[test]
fn enumerate_devices() {
    let devices = VkContext::enumerate_devices().expect("enumerazione device");
    assert!(!devices.is_empty(), "no Vulkan device found");
    for (name, vendor, ty) in &devices {
        eprintln!("device: {name} (vendor 0x{vendor:04x}, {ty:?})");
    }
}
