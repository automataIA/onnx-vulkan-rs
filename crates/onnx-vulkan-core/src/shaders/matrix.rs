//! Persistent identity shared by matrix runners and runtime dispatch.
use crate::tuning::{
    ImplementationFingerprint, ImplementationFingerprintInputs, NamedShaderSource,
};
use std::collections::BTreeMap;

pub fn implementation_fingerprint() -> ImplementationFingerprint {
    static VALUE: std::sync::OnceLock<ImplementationFingerprint> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        ImplementationFingerprint::compute(&ImplementationFingerprintInputs {
            schema_version: 1,
            shader_sources: &[
                NamedShaderSource {
                    name: "gemm16",
                    contents: GEMM16.as_bytes(),
                },
                NamedShaderSource {
                    name: "matmul64",
                    contents: super::matmul_fp32::MATMUL.as_bytes(),
                },
                NamedShaderSource {
                    name: "matmul16",
                    contents: super::matmul_fp32::MATMUL_SMALL.as_bytes(),
                },
                NamedShaderSource {
                    name: "gemm64",
                    contents: super::gemm::GEMM.as_bytes(),
                },
            ],
            specialization_constants: &BTreeMap::new(),
            generator_logic: include_bytes!("matrix.rs"),
            dispatch_logic: b"matrix-f32:v1;rank2;matmul-m>1;gemm-transpose-bias-alpha-beta",
            compiler_settings: b"naga=30;wgsl-in;spv-out;entry=main",
            runtime_abi: b"matmul-bindings=3;push=112;gemm-bindings=4;push=32",
        })
        .expect("committed matrix fingerprint inputs are complete")
    })
}

/// Rank-2 Gemm with one output per thread, sharing the Gemm push ABI.
pub const GEMM16: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct Push { m:u32, k:u32, n:u32, flags:u32, alpha:f32, beta:f32, c_rows:u32, c_cols:u32 }
var<immediate> pc: Push;
var<workgroup> aa: array<f32,256>;
var<workgroup> bb: array<f32,256>;
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) w:vec3<u32>, @builtin(local_invocation_id) l:vec3<u32>) {
    let row=w.y*16u+l.y; let col=w.x*16u+l.x;
    let tid=l.y*16u+l.x;
    var sum=0.0;
    for(var base=0u; base<pc.k; base+=16u) {
        let ak=base+l.x; let bk=base+l.y;
        aa[tid]=0.0; bb[tid]=0.0;
        if(row<pc.m && ak<pc.k) { aa[tid]=a[select(row*pc.k+ak,ak*pc.m+row,(pc.flags&1u)!=0u)]; }
        if(col<pc.n && bk<pc.k) { bb[tid]=b[select(bk*pc.n+col,col*pc.k+bk,(pc.flags&2u)!=0u)]; }
        workgroupBarrier();
        for(var j=0u;j<16u;j+=1u) { sum=fma(aa[l.y*16u+j],bb[j*16u+l.x],sum); }
        workgroupBarrier();
    }
    if(row<pc.m && col<pc.n) {
        var value=pc.alpha*sum;
        if((pc.flags&4u)!=0u) {
            let cr=select(0u,row,pc.c_rows>1u); let cc=select(0u,col,pc.c_cols>1u);
            value+=pc.beta*c[cr*pc.c_cols+cc];
        }
        out[row*pc.n+col]=value;
    }
}
"#;
