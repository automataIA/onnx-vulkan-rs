//! WGSL → SPIR-V compilation via naga.

use anyhow::{Result, anyhow};

pub fn compile_wgsl(source: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| anyhow!("WGSL parse error: {}", e.emit_to_string(source)))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| anyhow!("WGSL validation error: {}", e.emit_to_string(source)))?;

    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .map_err(|e| anyhow!("SPIR-V backend error: {e}"))
}
