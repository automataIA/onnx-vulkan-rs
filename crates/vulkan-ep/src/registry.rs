//! Kernel registry construction: kernel definitions + create function.

use crate::kernels;
use crate::ort_util::{apis, check};
use anyhow::Result;
use ort_ep_sys as sys;
use std::ffi::CStr;

pub const EP_NAME: &CStr = c"VulkanEP";

/// Ops claimed by the EP (must match the registered kernels).
pub const SUPPORTED_OPS: &[&str] = &[
    "MatMulInteger",
    "DynamicQuantizeLinear",
    "Mul",
    "Add",
    "Sigmoid",
    "Relu",
    "Cast",
    "LayerNormalization",
    "Softmax",
    "MatMul",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
];

fn tensor_type(t: sys::ONNXTensorElementDataType) -> Result<*const sys::OrtDataType> {
    let ep = apis().ep;
    let mut out: *const sys::OrtDataType = std::ptr::null();
    check(
        unsafe { (ep.GetTensorDataType.expect("GetTensorDataType"))(t, &mut out) },
        "GetTensorDataType",
    )?;
    Ok(out)
}

struct KernelSpec {
    op_type: &'static CStr,
    since_version: (i32, i32),
    /// (constraint name, allowed tensor types)
    constraints: &'static [(&'static CStr, &'static [sys::ONNXTensorElementDataType])],
    create_func: sys::OrtKernelCreateFunc,
    /// (input index, mem type) — e.g. MemcpyFromHost: input 0 on CPU
    input_mem: Option<(usize, sys::OrtMemType)>,
    /// (output index, mem type) — e.g. MemcpyToHost: output 0 on CPU
    output_mem: Option<(usize, sys::OrtMemType)>,
}

use sys::ONNXTensorElementDataType as T;

const U8: &[T] = &[T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8];
const F32: &[T] = &[T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT];
const I32: &[T] = &[T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32];
/// Tensor types transferable by the Memcpy kernels (all those used by the encoder).
const MEMCPY_TYPES: &[T] = &[
    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
];

fn kernel_specs() -> Vec<KernelSpec> {
    vec![
        KernelSpec {
            op_type: c"MatMulInteger",
            since_version: (10, i32::MAX),
            constraints: &[(c"T1", U8), (c"T2", U8), (c"T3", I32)],
            create_func: Some(kernels::matmul_integer::create_kernel),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"DynamicQuantizeLinear",
            since_version: (11, i32::MAX),
            constraints: &[(c"T1", F32), (c"T2", U8)],
            create_func: Some(kernels::dynamic_quantize::create_kernel),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Mul",
            since_version: (14, i32::MAX),
            constraints: &[(c"T", F32)],
            create_func: Some(kernels::elementwise::create_mul),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Add",
            since_version: (14, i32::MAX),
            constraints: &[(c"T", F32)],
            create_func: Some(kernels::elementwise::create_add),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Sigmoid",
            since_version: (13, i32::MAX),
            constraints: &[(c"T", F32)],
            create_func: Some(kernels::elementwise::create_sigmoid),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Relu",
            since_version: (14, i32::MAX),
            constraints: &[(c"T", F32)],
            create_func: Some(kernels::elementwise::create_relu),
            input_mem: None,
            output_mem: None,
        },
        // only the i32→f32 Casts of the dequant chain (constraint on types)
        KernelSpec {
            op_type: c"Cast",
            since_version: (13, i32::MAX),
            constraints: &[(c"T1", I32), (c"T2", F32)],
            create_func: Some(kernels::elementwise::create_cast),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"LayerNormalization",
            since_version: (17, i32::MAX),
            constraints: &[(c"T", F32), (c"U", F32)],
            create_func: Some(kernels::layernorm::create_kernel),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Softmax",
            since_version: (13, i32::MAX),
            constraints: &[(c"T", F32)],
            create_func: Some(kernels::softmax::create_kernel),
            input_mem: None,
            output_mem: None,
        },
        KernelSpec {
            op_type: c"MatMul",
            since_version: (13, i32::MAX),
            constraints: &[(c"T", F32)],
            create_func: Some(kernels::matmul_fp32::create_kernel),
            input_mem: None,
            output_mem: None,
        },
        // movement ops: keep chains on GPU (fewer CPU↔GPU boundaries).
        // shape/axes (input 1) stays on CPU.
        KernelSpec {
            op_type: c"Reshape",
            since_version: (13, i32::MAX),
            constraints: &[(c"T", MEMCPY_TYPES)],
            create_func: Some(kernels::movement::create_reshape),
            input_mem: Some((1, sys::OrtMemType::OrtMemTypeCPUInput)),
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Squeeze",
            since_version: (13, i32::MAX),
            constraints: &[(c"T", MEMCPY_TYPES)],
            create_func: Some(kernels::movement::create_squeeze),
            input_mem: Some((1, sys::OrtMemType::OrtMemTypeCPUInput)),
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Unsqueeze",
            since_version: (13, i32::MAX),
            constraints: &[(c"T", MEMCPY_TYPES)],
            create_func: Some(kernels::movement::create_unsqueeze),
            input_mem: Some((1, sys::OrtMemType::OrtMemTypeCPUInput)),
            output_mem: None,
        },
        KernelSpec {
            op_type: c"Transpose",
            since_version: (13, i32::MAX),
            constraints: &[(
                c"T",
                &[
                    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                    T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
                ],
            )],
            create_func: Some(kernels::movement::create_transpose),
            input_mem: None,
            output_mem: None,
        },
        // inserted by ORT at device↔CPU boundaries (MemcpyTransformer)
        KernelSpec {
            op_type: c"MemcpyFromHost",
            since_version: (1, i32::MAX),
            constraints: &[(c"T", MEMCPY_TYPES)],
            create_func: Some(kernels::memcpy::create_from_host),
            input_mem: Some((0, sys::OrtMemType::OrtMemTypeCPUInput)),
            output_mem: None,
        },
        KernelSpec {
            op_type: c"MemcpyToHost",
            since_version: (1, i32::MAX),
            constraints: &[(c"T", MEMCPY_TYPES)],
            create_func: Some(kernels::memcpy::create_to_host),
            input_mem: None,
            output_mem: Some((0, sys::OrtMemType::OrtMemTypeCPUOutput)),
        },
    ]
}

/// Creates and populates the plugin's `OrtKernelRegistry`.
pub fn create_registry() -> Result<*mut sys::OrtKernelRegistry> {
    let ep = apis().ep;
    let mut registry: *mut sys::OrtKernelRegistry = std::ptr::null_mut();
    check(
        unsafe { (ep.CreateKernelRegistry.expect("CreateKernelRegistry"))(&mut registry) },
        "CreateKernelRegistry",
    )?;

    let result = (|| -> Result<()> {
        for spec in kernel_specs() {
            let mut builder: *mut sys::OrtKernelDefBuilder = std::ptr::null_mut();
            check(
                unsafe {
                    (ep.CreateKernelDefBuilder.expect("CreateKernelDefBuilder"))(&mut builder)
                },
                "CreateKernelDefBuilder",
            )?;
            let def = (|| -> Result<*mut sys::OrtKernelDef> {
                unsafe {
                    check(
                        (ep.KernelDefBuilder_SetOperatorType
                            .expect("SetOperatorType"))(
                            builder, spec.op_type.as_ptr()
                        ),
                        "KernelDefBuilder_SetOperatorType",
                    )?;
                    check(
                        (ep.KernelDefBuilder_SetDomain.expect("SetDomain"))(builder, c"".as_ptr()),
                        "KernelDefBuilder_SetDomain",
                    )?;
                    check(
                        (ep.KernelDefBuilder_SetSinceVersion
                            .expect("SetSinceVersion"))(
                            builder,
                            spec.since_version.0,
                            spec.since_version.1,
                        ),
                        "KernelDefBuilder_SetSinceVersion",
                    )?;
                    check(
                        (ep.KernelDefBuilder_SetExecutionProvider
                            .expect("SetExecutionProvider"))(
                            builder, EP_NAME.as_ptr()
                        ),
                        "KernelDefBuilder_SetExecutionProvider",
                    )?;
                    for (name, tys) in spec.constraints {
                        let types: Vec<*const sys::OrtDataType> =
                            tys.iter().map(|t| tensor_type(*t)).collect::<Result<_>>()?;
                        check(
                            (ep.KernelDefBuilder_AddTypeConstraint
                                .expect("AddTypeConstraint"))(
                                builder,
                                name.as_ptr(),
                                types.as_ptr(),
                                types.len(),
                            ),
                            "KernelDefBuilder_AddTypeConstraint",
                        )?;
                    }
                    if let Some((idx, mem)) = spec.input_mem {
                        check(
                            (ep.KernelDefBuilder_SetInputMemType
                                .expect("SetInputMemType"))(
                                builder, idx, mem
                            ),
                            "KernelDefBuilder_SetInputMemType",
                        )?;
                    }
                    if let Some((idx, mem)) = spec.output_mem {
                        check(
                            (ep.KernelDefBuilder_SetOutputMemType
                                .expect("SetOutputMemType"))(
                                builder, idx, mem
                            ),
                            "KernelDefBuilder_SetOutputMemType",
                        )?;
                    }
                    let mut def: *mut sys::OrtKernelDef = std::ptr::null_mut();
                    check(
                        (ep.KernelDefBuilder_Build.expect("Build"))(builder, &mut def),
                        "KernelDefBuilder_Build",
                    )?;
                    Ok(def)
                }
            })();
            unsafe { (ep.ReleaseKernelDefBuilder.expect("ReleaseKernelDefBuilder"))(builder) };
            let def = def?;

            let add_result = check(
                unsafe {
                    (ep.KernelRegistry_AddKernel
                        .expect("KernelRegistry_AddKernel"))(
                        registry,
                        def,
                        spec.create_func,
                        std::ptr::null_mut(),
                    )
                },
                "KernelRegistry_AddKernel",
            );
            unsafe { (ep.ReleaseKernelDef.expect("ReleaseKernelDef"))(def) };
            add_result?;
        }
        Ok(())
    })();

    match result {
        Ok(()) => Ok(registry),
        Err(e) => {
            unsafe { (ep.ReleaseKernelRegistry.expect("ReleaseKernelRegistry"))(registry) };
            Err(e)
        }
    }
}
