//! Adapter from `OrtGraph` to `onnx-vulkan-core` owned IR.
//!
//! ORT warning: `OrtGraph`/`OrtNode` passed to `Compile` are valid only for
//! call duration. We copy everything needed by the interpreter
//! (value names, attributes, initializer bytes) into owned Rust structs,
//! allowing `Compute` to execute without referencing ORT graph pointers.

use crate::ort_util::{apis, check};
use anyhow::Result;
pub use onnx_vulkan_core::{AttrValue, GraphIr, InitializerIr, NodeIr, fold_constant_params};
use ort_ep_sys as sys;
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::ptr;

/// Name of `OrtValueInfo` (empty string if null pointer).
unsafe fn value_info_name(vi: *const sys::OrtValueInfo) -> Result<String> {
    if vi.is_null() {
        return Ok(String::new());
    }
    let api = apis().ort;
    let mut name: *const c_char = ptr::null();
    check(
        unsafe { (api.GetValueInfoName.expect("GetValueInfoName"))(vi, &mut name) },
        "GetValueInfoName",
    )?;
    if name.is_null() {
        return Ok(String::new());
    }
    Ok(unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned())
}

/// Reads dtype+shape+bytes of CPU-resident `OrtValue` tensor.
unsafe fn read_tensor(value: *const sys::OrtValue) -> Result<InitializerIr> {
    let api = apis().ort;
    unsafe {
        let mut info: *mut sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
        check(
            (api.GetTensorTypeAndShape.expect("GetTensorTypeAndShape"))(value, &mut info),
            "GetTensorTypeAndShape",
        )?;
        let mut dtype = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        check(
            (api.GetTensorElementType.expect("GetTensorElementType"))(info, &mut dtype),
            "GetTensorElementType",
        )?;
        let mut num_dims = 0usize;
        check(
            (api.GetDimensionsCount.expect("GetDimensionsCount"))(info, &mut num_dims),
            "GetDimensionsCount",
        )?;
        let mut shape = vec![0i64; num_dims];
        check(
            (api.GetDimensions.expect("GetDimensions"))(info, shape.as_mut_ptr(), num_dims),
            "GetDimensions",
        )?;
        let mut elem_count = 0usize;
        check(
            (api.GetTensorShapeElementCount
                .expect("GetTensorShapeElementCount"))(info, &mut elem_count),
            "GetTensorShapeElementCount",
        )?;
        (api.ReleaseTensorTypeAndShapeInfo
            .expect("ReleaseTensorTypeAndShapeInfo"))(info);

        let dtype = dtype as i32;
        let nbytes = onnx_vulkan_core::storage_len(dtype, elem_count).unwrap_or(0);
        let mut src: *mut std::ffi::c_void = ptr::null_mut();
        check(
            (api.GetTensorMutableData.expect("GetTensorMutableData"))(value.cast_mut(), &mut src),
            "GetTensorMutableData",
        )?;
        let data = if nbytes == 0 || src.is_null() {
            Vec::new()
        } else {
            std::slice::from_raw_parts(src.cast::<u8>(), nbytes).to_vec()
        };
        Ok(InitializerIr { dtype, shape, data })
    }
}

/// Reads `OrtOpAttr` attribute into [`AttrValue`] (supported types).
unsafe fn read_attr(attr: *const sys::OrtOpAttr) -> Result<Option<(String, AttrValue)>> {
    let api = apis().ort;
    unsafe {
        let mut name: *const c_char = ptr::null();
        check(
            (api.OpAttr_GetName.expect("OpAttr_GetName"))(attr, &mut name),
            "OpAttr_GetName",
        )?;
        let name = CStr::from_ptr(name).to_string_lossy().into_owned();

        let mut ty = sys::OrtOpAttrType::ORT_OP_ATTR_UNDEFINED;
        check(
            (api.OpAttr_GetType.expect("OpAttr_GetType"))(attr, &mut ty),
            "OpAttr_GetType",
        )?;
        let read = api.ReadOpAttr.expect("ReadOpAttr");

        // first pass: bytes needed (data=null, len=0 → out=required bytes)
        let mut need = 0usize;
        let s = read(attr, ty, ptr::null_mut(), 0, &mut need);
        if !s.is_null() {
            (api.ReleaseStatus.expect("ReleaseStatus"))(s);
        }

        let value = match ty {
            sys::OrtOpAttrType::ORT_OP_ATTR_INT => {
                let mut v = 0i64;
                let mut out = 0usize;
                check(
                    read(attr, ty, (&mut v as *mut i64).cast(), 8, &mut out),
                    "ReadOpAttr INT",
                )?;
                Some(AttrValue::Int(v))
            }
            sys::OrtOpAttrType::ORT_OP_ATTR_FLOAT => {
                let mut v = 0f32;
                let mut out = 0usize;
                check(
                    read(attr, ty, (&mut v as *mut f32).cast(), 4, &mut out),
                    "ReadOpAttr FLOAT",
                )?;
                Some(AttrValue::Float(v))
            }
            sys::OrtOpAttrType::ORT_OP_ATTR_INTS => {
                let n = need / 8;
                let mut v = vec![0i64; n];
                let mut out = need;
                if n > 0 {
                    check(
                        read(attr, ty, v.as_mut_ptr().cast(), need, &mut out),
                        "ReadOpAttr INTS",
                    )?;
                }
                Some(AttrValue::Ints(v))
            }
            sys::OrtOpAttrType::ORT_OP_ATTR_FLOATS => {
                let n = need / 4;
                let mut v = vec![0f32; n];
                let mut out = need;
                if n > 0 {
                    check(
                        read(attr, ty, v.as_mut_ptr().cast(), need, &mut out),
                        "ReadOpAttr FLOATS",
                    )?;
                }
                Some(AttrValue::Floats(v))
            }
            sys::OrtOpAttrType::ORT_OP_ATTR_STRING => {
                let mut buf = vec![0u8; need];
                let mut out = need;
                if need > 0 {
                    check(
                        read(attr, ty, buf.as_mut_ptr().cast(), need, &mut out),
                        "ReadOpAttr STRING",
                    )?;
                }
                Some(AttrValue::String(
                    String::from_utf8_lossy(&buf[..out.min(buf.len())]).into_owned(),
                ))
            }
            sys::OrtOpAttrType::ORT_OP_ATTR_TENSOR => {
                let get = api
                    .OpAttr_GetTensorAttributeAsOrtValue
                    .expect("OpAttr_GetTensorAttributeAsOrtValue");
                let mut val: *mut sys::OrtValue = ptr::null_mut();
                check(get(attr, &mut val), "OpAttr_GetTensorAttributeAsOrtValue")?;
                if val.is_null() {
                    None
                } else {
                    let t = read_tensor(val)?;
                    (api.ReleaseValue.expect("ReleaseValue"))(val);
                    Some(AttrValue::Tensor(t))
                }
            }
            // GRAPH/STRINGS not used by the encoder kernels
            _ => None,
        };
        Ok(value.map(|v| (name, v)))
    }
}

/// Extracts complete IR from fused `OrtGraph`.
///
/// # Safety
/// `graph` valid for call duration (`Compile` contract).
pub unsafe fn extract(graph: *const sys::OrtGraph) -> Result<GraphIr> {
    let api = apis().ort;
    unsafe {
        let mut ir = GraphIr {
            initializers: extract_initializers(graph)?,
            // --- graph inputs / outputs ---
            inputs: read_io(api, graph, true)?,
            outputs: read_io(api, graph, false)?,
            ..Default::default()
        };

        // --- nodes ---
        let mut num_nodes = 0usize;
        check(
            (api.Graph_GetNumNodes.expect("Graph_GetNumNodes"))(graph, &mut num_nodes),
            "Graph_GetNumNodes",
        )?;
        let mut nodes: Vec<*const sys::OrtNode> = vec![ptr::null(); num_nodes];
        check(
            (api.Graph_GetNodes.expect("Graph_GetNodes"))(graph, nodes.as_mut_ptr(), num_nodes),
            "Graph_GetNodes",
        )?;
        for node in nodes {
            let mut n = extract_node(api, node)?;
            // same normalization applied in `GetCapability`: the node that
            // runs must be the one we declared we can handle
            fold_constant_params(&mut n, &ir.initializers);
            ir.nodes.push(n);
        }

        Ok(ir)
    }
}

/// Graph initializers, copied into host memory.
///
/// Also needed in `GetCapability`, where the graph is not yet fully extracted
/// but constant values decide whether a node is supported.
///
/// # Safety
/// `graph` valid for the call duration.
pub unsafe fn extract_initializers(
    graph: *const sys::OrtGraph,
) -> Result<HashMap<String, InitializerIr>> {
    let api = apis().ort;
    unsafe {
        let mut out = HashMap::new();
        let mut num_init = 0usize;
        check(
            (api.Graph_GetNumInitializers
                .expect("Graph_GetNumInitializers"))(graph, &mut num_init),
            "Graph_GetNumInitializers",
        )?;
        if num_init == 0 {
            return Ok(out);
        }
        let mut vis: Vec<*const sys::OrtValueInfo> = vec![ptr::null(); num_init];
        check(
            (api.Graph_GetInitializers.expect("Graph_GetInitializers"))(
                graph,
                vis.as_mut_ptr(),
                num_init,
            ),
            "Graph_GetInitializers",
        )?;
        for vi in vis {
            let name = value_info_name(vi)?;
            let mut value: *const sys::OrtValue = ptr::null();
            check(
                (api.ValueInfo_GetInitializerValue
                    .expect("ValueInfo_GetInitializerValue"))(vi, &mut value),
                "ValueInfo_GetInitializerValue",
            )?;
            if value.is_null() {
                continue;
            }
            out.insert(name, read_tensor(value)?);
        }
        Ok(out)
    }
}

/// Reads the names of the graph's inputs or outputs.
unsafe fn read_io(
    api: &sys::OrtApi,
    graph: *const sys::OrtGraph,
    inputs: bool,
) -> Result<Vec<String>> {
    unsafe {
        let mut n = 0usize;
        if inputs {
            check(
                (api.Graph_GetNumInputs.expect("Graph_GetNumInputs"))(graph, &mut n),
                "Graph_GetNumInputs",
            )?;
        } else {
            check(
                (api.Graph_GetNumOutputs.expect("Graph_GetNumOutputs"))(graph, &mut n),
                "Graph_GetNumOutputs",
            )?;
        }
        let mut vis: Vec<*const sys::OrtValueInfo> = vec![ptr::null(); n];
        if n > 0 {
            if inputs {
                check(
                    (api.Graph_GetInputs.expect("Graph_GetInputs"))(graph, vis.as_mut_ptr(), n),
                    "Graph_GetInputs",
                )?;
            } else {
                check(
                    (api.Graph_GetOutputs.expect("Graph_GetOutputs"))(graph, vis.as_mut_ptr(), n),
                    "Graph_GetOutputs",
                )?;
            }
        }
        vis.into_iter().map(|vi| value_info_name(vi)).collect()
    }
}

/// Extracts a single node into [`NodeIr`].
///
/// # Safety
/// `node` valid for the call duration.
pub unsafe fn extract_node(api: &sys::OrtApi, node: *const sys::OrtNode) -> Result<NodeIr> {
    unsafe {
        let mut op: *const c_char = ptr::null();
        check(
            (api.Node_GetOperatorType.expect("Node_GetOperatorType"))(node, &mut op),
            "Node_GetOperatorType",
        )?;
        let op = CStr::from_ptr(op).to_string_lossy().into_owned();

        let mut domain: *const c_char = ptr::null();
        check(
            (api.Node_GetDomain.expect("Node_GetDomain"))(node, &mut domain),
            "Node_GetDomain",
        )?;
        let domain = if domain.is_null() {
            String::new()
        } else {
            CStr::from_ptr(domain).to_string_lossy().into_owned()
        };

        let mut since_version = 0;
        check(
            (api.Node_GetSinceVersion.expect("Node_GetSinceVersion"))(node, &mut since_version),
            "Node_GetSinceVersion",
        )?;

        let mut nm: *const c_char = ptr::null();
        check(
            (api.Node_GetName.expect("Node_GetName"))(node, &mut nm),
            "Node_GetName",
        )?;
        let name = if nm.is_null() {
            String::new()
        } else {
            CStr::from_ptr(nm).to_string_lossy().into_owned()
        };

        let inputs = node_value_names(api, node, true)?;
        let outputs = node_value_names(api, node, false)?;

        let mut num_attr = 0usize;
        check(
            (api.Node_GetNumAttributes.expect("Node_GetNumAttributes"))(node, &mut num_attr),
            "Node_GetNumAttributes",
        )?;
        let mut attrs = HashMap::new();
        if num_attr > 0 {
            let mut list: Vec<*const sys::OrtOpAttr> = vec![ptr::null(); num_attr];
            check(
                (api.Node_GetAttributes.expect("Node_GetAttributes"))(
                    node,
                    list.as_mut_ptr(),
                    num_attr,
                ),
                "Node_GetAttributes",
            )?;
            for a in list {
                if let Some((k, v)) = read_attr(a)? {
                    attrs.insert(k, v);
                }
            }
        }

        Ok(NodeIr {
            domain,
            op,
            since_version,
            name,
            inputs,
            outputs,
            attrs,
        })
    }
}

/// Value names of the inputs and outputs of the fused node (ORT binding order).
///
/// # Safety
/// `node` valid for the call duration.
pub unsafe fn fused_node_io(node: *const sys::OrtNode) -> Result<(Vec<String>, Vec<String>)> {
    let api = apis().ort;
    let inputs = unsafe { node_value_names(api, node, true)? };
    let outputs = unsafe { node_value_names(api, node, false)? };
    Ok((inputs, outputs))
}

/// Names of a node's input or output values.
unsafe fn node_value_names(
    api: &sys::OrtApi,
    node: *const sys::OrtNode,
    inputs: bool,
) -> Result<Vec<String>> {
    unsafe {
        let mut n = 0usize;
        if inputs {
            check(
                (api.Node_GetNumInputs.expect("Node_GetNumInputs"))(node, &mut n),
                "Node_GetNumInputs",
            )?;
        } else {
            check(
                (api.Node_GetNumOutputs.expect("Node_GetNumOutputs"))(node, &mut n),
                "Node_GetNumOutputs",
            )?;
        }
        let mut vis: Vec<*const sys::OrtValueInfo> = vec![ptr::null(); n];
        if n > 0 {
            if inputs {
                check(
                    (api.Node_GetInputs.expect("Node_GetInputs"))(node, vis.as_mut_ptr(), n),
                    "Node_GetInputs",
                )?;
            } else {
                check(
                    (api.Node_GetOutputs.expect("Node_GetOutputs"))(node, vis.as_mut_ptr(), n),
                    "Node_GetOutputs",
                )?;
            }
        }
        vis.into_iter().map(|vi| value_info_name(vi)).collect()
    }
}
