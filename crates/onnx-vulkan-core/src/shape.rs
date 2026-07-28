//! Shape primitives shared between host execution and GPU kernels.

use crate::{Error, Result};

/// Result of ONNX multidirectional broadcasting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Broadcast {
    pub out_shape: Vec<i64>,
    pub out_strides: Vec<u32>,
    pub a_strides: Vec<u32>,
    pub b_strides: Vec<u32>,
}

/// Number of elements in a static shape, checking for sign and overflow.
pub fn element_count(shape: &[i64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        let dimension = usize::try_from(dimension).map_err(|_| {
            Error::InvalidShape(format!("dynamic or negative dimension: {shape:?}"))
        })?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| Error::InvalidShape(format!("overflow in element count: {shape:?}")))
    })
}

/// Computes output shape and strides for ONNX multidirectional broadcasting.
///
/// Strides are `u32` because consumed directly by Vulkan push
/// constants. Negative dimensions or tensors exceeding `u32::MAX`
/// elements are rejected prior to dispatch.
pub fn broadcast(a: &[i64], b: &[i64]) -> Result<Broadcast> {
    let rank = a.len().max(b.len());
    let get = |shape: &[i64], dimension: usize| -> i64 {
        let offset = rank - shape.len();
        if dimension < offset {
            1
        } else {
            shape[dimension - offset]
        }
    };

    let mut out_shape = vec![0i64; rank];
    for (dimension, output) in out_shape.iter_mut().enumerate() {
        let (a_dim, b_dim) = (get(a, dimension), get(b, dimension));
        if a_dim < 0 || b_dim < 0 {
            return Err(Error::InvalidShape(format!(
                "broadcast requires concrete dimensions: {a:?} vs {b:?}"
            )));
        }
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(Error::InvalidShape(format!(
                "incompatible broadcast: {a:?} vs {b:?}"
            )));
        }
        *output = if a_dim == 1 { b_dim } else { a_dim };
    }

    let strides = |shape: &[i64]| -> Result<Vec<u32>> {
        let mut result = vec![0u32; rank];
        let mut accumulator = 1u32;
        for dimension in (0..rank).rev() {
            let dim = get(shape, dimension);
            result[dimension] = if dim == 1 && out_shape[dimension] != 1 {
                0
            } else {
                accumulator
            };
            let dim = u32::try_from(dim).map_err(|_| {
                Error::InvalidShape(format!("dimension {dim} not representable as u32"))
            })?;
            accumulator = accumulator.checked_mul(dim).ok_or_else(|| {
                Error::InvalidShape(format!("tensor too large for Vulkan: {shape:?}"))
            })?;
        }
        Ok(result)
    };

    let mut out_strides = vec![0u32; rank];
    let mut accumulator = 1u32;
    for dimension in (0..rank).rev() {
        out_strides[dimension] = accumulator;
        let dim = u32::try_from(out_shape[dimension]).map_err(|_| {
            Error::InvalidShape(format!(
                "dimension {} not representable as u32",
                out_shape[dimension]
            ))
        })?;
        accumulator = accumulator.checked_mul(dim).ok_or_else(|| {
            Error::InvalidShape(format!("broadcast output too large: {out_shape:?}"))
        })?;
    }

    Ok(Broadcast {
        a_strides: strides(a)?,
        b_strides: strides(b)?,
        out_shape,
        out_strides,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasts_from_the_right() {
        let result = broadcast(&[2, 3, 4], &[4]).expect("compatible shapes");

        assert_eq!(result.out_shape, vec![2, 3, 4]);
        assert_eq!(result.out_strides, vec![12, 4, 1]);
        assert_eq!(result.a_strides, vec![12, 4, 1]);
        assert_eq!(result.b_strides, vec![0, 0, 1]);
    }

    #[test]
    fn rejects_incompatible_or_dynamic_shapes() {
        assert!(matches!(
            broadcast(&[2, 3], &[4, 3]),
            Err(Error::InvalidShape(_))
        ));
        assert!(matches!(
            broadcast(&[-1, 3], &[1, 3]),
            Err(Error::InvalidShape(_))
        ));
    }

    #[test]
    fn preserves_a_zero_dimension_when_broadcasting_one() {
        let result = broadcast(&[0, 4], &[1, 4]).expect("compatible shapes");

        assert_eq!(result.out_shape, vec![0, 4]);
    }

    #[test]
    fn element_count_checks_dynamic_shapes_and_overflow() {
        assert_eq!(element_count(&[]).expect("scalar"), 1);
        assert_eq!(element_count(&[2, 0, 4]).expect("empty tensor"), 0);
        assert!(matches!(
            element_count(&[-1, 4]),
            Err(Error::InvalidShape(_))
        ));
    }
}
