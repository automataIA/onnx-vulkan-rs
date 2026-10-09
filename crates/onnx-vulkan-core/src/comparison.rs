//! Dtype-aware correctness comparison for tuning and backend diagnostics.
//!
//! Integer outputs are always compared exactly. Floating-point outputs use an
//! explicit `atol + rtol * |reference|` contract, while NaN and infinity
//! classification is accounted for separately instead of being hidden by the
//! finite error metrics.

/// Tolerances used only for finite floating-point values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatTolerance {
    pub absolute: f64,
    pub relative: f64,
    pub relative_floor: f64,
}

impl FloatTolerance {
    pub const fn new(absolute: f64, relative: f64) -> Self {
        Self {
            absolute,
            relative,
            relative_floor: 0.0,
        }
    }

    /// Uses `max(|reference|, floor)` as the relative scale.
    pub const fn with_relative_floor(mut self, floor: f64) -> Self {
        self.relative_floor = floor;
        self
    }
}

/// Machine-readable comparison summary shared by operator runners and hosts.
#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    pub dtype: &'static str,
    pub reference_len: usize,
    pub candidate_len: usize,
    pub mismatches: usize,
    pub max_abs: f64,
    pub max_relative: f64,
    /// Largest `|Δ| / (atol + rtol·|ref|)`: at most 1 passes. Unlike
    /// `max_relative`, it stays finite where the reference is zero; exact
    /// comparisons report infinity on any mismatch.
    pub max_tolerance_ratio: f64,
    pub matching_nan: usize,
    pub nan_mismatches: usize,
    pub matching_infinity: usize,
    pub infinity_mismatches: usize,
}

impl ComparisonReport {
    pub fn passed(&self) -> bool {
        self.reference_len == self.candidate_len && self.mismatches == 0
    }
}

/// Compares f32 outputs without allowing non-finite values into error math.
pub fn compare_f32(
    reference: &[f32],
    candidate: &[f32],
    tolerance: FloatTolerance,
) -> ComparisonReport {
    compare_float("float32", reference, candidate, tolerance, |value| {
        f64::from(*value)
    })
}

/// Compares f64 outputs without allowing non-finite values into error math.
pub fn compare_f64(
    reference: &[f64],
    candidate: &[f64],
    tolerance: FloatTolerance,
) -> ComparisonReport {
    compare_float("float64", reference, candidate, tolerance, |value| *value)
}

fn compare_float<T>(
    dtype: &'static str,
    reference: &[T],
    candidate: &[T],
    tolerance: FloatTolerance,
    convert: impl Fn(&T) -> f64,
) -> ComparisonReport {
    let mut report = empty_report(dtype, reference.len(), candidate.len());
    for (expected, actual) in reference.iter().zip(candidate) {
        let expected = convert(expected);
        let actual = convert(actual);
        if expected.is_nan() || actual.is_nan() {
            if expected.is_nan() && actual.is_nan() {
                report.matching_nan += 1;
            } else {
                report.nan_mismatches += 1;
                report.mismatches += 1;
            }
            continue;
        }
        if expected.is_infinite() || actual.is_infinite() {
            if expected == actual {
                report.matching_infinity += 1;
            } else {
                report.infinity_mismatches += 1;
                report.mismatches += 1;
            }
            continue;
        }
        let difference = (expected - actual).abs();
        let scale = expected.abs().max(tolerance.relative_floor);
        let relative = if scale == 0.0 {
            if difference == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            difference / scale
        };
        let allowed = tolerance.absolute + tolerance.relative * scale;
        let ratio = if allowed > 0.0 {
            difference / allowed
        } else if difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        report.max_abs = report.max_abs.max(difference);
        report.max_relative = report.max_relative.max(relative);
        report.max_tolerance_ratio = report.max_tolerance_ratio.max(ratio);
        if difference > allowed {
            report.mismatches += 1;
        }
    }
    report
}

// `max_abs` reports the largest integer distance for diagnostics only; the
// verdict stays exact equality, which `f64` could not represent for `i64`
macro_rules! exact_comparator {
    ($name:ident, $ty:ty, $label:literal, $widen:expr) => {
        pub fn $name(reference: &[$ty], candidate: &[$ty]) -> ComparisonReport {
            let widen: fn(&$ty) -> i128 = $widen;
            let mut report = empty_report($label, reference.len(), candidate.len());
            for (expected, actual) in reference.iter().zip(candidate) {
                if expected != actual {
                    report.mismatches += 1;
                    report.max_tolerance_ratio = f64::INFINITY;
                    let distance = widen(expected).abs_diff(widen(actual)) as f64;
                    report.max_abs = report.max_abs.max(distance);
                }
            }
            report
        }
    };
}

exact_comparator!(compare_i64, i64, "int64", |v| i128::from(*v));
exact_comparator!(compare_i32, i32, "int32", |v| i128::from(*v));
exact_comparator!(compare_i8, i8, "int8", |v| i128::from(*v));
exact_comparator!(compare_u8, u8, "uint8", |v| i128::from(*v));
exact_comparator!(compare_bool, bool, "bool", |v| i128::from(*v));

fn empty_report(
    dtype: &'static str,
    reference_len: usize,
    candidate_len: usize,
) -> ComparisonReport {
    ComparisonReport {
        dtype,
        reference_len,
        candidate_len,
        mismatches: reference_len.abs_diff(candidate_len),
        max_abs: 0.0,
        max_relative: 0.0,
        max_tolerance_ratio: 0.0,
        matching_nan: 0,
        nan_mismatches: 0,
        matching_infinity: 0,
        infinity_mismatches: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_are_exact_even_beyond_f64_precision() {
        let reference = [9_007_199_254_740_992_i64];
        let candidate = [9_007_199_254_740_993_i64];
        let report = compare_i64(&reference, &candidate);
        assert_eq!(report.mismatches, 1);
        assert!(!report.passed());
    }

    #[test]
    fn integer_mismatches_report_their_distance() {
        let report = compare_u8(&[31, 7, 200], &[28, 7, 201]);
        assert_eq!(report.mismatches, 2);
        assert_eq!(report.max_abs, 3.0);
        let report = compare_i64(&[i64::MIN], &[i64::MAX]);
        assert_eq!(report.max_abs, u64::MAX as f64);
        assert_eq!(compare_i8(&[-3], &[-3]).max_abs, 0.0);
    }

    #[test]
    fn float_tolerance_and_lengths_are_counted() {
        let report = compare_f32(
            &[0.0, 100.0, 3.0],
            &[0.000_01, 100.2],
            FloatTolerance::new(1e-4, 1e-3),
        );
        assert_eq!(report.mismatches, 2);
        // 0.2 against 1e-4 + 1e-3·100: twice the allowance; the 1e-5 at a
        // zero reference is an infinite relative error but a tenth of atol.
        assert!((report.max_tolerance_ratio - 0.2 / 0.1001).abs() < 1e-4);
        assert!(report.max_relative.is_infinite());
        assert_eq!(report.reference_len, 3);
        assert_eq!(report.candidate_len, 2);
    }

    #[test]
    fn non_finite_classes_are_explicit() {
        let report = compare_f64(
            &[f64::NAN, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            &[f64::NAN, 0.0, f64::INFINITY, f64::INFINITY],
            FloatTolerance::new(0.0, 0.0),
        );
        assert_eq!(report.matching_nan, 1);
        assert_eq!(report.nan_mismatches, 1);
        assert_eq!(report.matching_infinity, 1);
        assert_eq!(report.infinity_mismatches, 1);
        assert_eq!(report.mismatches, 2);
    }
}
