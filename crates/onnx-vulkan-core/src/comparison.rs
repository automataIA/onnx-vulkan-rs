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
        report.max_abs = report.max_abs.max(difference);
        report.max_relative = report.max_relative.max(relative);
        if difference > tolerance.absolute + tolerance.relative * scale {
            report.mismatches += 1;
        }
    }
    report
}

macro_rules! exact_comparator {
    ($name:ident, $ty:ty, $label:literal) => {
        pub fn $name(reference: &[$ty], candidate: &[$ty]) -> ComparisonReport {
            let mut report = empty_report($label, reference.len(), candidate.len());
            report.mismatches = reference
                .iter()
                .zip(candidate)
                .filter(|(expected, actual)| expected != actual)
                .count();
            report
        }
    };
}

exact_comparator!(compare_i64, i64, "int64");
exact_comparator!(compare_i32, i32, "int32");
exact_comparator!(compare_i8, i8, "int8");
exact_comparator!(compare_u8, u8, "uint8");
exact_comparator!(compare_bool, bool, "bool");

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
    fn float_tolerance_and_lengths_are_counted() {
        let report = compare_f32(
            &[0.0, 100.0, 3.0],
            &[0.000_01, 100.2],
            FloatTolerance::new(1e-4, 1e-3),
        );
        assert_eq!(report.mismatches, 2);
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
