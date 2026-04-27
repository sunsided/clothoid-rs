//! Angle utilities for clothoid optimization.
//!
//! Provides signed, wrapped angle difference computation normalized to `(-π, π]`.

/// Returns the signed, wrapped difference between two angles (in radians), in `(-π, π]`.
#[must_use]
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let d = (a - b) % (2.0 * pi);
    if d < -pi {
        d + 2.0 * pi
    } else if d > pi {
        d - 2.0 * pi
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_lossless,
        clippy::field_reassign_with_default,
        clippy::doc_markdown,
        clippy::needless_range_loop
    )]

    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn angle_diff_same_angle() {
        assert_eq!(angle_diff(0.0, 0.0), 0.0);
        assert_eq!(angle_diff(1.0, 1.0), 0.0);
        assert_eq!(angle_diff(-2.0, -2.0), 0.0);
    }

    #[test]
    fn angle_diff_small_positive() {
        assert!((angle_diff(0.2, 0.0) - 0.2).abs() < 1e-10);
        assert!((angle_diff(0.1, -0.1) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn angle_diff_small_negative() {
        assert!((angle_diff(0.0, 0.2) + 0.2).abs() < 1e-10);
        assert!((angle_diff(-0.1, 0.1) + 0.2).abs() < 1e-10);
    }

    #[test]
    fn angle_diff_wrap_positive() {
        let eps = 0.1;
        let d = angle_diff(2.0 * PI + eps, 0.0);
        assert!((d - eps).abs() < 1e-10, "expected {eps}, got {d}");
    }

    #[test]
    fn angle_diff_wrap_negative() {
        let eps = 0.1;
        let d = angle_diff(0.0, 2.0 * PI - eps);
        assert!((d - eps).abs() < 1e-10, "expected {eps}, got {d}");
    }

    #[test]
    fn angle_diff_half_circle() {
        let d = angle_diff(PI, 0.0);
        assert!((d - PI).abs() < 1e-10);
    }
}
