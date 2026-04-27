//! Shared projection helpers for sampling-based closest-point search.
//!
//! Provides `sample_distances` to find the nearest bucket and
//! `golden_section_refine` for local refinement.

use path_traits::{Path, PathError, Point};

/// Coarse samples a clothoid arc and returns the arc-length `s` of the nearest sample.
///
/// Returns `(s_best, d_best², bucket_start_s, bucket_end_s)` where the bucket
/// brackets the nearest sample.
pub(crate) fn coarse_project(
    arc: &crate::path_traits_impls::arc::ClothoidArc,
    query: crate::Point2,
    n_steps: usize,
) -> Result<(f64, f64, f64, f64), PathError> {
    let length = arc.length;
    if length <= 0.0 {
        return Err(PathError::Degenerate);
    }

    let mut best_s = 0.0;
    let mut best_d2 = f64::MAX;

    for i in 0..=n_steps {
        let s = (i as f64) * length / (n_steps as f64);
        let pt = arc.sample_at(s)?;
        let d = query.displacement(pt);
        let d2 = d.dot(d);
        if d2 < best_d2 {
            best_d2 = d2;
            best_s = s;
        }
    }

    let bucket_half = length / (n_steps as f64);
    let bucket_start = (best_s - bucket_half).max(0.0);
    let bucket_end = (best_s + bucket_half).min(length);

    Ok((best_s, best_d2, bucket_start, bucket_end))
}

/// Refine the closest point using golden-section search on `[s_lo, s_hi]`.
///
/// Minimises the squared distance from `query` to `arc.sample_at(s)`.
/// Returns `(s_best, d_best²)`.
pub(crate) fn golden_section_refine(
    arc: &crate::path_traits_impls::arc::ClothoidArc,
    query: crate::Point2,
    s_lo: f64,
    s_hi: f64,
    tol: f64,
) -> Result<(f64, f64), PathError> {
    let gr = (3.0 - 5.0_f64.sqrt()) / 2.0;

    let mut a = s_lo;
    let mut b = s_hi;
    let mut c = a + gr * (b - a);
    let mut d = b - gr * (b - a);

    let dist2 = |s: f64| -> f64 {
        arc.sample_at(s)
            .map(|pt| {
                let dv = query.displacement(pt);
                dv.dot(dv)
            })
            .unwrap_or(f64::MAX)
    };

    let mut fc = dist2(c);
    let mut fd = dist2(d);

    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = a + gr * (b - a);
            fc = dist2(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - gr * (b - a);
            fd = dist2(d);
        }
    }

    let best_s = (a + b) / 2.0;
    let best_d2 = dist2(best_s);
    Ok((best_s, best_d2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Pose;
    use crate::path_traits_impls::arc::ClothoidArc;

    #[test]
    fn coarse_project_on_arc_near_straight() {
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 0.0,
            length: 10.0,
            n_steps: 100,
        };
        let query = crate::Point2 { x: 5.0, y: 0.0 };
        let (s, d2, _, _) = coarse_project(&arc, query, 100).unwrap();
        assert!((s - 5.0).abs() < 0.2);
        assert!(d2 < 1e-10);
    }

    #[test]
    fn golden_section_refinement_on_arc() {
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 0.0,
            length: 10.0,
            n_steps: 256,
        };
        let query = crate::Point2 { x: 3.7, y: 0.5 };
        let (s, d2) = golden_section_refine(&arc, query, 0.0, 10.0, 1e-10).unwrap();
        assert!((s - 3.7).abs() < 1e-4, "s={s} expected near 3.7");
        assert!((d2 - 0.25).abs() < 1e-4);
    }

    #[test]
    fn degenerate_arc_returns_error() {
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 1.0,
            length: 0.0,
            n_steps: 10,
        };
        let query = crate::Point2 { x: 1.0, y: 1.0 };
        assert!(coarse_project(&arc, query, 10).is_err());
    }
}
