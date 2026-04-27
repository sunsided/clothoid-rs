//! `ClothoidArc` — a clothoid (Euler spiral) arc implementing `path_traits`.
//!
//! Curvature varies linearly from `ks` to `ke` over `length`.
//! Heading: `θ(s) = θ₀ + ks·s + ½·(ke−ks)·s²/L`
//! Curvature: `κ(s) = ks + (ke−ks)·s/L`

use path_traits::{
    Curved, FrenetFrame, Heading, ParametricPath, Path, PathError, PathSegment, Point, Project,
    Tangent,
};

use crate::optimizer::{Pose, integrate_clothoid};
use crate::path_traits_impls::primitive::Vec2;
use crate::path_traits_impls::project::{coarse_project, golden_section_refine};

/// A clothoid arc with linearly varying curvature.
///
/// The curvature at arc-length `s ∈ [0, length]` is
/// `κ(s) = ks + (ke - ks) * s / length`.
#[derive(Debug, Clone, Copy)]
pub struct ClothoidArc {
    /// Starting pose (position and heading).
    pub start: Pose,
    /// Curvature at `s = 0`.
    pub ks: f64,
    /// Curvature at `s = length`.
    pub ke: f64,
    /// Total arc-length.
    pub length: f64,
    /// Number of RK4 steps for sampling. Default: 256.
    pub n_steps: usize,
}

impl ClothoidArc {
    /// Returns the heading angle `θ(s)` at arc-length `s` (closed-form).
    #[inline]
    #[must_use]
    pub fn heading_at_s(&self, s: f64) -> f64 {
        let dk = self.ke - self.ks;
        self.start.angle + self.ks * s + 0.5 * dk * s * s / self.length
    }

    /// Returns the curvature `κ(s)` at arc-length `s` (closed-form).
    #[inline]
    #[must_use]
    pub fn curvature_at_s(&self, s: f64) -> f64 {
        self.ks + (self.ke - self.ks) * s / self.length
    }

    /// Integrates from `start` to arc-length `s` using RK4.
    fn integrate_to(&self, s_query: f64) -> (f64, f64, f64) {
        if s_query <= 0.0 {
            return (self.start.x, self.start.y, self.start.angle);
        }
        let n_eff = (self.n_steps as f64 * s_query / self.length)
            .ceil()
            .max(1.0) as usize;
        let states = integrate_clothoid(
            self.start.x,
            self.start.y,
            self.start.angle,
            self.ks,
            self.ke,
            s_query,
            n_eff,
        );
        let last = states
            .last()
            .copied()
            .unwrap_or(crate::optimizer::ClothoidState {
                x: self.start.x,
                y: self.start.y,
                theta: self.start.angle,
            });
        (last.x, last.y, last.theta)
    }
}

impl Path for ClothoidArc {
    type Point = crate::Point2;
    type Error = PathError<f64>;
    type Scalar = f64;

    fn length(&self) -> f64 {
        self.length
    }

    fn sample_at(&self, s: f64) -> Result<Self::Point, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::out_of_domain(s, self.domain()));
        }
        let (x, y, _) = self.integrate_to(s);
        Ok(crate::Point2 { x, y })
    }
}

impl ParametricPath for ClothoidArc {
    fn sample_t(&self, t: f64) -> Result<Self::Point, Self::Error> {
        if !(0.0..=1.0).contains(&t) {
            return Err(PathError::out_of_domain(t, 0.0..=1.0));
        }
        self.sample_at(t * self.length)
    }
}

impl Tangent for ClothoidArc {
    fn tangent_at(&self, s: f64) -> Result<<Self::Point as Point>::Vector, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::out_of_domain(s, self.domain()));
        }
        let theta = self.heading_at_s(s);
        Ok(Vec2::new(theta.cos(), theta.sin()))
    }
}

impl Heading for ClothoidArc {
    fn heading_at(&self, s: f64) -> Result<f64, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::out_of_domain(s, self.domain()));
        }
        Ok(self.heading_at_s(s))
    }
}

impl Curved for ClothoidArc {
    type Curvature = f64;

    fn curvature_at(&self, s: f64) -> Result<Self::Curvature, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::out_of_domain(s, self.domain()));
        }
        Ok(self.curvature_at_s(s))
    }
}

impl FrenetFrame for ClothoidArc {
    type Frame = (Vec2, Vec2);

    fn frame_at(&self, s: f64) -> Result<Self::Frame, Self::Error> {
        let t = self.tangent_at(s)?;
        let n = Vec2::new(-t.y, t.x);
        Ok((t, n))
    }
}

impl Project for ClothoidArc {
    fn project(&self, p: Self::Point) -> Result<f64, Self::Error> {
        if self.length <= 0.0 {
            return Err(PathError::degenerate("zero-length clothoid arc"));
        }

        let (_best_s, _, bucket_lo, bucket_hi) = coarse_project(self, p, self.n_steps)?;

        let tol = self.length / (self.n_steps as f64) * 0.01;
        let (s_refined, _) = golden_section_refine(self, p, bucket_lo, bucket_hi, tol)?;

        Ok(s_refined.clamp(0.0, self.length))
    }
}

impl PathSegment for ClothoidArc {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::ClothoidState;

    fn make_arc() -> ClothoidArc {
        ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 1.0,
            length: 5.0,
            n_steps: 256,
        }
    }

    #[test]
    fn clothoid_arc_start() {
        let arc = make_arc();
        let pt = arc.sample_at(0.0).unwrap();
        assert!((pt.x - 0.0).abs() < 1e-10);
        assert!((pt.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn clothoid_arc_end_matches_integrate() {
        let arc = make_arc();
        let pt = arc.sample_at(arc.length).unwrap();
        let states = integrate_clothoid(
            arc.start.x,
            arc.start.y,
            arc.start.angle,
            arc.ks,
            arc.ke,
            arc.length,
            4096,
        );
        let last: &ClothoidState = states.last().unwrap();
        assert!((pt.x - last.x).abs() < 1e-6);
        assert!((pt.y - last.y).abs() < 1e-6);
    }

    #[test]
    fn clothoid_arc_straight_line() {
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 0.0,
            length: 5.0,
            n_steps: 256,
        };
        let pt = arc.sample_at(5.0).unwrap();
        assert!((pt.x - 5.0).abs() < 1e-6);
        assert!(pt.y.abs() < 1e-6);
    }

    #[test]
    fn clothoid_arc_constant_curvature_heading() {
        let k = 0.5;
        let len = std::f64::consts::PI;
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: k,
            ke: k,
            length: len,
            n_steps: 512,
        };
        let expected_theta = k * len;
        assert!((arc.heading_at_s(len) - expected_theta).abs() < 1e-10);
    }

    #[test]
    fn clothoid_arc_linear_curvature() {
        let arc = make_arc();
        let k0 = arc.curvature_at_s(0.0);
        let k1 = arc.curvature_at_s(arc.length);
        let kmid = arc.curvature_at_s(arc.length / 2.0);
        assert!((k0 - 0.0).abs() < 1e-10);
        assert!((k1 - 1.0).abs() < 1e-10);
        assert!((kmid - 0.5).abs() < 1e-10);
    }

    #[test]
    fn clothoid_arc_heading_closed_form() {
        let arc = make_arc();
        for s_val in [0.0, 1.0, 2.5, 5.0] {
            let h = arc.heading_at(s_val).unwrap();
            let expected = arc.heading_at_s(s_val);
            assert!((h - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn clothoid_arc_tangent_unit_length() {
        let arc = make_arc();
        for s_val in [0.0, 1.0, 2.5, 5.0] {
            let t = arc.tangent_at(s_val).unwrap();
            let len = (t.x * t.x + t.y * t.y).sqrt();
            assert!((len - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn clothoid_arc_frenet_frame_orthonormal() {
        let arc = make_arc();
        for s_val in [0.0, 2.5, 5.0] {
            let (t, n) = arc.frame_at(s_val).unwrap();
            assert!((t.dot(t) - 1.0).abs() < 1e-10);
            assert!((n.dot(n) - 1.0).abs() < 1e-10);
            assert!((t.dot(n)).abs() < 1e-10);
        }
    }

    #[test]
    fn clothoid_arc_out_of_domain() {
        let arc = make_arc();
        assert!(arc.sample_at(-0.1).is_err());
        assert!(arc.sample_at(5.1).is_err());
    }

    #[test]
    fn clothoid_arc_project_recovers_s() {
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 1.0,
            length: 5.0,
            n_steps: 512,
        };
        let target_s = 2.5;
        let pt = arc.sample_at(target_s).unwrap();
        let heading = arc.heading_at(target_s).unwrap();
        let normal = Vec2::new(-heading.sin(), heading.cos());
        let offset = crate::Point2 {
            x: pt.x + normal.x * 0.1,
            y: pt.y + normal.y * 0.1,
        };
        let s = arc.project(offset).unwrap();
        let tol = arc.length / (arc.n_steps as f64) * 3.0;
        assert!(
            (s - target_s).abs() < tol,
            "project s={s} expected near {target_s}"
        );
    }

    #[test]
    fn clothoid_arc_degenerate_project() {
        let arc = ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 1.0,
            length: 0.0,
            n_steps: 256,
        };
        let p = crate::Point2 { x: 1.0, y: 0.0 };
        assert!(arc.project(p).is_err());
    }

    #[test]
    fn clothoid_arc_parametric() {
        let arc = make_arc();
        let pt_half_t = arc.sample_t(0.5).unwrap();
        let pt_half_s = arc.sample_at(2.5).unwrap();
        assert!((pt_half_t.x - pt_half_s.x).abs() < 1e-6);
        assert!((pt_half_t.y - pt_half_s.y).abs() < 1e-6);
    }

    #[test]
    fn clothoid_arc_curvature_impl_at() {
        let arc = make_arc();
        for s_val in [0.0, 1.0, 2.5, 5.0] {
            let k_impl = arc.curvature_at(s_val).unwrap();
            let k_direct = arc.curvature_at_s(s_val);
            assert!((k_impl - k_direct).abs() < 1e-10);
        }
    }
}
