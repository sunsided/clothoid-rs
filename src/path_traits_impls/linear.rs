//! `LinearSegment` — a straight-line segment implementing `path_traits`.
//!
//! A linear segment has constant heading and zero curvature.

use path_traits::{
    Curved, FrenetFrame, Heading, ParametricPath, Path, PathError, PathSegment, Point, Project,
    Tangent,
};

use crate::optimizer::Pose;
use crate::path_traits_impls::primitive::Vec2;

/// A straight-line segment starting at `start` with given `length`.
///
/// The heading stays constant along the segment.
#[derive(Debug, Clone, Copy)]
pub struct LinearSegment {
    /// Starting pose (position and heading).
    pub start: Pose,
    /// Arc-length of the segment.
    pub length: f64,
}

impl LinearSegment {
    /// Computes the state (x, y, theta) at arc-length `s` along the segment.
    #[inline]
    fn state_at(&self, s: f64) -> (f64, f64, f64) {
        let theta = self.start.angle;
        let x = self.start.x + s * theta.cos();
        let y = self.start.y + s * theta.sin();
        (x, y, theta)
    }
}

impl Path for LinearSegment {
    type Point = crate::Point2;
    type Error = PathError;
    type Scalar = f64;

    fn length(&self) -> f64 {
        self.length
    }

    fn sample_at(&self, s: f64) -> Result<Self::Point, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::OutOfDomain);
        }
        let (x, y, _) = self.state_at(s);
        Ok(crate::Point2 { x, y })
    }
}

impl ParametricPath for LinearSegment {
    fn sample_t(&self, t: f64) -> Result<Self::Point, Self::Error> {
        if !(0.0..=1.0).contains(&t) {
            return Err(PathError::OutOfDomain);
        }
        self.sample_at(t * self.length)
    }
}

impl Tangent for LinearSegment {
    fn tangent_at(&self, s: f64) -> Result<<Self::Point as Point>::Vector, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::OutOfDomain);
        }
        let theta = self.start.angle;
        Ok(Vec2::new(theta.cos(), theta.sin()))
    }
}

impl Heading for LinearSegment {
    fn heading_at(&self, s: f64) -> Result<f64, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::OutOfDomain);
        }
        Ok(self.start.angle)
    }
}

impl Curved for LinearSegment {
    type Curvature = f64;

    fn curvature_at(&self, s: f64) -> Result<Self::Curvature, Self::Error> {
        if s < 0.0 || s > self.length {
            return Err(PathError::OutOfDomain);
        }
        Ok(0.0)
    }
}

impl FrenetFrame for LinearSegment {
    type Frame = (Vec2, Vec2);

    fn frame_at(&self, s: f64) -> Result<Self::Frame, Self::Error> {
        let t = self.tangent_at(s)?;
        let n = Vec2::new(-t.y, t.x);
        Ok((t, n))
    }
}

impl Project for LinearSegment {
    fn project(&self, p: Self::Point) -> Result<f64, Self::Error> {
        if self.length <= 0.0 {
            return Ok(0.0);
        }
        let dir = Vec2::new(self.start.angle.cos(), self.start.angle.sin());
        let dx = p.x - self.start.x;
        let dy = p.y - self.start.y;
        let t = dx * dir.x + dy * dir.y;
        Ok(t.clamp(0.0, self.length))
    }
}

impl PathSegment for LinearSegment {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment() -> LinearSegment {
        LinearSegment {
            start: Pose::new(1.0, 2.0, std::f64::consts::FRAC_PI_4),
            length: 10.0,
        }
    }

    #[test]
    fn linear_length() {
        let seg = make_segment();
        assert!((seg.length() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn linear_start_end() {
        let seg = make_segment();
        let start = seg.start().unwrap();
        assert!((start.x - 1.0).abs() < 1e-10);
        assert!((start.y - 2.0).abs() < 1e-10);

        let end = seg.end().unwrap();
        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();
        assert!((end.x - (1.0 + 10.0 * c)).abs() < 1e-6);
        assert!((end.y - (2.0 + 10.0 * s)).abs() < 1e-6);
    }

    #[test]
    fn linear_out_of_domain() {
        let seg = make_segment();
        assert!(seg.sample_at(-0.1).is_err());
        assert!(seg.sample_at(10.1).is_err());
        let err = seg.sample_at(-0.1).unwrap_err();
        assert_eq!(err, PathError::OutOfDomain);
    }

    #[test]
    fn linear_tangent_constant() {
        let seg = make_segment();
        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();
        for s_val in [0.0, 5.0, 10.0] {
            let t = seg.tangent_at(s_val).unwrap();
            assert!((t.x - c).abs() < 1e-10);
            assert!((t.y - s).abs() < 1e-10);
        }
    }

    #[test]
    fn linear_heading_constant() {
        let seg = make_segment();
        for s_val in [0.0, 3.0, 10.0] {
            let h = seg.heading_at(s_val).unwrap();
            assert!((h - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        }
    }

    #[test]
    fn linear_curvature_zero() {
        let seg = make_segment();
        for s_val in [0.0, 5.0, 10.0] {
            let k = seg.curvature_at(s_val).unwrap();
            assert!(k.abs() < 1e-10);
        }
    }

    #[test]
    fn linear_frenet_frame_orthonormal() {
        let seg = make_segment();
        let (t, n) = seg.frame_at(5.0).unwrap();
        assert!((t.dot(t) - 1.0).abs() < 1e-10);
        assert!((n.dot(n) - 1.0).abs() < 1e-10);
        assert!((t.dot(n)).abs() < 1e-10);
    }

    #[test]
    fn linear_project_on_line() {
        let seg = LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 10.0,
        };
        let p = crate::Point2 { x: 3.5, y: 0.0 };
        let s = seg.project(p).unwrap();
        assert!((s - 3.5).abs() < 1e-10);
    }

    #[test]
    fn linear_project_clamped_before() {
        let seg = LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 10.0,
        };
        let p = crate::Point2 { x: -5.0, y: 0.0 };
        let s = seg.project(p).unwrap();
        assert!((s - 0.0).abs() < 1e-10);
    }

    #[test]
    fn linear_project_clamped_after() {
        let seg = LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 10.0,
        };
        let p = crate::Point2 { x: 15.0, y: 0.0 };
        let s = seg.project(p).unwrap();
        assert!((s - 10.0).abs() < 1e-10);
    }

    #[test]
    fn linear_project_off_line() {
        let seg = LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 10.0,
        };
        let p = crate::Point2 { x: 3.0, y: 4.0 };
        let s = seg.project(p).unwrap();
        assert!((s - 3.0).abs() < 1e-10);
    }

    #[test]
    fn linear_degenerate_project_ok() {
        let seg = LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 0.0,
        };
        let p = crate::Point2 { x: 1.0, y: 0.0 };
        let s = seg.project(p).unwrap();
        assert!((s - 0.0).abs() < 1e-10);
    }

    #[test]
    fn linear_parametric_t_to_s() {
        let seg = make_segment();
        let pt = seg.sample_t(0.5).unwrap();
        let pt2 = seg.sample_at(5.0).unwrap();
        assert!((pt.x - pt2.x).abs() < 1e-10);
        assert!((pt.y - pt2.y).abs() < 1e-10);
    }

    #[test]
    fn linear_parametric_out_of_domain() {
        let seg = make_segment();
        assert!(seg.sample_t(-0.1).is_err());
        assert!(seg.sample_t(1.1).is_err());
    }
}
