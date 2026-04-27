//! `ArcSegment` — enum covering both `LinearSegment` and `ClothoidArc`.
//!
//! Used as the `Segment` associated type for `ClothoidPath`.

use path_traits::{
    Curved, FrenetFrame, Heading, ParametricPath, Path, PathError, PathSegment, Point, Project,
    Tangent,
};

use crate::path_traits_impls::arc::ClothoidArc;
use crate::path_traits_impls::linear::LinearSegment;
use crate::path_traits_impls::primitive::Vec2;

/// A segment that is either a straight line or a clothoid arc.
#[derive(Debug, Clone, Copy)]
pub enum ArcSegment {
    /// A straight-line segment.
    Linear(LinearSegment),
    /// A clothoid (Euler spiral) arc.
    Clothoid(ClothoidArc),
}

impl ArcSegment {
    /// Returns the length of this segment.
    #[must_use]
    pub fn segment_length(&self) -> f64 {
        match self {
            ArcSegment::Linear(s) => s.length,
            ArcSegment::Clothoid(a) => a.length,
        }
    }
}

impl Path for ArcSegment {
    type Point = crate::Point2;
    type Error = PathError<f64>;
    type Scalar = f64;

    fn length(&self) -> f64 {
        match self {
            ArcSegment::Linear(s) => s.length(),
            ArcSegment::Clothoid(a) => a.length(),
        }
    }

    fn sample_at(&self, s: f64) -> Result<Self::Point, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.sample_at(s),
            ArcSegment::Clothoid(arc) => arc.sample_at(s),
        }
    }

    fn start(&self) -> Result<Self::Point, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.start(),
            ArcSegment::Clothoid(arc) => arc.start(),
        }
    }

    fn end(&self) -> Result<Self::Point, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.end(),
            ArcSegment::Clothoid(arc) => arc.end(),
        }
    }

    fn domain(&self) -> core::ops::RangeInclusive<f64> {
        match self {
            ArcSegment::Linear(seg) => seg.domain(),
            ArcSegment::Clothoid(arc) => arc.domain(),
        }
    }
}

impl ParametricPath for ArcSegment {
    fn sample_t(&self, t: f64) -> Result<Self::Point, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.sample_t(t),
            ArcSegment::Clothoid(arc) => arc.sample_t(t),
        }
    }
}

impl Tangent for ArcSegment {
    fn tangent_at(&self, s: f64) -> Result<<Self::Point as Point>::Vector, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.tangent_at(s),
            ArcSegment::Clothoid(arc) => arc.tangent_at(s),
        }
    }
}

impl Heading for ArcSegment {
    fn heading_at(&self, s: f64) -> Result<f64, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.heading_at(s),
            ArcSegment::Clothoid(arc) => arc.heading_at(s),
        }
    }
}

impl Curved for ArcSegment {
    type Curvature = f64;

    fn curvature_at(&self, s: f64) -> Result<Self::Curvature, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.curvature_at(s),
            ArcSegment::Clothoid(arc) => arc.curvature_at(s),
        }
    }
}

impl FrenetFrame for ArcSegment {
    type Frame = (Vec2, Vec2);

    fn frame_at(&self, s: f64) -> Result<Self::Frame, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.frame_at(s),
            ArcSegment::Clothoid(arc) => arc.frame_at(s),
        }
    }
}

impl Project for ArcSegment {
    fn project(&self, p: Self::Point) -> Result<f64, Self::Error> {
        match self {
            ArcSegment::Linear(seg) => seg.project(p),
            ArcSegment::Clothoid(arc) => arc.project(p),
        }
    }
}

impl PathSegment for ArcSegment {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Pose;

    #[test]
    fn arc_segment_linear_length() {
        let seg = ArcSegment::Linear(LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 5.0,
        });
        assert!((seg.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn arc_segment_clothoid_length() {
        let seg = ArcSegment::Clothoid(ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 1.0,
            length: 3.0,
            n_steps: 64,
        });
        assert!((seg.length() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn arc_segment_sample_linear() {
        let seg = ArcSegment::Linear(LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 10.0,
        });
        let pt = seg.sample_at(5.0).unwrap();
        assert!((pt.x - 5.0).abs() < 1e-10);
        assert!(pt.y.abs() < 1e-10);
    }

    #[test]
    fn arc_segment_tangent_clothoid() {
        let seg = ArcSegment::Clothoid(ClothoidArc {
            start: Pose::new(0.0, 0.0, 0.0),
            ks: 0.0,
            ke: 0.0,
            length: 5.0,
            n_steps: 64,
        });
        let t = seg.tangent_at(2.5).unwrap();
        assert!((t.x - 1.0).abs() < 1e-6);
        assert!(t.y.abs() < 1e-6);
    }

    #[test]
    fn arc_segment_project_linear() {
        let seg = ArcSegment::Linear(LinearSegment {
            start: Pose::new(0.0, 0.0, 0.0),
            length: 10.0,
        });
        let p = crate::Point2 { x: 7.0, y: 0.0 };
        let s = seg.project(p).unwrap();
        assert!((s - 7.0).abs() < 1e-10);
    }
}
