//! Primitive geometric types for `path-traits` integration.
//!
//! Provides `Vec2` implementing `path_traits::Vector` and a `path_traits::Point`
//! impl for the existing `crate::Point2`.

use std::ops::{Add, Mul, Sub};

/// A 2D vector with `f64` components.
#[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
pub struct Vec2 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
}

impl Vec2 {
    /// Creates a new `Vec2`.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Dot product of two vectors.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Euclidean norm (length).
    #[must_use]
    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl path_traits::Vector for Vec2 {
    type Scalar = f64;

    fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    fn dot(self, other: Self) -> Self::Scalar {
        self.dot(other)
    }

    fn norm(self) -> Self::Scalar {
        self.norm()
    }
}

impl path_traits::Point for crate::Point2 {
    type Scalar = f64;
    type Vector = Vec2;

    fn displacement(self, other: Self) -> Self::Vector {
        Vec2 {
            x: other.x - self.x,
            y: other.y - self.y,
        }
    }

    fn translate(self, v: Self::Vector) -> Self {
        Self {
            x: self.x + v.x,
            y: self.y + v.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use path_traits::Point;

    #[test]
    fn vec2_add() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a + b;
        assert!((c.x - 4.0).abs() < 1e-10);
        assert!((c.y - 6.0).abs() < 1e-10);
    }

    #[test]
    fn vec2_sub() {
        let a = Vec2::new(5.0, 7.0);
        let b = Vec2::new(2.0, 3.0);
        let c = a - b;
        assert!((c.x - 3.0).abs() < 1e-10);
        assert!((c.y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn vec2_mul_scalar() {
        let a = Vec2::new(1.0, 2.0);
        let c = a * 3.0;
        assert!((c.x - 3.0).abs() < 1e-10);
        assert!((c.y - 6.0).abs() < 1e-10);
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dot(b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn vec2_norm() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn vec2_zero() {
        let z = <Vec2 as path_traits::Vector>::zero();
        assert!((z.x - 0.0).abs() < 1e-10);
        assert!((z.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn point_displacement_translate_roundtrip() {
        let a = crate::Point2 { x: 1.0, y: 2.0 };
        let b = crate::Point2 { x: 4.0, y: 6.0 };
        let d = a.displacement(b);
        let c = a.translate(d);
        assert!((c.x - b.x).abs() < 1e-10);
        assert!((c.y - b.y).abs() < 1e-10);
    }

    #[test]
    fn point_distance_nonnegative() {
        let a = crate::Point2 { x: 0.0, y: 0.0 };
        let b = crate::Point2 { x: 3.0, y: 4.0 };
        let d = a.distance(b);
        assert!((d - 5.0).abs() < 1e-10);
    }
}
