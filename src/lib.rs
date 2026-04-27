//! # clothoid
//!
//! A Rust library for computing and fitting **Clothoids** (also known as
//! **Euler spirals** or **Cornu spirals**). A clothoid is a curve whose
//! curvature changes linearly with its arc length.
//!
//! ## Usage
//!
//! ```
//! use clothoid::Clothoid;
//!
//! let clothoid = Clothoid::new(1.0);
//! let angle = clothoid.direction_angle(0.5);
//! ```
//!
//! ## Choosing an Optimizer
//!
//! The `FitState` supports two derivative-free optimizers for path fitting:
//!
//! - **Nelder-Mead** (default) — The original simplex method. Fast per-iteration
//!   (500 evaluations per step), good for simple paths. Created with `FitState::new()`.
//! - **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy. More robust on
//!   difficult, non-convex landscapes but evaluates more candidates per generation.
//!   Created with `FitState::cma_es()`.
//!
//! Both share the same `FitState` API. Press `O` in the interactive demo to toggle
//! between them.
//!
//! ## Features
//!
//! - `fresnel` — enables high-precision Fresnel integral computation via the
//!   external `fresnel` crate. When disabled, uses an approximation based on
//!   auxiliary functions (Wilde 2009 / Abramowitz & Stegun).

pub mod fit;
pub mod optimizer;

pub use fit::{DefaultPlanner, Planner};
pub use optimizer::{CmaEs, NelderMead, Optimizer, PlanObjective, SymmetryMode};

/// The square root of π (`√π ≈ 1.77245`).
///
/// Used in the scaling of Fresnel integral computations.
#[allow(dead_code)]
const PI_SQRT: f64 =
    1.772_453_850_905_516_027_298_167_483_341_145_182_797_549_456_122_387_128_213_807_789_8_f64;

/// The inverse of the square root of π (`1/√π ≈ 0.56419`).
///
/// Used to normalize arguments before passing them to Fresnel integral
/// computations (see [`PI_SQRT`]).
#[allow(dead_code)]
const INV_PI_SQRT: f64 =
    0.564_189_583_547_756_286_948_079_451_560_772_585_844_050_629_328_998_856_844_085_721_7_f64;

/// A point in 2D Cartesian space.
///
/// Stores `x` and `y` coordinates as 64-bit floating-point values.
#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct Point2 {
    /// The x-coordinate.
    pub x: f64,
    /// The y-coordinate.
    pub y: f64,
}

/// A Clothoid (Euler spiral) curve.
///
/// A clothoid is a curve whose curvature changes linearly with its arc length.
/// This type provides methods to compute points on the spiral using Fresnel
/// integrals, either via the optional `fresnel` crate (high precision) or an
/// built-in approximation.
///
/// The scaling factor `a` determines the "tightness" of the spiral: larger
/// values produce gentler curves.
pub struct Clothoid {
    /// The scaling factor of the clothoid.
    ///
    /// Controls the rate at which curvature changes with arc length.
    a: f64,
}

impl Clothoid {
    /// Creates a new [`Clothoid`] with the given scaling factor `a`.
    ///
    /// # Arguments
    ///
    /// * `a` — The scaling factor. Larger values produce gentler curves.
    #[must_use]
    pub fn new(a: f64) -> Self {
        Self { a }
    }

    /// Computes the direction angle at a given arc length along the clothoid.
    ///
    /// The direction angle is given by `θ(s) = s² / (2a²)`, where `s` is the
    /// arc length and `a` is the scaling factor.
    ///
    /// # Arguments
    ///
    /// * `arc_length` — The arc length `s` along the clothoid.
    ///
    /// # Returns
    ///
    /// The direction angle in radians.
    #[inline]
    #[must_use]
    pub fn direction_angle(&self, arc_length: f64) -> f64 {
        0.5 * (arc_length * arc_length) / (self.a * self.a)
    }

    /// Computes a point on the clothoid at parameter `t`.
    ///
    /// Dispatches to either the high-precision `fresnel` crate implementation
    /// (when the `fresnel` feature is enabled) or the built-in approximation.
    ///
    /// # Arguments
    ///
    /// * `t` — The parameter value (proportional to arc length).
    ///
    /// # Returns
    ///
    /// A [`Point2`] on the clothoid curve.
    #[inline]
    #[allow(dead_code)]
    fn calculate(&self, t: f64) -> Point2 {
        #[cfg(feature = "fresnel")]
        {
            self.calculate_fresnl(t)
        }

        #[cfg(not(feature = "fresnel"))]
        {
            self.calculate_approx(t)
        }
    }

    /// Computes a point on the clothoid using the external `fresnel` crate.
    ///
    /// This method provides high-precision Fresnel integral computation.
    /// Only available when the `fresnel` feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `t` — The parameter value.
    ///
    /// # Returns
    ///
    /// A [`Point2`] on the clothoid curve.
    #[cfg(feature = "fresnel")]
    fn calculate_fresnl(&self, t: f64) -> Point2 {
        let (s, c) = fresnel::fresnl(t * INV_PI_SQRT);
        Point2 {
            x: self.a * PI_SQRT * s,
            y: self.a * PI_SQRT * c,
        }
    }

    /// Computes a point on the clothoid using the built-in approximation.
    ///
    /// Uses auxiliary functions `f(x)` and `g(x)` to approximate the Fresnel
    /// integrals (see [`AuxFg::compute`]).
    ///
    /// # Arguments
    ///
    /// * `t` — The parameter value.
    ///
    /// # Returns
    ///
    /// A [`Point2`] on the clothoid curve.
    #[allow(dead_code)]
    fn calculate_approx(&self, t: f64) -> Point2 {
        let fsc = FresnelSinCos::compute(t * INV_PI_SQRT);
        Point2 {
            x: self.a * PI_SQRT * fsc.sin,
            y: self.a * PI_SQRT * fsc.cos,
        }
    }
}

/// Auxiliary functions `f(x)` and `g(x)` for computing Fresnel integrals.
///
/// These functions are used in the approximation of the Fresnel sine and
/// cosine integrals `S(x)` and `C(x)`.
///
/// ## Sources
///
/// Doran K. Wilde, "Computing Clothoid Segments for Trajectory Generation".
/// IEEE/RSJ International Conference on Intelligent Robots and Systems, October 2009.
///
/// Abramowitz, Milton and Stegun, Irene A., (Editors), "Handbook of Mathematical
/// Functions with Formulas, Graphs, and Mathematical Tables".
/// National Bureau of Standards Applied Mathematics Series, No. 55, June 1964, pp. 295-303.
#[allow(dead_code)]
struct AuxFg {
    /// The value of the auxiliary function `f(x)`.
    pub f: f64,
    /// The value of the auxiliary function `g(x)`.
    pub g: f64,
}

impl AuxFg {
    /// Computes the auxiliary functions `f(x)` and `g(x)` for a given input.
    ///
    /// Uses rational polynomial approximations from Abramowitz & Stegun.
    ///
    /// ## Sources
    ///
    /// Doran K. Wilde, "Computing Clothoid Segments for Trajectory Generation".
    /// IEEE/RSJ International Conference on Intelligent Robots and Systems, October 2009.
    ///
    /// Abramowitz, Milton and Stegun, Irene A., (Editors), "Handbook of Mathematical
    /// Functions with Formulas, Graphs, and Mathematical Tables".
    /// National Bureau of Standards Applied Mathematics Series, No. 55, June 1964, pp. 295-303.
    ///
    /// # Arguments
    ///
    /// * `x` — The input value (typically normalized by `1/√π`).
    ///
    /// # Returns
    ///
    /// An [`AuxFg`] containing the computed `f` and `g` values.
    #[allow(dead_code)]
    pub fn compute(x: f64) -> Self {
        let x2 = x * x;
        let x3 = x * x * x;

        let f = (1. + 0.926 * x) / (2. + 1.792 * x + 3.104 * x2);
        let g = 1. / (2. + 4.142 * x + 3.492 * x2 + 6.670 * x3);

        Self { f, g }
    }
}

/// Fresnel sine and cosine integral values `S(x)` and `C(x)`.
///
/// These integrals are fundamental to computing points on a clothoid curve:
/// - `S(x) = ∫₀ˣ sin(πt²/2) dt`
/// - `C(x) = ∫₀ˣ cos(πt²/2) dt`
#[allow(dead_code)]
struct FresnelSinCos {
    /// The Fresnel cosine integral `C(x)`.
    pub cos: f64,
    /// The Fresnel sine integral `S(x)`.
    pub sin: f64,
}

impl FresnelSinCos {
    /// Computes the Fresnel sine and cosine integrals `S(x)` and `C(x)`.
    ///
    /// Uses the auxiliary functions `f(x)` and `g(x)` (see [`AuxFg::compute`])
    /// to approximate the integrals via:
    /// - `C(x) = 0.5 + f(x)·sin(πx²/2) - g(x)·cos(πx²/2)`
    /// - `S(x) = 0.5 - f(x)·cos(πx²/2) - g(x)·sin(πx²/2)`
    ///
    /// # Arguments
    ///
    /// * `x` — The input value (typically normalized by `1/√π`).
    ///
    /// # Returns
    ///
    /// A [`FresnelSinCos`] containing the computed `sin` and `cos` values.
    #[allow(dead_code)]
    pub fn compute(x: f64) -> Self {
        let aux = AuxFg::compute(x);
        let (sin, cos) = (x * x * std::f64::consts::FRAC_PI_2).sin_cos();
        Self {
            cos: 0.5 + aux.f * sin - aux.g * cos,
            sin: 0.5 - aux.f * cos - aux.g * sin,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::unreadable_literal
    )]

    use super::*;
    use assert_float_eq::*;

    #[test]
    fn it_works() {
        let clothoid = Clothoid::new(1.);
        let alpha = clothoid.direction_angle(0.);
        assert_eq!(alpha, 0.);
    }

    #[test]
    fn calculate() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_approx(std::f64::consts::PI);
        assert!((pt.x - 6.77).abs() < 0.01);
        assert!((pt.y - 4.59).abs() < 0.01);
    }

    #[test]
    fn clothoid_state_defaults() {
        use crate::optimizer::ClothoidState;
        let state = ClothoidState::default();
        assert_eq!(state.x, 0.0);
        assert_eq!(state.y, 0.0);
        assert_eq!(state.theta, 0.0);
    }

    #[test]
    #[cfg(feature = "fresnel")]
    fn calculate_fresnl_works() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate_fresnl(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_fresnl(std::f64::consts::PI);
        assert_f64_near!(pt.x, 6.7669799976205);
        assert_f64_near!(pt.y, 4.615663254508842);

        // http://jsxgraph.uni-bayreuth.de/wiki/index.php/Euler's_spiral_(Clothoid)
    }

    #[test]
    fn calculate_approx_works() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate_approx(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_approx(std::f64::consts::PI);
        assert_f64_near!(pt.x, 6.777113091819308);
        assert_f64_near!(pt.y, 4.588251163366395);
    }

    #[test]
    fn pochhammer() {
        // https://dlmf.nist.gov/7.12
        // https://dlmf.nist.gov/5.2#iii

        fn p(a: f64, n: usize) -> f64 {
            if n == 0 {
                return 1.;
            }
            let mut product = 1.;
            for i in 0..n {
                product *= a + (i as f64);
            }
            product
        }

        assert_eq!(p(0.5, 0), 1.);
        assert_eq!(p(0.5, 1), 0.5);
        assert_eq!(p(0.5, 2), 0.75);
        assert_eq!(p(0.5, 3), 1.875);
        assert_eq!(p(0.5, 7), 1055.7421875);

        // https://docs.google.com/spreadsheets/d/1xQJsACKpuro7ReS3RGYlTHNxuw4o7TUoN4E6i2s_Iwo/edit#gid=0
    }
}
