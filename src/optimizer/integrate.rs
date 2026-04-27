//! RK4 integration for clothoid arcs with linearly varying curvature.
//!
//! Provides 4th-order Runge-Kutta stepping and full-path integration,
//! returning sampled `(x, y, θ)` states along the curve.

use super::state::ClothoidState;

/// Performs a single 4th-order Runge-Kutta integration step along a clothoid arc.
///
/// The clothoid has linearly varying curvature:
/// `κ(s) = ks + (ke - ks) * s / length`.
///
/// State: `(x, y, θ)`. Derivatives: `ẋ = cos θ`, `ẏ = sin θ`, `θ̇ = κ(s)`.
///
/// # Arguments
/// * `state` - current state (x, y, theta)
/// * `ks`, `ke` - curvature at start and end of the clothoid segment
/// * `length` - total arc length of the segment (used for curvature interpolation)
/// * `s` - current arc-length position within the segment
/// * `h` - step size
///
/// # Returns
/// Next state `(x, y, θ)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn rk4_step(
    state: ClothoidState,
    ks: f64,
    ke: f64,
    length: f64,
    s: f64,
    h: f64,
) -> ClothoidState {
    let ClothoidState { x, y, theta } = state;
    let kappa = |s: f64| ks + (ke - ks) * s / length;

    let (dx1, dy1, dth1) = (theta.cos(), theta.sin(), kappa(s));
    let th2 = theta + h / 2.0 * dth1;
    let (dx2, dy2, dth2) = (th2.cos(), th2.sin(), kappa(s + h / 2.0));
    let th3 = theta + h / 2.0 * dth2;
    let (dx3, dy3, dth3) = (th3.cos(), th3.sin(), kappa(s + h / 2.0));
    let th4 = theta + h * dth3;
    let (dx4, dy4, dth4) = (th4.cos(), th4.sin(), kappa(s + h));

    let nx = x + h / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
    let ny = y + h / 6.0 * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4);
    let nt = theta + h / 6.0 * (dth1 + 2.0 * dth2 + 2.0 * dth3 + dth4);
    ClothoidState {
        x: nx,
        y: ny,
        theta: nt,
    }
}

/// Integrates a clothoid arc using RK4, returning all sampled `(x, y, θ)` states.
///
/// The curvature varies linearly from `ks` at `s=0` to `ke` at `s=length`.
///
/// Returns a vector of length `n_steps + 1` (including the initial state).
/// If `length <= 0` or `n_steps == 0`, returns a single-element vector containing
/// the start state.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn integrate_clothoid(
    x0: f64,
    y0: f64,
    theta0: f64,
    ks: f64,
    ke: f64,
    length: f64,
    n_steps: usize,
) -> Vec<ClothoidState> {
    if length <= 0.0 || n_steps == 0 {
        return vec![ClothoidState {
            x: x0,
            y: y0,
            theta: theta0,
        }];
    }
    let h = length / n_steps as f64;
    let mut state = ClothoidState {
        x: x0,
        y: y0,
        theta: theta0,
    };
    let mut pts = Vec::with_capacity(n_steps + 1);
    pts.push(state);
    for step in 0..n_steps {
        let s = step as f64 * h;
        state = rk4_step(state, ks, ke, length, s, h);
        pts.push(state);
    }
    pts
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
    fn rk4_step_straight_x() {
        let result = rk4_step(
            ClothoidState {
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            },
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        );
        assert!(
            (result.x - 1.0).abs() < 1e-10,
            "x should advance by step h=1"
        );
        assert!(result.y.abs() < 1e-10, "y should not change");
        assert!(result.theta.abs() < 1e-10, "θ should not change");
    }

    #[test]
    fn rk4_step_straight_y() {
        let result = rk4_step(
            ClothoidState {
                x: 0.0,
                y: 0.0,
                theta: PI / 2.0,
            },
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        );
        assert!(result.x.abs() < 1e-10, "x should not change");
        assert!((result.y - 1.0).abs() < 1e-10, "y should advance by h=1");
        assert!(
            (result.theta - PI / 2.0).abs() < 1e-10,
            "θ should not change"
        );
    }

    #[test]
    fn rk4_step_constant_curvature_angle() {
        let kappa = 1.0;
        let h = 0.1;
        let result = rk4_step(
            ClothoidState {
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            },
            kappa,
            kappa,
            1.0,
            0.0,
            h,
        );
        assert!((result.theta - kappa * h).abs() < 1e-6);
    }

    #[test]
    fn rk4_step_non_origin_start() {
        let result = rk4_step(
            ClothoidState {
                x: 3.0,
                y: -2.0,
                theta: 0.0,
            },
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
        );
        assert!((result.x - 5.0).abs() < 1e-10);
        assert!((result.y + 2.0).abs() < 1e-10);
        assert!(result.theta.abs() < 1e-10);
    }

    #[test]
    fn integrate_clothoid_zero_length() {
        let pts = integrate_clothoid(1.0, 2.0, 0.5, 1.0, -1.0, 0.0, 100);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].x, 1.0);
        assert_eq!(pts[0].y, 2.0);
        assert_eq!(pts[0].theta, 0.5);
    }

    #[test]
    fn integrate_clothoid_zero_steps() {
        let pts = integrate_clothoid(0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0);
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn integrate_clothoid_straight_line_x() {
        let length = 5.0;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, 0.0, 0.0, length, 200);
        let last = pts.last().unwrap();
        assert!((last.x - length).abs() < 1e-6, "x should equal arc length");
        assert!(last.y.abs() < 1e-6, "y should be zero");
        assert!(last.theta.abs() < 1e-6, "heading should be unchanged");
    }

    #[test]
    fn integrate_clothoid_straight_line_y() {
        let length = 3.0;
        let pts = integrate_clothoid(0.0, 0.0, PI / 2.0, 0.0, 0.0, length, 200);
        let last = pts.last().unwrap();
        assert!(last.x.abs() < 1e-6);
        assert!((last.y - length).abs() < 1e-6);
        assert!((last.theta - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn integrate_clothoid_constant_kappa_angle_change() {
        let kappa = 1.0;
        let arc = PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 1000);
        let last = pts.last().unwrap();
        assert!(
            (last.theta - PI).abs() < 1e-3,
            "heading should rotate by π rad"
        );
    }

    #[test]
    fn integrate_clothoid_half_circle_position() {
        let kappa = 1.0;
        let arc = PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 2000);
        let last = pts.last().unwrap();
        assert!(
            last.x.abs() < 1e-3,
            "x should return to ~0 after half circle"
        );
        assert!((last.y - 2.0).abs() < 1e-3, "y should be 2*radius = 2");
    }

    #[test]
    fn integrate_clothoid_full_circle() {
        let kappa = 1.0;
        let arc = 2.0 * PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 4000);
        let last = pts.last().unwrap();
        assert!(last.x.abs() < 1e-2, "x should return near 0");
        assert!(last.y.abs() < 1e-2, "y should return near 0");
        assert!((last.theta - 2.0 * PI).abs() < 1e-2);
    }

    #[test]
    fn integrate_clothoid_linearly_varying_kappa_angle() {
        let ks = 0.0;
        let ke = 2.0;
        let length = 1.0;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, ks, ke, length, 1000);
        let last = pts.last().unwrap();
        let expected = f64::midpoint(ks, ke) * length;
        assert!((last.theta - expected).abs() < 1e-4);
    }

    #[test]
    fn integrate_clothoid_point_count() {
        let n = 50;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, n);
        assert_eq!(pts.len(), n + 1);
    }
}
