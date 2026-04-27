//! Optimization objective functions for clothoid path fitting.
//!
//! Provides configurable weighted objectives that penalize endpoint errors,
//! negative lengths, curvature limits, sign flips, G2 discontinuities,
//! symmetry deviations, and other geometric properties.

use super::angle::angle_diff;
use super::path::eval_path;
use super::state::ClothoidState;

/// Symmetry mode for the objective function.
///
/// Controls when the symmetry penalty term is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SymmetryMode {
    /// Enable symmetry penalty only when start/end poses are mirror-symmetric.
    #[default]
    Auto,
    /// Always disable symmetry penalty.
    Off,
    /// Always enable symmetry penalty.
    On,
}

/// Configurable weighted objective for clothoid path optimization.
///
/// Each term has an associated weight; setting a weight to `0.0` disables that term.
///
/// Use `PlanObjective::default()` for legacy-compatible behaviour (bit-compatible
/// with the original `compute_error`). Use `PlanObjective::recommended()` for the
/// full set of tuned penalties.
#[derive(Clone, Debug)]
pub struct PlanObjective {
    pub w_end_pos: f64,
    pub w_end_angle: f64,
    pub w_neg_length: f64,
    pub w_micro_arc: f64,
    pub micro_arc_threshold: f64,
    pub max_kappa: f64,
    pub w_max_kappa: f64,
    pub w_sign_flips: f64,
    pub w_kappa_rate: f64,
    pub w_g2: f64,
    pub w_kappa_start_zero: f64,
    pub w_kappa_end_zero: f64,
    pub min_seg_len: f64,
    pub w_min_seg_len: f64,
    pub w_total_length: f64,
    pub target_length: Option<f64>,
    pub symmetry: SymmetryMode,
    pub w_symmetry: f64,
}

impl Default for PlanObjective {
    fn default() -> Self {
        Self {
            w_end_pos: 10.0,
            w_end_angle: 5.0,
            w_neg_length: 100.0,
            w_micro_arc: 10.0,
            micro_arc_threshold: 0.1,
            max_kappa: 1e30,
            w_max_kappa: 0.0,
            w_sign_flips: 0.0,
            w_kappa_rate: 0.0,
            w_g2: 0.0,
            w_kappa_start_zero: 0.0,
            w_kappa_end_zero: 0.0,
            min_seg_len: 0.0,
            w_min_seg_len: 0.0,
            w_total_length: 0.001,
            target_length: None,
            symmetry: SymmetryMode::Auto,
            w_symmetry: 0.0,
        }
    }
}

impl PlanObjective {
    /// Returns an objective with sensible defaults for all new terms enabled.
    ///
    /// This is the recommended starting point for interactive use.
    #[must_use]
    pub fn recommended() -> Self {
        Self {
            w_end_pos: 10.0,
            w_end_angle: 5.0,
            w_neg_length: 100.0,
            w_micro_arc: 10.0,
            micro_arc_threshold: 0.1,
            max_kappa: 2.0,
            w_max_kappa: 5.0,
            w_sign_flips: 0.5,
            w_kappa_rate: 0.1,
            w_g2: 1.0,
            w_kappa_start_zero: 0.0,
            w_kappa_end_zero: 0.0,
            min_seg_len: 0.0,
            w_min_seg_len: 10.0,
            w_total_length: 0.001,
            target_length: None,
            symmetry: SymmetryMode::Auto,
            w_symmetry: 1.0,
        }
    }

    /// Computes the weighted objective value for the given path parameters.
    ///
    /// # Arguments
    /// * `params` — flat parameter slice of length `4 * n_clothoids + 1`
    /// * `n_clothoids` — number of clothoid segments
    /// * `start` — start pose
    /// * `end` — end pose
    ///
    /// # Returns
    /// The scalar objective value (lower is better).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compute(
        &self,
        params: &[f64],
        n_clothoids: usize,
        start: &super::pose::Pose,
        end: &super::pose::Pose,
    ) -> f64 {
        let mut neg_penalty = 0.0f64;
        let mut len_penalty = 0.0f64;
        let mut total_length = 0.0f64;
        let mut max_kappa_penalty = 0.0f64;
        let mut sign_flip_penalty = 0.0f64;
        let mut kappa_rate_penalty = 0.0f64;
        let mut g2_penalty = 0.0f64;
        let mut min_seg_penalty = 0.0f64;

        for i in 0..n_clothoids {
            let base = 4 * i;
            let l = params[base];
            let ks = params[base + 1];
            let ke = params[base + 2];
            let clen = params[base + 3];

            if l < 0.0 {
                neg_penalty += self.w_neg_length * l * l;
            }
            if clen < 0.0 {
                neg_penalty += self.w_neg_length * clen * clen;
            }
            if (0.0..self.micro_arc_threshold).contains(&clen) {
                len_penalty += self.w_micro_arc * (self.micro_arc_threshold - clen).powi(2);
            }
            total_length += l.max(0.0) + clen.max(0.0);

            if self.w_max_kappa > 0.0 && self.max_kappa < 1e20 {
                let mk = self.max_kappa;
                if ks.abs() > mk {
                    max_kappa_penalty += self.w_max_kappa * (ks.abs() - mk).powi(2);
                }
                if ke.abs() > mk {
                    max_kappa_penalty += self.w_max_kappa * (ke.abs() - mk).powi(2);
                }
            }

            if self.w_kappa_rate > 0.0 {
                let clen_abs = clen.abs().max(1e-12);
                let rate = (ke - ks) / clen_abs;
                kappa_rate_penalty += self.w_kappa_rate * rate * rate;
            }

            if self.w_min_seg_len > 0.0 && self.min_seg_len > 0.0 {
                let ml = self.min_seg_len;
                if l < ml {
                    min_seg_penalty += self.w_min_seg_len * (ml - l).powi(2);
                }
                if clen < ml {
                    min_seg_penalty += self.w_min_seg_len * (ml - clen).powi(2);
                }
            }
        }

        let l_final = params[4 * n_clothoids];
        if l_final < 0.0 {
            neg_penalty += self.w_neg_length * l_final * l_final;
        }
        total_length += l_final.max(0.0);

        if self.w_min_seg_len > 0.0 && self.min_seg_len > 0.0 && l_final < self.min_seg_len {
            min_seg_penalty += self.w_min_seg_len * (self.min_seg_len - l_final).powi(2);
        }

        let pts = eval_path(params, n_clothoids, start, 20);
        let default = ClothoidState {
            x: start.x,
            y: start.y,
            theta: start.angle,
        };
        let last = pts.last().unwrap_or(&default);

        if last.x.is_nan() || last.y.is_nan() || last.theta.is_nan() {
            return 1e10;
        }

        let dx = last.x - end.x;
        let dy = last.y - end.y;
        let dist_sq = dx * dx + dy * dy;
        let ad = angle_diff(last.theta, end.angle);

        let end_pos_term = self.w_end_pos * dist_sq;
        let end_angle_term = self.w_end_angle * ad * ad;

        let length_term = match self.target_length {
            Some(t) => self.w_total_length * (total_length - t).powi(2),
            None => self.w_total_length * total_length,
        };

        let endpoint_kappa_term = {
            let mut t = 0.0;
            if self.w_kappa_start_zero > 0.0 {
                let ks0 = params[1];
                t += self.w_kappa_start_zero * ks0 * ks0;
            }
            if self.w_kappa_end_zero > 0.0 {
                let ke_last = params[4 * (n_clothoids - 1) + 2];
                t += self.w_kappa_end_zero * ke_last * ke_last;
            }
            t
        };

        if self.w_sign_flips > 0.0 && n_clothoids >= 2 {
            let kappa_avg: Vec<f64> = (0..n_clothoids)
                .map(|i| {
                    let base = 4 * i;
                    f64::midpoint(params[base + 1], params[base + 2])
                })
                .collect();
            for i in 0..n_clothoids - 1 {
                let ka = kappa_avg[i];
                let kb = kappa_avg[i + 1];
                if ka.abs() < 1e-3 || kb.abs() < 1e-3 {
                    continue;
                }
                let proxy = (-ka * kb).max(0.0);
                if proxy > 0.0 {
                    sign_flip_penalty += self.w_sign_flips * proxy;
                }
            }
        }

        if self.w_g2 > 0.0 && n_clothoids >= 2 {
            for i in 0..n_clothoids - 1 {
                let ke_i = params[4 * i + 2];
                let ks_next = params[4 * (i + 1) + 1];
                g2_penalty += self.w_g2 * (ke_i - ks_next).powi(2);
            }
        }

        let symmetry_term = if self.w_symmetry > 0.0 {
            let active = match self.symmetry {
                SymmetryMode::On => true,
                SymmetryMode::Off => false,
                SymmetryMode::Auto => is_symmetric_task(start, end),
            };
            if active {
                self.w_symmetry * symmetry_distance(params, n_clothoids)
            } else {
                0.0
            }
        } else {
            0.0
        };

        end_pos_term
            + end_angle_term
            + neg_penalty
            + len_penalty
            + max_kappa_penalty
            + sign_flip_penalty
            + kappa_rate_penalty
            + g2_penalty
            + endpoint_kappa_term
            + min_seg_penalty
            + length_term
            + symmetry_term
    }
}

/// Checks whether the start and end poses are mirror-symmetric around the midpoint.
///
/// Returns true when reflecting `end` through the midpoint and flipping the heading
/// reproduces `start` within a tight tolerance.
#[must_use]
pub fn is_symmetric_task(start: &super::pose::Pose, end: &super::pose::Pose) -> bool {
    let mid_x = f64::midpoint(start.x, end.x);
    let mid_y = f64::midpoint(start.y, end.y);

    let reflected_end_x = 2.0 * mid_x - end.x;
    let reflected_end_y = 2.0 * mid_y - end.y;

    let pos_tol = 1e-6;
    let angle_tol = 1e-6;

    let dx = (start.x - reflected_end_x).abs();
    let dy = (start.y - reflected_end_y).abs();

    let reflected_angle = (end.angle + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
    let angle_d = angle_diff(start.angle, reflected_angle).abs();

    dx < pos_tol && dy < pos_tol && angle_d < angle_tol
}

/// Computes the squared L2 distance between a parameter vector and its mirror image.
///
/// Mirror: reverse segment order, negate curvatures.
fn symmetry_distance(params: &[f64], n_clothoids: usize) -> f64 {
    if n_clothoids == 0 {
        return 0.0;
    }
    let n_params = 4 * n_clothoids + 1;
    let mut mirrored = vec![0.0f64; n_params];

    for i in 0..n_clothoids {
        let rev = n_clothoids - 1 - i;
        let src_base = 4 * rev;
        let dst_base = 4 * i;
        mirrored[dst_base] = params[src_base];
        mirrored[dst_base + 1] = -params[src_base + 2];
        mirrored[dst_base + 2] = -params[src_base + 1];
        mirrored[dst_base + 3] = params[src_base + 3];
    }
    mirrored[n_params - 1] = params[n_params - 1];

    let mut sum = 0.0;
    for i in 0..n_params {
        let d = params[i] - mirrored[i];
        sum += d * d;
    }
    sum
}

/// Computes the scalar optimization objective for a path described by `params`.
///
/// The objective penalises:
/// - Distance² from the path endpoint to `end.x/y` (weight 10)
/// - Angle error² at the endpoint (weight 5)
/// - Negative segment lengths (soft constraint, weight 100)
/// - Very short clothoid arcs (soft constraint, weight 10)
/// - Total path length (regularisation, weight 0.001)
#[must_use]
pub fn compute_error(
    params: &[f64],
    n_clothoids: usize,
    start: &super::pose::Pose,
    end: &super::pose::Pose,
) -> f64 {
    PlanObjective::default().compute(params, n_clothoids, start, end)
}

/// Returns the un-weighted `(position_error, |angle_error|)` for a candidate path.
///
/// * `position_error` — Euclidean distance from path endpoint to `end.x/y`
/// * `angle_error` — absolute wrapped angle difference at the endpoint
#[must_use]
pub fn compute_end_errors(
    params: &[f64],
    n_clothoids: usize,
    start: &super::pose::Pose,
    end: &super::pose::Pose,
) -> (f64, f64) {
    let pts = eval_path(params, n_clothoids, start, 20);
    let default = ClothoidState {
        x: start.x,
        y: start.y,
        theta: start.angle,
    };
    let last = pts.last().unwrap_or(&default);
    let dist = ((last.x - end.x).powi(2) + (last.y - end.y).powi(2)).sqrt();
    let ad = angle_diff(last.theta, end.angle).abs();
    (dist, ad)
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

    use super::super::pose::Pose;
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn compute_end_errors_perfect_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let (pos_err, angle_err) = compute_end_errors(&params, 1, &start, &end);
        assert!(pos_err < 1e-5, "pos_err = {pos_err}");
        assert!(angle_err < 1e-5, "angle_err = {angle_err}");
    }

    #[test]
    fn compute_end_errors_miss_target() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(10.0, 5.0, 1.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let (pos_err, angle_err) = compute_end_errors(&params, 1, &start, &end);
        assert!(pos_err > 1.0, "pos_err should be large");
        assert!(angle_err > 0.5, "angle_err should be non-trivial");
    }

    #[test]
    fn compute_error_perfect_solution() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let err = compute_error(&params, 1, &start, &end);
        assert!(err < 0.01, "err = {err}");
    }

    #[test]
    fn compute_error_negative_length_penalty() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params_ok = [0.0, 0.0, 0.0, 5.0, 0.0];
        let params_neg = [-2.0, 0.0, 0.0, 5.0, 0.0];
        let err_ok = compute_error(&params_ok, 1, &start, &end);
        let err_neg = compute_error(&params_neg, 1, &start, &end);
        assert!(err_neg > err_ok, "negative l should incur extra penalty");
    }

    #[test]
    fn compute_error_wrong_end() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(10.0, 5.0, 1.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let err = compute_error(&params, 1, &start, &end);
        assert!(err > 100.0, "err = {err}");
    }

    #[test]
    fn plan_objective_default_matches_compute_error() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let err_old = compute_error(&params, 1, &start, &end);
        let err_new = PlanObjective::default().compute(&params, 1, &start, &end);
        assert!(
            (err_old - err_new).abs() < 1e-12,
            "default mismatch: {err_old} vs {err_new}"
        );

        let params2 = [-2.0, 0.5, -0.3, 0.05, 1.0];
        let err_old2 = compute_error(&params2, 1, &start, &end);
        let err_new2 = PlanObjective::default().compute(&params2, 1, &start, &end);
        assert!(
            (err_old2 - err_new2).abs() < 1e-12,
            "default mismatch on penalized params: {err_old2} vs {err_new2}"
        );
    }

    #[test]
    fn plan_objective_max_kappa_penalty_fires() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let mut obj = PlanObjective::default();
        obj.max_kappa = 1.0;
        obj.w_max_kappa = 10.0;

        let err_no_penalty = obj.compute(&params, 1, &start, &end);

        let params_high_kappa = [0.0, 1.5, 1.5, 5.0, 0.0];
        let err_with_penalty = obj.compute(&params_high_kappa, 1, &start, &end);

        assert!(
            err_with_penalty > err_no_penalty,
            "max_kappa penalty should fire: {err_with_penalty} vs {err_no_penalty}"
        );
    }

    #[test]
    fn plan_objective_sign_flip_counts_inflections() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(10.0, 0.0, 0.0);
        let params: Vec<f64> = vec![
            0.0, 2.0, 2.0, 3.0, 0.0, -2.0, -2.0, 3.0, 0.0, 2.0, 2.0, 3.0, 0.0,
        ];
        let mut obj = PlanObjective::default();
        obj.w_sign_flips = 1.0;
        obj.w_end_pos = 0.0;
        obj.w_end_angle = 0.0;
        obj.w_total_length = 0.0;

        let err = obj.compute(&params, 3, &start, &end);
        assert!(err > 0.0, "sign flip penalty should fire, err = {err}");

        let params_no_flip: Vec<f64> = vec![
            0.0, 2.0, 2.0, 3.0, 0.0, 2.0, 2.0, 3.0, 0.0, 2.0, 2.0, 3.0, 0.0,
        ];
        let err_no_flip = obj.compute(&params_no_flip, 3, &start, &end);
        assert!(
            err_no_flip < err,
            "no-flip should have less penalty: {err_no_flip} vs {err}"
        );
    }

    #[test]
    fn plan_objective_kappa_rate_quadratic() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params_small = [0.0, 1.0, 2.0, 5.0, 0.0];
        let params_large = [0.0, 1.0, 3.0, 5.0, 0.0];

        let mut obj = PlanObjective::default();
        obj.w_kappa_rate = 1.0;
        obj.w_end_pos = 0.0;
        obj.w_end_angle = 0.0;
        obj.w_total_length = 0.0;

        let err_small = obj.compute(&params_small, 1, &start, &end);
        let err_large = obj.compute(&params_large, 1, &start, &end);

        let rate_small = 1.0 / 5.0;
        let rate_large = 2.0 / 5.0;
        let expected_ratio = (rate_large * rate_large) / (rate_small * rate_small);
        let actual_ratio = err_large / err_small;

        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "ratio should be ~{expected_ratio}, got {actual_ratio}"
        );
    }

    #[test]
    fn plan_objective_g2_penalises_boundary_jump() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(10.0, 0.0, 0.0);
        let params_discontinuous: Vec<f64> = vec![0.0, 1.0, 3.0, 3.0, 0.0, 5.0, 5.0, 3.0, 0.0];
        let params_continuous: Vec<f64> = vec![0.0, 1.0, 3.0, 3.0, 0.0, 3.0, 5.0, 3.0, 0.0];

        let mut obj = PlanObjective::default();
        obj.w_g2 = 10.0;
        obj.w_end_pos = 0.0;
        obj.w_end_angle = 0.0;
        obj.w_total_length = 0.0;

        let err_disc = obj.compute(&params_discontinuous, 2, &start, &end);
        let err_cont = obj.compute(&params_continuous, 2, &start, &end);
        assert!(
            err_disc > err_cont,
            "G2 discontinuity should penalize more: {err_disc} vs {err_cont}"
        );
    }

    #[test]
    fn plan_objective_endpoint_zero_kappa() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 2.0, 0.0, 5.0, 0.0];

        let mut obj = PlanObjective::default();
        obj.w_kappa_start_zero = 10.0;
        obj.w_kappa_end_zero = 0.0;
        obj.w_end_pos = 0.0;
        obj.w_end_angle = 0.0;
        obj.w_total_length = 0.0;

        let err_start = obj.compute(&params, 1, &start, &end);
        assert!(err_start > 0.0, "start κ penalty should fire: {err_start}");

        obj.w_kappa_start_zero = 0.0;
        obj.w_kappa_end_zero = 10.0;
        let err_end = obj.compute(&params, 1, &start, &end);
        assert!(err_end < 1e-10, "end κ penalty should not fire: {err_end}");
    }

    #[test]
    fn plan_objective_min_seg_len() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params_short = [0.0, 0.0, 0.0, 0.01, 0.0];
        let params_long = [0.0, 0.0, 0.0, 5.0, 0.0];

        let mut obj = PlanObjective::default();
        obj.min_seg_len = 1.0;
        obj.w_min_seg_len = 10.0;
        obj.w_end_pos = 0.0;
        obj.w_end_angle = 0.0;
        obj.w_total_length = 0.0;

        let err_short = obj.compute(&params_short, 1, &start, &end);
        let err_long = obj.compute(&params_long, 1, &start, &end);
        assert!(
            err_short > err_long,
            "short segment should penalize: {err_short} vs {err_long}"
        );
    }

    #[test]
    fn plan_objective_total_length_target() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];

        let mut obj = PlanObjective::default();
        obj.target_length = Some(10.0);
        obj.w_total_length = 1.0;
        obj.w_end_pos = 0.0;
        obj.w_end_angle = 0.0;

        let err = obj.compute(&params, 1, &start, &end);
        let expected = 25.0f64;
        assert!(
            (err - expected).abs() < 0.01,
            "expected ~{expected}, got {err}"
        );
    }

    #[test]
    fn plan_objective_symmetry_detection_auto() {
        let start = Pose::new(-3.0, 0.0, 0.0);
        let end = Pose::new(3.0, 0.0, PI);

        assert!(is_symmetric_task(&start, &end), "should detect symmetry");

        let asymmetric_end = Pose::new(3.0, 0.0, 0.0);
        assert!(
            !is_symmetric_task(&start, &asymmetric_end),
            "should not detect asymmetry"
        );
    }

    #[test]
    fn plan_objective_symmetry_distance_zero_for_mirror() {
        let params: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 0.0, -2.0, -1.0, 3.0, 0.0];
        let dist = symmetry_distance(&params, 2);
        assert!(dist < 1e-12, "mirror distance should be ~0, got {dist}");
    }

    #[test]
    fn plan_objective_symmetry_mode_traits() {
        let a = SymmetryMode::Auto;
        let b = SymmetryMode::Auto;
        let c = SymmetryMode::On;
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(format!("{a:?}"), "Auto");
    }
}
