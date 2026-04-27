//! Nelder-Mead simplex optimizer.
//!
//! Implements the standard Nelder-Mead method with reflection (α=1),
//! expansion (γ=2), contraction (ρ=0.5), and shrink (σ=0.5) coefficients.

use super::solver::Optimizer;

/// The Nelder-Mead simplex method optimizer.
///
/// Standard coefficients: reflection α=1, expansion γ=2, contraction ρ=0.5,
/// shrink σ=0.5.  NaN / ±∞ function values are replaced with `1e10`.
#[derive(Clone, Debug)]
pub struct NelderMead;

impl Default for NelderMead {
    fn default() -> Self {
        NelderMead
    }
}

impl Optimizer for NelderMead {
    fn minimize(&mut self, f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
        nelder_mead_impl(f, x0, max_iter)
    }

    fn name(&self) -> &'static str {
        "Nelder-Mead"
    }
}

fn nelder_mead_impl(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
    let n = x0.len();
    if n == 0 {
        return vec![];
    }

    let alpha = 1.0f64;
    let gamma = 2.0f64;
    let rho = 0.5f64;
    let sigma = 0.5f64;

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut p = x0.to_vec();
        let delta = if p[i].abs() > 1e-10 { p[i] * 0.05 } else { 0.1 };
        p[i] += delta;
        simplex.push(p);
    }

    let sanitize = |v: f64| {
        if v.is_nan() || v.is_infinite() {
            1e10
        } else {
            v
        }
    };
    let mut fvals: Vec<f64> = simplex.iter().map(|p| sanitize(f(p))).collect();

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        let mut centroid = vec![0.0f64; n];
        for &i in &order[..n] {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        #[allow(clippy::cast_precision_loss)]
        for c in &mut centroid {
            *c /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst][j]))
            .collect();
        let fr = sanitize(f(&reflected));

        if fr < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let fe = sanitize(f(&expanded));
            if fe < fr {
                simplex[worst] = expanded;
                fvals[worst] = fe;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = fr;
            }
        } else if fr < fvals[second_worst] {
            simplex[worst] = reflected;
            fvals[worst] = fr;
        } else {
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[worst][j] - centroid[j]))
                .collect();
            let fc = sanitize(f(&contracted));
            if fc < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = fc;
            } else {
                #[allow(clippy::needless_range_loop)]
                for i in 1..=n {
                    let idx = order[i];
                    for j in 0..n {
                        let val = simplex[idx][j];
                        let best_val = simplex[best][j];
                        simplex[idx][j] = best_val + sigma * (val - best_val);
                    }
                    fvals[idx] = sanitize(f(&simplex[idx]));
                }
            }
        }
    }

    let best_idx = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    simplex[best_idx].clone()
}

/// Minimises `f` starting from `x0` using the Nelder-Mead simplex method.
///
/// Standard coefficients: reflection α=1, expansion γ=2, contraction ρ=0.5,
/// shrink σ=0.5.  NaN / ±∞ function values are replaced with `1e10`.
///
/// Returns the best parameter vector found within `max_iter` iterations.
/// Returns an empty vector when `x0` is empty.
pub fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
    NelderMead.minimize(f, x0, max_iter)
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
    use crate::optimizer::{compute_end_errors, compute_error, pose::Pose};

    #[test]
    fn nelder_mead_empty() {
        let result = nelder_mead(&|_: &[f64]| 0.0, &[], 100);
        assert!(result.is_empty());
    }

    #[test]
    fn nelder_mead_1d_quadratic() {
        let result = nelder_mead(&|x: &[f64]| (x[0] - 3.0).powi(2), &[0.0], 2000);
        assert!((result[0] - 3.0).abs() < 0.01, "result = {}", result[0]);
    }

    #[test]
    fn nelder_mead_2d_bowl() {
        let result = nelder_mead(&|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[5.0, -3.0], 2000);
        assert!(result[0].abs() < 0.01, "x = {}", result[0]);
        assert!(result[1].abs() < 0.01, "y = {}", result[1]);
    }

    #[test]
    fn nelder_mead_3d_bowl() {
        let f = |x: &[f64]| x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2);
        let result = nelder_mead(&f, &[-2.0, 4.0, 1.0], 3000);
        assert!(result[0].abs() < 0.05);
        assert!(result[1].abs() < 0.05);
        assert!(result[2].abs() < 0.05);
    }

    #[test]
    fn nelder_mead_nan_objective() {
        let result = nelder_mead(&|_: &[f64]| f64::NAN, &[1.0, 2.0], 50);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn optimizer_finds_straight_line_solution() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let x0 = vec![0.0, 0.0, 0.0, 5.0, 0.0];
        let n = 1;
        let params = nelder_mead(&|p: &[f64]| compute_error(p, n, &start, &end), &x0, 1000);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.01, "pos_err = {pos_err}");
        assert!(angle_err < 0.01, "angle_err = {angle_err}");
    }

    #[test]
    fn optimizer_finds_right_angle_turn() {
        use std::f64::consts::PI;
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(1.0, 1.0, PI / 2.0);
        let x0 = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let n = 1;
        let params = nelder_mead(&|p: &[f64]| compute_error(p, n, &start, &end), &x0, 2000);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.1, "pos_err = {pos_err}");
        assert!(angle_err < 0.1, "angle_err = {angle_err}");
    }
}
