//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
//!
//! Implements the (μ/μ_w, λ) variant with standard Hansen defaults for
//! strategy parameters, eigendecomposition, and step-size adaptation.

use super::linalg::jacobi_eigen;
use super::rng::{DEFAULT_RNG_SEED, Lcg, randn};
use super::solver::Optimizer;

/// Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
///
/// Implements the (`μ/μ_w`, λ) variant with standard Hansen defaults.
#[derive(Clone, Debug)]
pub struct CmaEs {
    seed: u64,
    sigma0: f64,
}

impl Default for CmaEs {
    fn default() -> Self {
        CmaEs {
            seed: DEFAULT_RNG_SEED,
            sigma0: 0.5,
        }
    }
}

impl CmaEs {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        CmaEs { seed, sigma0: 0.5 }
    }

    #[must_use]
    pub fn with_sigma0(mut self, sigma: f64) -> Self {
        self.sigma0 = sigma;
        self
    }
}

impl Optimizer for CmaEs {
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn minimize(&mut self, f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
        let n = x0.len();
        if n == 0 {
            return vec![];
        }

        let mut lcg = Lcg::new(self.seed);

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let lambda = (4.0 + (3.0 * (n as f64).ln()).floor()) as usize;
        let mu = lambda / 2;
        let mut weights = vec![0.0; mu];
        for (i, w) in weights.iter_mut().enumerate().take(mu) {
            #[allow(clippy::cast_precision_loss)]
            let w_val = ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(1e-12);
            *w = w_val;
        }
        let wsum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= wsum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        #[allow(clippy::cast_precision_loss)]
        let n_d = n as f64;
        let cc = (4.0 + mu_eff / n_d) / (n_d + 4.0 + 2.0 * mu_eff / n_d);
        let cs = (mu_eff + 2.0) / (n_d + mu_eff + 5.0);
        let damps = 1.0 + 2.0 * (mu_eff / (n_d + 1.0) - 1.0).max(0.0) + cs;
        let c1 = 2.0 / ((n_d + 1.3).powi(2) + mu_eff);
        let cmu =
            (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n_d + 2.0).powi(2) + mu_eff));
        let cmu = cmu.max(0.0);

        let mut m = x0.to_vec();
        let mut sigma = self.sigma0;

        let mut pc = vec![0.0; n];
        let mut ps = vec![0.0; n];
        let mut c: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            c[i * n + i] = 1.0;
        }

        let mut bd: Option<(Vec<f64>, Vec<f64>)> = None;
        let mut eig_count = 0;

        let sanitize = |v: f64| {
            if v.is_nan() || v.is_infinite() {
                1e10
            } else {
                v
            }
        };

        let mut best_x = m.clone();
        let mut best_val = sanitize(f(&m));

        for generation in 0..max_iter {
            let (b, d) = if let Some((bb, dd)) = &bd {
                (bb.clone(), dd.clone())
            } else {
                let mut a_copy = c.clone();
                let mut eigs = vec![0.0; n];
                let mut v_mat = vec![0.0; n * n];
                jacobi_eigen(&mut a_copy, n, &mut eigs, &mut v_mat, 50);
                for val in &mut eigs {
                    if *val < 1e-30 {
                        *val = 1e-30;
                    }
                }
                let d_vec: Vec<f64> = eigs.iter().map(|v| v.sqrt()).collect();
                bd = Some((v_mat.clone(), d_vec.clone()));
                eig_count = 0;
                (v_mat, d_vec)
            };

            let mut arz: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut arx: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut arfitness: Vec<f64> = Vec::with_capacity(lambda);

            for _k in 0..lambda {
                let mut z = vec![0.0; n];
                for zi in z.iter_mut().take(n) {
                    *zi = randn(&mut lcg);
                }
                let mut x = m.clone();
                for i in 0..n {
                    let mut bdz = 0.0;
                    for j in 0..n {
                        bdz += b[i * n + j] * d[j] * z[j];
                    }
                    x[i] += sigma * bdz;
                }
                arz.push(z);
                arx.push(x);
            }

            for x in arx.iter().take(lambda) {
                arfitness.push(sanitize(f(x)));
            }

            let mut idx: Vec<usize> = (0..lambda).collect();
            idx.sort_by(|&a, &b| {
                arfitness[a]
                    .partial_cmp(&arfitness[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (i, &best_idx) in idx.iter().enumerate().take(mu) {
                let fv = arfitness[best_idx];
                if i == 0 && fv < best_val {
                    best_val = fv;
                    best_x.clone_from(&arx[best_idx]);
                }
            }

            let mut m_new = vec![0.0; n];
            for i in 0..n {
                for j in 0..mu {
                    m_new[i] += weights[j] * arx[idx[j]][i];
                }
            }
            let mut m_delta = vec![0.0; n];
            for i in 0..n {
                m_delta[i] = m_new[i] - m[i];
            }
            m = m_new;

            let inv_sigma = 1.0 / sigma;
            for i in 0..n {
                let mut tmp = 0.0;
                for j in 0..n {
                    tmp += b[j * n + i] * m_delta[j];
                }
                ps[i] = (1.0 - cs) * ps[i] + (cs * (2.0 - cs) * mu_eff).sqrt() * inv_sigma * tmp;
            }

            let mut ps_norm2 = 0.0;
            for v in &ps {
                ps_norm2 += v * v;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let hsig = if ps_norm2 / (1.0 - (1.0 - cs).powi(2 * generation as i32))
                < (2.4 + 4.0 / (n_d + 1.0)) * n_d
            {
                1.0
            } else {
                0.0
            };

            for i in 0..n {
                let mut tmp = 0.0;
                for j in 0..n {
                    tmp += b[j * n + i] * m_delta[j];
                }
                pc[i] = (1.0 - cc) * pc[i] + hsig * (cc * (2.0 - cc) * mu_eff).sqrt() * tmp;
            }

            for i in 0..n {
                for j in 0..=i {
                    let rank1 = pc[i] * pc[j];
                    let mut rank_mu = 0.0;
                    for k in 0..mu {
                        let arz_k = &arz[idx[k]];
                        rank_mu += weights[k] * arz_k[i] * arz_k[j];
                    }
                    c[i * n + j] = (1.0 - c1 - cmu) * c[i * n + j] + c1 * rank1 + cmu * rank_mu;
                    c[j * n + i] = c[i * n + j];
                }
            }

            let chi_n = n_d.sqrt() * (1.0 - 1.0 / (4.0 * n_d) + 1.0 / (21.0 * n_d * n_d));
            let ps_norm_expected = chi_n;
            sigma *= ((cs / damps) * (ps_norm2.sqrt() / ps_norm_expected - 1.0)).exp();

            if !(1e-20..=1e20).contains(&sigma) {
                break;
            }

            eig_count += 1;
            if eig_count >= 1 {
                bd = None;
            }
        }

        let final_val = sanitize(f(&best_x));
        if sanitize(f(&m)) < final_val {
            m
        } else {
            best_x
        }
    }

    fn name(&self) -> &'static str {
        "CMA-ES"
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
    use crate::optimizer::{compute_end_errors, compute_error, pose::Pose};
    use std::f64::consts::PI;

    #[test]
    fn cmaes_empty() {
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED);
        let result = cma.minimize(&|_: &[f64]| 0.0, &[], 100);
        assert!(result.is_empty());
    }

    #[test]
    fn cmaes_1d_quadratic() {
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED).with_sigma0(1.0);
        let result = cma.minimize(&|x: &[f64]| (x[0] - 3.0).powi(2), &[0.0], 200);
        assert!((result[0] - 3.0).abs() < 0.05, "result = {}", result[0]);
    }

    #[test]
    fn cmaes_2d_bowl() {
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED).with_sigma0(1.0);
        let result = cma.minimize(&|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[5.0, -3.0], 300);
        assert!(result[0].abs() < 0.05, "x = {}", result[0]);
        assert!(result[1].abs() < 0.05, "y = {}", result[1]);
    }

    #[test]
    fn cmaes_3d_bowl() {
        let f = |x: &[f64]| x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2);
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED).with_sigma0(1.0);
        let result = cma.minimize(&f, &[-2.0, 4.0, 1.0], 400);
        assert!(result[0].abs() < 0.1);
        assert!(result[1].abs() < 0.1);
        assert!(result[2].abs() < 0.1);
    }

    #[test]
    fn cmaes_nan_objective() {
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED).with_sigma0(0.5);
        let result = cma.minimize(&|_: &[f64]| f64::NAN, &[1.0, 2.0], 50);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn cmaes_deterministic() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let mut cma1 = CmaEs::new(42).with_sigma0(1.0);
        let mut cma2 = CmaEs::new(42).with_sigma0(1.0);
        let r1 = cma1.minimize(&f, &[1.0, -1.0], 100);
        let r2 = cma2.minimize(&f, &[1.0, -1.0], 100);
        assert!(
            (r1[0] - r2[0]).abs() < 1e-12 && (r1[1] - r2[1]).abs() < 1e-12,
            "determinism failed: {r1:?} vs {r2:?}"
        );
    }

    #[test]
    fn cmaes_optimizer_name() {
        let cma = CmaEs::new(DEFAULT_RNG_SEED);
        assert_eq!(cma.name(), "CMA-ES");
    }

    #[test]
    fn cmaes_finds_straight_line_solution() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let x0 = vec![0.0, 0.0, 0.0, 5.0, 0.0];
        let n = 1;
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED).with_sigma0(1.0);
        let params = cma.minimize(&|p: &[f64]| compute_error(p, n, &start, &end), &x0, 500);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.1, "pos_err = {pos_err}");
        assert!(angle_err < 0.1, "angle_err = {angle_err}");
    }

    #[test]
    fn cmaes_finds_right_angle_turn() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(1.0, 1.0, PI / 2.0);
        let x0 = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let n = 1;
        let mut cma = CmaEs::new(DEFAULT_RNG_SEED).with_sigma0(0.5);
        let params = cma.minimize(&|p: &[f64]| compute_error(p, n, &start, &end), &x0, 800);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.2, "pos_err = {pos_err}");
        assert!(angle_err < 0.2, "angle_err = {angle_err}");
    }
}
