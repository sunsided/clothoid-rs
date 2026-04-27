//! Linear algebra utilities for the CMA-ES optimizer.
//!
//! Provides a Jacobi eigenvalue solver for symmetric matrices,
//! used to decompose the covariance matrix during optimization.

/// Jacobi eigenvalue solver for symmetric matrices.
///
/// Given a symmetric n×n matrix `a`, computes eigenvalues `e` and eigenvector
/// matrix `v` such that `a ≈ v * diag(e) * v^T`.
///
/// The matrix `a` is modified in-place; `e` and `v` are written to.
/// Iterates up to `max_iter` sweeps (50 is plenty for n≤20).
#[allow(clippy::many_single_char_names)]
pub fn jacobi_eigen(a: &mut [f64], n: usize, e: &mut [f64], v: &mut [f64], max_iter: usize) {
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
        e[i] = a[i * n + i];
    }

    let mut b = e.to_vec();
    let mut z = vec![0.0; n];

    for _sweep in 0..max_iter {
        let mut sm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                sm += a[i * n + j].abs();
            }
        }
        if sm < 1e-40 {
            break;
        }

        for ip in 0..n {
            for iq in (ip + 1)..n {
                let g = a[ip * n + iq];
                if g.abs() < 1e-40 {
                    continue;
                }
                let h = e[iq] - e[ip];
                let t = if h.abs() < 1e-30 {
                    g / h.abs().max(1e-30)
                } else {
                    let theta = h / (2.0 * g);
                    let sgn = if theta > 0.0 { 1.0 } else { -1.0 };
                    sgn / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                let h_val = t * g;
                z[ip] -= h_val;
                z[iq] += h_val;
                e[ip] -= h_val;
                e[iq] += h_val;
                a[ip * n + iq] = 0.0;

                for j in 0..ip {
                    let gj = a[j * n + ip];
                    let hj = a[j * n + iq];
                    a[j * n + ip] = gj - s * (hj + tau * gj);
                    a[j * n + iq] = hj + s * (gj - tau * hj);
                }
                for j in (ip + 1)..iq {
                    let gj = a[ip * n + j];
                    let hj = a[j * n + iq];
                    a[ip * n + j] = gj - s * (hj + tau * gj);
                    a[j * n + iq] = hj + s * (gj - tau * hj);
                }
                for j in (iq + 1)..n {
                    let gj = a[ip * n + j];
                    let hj = a[iq * n + j];
                    a[ip * n + j] = gj - s * (hj + tau * gj);
                    a[iq * n + j] = hj + s * (gj - tau * hj);
                }
                for j in 0..n {
                    let gj = v[j * n + ip];
                    let hj = v[j * n + iq];
                    v[j * n + ip] = gj - s * (hj + tau * gj);
                    v[j * n + iq] = hj + s * (gj - tau * hj);
                }
            }
        }
        for i in 0..n {
            b[i] += z[i];
            z[i] = 0.0;
        }
    }
    e[..n].copy_from_slice(&b[..n]);
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

    #[test]
    fn jacobi_eigen_identity() {
        let n = 3;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let mut e = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        jacobi_eigen(&mut a, n, &mut e, &mut v, 50);
        e.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..n {
            assert!((e[i] - 1.0).abs() < 1e-8, "eigenvalue[{i}] = {}", e[i]);
        }
    }

    #[test]
    fn jacobi_eigen_3x3_symmetric_reconstruction() {
        let n = 3;
        let orig = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let mut a = orig.to_vec();
        let mut e = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        jacobi_eigen(&mut a, n, &mut e, &mut v, 100);

        let mut reconstructed = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[i * n + k] * e[k] * v[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }
        for i in 0..n * n {
            assert!(
                (reconstructed[i] - orig[i]).abs() < 1e-8,
                "reconstruction mismatch at {i}: {} vs {}",
                reconstructed[i],
                orig[i]
            );
        }
    }

    #[test]
    fn jacobi_eigen_5x5_symmetric() {
        let n = 5;
        let mut orig = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let val = (i * 7 + j * 13) as f64 * 0.1 - 1.0;
                orig[i * n + j] = val;
                orig[j * n + i] = val;
            }
        }
        let mut a = orig.clone();
        let mut e = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        jacobi_eigen(&mut a, n, &mut e, &mut v, 200);

        let mut reconstructed = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[i * n + k] * e[k] * v[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }
        for i in 0..n * n {
            assert!(
                (reconstructed[i] - orig[i]).abs() < 1e-6,
                "5x5 mismatch at {i}"
            );
        }
    }
}
