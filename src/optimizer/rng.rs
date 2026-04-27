//! Random number generation for optimizer initialization.
//!
//! Provides a linear congruential generator (LCG) for reproducible
//! parameter seeding and a Box-Muller transform for normal variates.

/// Default seed for the linear congruential generator (LCG) used by the optimizer.
///
/// Use a fixed seed for reproducible optimization results.
pub const DEFAULT_RNG_SEED: u64 = 42;

/// A fast, deterministic linear congruential generator (LCG).
///
/// Uses the constants from Numerical Recipes (a = 6364136223846793005,
/// c = 1442695040888963407) to produce values in `[0, 1)`.
///
/// Suitable for random initialization of optimizer parameters. Use a fixed
/// seed for reproducibility (see [`DEFAULT_RNG_SEED`]).
pub struct Lcg {
    /// The internal state of the generator.
    state: u64,
}

impl Lcg {
    /// Creates a new [`Lcg`] with the given `seed`.
    ///
    /// # Arguments
    ///
    /// * `seed` — The initial state. Same seeds produce identical sequences.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the generator and returns the next value in `[0, 1)`.
    ///
    /// # Returns
    ///
    /// A floating-point value `v` where `0.0 ≤ v < 1.0`.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_val(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as f64 / 2_147_483_648.0
    }
}

#[cfg(feature = "cma-es")]
/// Generate a standard normal variate using Box-Muller transform over the given LCG.
///
/// Returns a value drawn from N(0, 1).
pub fn randn(lcg: &mut Lcg) -> f64 {
    let u1 = lcg.next_val().max(f64::MIN_POSITIVE);
    let u2 = lcg.next_val();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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
    fn lcg_reproducible() {
        let mut a = Lcg::new(DEFAULT_RNG_SEED);
        let mut b = Lcg::new(DEFAULT_RNG_SEED);
        for _ in 0..200 {
            assert_eq!(a.next_val().to_bits(), b.next_val().to_bits());
        }
    }

    #[test]
    fn lcg_range() {
        let mut lcg = Lcg::new(12345);
        for _ in 0..10_000 {
            let v = lcg.next_val();
            assert!(v >= 0.0, "value {v} is negative");
            assert!(v < 1.0, "value {v} is ≥ 1.0");
        }
    }

    #[test]
    fn lcg_different_seeds() {
        let mut a = Lcg::new(1);
        let mut b = Lcg::new(2);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next_val().to_bits()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next_val().to_bits()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[cfg(feature = "cma-es")]
    #[test]
    fn randn_mean_near_zero() {
        let mut lcg = Lcg::new(99);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| randn(&mut lcg)).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "randn mean = {mean}");
    }

    #[cfg(feature = "cma-es")]
    #[test]
    fn randn_variance_near_one() {
        let mut lcg = Lcg::new(99);
        let n = 10_000;
        let vals: Vec<f64> = (0..n).map(|_| randn(&mut lcg)).collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((var - 1.0).abs() < 0.1, "randn variance = {var}");
    }
}
