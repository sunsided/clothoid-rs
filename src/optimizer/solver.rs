//! Optimizer trait abstraction for derivative-free minimization.
//!
//! Defines the [`Optimizer`] trait implemented by concrete solvers
//! such as Nelder-Mead and CMA-ES.

/// A derivative-free minimizer for flat parameter vectors.
pub trait Optimizer: Send {
    /// Minimize `f` starting from `x0`. Returns the best parameter vector found.
    fn minimize(&mut self, f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64>;
    /// Human-readable name of this optimizer.
    fn name(&self) -> &'static str;
}
