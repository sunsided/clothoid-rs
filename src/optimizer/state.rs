//! State representation for clothoid integration.
//!
//! Defines the `(x, y, θ)` state vector used during RK4 stepping
//! and path evaluation.

/// State for a clothoid integration step.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClothoidState {
    /// The x-coordinate.
    pub x: f64,
    /// The y-coordinate.
    pub y: f64,
    /// The heading angle in radians.
    pub theta: f64,
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
    fn clothoid_state_defaults() {
        let state = ClothoidState::default();
        assert_eq!(state.x, 0.0);
        assert_eq!(state.y, 0.0);
        assert_eq!(state.theta, 0.0);
    }
}
