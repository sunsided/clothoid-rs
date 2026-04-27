//! 2D pose representation for clothoid path endpoints.
//!
//! Defines a position `(x, y)` and heading angle measured counter-clockwise
//! from the positive X axis.

/// A 2D pose consisting of a position and a heading angle.
///
/// The heading angle is measured in radians, counter-clockwise (CCW) from the
/// positive X axis.
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct Pose {
    /// The x-coordinate of the position.
    pub x: f64,
    /// The y-coordinate of the position.
    pub y: f64,
    /// Heading angle in radians, measured CCW from +X.
    pub angle: f64,
}

impl Pose {
    /// Creates a new [`Pose`] with the given position and heading angle.
    ///
    /// # Arguments
    ///
    /// * `x` — The x-coordinate.
    /// * `y` — The y-coordinate.
    /// * `angle` — The heading angle in radians (CCW from +X).
    #[must_use]
    pub fn new(x: f64, y: f64, angle: f64) -> Self {
        Self { x, y, angle }
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

    #[test]
    fn pose_new() {
        let p = Pose::new(1.0, 2.0, 0.5);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.angle, 0.5);
    }
}
