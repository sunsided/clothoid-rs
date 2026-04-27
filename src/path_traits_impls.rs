//! Implementation of `path_traits` for clothoid types.
//!
//! This module is only available when the `path-traits` feature is enabled.
//!
//! ## Types
//!
//! - [`ClothoidArc`] — a clothoid arc with linearly varying curvature.
//! - [`LinearSegment`] — a straight-line segment.
//! - [`ArcSegment`] — enum covering both segment kinds.
//! - [`ClothoidPath`] — a compound path of multiple segments.
//! - [`Vec2`] — a 2D vector used as the `Vector` associated type.
//!
//! ## Example
//!
//! ```
//! use clothoid::ClothoidArc;
//! use clothoid::optimizer::Pose;
//! use path_traits::Path;
//!
//! let arc = ClothoidArc {
//!     start: Pose::new(0.0, 0.0, 0.0),
//!     ks: 0.0,
//!     ke: 1.0,
//!     length: 5.0,
//!     n_steps: 256,
//! };
//!
//! let pt = arc.sample_at(2.5).unwrap();
//! println!("point at s=2.5: ({}, {})", pt.x, pt.y);
//! ```

pub mod arc;
pub mod linear;
pub mod multi;
pub mod primitive;
pub mod project;
pub mod segment;

pub use arc::ClothoidArc;
pub use linear::LinearSegment;
pub use multi::ClothoidPath;
pub use primitive::Vec2;
pub use segment::ArcSegment;

/// Re-export the `path_traits` crate for convenience.
pub use path_traits;
