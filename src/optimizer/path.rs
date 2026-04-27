//! Path evaluation from flat parameter vectors.
//!
//! Converts parameter vectors describing straight segments and clothoid arcs
//! into sampled state sequences, with segmented output for rendering.

use super::integrate::integrate_clothoid;
use super::state::ClothoidState;

/// The type of a path segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    /// A straight-line segment.
    Linear,
    /// A clothoid (Euler spiral) curve segment.
    Curve,
}

/// A segment of a path, consisting of sampled points and metadata.
#[derive(Clone, Debug)]
pub struct PathSegment {
    /// The kind of this segment (linear or curve).
    pub kind: SegmentKind,
    /// Sampled states along the segment.
    pub points: Vec<ClothoidState>,
    /// The heading angle at the segment boundary.
    pub boundary_theta: f64,
}

/// Evaluates the full path described by a flat parameter vector.
///
/// ## Parameter layout
/// `[l₀, ks₀, ke₀, clen₀,  l₁, ks₁, ke₁, clen₁,  …,  lₙ]`
///
/// Length: `4 * n_clothoids + 1`.
///
/// For each clothoid segment `i`:
/// - `lᵢ` — length of the preceding straight segment (clamped to ≥ 0)
/// - `ksᵢ`, `keᵢ` — start / end curvature of the clothoid arc
/// - `clenᵢ` — arc length of the clothoid (clamped to ≥ 1e-6)
/// - `lₙ` — trailing straight segment after the last clothoid
///
/// # Arguments
/// * `params` — flat parameter slice of length `4 * n_clothoids + 1`
/// * `n_clothoids` — number of clothoid arcs in the path
/// * `start` — initial pose
/// * `n_steps` — RK4 integration steps per clothoid arc
#[must_use]
pub fn eval_path(
    params: &[f64],
    n_clothoids: usize,
    start: &super::pose::Pose,
    n_steps: usize,
) -> Vec<ClothoidState> {
    let mut state = ClothoidState {
        x: start.x,
        y: start.y,
        theta: start.angle,
    };
    let mut all: Vec<ClothoidState> = vec![state];

    for i in 0..n_clothoids {
        let base = 4 * i;
        let l = params[base].max(0.0);
        let ks = params[base + 1];
        let ke = params[base + 2];
        let clen = params[base + 3].max(1e-6);

        if l > 1e-10 {
            state.x += l * state.theta.cos();
            state.y += l * state.theta.sin();
            all.push(state);
        }

        let pts = integrate_clothoid(state.x, state.y, state.theta, ks, ke, clen, n_steps);
        if let Some(last) = pts.last() {
            state = *last;
        }
        if pts.len() > 1 {
            all.extend_from_slice(&pts[1..]);
        }
    }

    let l_final = params[4 * n_clothoids].max(0.0);
    if l_final > 1e-10 {
        state.x += l_final * state.theta.cos();
        state.y += l_final * state.theta.sin();
        all.push(state);
    }

    all
}

/// Evaluates the full path described by a flat parameter vector, returning
/// segmented results with [`SegmentKind`] tags.
///
/// The parameter layout is identical to [`eval_path`], but the result is
/// structured as [`PathSegment`]s for rendering or further processing.
///
/// # Arguments
///
/// * `params` — Flat parameter slice of length `4 * n_clothoids + 1`.
/// * `n_clothoids` — Number of clothoid arcs in the path.
/// * `start` — Initial pose.
/// * `n_steps` — RK4 integration steps per clothoid arc.
///
/// # Returns
///
/// A vector of [`PathSegment`]s, each tagged as [`SegmentKind::Linear`] or
/// [`SegmentKind::Curve`].
#[must_use]
pub fn eval_path_segmented(
    params: &[f64],
    n_clothoids: usize,
    start: &super::pose::Pose,
    n_steps: usize,
) -> Vec<PathSegment> {
    let mut state = ClothoidState {
        x: start.x,
        y: start.y,
        theta: start.angle,
    };
    let mut segments: Vec<PathSegment> = Vec::new();

    for i in 0..n_clothoids {
        let base = 4 * i;
        let l = params[base].max(0.0);
        let ks = params[base + 1];
        let ke = params[base + 2];
        let clen = params[base + 3].max(1e-6);

        if l > 1e-10 {
            let entry = state;
            let exit = ClothoidState {
                x: state.x + l * state.theta.cos(),
                y: state.y + l * state.theta.sin(),
                theta: state.theta,
            };
            state = exit;
            segments.push(PathSegment {
                kind: SegmentKind::Linear,
                points: vec![entry, exit],
                boundary_theta: state.theta,
            });
        }

        let pts = integrate_clothoid(state.x, state.y, state.theta, ks, ke, clen, n_steps);
        if let Some(last) = pts.last() {
            let cloth_pts: Vec<ClothoidState> = if pts.len() > 1 {
                pts[1..].to_vec()
            } else {
                pts.clone()
            };
            if !cloth_pts.is_empty() {
                segments.push(PathSegment {
                    kind: SegmentKind::Curve,
                    points: cloth_pts,
                    boundary_theta: last.theta,
                });
            }
            state = *last;
        }
    }

    let l_final = params[4 * n_clothoids].max(0.0);
    if l_final > 1e-10 {
        let entry = state;
        let exit = ClothoidState {
            x: state.x + l_final * state.theta.cos(),
            y: state.y + l_final * state.theta.sin(),
            theta: state.theta,
        };
        segments.push(PathSegment {
            kind: SegmentKind::Linear,
            points: vec![entry, exit],
            boundary_theta: exit.theta,
        });
    }

    if segments.is_empty() {
        segments.push(PathSegment {
            kind: SegmentKind::Linear,
            points: vec![state, state],
            boundary_theta: state.theta,
        });
    }

    segments
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

    #[test]
    fn eval_path_single_clothoid_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
        assert!(last.theta.abs() < 1e-6);
    }

    #[test]
    fn eval_path_initial_straight_plus_clothoid() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [2.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    #[test]
    fn eval_path_trailing_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 3.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    #[test]
    fn eval_path_non_origin_start() {
        let start = Pose::new(1.0, 2.0, 0.0);
        let params = [0.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 4.0).abs() < 1e-6);
        assert!((last.y - 2.0).abs() < 1e-6);
    }

    #[test]
    fn eval_path_two_clothoids_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 2, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    #[test]
    fn eval_path_negative_lengths_clamped() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [-1.0, 0.0, 0.0, 3.0, -0.5];
        let pts = eval_path(&params, 1, &start, 50);
        let last = pts.last().unwrap();
        assert!((last.x - 3.0).abs() < 1e-5);
        assert!(last.y.abs() < 1e-5);
    }

    #[test]
    fn eval_path_segmented_pure_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let segs = eval_path_segmented(&params, 1, &start, 100);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].kind, SegmentKind::Curve);
        assert!(segs[0].points.len() >= 2);
        let last = segs[0].points.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    #[test]
    fn eval_path_segmented_curved_clothoid() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 1.0, 0.5, 3.0, 0.0];
        let segs = eval_path_segmented(&params, 1, &start, 100);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].kind, SegmentKind::Curve);
        assert!(segs[0].points.len() >= 2);
    }

    #[test]
    fn eval_path_segmented_prefix_straight_plus_clothoid() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [2.0, 1.0, 0.5, 3.0, 0.0];
        let segs = eval_path_segmented(&params, 1, &start, 100);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].kind, SegmentKind::Linear);
        assert_eq!(segs[1].kind, SegmentKind::Curve);
        let lin_end = segs[0].points.last().unwrap();
        assert!((lin_end.x - 2.0).abs() < 1e-10);
        assert!(lin_end.y.abs() < 1e-10);
        let cur_start = segs[1].points.first().unwrap();
        assert!((cur_start.x - 2.0).abs() < 0.1);
    }

    #[test]
    fn eval_path_segmented_trailing_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 3.0];
        let segs = eval_path_segmented(&params, 1, &start, 100);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].kind, SegmentKind::Curve);
        assert_eq!(segs[1].kind, SegmentKind::Linear);
        let last = segs[1].points.last().unwrap();
        assert_eq!(segs[1].boundary_theta, last.theta);
    }

    #[test]
    fn eval_path_segmented_boundary_theta() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [2.0, 1.0, 1.0, 3.0, 1.0];
        let segs = eval_path_segmented(&params, 1, &start, 100);
        assert_eq!(segs.len(), 3);
        assert!((segs[0].boundary_theta - 0.0).abs() < 1e-10);
        let last_seg = segs.last().unwrap();
        let last_pt = last_seg.points.last().unwrap();
        assert!((last_seg.boundary_theta - last_pt.theta).abs() < 1e-10);
    }
}
