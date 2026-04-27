//! Path evaluation and optimization utilities for clothoid fitting.
//!
//! This module provides:
//! - RK4 integration for clothoid arcs with linearly varying curvature
//! - Path evaluation from flat parameter vectors
//! - A Nelder-Mead simplex optimizer for derivative-free minimization
//! - A linear congruential generator (LCG) for reproducible random initialization

/// Default seed for the linear congruential generator (LCG) used by the optimizer.
///
/// Use a fixed seed for reproducible optimization results.
pub const DEFAULT_RNG_SEED: u64 = 42;

// ============================================================================
// Pose
// ============================================================================

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
    /// Heading angle in radians, measured CCW from the positive X axis.
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

// ============================================================================
// Clothoid math (RK4)
// ============================================================================

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

/// Performs a single 4th-order Runge-Kutta integration step along a clothoid arc.
///
/// The clothoid has linearly varying curvature:
/// `κ(s) = ks + (ke - ks) * s / length`.
///
/// State: `(x, y, θ)`. Derivatives: `ẋ = cos θ`, `ẏ = sin θ`, `θ̇ = κ(s)`.
///
/// # Arguments
/// * `state` - current state (x, y, theta)
/// * `ks`, `ke` - curvature at start and end of the clothoid segment
/// * `length` - total arc length of the segment (used for curvature interpolation)
/// * `s` - current arc-length position within the segment
/// * `h` - step size
///
/// # Returns
/// Next state `(x, y, θ)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn rk4_step(
    state: ClothoidState,
    ks: f64,
    ke: f64,
    length: f64,
    s: f64,
    h: f64,
) -> ClothoidState {
    let ClothoidState { x, y, theta } = state;
    let kappa = |s: f64| ks + (ke - ks) * s / length;

    let (dx1, dy1, dth1) = (theta.cos(), theta.sin(), kappa(s));
    let th2 = theta + h / 2.0 * dth1;
    let (dx2, dy2, dth2) = (th2.cos(), th2.sin(), kappa(s + h / 2.0));
    let th3 = theta + h / 2.0 * dth2;
    let (dx3, dy3, dth3) = (th3.cos(), th3.sin(), kappa(s + h / 2.0));
    let th4 = theta + h * dth3;
    let (dx4, dy4, dth4) = (th4.cos(), th4.sin(), kappa(s + h));

    let nx = x + h / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
    let ny = y + h / 6.0 * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4);
    let nt = theta + h / 6.0 * (dth1 + 2.0 * dth2 + 2.0 * dth3 + dth4);
    ClothoidState {
        x: nx,
        y: ny,
        theta: nt,
    }
}

/// Integrates a clothoid arc using RK4, returning all sampled `(x, y, θ)` states.
///
/// The curvature varies linearly from `ks` at `s=0` to `ke` at `s=length`.
///
/// Returns a vector of length `n_steps + 1` (including the initial state).
/// If `length <= 0` or `n_steps == 0`, returns a single-element vector containing
/// the start state.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn integrate_clothoid(
    x0: f64,
    y0: f64,
    theta0: f64,
    ks: f64,
    ke: f64,
    length: f64,
    n_steps: usize,
) -> Vec<ClothoidState> {
    if length <= 0.0 || n_steps == 0 {
        return vec![ClothoidState {
            x: x0,
            y: y0,
            theta: theta0,
        }];
    }
    let h = length / n_steps as f64;
    let mut state = ClothoidState {
        x: x0,
        y: y0,
        theta: theta0,
    };
    let mut pts = Vec::with_capacity(n_steps + 1);
    pts.push(state);
    for step in 0..n_steps {
        let s = step as f64 * h;
        state = rk4_step(state, ks, ke, length, s, h);
        pts.push(state);
    }
    pts
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
    start: &Pose,
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

// ============================================================================
// Segmented path evaluation
// ============================================================================

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
    start: &Pose,
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

/// Returns the signed, wrapped difference between two angles (in radians), in `(-π, π]`.
#[must_use]
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let d = (a - b) % (2.0 * pi);
    if d < -pi {
        d + 2.0 * pi
    } else if d > pi {
        d - 2.0 * pi
    } else {
        d
    }
}

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
    pub fn compute(&self, params: &[f64], n_clothoids: usize, start: &Pose, end: &Pose) -> f64 {
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
pub fn is_symmetric_task(start: &Pose, end: &Pose) -> bool {
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
pub fn compute_error(params: &[f64], n_clothoids: usize, start: &Pose, end: &Pose) -> f64 {
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
    start: &Pose,
    end: &Pose,
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

// ============================================================================
// Optimizer trait
// ============================================================================

/// A derivative-free minimizer for flat parameter vectors.
pub trait Optimizer: Send {
    /// Minimize `f` starting from `x0`. Returns the best parameter vector found.
    fn minimize(&mut self, f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64>;
    /// Human-readable name of this optimizer.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Nelder-Mead optimizer
// ============================================================================

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

// ============================================================================
// LCG random number generator
// ============================================================================

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

/// Generate a standard normal variate using Box-Muller transform over the given LCG.
///
/// Returns a value drawn from N(0, 1).
pub fn randn(lcg: &mut Lcg) -> f64 {
    let u1 = lcg.next_val().max(f64::MIN_POSITIVE);
    let u2 = lcg.next_val();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

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

// ============================================================================
// CMA-ES optimizer
// ============================================================================

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

        for gen in 0..max_iter {
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
            let hsig = if ps_norm2 / (1.0 - (1.0 - cs).powi(2 * gen as i32))
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

// ============================================================================
// Tests
// ============================================================================

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
    use std::f64::consts::PI;

    // ------------------------------------------------------------------
    // rk4_step
    // ------------------------------------------------------------------

    /// Zero curvature, θ=0: state should advance purely in +x.
    #[test]
    fn rk4_step_straight_x() {
        let result = rk4_step(
            ClothoidState {
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            },
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        );
        assert!(
            (result.x - 1.0).abs() < 1e-10,
            "x should advance by step h=1"
        );
        assert!(result.y.abs() < 1e-10, "y should not change");
        assert!(result.theta.abs() < 1e-10, "θ should not change");
    }

    /// Zero curvature, θ=π/2: state should advance purely in +y.
    #[test]
    fn rk4_step_straight_y() {
        let result = rk4_step(
            ClothoidState {
                x: 0.0,
                y: 0.0,
                theta: PI / 2.0,
            },
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        );
        assert!(result.x.abs() < 1e-10, "x should not change");
        assert!((result.y - 1.0).abs() < 1e-10, "y should advance by h=1");
        assert!(
            (result.theta - PI / 2.0).abs() < 1e-10,
            "θ should not change"
        );
    }

    /// Non-zero constant curvature: θ changes by κ·h.
    #[test]
    fn rk4_step_constant_curvature_angle() {
        let kappa = 1.0;
        let h = 0.1;
        let result = rk4_step(
            ClothoidState {
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            },
            kappa,
            kappa,
            1.0,
            0.0,
            h,
        );
        assert!((result.theta - kappa * h).abs() < 1e-6);
    }

    /// Step from a non-origin position with θ=0 advances x by h.
    #[test]
    fn rk4_step_non_origin_start() {
        let result = rk4_step(
            ClothoidState {
                x: 3.0,
                y: -2.0,
                theta: 0.0,
            },
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
        );
        assert!((result.x - 5.0).abs() < 1e-10);
        assert!((result.y + 2.0).abs() < 1e-10);
        assert!(result.theta.abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // integrate_clothoid
    // ------------------------------------------------------------------

    /// Zero-length arc returns only the start state.
    #[test]
    fn integrate_clothoid_zero_length() {
        let pts = integrate_clothoid(1.0, 2.0, 0.5, 1.0, -1.0, 0.0, 100);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].x, 1.0);
        assert_eq!(pts[0].y, 2.0);
        assert_eq!(pts[0].theta, 0.5);
    }

    /// n_steps=0 returns only the start state even if length > 0.
    #[test]
    fn integrate_clothoid_zero_steps() {
        let pts = integrate_clothoid(0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0);
        assert_eq!(pts.len(), 1);
    }

    /// Zero curvature, θ=0: should trace a straight line of the given length.
    #[test]
    fn integrate_clothoid_straight_line_x() {
        let length = 5.0;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, 0.0, 0.0, length, 200);
        let last = pts.last().unwrap();
        assert!((last.x - length).abs() < 1e-6, "x should equal arc length");
        assert!(last.y.abs() < 1e-6, "y should be zero");
        assert!(last.theta.abs() < 1e-6, "heading should be unchanged");
    }

    /// Zero curvature, θ=π/2: should trace a straight line in +y.
    #[test]
    fn integrate_clothoid_straight_line_y() {
        let length = 3.0;
        let pts = integrate_clothoid(0.0, 0.0, PI / 2.0, 0.0, 0.0, length, 200);
        let last = pts.last().unwrap();
        assert!(last.x.abs() < 1e-6);
        assert!((last.y - length).abs() < 1e-6);
        assert!((last.theta - PI / 2.0).abs() < 1e-6);
    }

    /// Constant curvature κ=1, arc=π: heading should change by π.
    #[test]
    fn integrate_clothoid_constant_kappa_angle_change() {
        let kappa = 1.0;
        let arc = PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 1000);
        let last = pts.last().unwrap();
        assert!(
            (last.theta - PI).abs() < 1e-3,
            "heading should rotate by π rad"
        );
    }

    /// Constant curvature κ=1, arc=π: traces a half-circle of radius 1.
    /// Starting at (0,0) heading +x, the centre is at (0,1) and the endpoint is (0,2).
    #[test]
    fn integrate_clothoid_half_circle_position() {
        let kappa = 1.0;
        let arc = PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 2000);
        let last = pts.last().unwrap();
        assert!(
            last.x.abs() < 1e-3,
            "x should return to ~0 after half circle"
        );
        assert!((last.y - 2.0).abs() < 1e-3, "y should be 2*radius = 2");
    }

    /// Constant curvature κ=1, arc=2π: full circle — should return to near (0,0)
    /// with heading equal to 0 (modulo 2π).
    #[test]
    fn integrate_clothoid_full_circle() {
        let kappa = 1.0;
        let arc = 2.0 * PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 4000);
        let last = pts.last().unwrap();
        assert!(last.x.abs() < 1e-2, "x should return near 0");
        assert!(last.y.abs() < 1e-2, "y should return near 0");
        assert!((last.theta - 2.0 * PI).abs() < 1e-2);
    }

    /// Linearly varying curvature (clothoid spiral): heading change = integral of κ(s) ds
    /// = (ks + ke) / 2 * length.
    #[test]
    fn integrate_clothoid_linearly_varying_kappa_angle() {
        let ks = 0.0;
        let ke = 2.0;
        let length = 1.0;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, ks, ke, length, 1000);
        let last = pts.last().unwrap();
        let expected = f64::midpoint(ks, ke) * length;
        assert!((last.theta - expected).abs() < 1e-4);
    }

    /// Returned vector has n_steps+1 elements when length > 0 and n_steps > 0.
    #[test]
    fn integrate_clothoid_point_count() {
        let n = 50;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, n);
        assert_eq!(pts.len(), n + 1);
    }

    // ------------------------------------------------------------------
    // eval_path
    // ------------------------------------------------------------------

    /// Single clothoid with zero curvature: straight line from start.
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

    /// Initial straight segment + zero-curvature clothoid: total length is sum.
    #[test]
    fn eval_path_initial_straight_plus_clothoid() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [2.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    /// Final trailing straight segment advances position.
    #[test]
    fn eval_path_trailing_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 3.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    /// Non-origin start pose: position offset is respected.
    #[test]
    fn eval_path_non_origin_start() {
        let start = Pose::new(1.0, 2.0, 0.0);
        let params = [0.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 4.0).abs() < 1e-6);
        assert!((last.y - 2.0).abs() < 1e-6);
    }

    /// Two clothoid segments, both zero-curvature: lengths add up correctly.
    #[test]
    fn eval_pathtwo_clothoids_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 2, &start, 100);
        let last = pts.last().unwrap();
        assert!((last.x - 5.0).abs() < 1e-6);
        assert!(last.y.abs() < 1e-6);
    }

    /// Negative lengths are clamped to zero.
    #[test]
    fn eval_path_negative_lengths_clamped() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [-1.0, 0.0, 0.0, 3.0, -0.5];
        let pts = eval_path(&params, 1, &start, 50);
        let last = pts.last().unwrap();
        assert!((last.x - 3.0).abs() < 1e-5);
        assert!(last.y.abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // eval_path_segmented
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // angle_diff
    // ------------------------------------------------------------------

    #[test]
    fn angle_diff_same_angle() {
        assert_eq!(angle_diff(0.0, 0.0), 0.0);
        assert_eq!(angle_diff(1.0, 1.0), 0.0);
        assert_eq!(angle_diff(-2.0, -2.0), 0.0);
    }

    #[test]
    fn angle_diff_small_positive() {
        assert!((angle_diff(0.2, 0.0) - 0.2).abs() < 1e-10);
        assert!((angle_diff(0.1, -0.1) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn angle_diff_small_negative() {
        assert!((angle_diff(0.0, 0.2) + 0.2).abs() < 1e-10);
        assert!((angle_diff(-0.1, 0.1) + 0.2).abs() < 1e-10);
    }

    /// Wrapping: 2π + ε vs 0 should give ε, not 2π + ε.
    #[test]
    fn angle_diff_wrap_positive() {
        let eps = 0.1;
        let d = angle_diff(2.0 * PI + eps, 0.0);
        assert!((d - eps).abs() < 1e-10, "expected {eps}, got {d}");
    }

    /// Wrapping: 0 vs 2π - ε should give ε (not -2π + ε).
    #[test]
    fn angle_diff_wrap_negative() {
        let eps = 0.1;
        let d = angle_diff(0.0, 2.0 * PI - eps);
        assert!((d - eps).abs() < 1e-10, "expected {eps}, got {d}");
    }

    /// π vs 0 should give exactly π.
    #[test]
    fn angle_diff_half_circle() {
        let d = angle_diff(PI, 0.0);
        assert!((d - PI).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // compute_end_errors
    // ------------------------------------------------------------------

    /// A straight-line solution has near-zero errors.
    #[test]
    fn compute_end_errors_perfect_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        // Straight clothoid of length 5, no curvature
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let (pos_err, angle_err) = compute_end_errors(&params, 1, &start, &end);
        assert!(pos_err < 1e-5, "pos_err = {pos_err}");
        assert!(angle_err < 1e-5, "angle_err = {angle_err}");
    }

    /// Errors are strictly positive when the path misses the target.
    #[test]
    fn compute_end_errors_miss_target() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(10.0, 5.0, 1.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0]; // goes to (5, 0, 0)
        let (pos_err, angle_err) = compute_end_errors(&params, 1, &start, &end);
        assert!(pos_err > 1.0, "pos_err should be large");
        assert!(angle_err > 0.5, "angle_err should be non-trivial");
    }

    // ------------------------------------------------------------------
    // compute_error
    // ------------------------------------------------------------------

    /// Perfect straight-line solution → error ≈ 0 (only tiny regularisation).
    #[test]
    fn compute_error_perfect_solution() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let err = compute_error(&params, 1, &start, &end);
        // Only 0.001 * total_length = 0.001 * 5 = 0.005 regularisation remains
        assert!(err < 0.01, "err = {err}");
    }

    /// Negative lengths incur penalty.
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

    /// Missing the target produces a large error.
    #[test]
    fn compute_error_wrong_end() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(10.0, 5.0, 1.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let err = compute_error(&params, 1, &start, &end);
        assert!(err > 100.0, "err = {err}");
    }

    // ------------------------------------------------------------------
    // nelder_mead
    // ------------------------------------------------------------------

    /// Empty initial point returns an empty vector.
    #[test]
    fn nelder_mead_empty() {
        let result = nelder_mead(&|_: &[f64]| 0.0, &[], 100);
        assert!(result.is_empty());
    }

    /// 1-D quadratic: (x − 3)² converges to x ≈ 3.
    #[test]
    fn nelder_mead_1d_quadratic() {
        let result = nelder_mead(&|x: &[f64]| (x[0] - 3.0).powi(2), &[0.0], 2000);
        assert!((result[0] - 3.0).abs() < 0.01, "result = {}", result[0]);
    }

    /// 2-D bowl: x² + y² converges to (0, 0).
    #[test]
    fn nelder_mead_2d_bowl() {
        let result = nelder_mead(&|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[5.0, -3.0], 2000);
        assert!(result[0].abs() < 0.01, "x = {}", result[0]);
        assert!(result[1].abs() < 0.01, "y = {}", result[1]);
    }

    /// 3-D bowl: minimises to origin from a random start.
    #[test]
    fn nelder_mead_3d_bowl() {
        let f = |x: &[f64]| x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2);
        let result = nelder_mead(&f, &[-2.0, 4.0, 1.0], 3000);
        assert!(result[0].abs() < 0.05);
        assert!(result[1].abs() < 0.05);
        assert!(result[2].abs() < 0.05);
    }

    /// NaN from the objective function is handled gracefully (replaced with 1e10).
    #[test]
    fn nelder_mead_nan_objective() {
        // f returns NaN for any argument; nelder_mead should not panic
        let result = nelder_mead(&|_: &[f64]| f64::NAN, &[1.0, 2.0], 50);
        assert_eq!(result.len(), 2);
    }

    // ------------------------------------------------------------------
    // Lcg
    // ------------------------------------------------------------------

    /// Same seed ⇒ same sequence.
    #[test]
    fn lcg_reproducible() {
        let mut a = Lcg::new(DEFAULT_RNG_SEED);
        let mut b = Lcg::new(DEFAULT_RNG_SEED);
        for _ in 0..200 {
            assert_eq!(a.next_val().to_bits(), b.next_val().to_bits());
        }
    }

    /// All values lie in [0, 1).
    #[test]
    fn lcg_range() {
        let mut lcg = Lcg::new(12345);
        for _ in 0..10_000 {
            let v = lcg.next_val();
            assert!(v >= 0.0, "value {v} is negative");
            assert!(v < 1.0, "value {v} is ≥ 1.0");
        }
    }

    /// Different seeds produce different sequences.
    #[test]
    fn lcg_different_seeds() {
        let mut a = Lcg::new(1);
        let mut b = Lcg::new(2);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next_val().to_bits()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next_val().to_bits()).collect();
        assert_ne!(seq_a, seq_b);
    }

    // ------------------------------------------------------------------
    // End-to-end optimizer integration test
    // ------------------------------------------------------------------

    /// The optimizer (Nelder-Mead on the clothoid error function) can find a
    /// near-perfect straight-line solution when start and end are co-linear.
    #[test]
    fn optimizer_finds_straight_line_solution() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(5.0, 0.0, 0.0);

        // Warm-start with something close: zero-curvature clothoid of length 5
        let x0 = vec![0.0, 0.0, 0.0, 5.0, 0.0];
        let n = 1;
        let params = nelder_mead(&|p: &[f64]| compute_error(p, n, &start, &end), &x0, 1000);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.01, "pos_err = {pos_err}");
        assert!(angle_err < 0.01, "angle_err = {angle_err}");
    }

    /// The optimizer can solve a path that requires a 90° turn.
    /// Start at (0,0) heading right (+x), end at (1,1) heading up (+y).
    #[test]
    fn optimizer_finds_right_angle_turn() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = Pose::new(1.0, 1.0, PI / 2.0);

        let x0 = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let n = 1;
        let params = nelder_mead(&|p: &[f64]| compute_error(p, n, &start, &end), &x0, 2000);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.1, "pos_err = {pos_err}");
        assert!(angle_err < 0.1, "angle_err = {angle_err}");
    }

    // ------------------------------------------------------------------
    // randn
    // ------------------------------------------------------------------

    #[test]
    fn randn_mean_near_zero() {
        let mut lcg = Lcg::new(99);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| randn(&mut lcg)).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "randn mean = {mean}");
    }

    #[test]
    fn randn_variance_near_one() {
        let mut lcg = Lcg::new(99);
        let n = 10_000;
        let vals: Vec<f64> = (0..n).map(|_| randn(&mut lcg)).collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((var - 1.0).abs() < 0.1, "randn variance = {var}");
    }

    // ------------------------------------------------------------------
    // jacobi_eigen
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // CmaEs
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // CMA-ES end-to-end integration tests
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // PlanObjective
    // ------------------------------------------------------------------

    /// PlanObjective::default() produces the same result as the original compute_error.
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

    /// max_kappa penalty fires when curvature exceeds bound.
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

    /// sign-flip penalty counts inflections on a 3-segment path with +,-,+ κ.
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

    /// kappa-rate penalty is quadratic in (ke-ks)/clen.
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

    /// G2 penalty fires when ke_0 != ks_1.
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

    /// endpoint zero-kappa only activates for selected endpoints.
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

    /// min_seg_len penalty fires below threshold, dormant above.
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

    /// total_length with a target produces a quadratic penalty around target.
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

    /// symmetry detection works for auto mode.
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

    /// symmetry distance is zero for perfectly mirrored parameters.
    #[test]
    fn plan_objective_symmetry_distance_zero_for_mirror() {
        let params: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 0.0, -2.0, -1.0, 3.0, 0.0];
        let dist = symmetry_distance(&params, 2);
        assert!(dist < 1e-12, "mirror distance should be ~0, got {dist}");
    }

    /// SymmetryMode derives Clone, PartialEq, Debug.
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
