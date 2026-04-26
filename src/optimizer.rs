/// Default seed for the LCG random number generator used by the optimizer.
pub const DEFAULT_RNG_SEED: u64 = 42;

// ============================================================================
// Pose
// ============================================================================

/// A 2D pose consisting of a position and a heading angle (in radians).
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    /// Heading angle in radians, measured CCW from the positive X axis.
    pub angle: f64,
}

impl Pose {
    pub fn new(x: f64, y: f64, angle: f64) -> Self {
        Self { x, y, angle }
    }
}

// ============================================================================
// Clothoid math (RK4)
// ============================================================================

/// Performs a single 4th-order Runge-Kutta integration step along a clothoid arc.
///
/// The clothoid has linearly varying curvature:
/// `κ(s) = ks + (ke - ks) * s / length`.
///
/// State: `(x, y, θ)`. Derivatives: `ẋ = cos θ`, `ẏ = sin θ`, `θ̇ = κ(s)`.
///
/// # Arguments
/// * `x`, `y`, `theta` - current state
/// * `ks`, `ke` - curvature at start and end of the clothoid segment
/// * `length` - total arc length of the segment (used for curvature interpolation)
/// * `s` - current arc-length position within the segment
/// * `h` - step size
///
/// # Returns
/// Next state `(x, y, θ)`.
pub fn rk4_step(
    x: f64,
    y: f64,
    theta: f64,
    ks: f64,
    ke: f64,
    length: f64,
    s: f64,
    h: f64,
) -> (f64, f64, f64) {
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
    (nx, ny, nt)
}

/// Integrates a clothoid arc using RK4, returning all sampled `(x, y, θ)` states.
///
/// The curvature varies linearly from `ks` at `s=0` to `ke` at `s=length`.
///
/// Returns a vector of length `n_steps + 1` (including the initial state).
/// If `length <= 0` or `n_steps == 0`, returns a single-element vector containing
/// the start state.
pub fn integrate_clothoid(
    x0: f64,
    y0: f64,
    theta0: f64,
    ks: f64,
    ke: f64,
    length: f64,
    n_steps: usize,
) -> Vec<(f64, f64, f64)> {
    if length <= 0.0 || n_steps == 0 {
        return vec![(x0, y0, theta0)];
    }
    let h = length / n_steps as f64;
    let mut x = x0;
    let mut y = y0;
    let mut theta = theta0;
    let mut pts = Vec::with_capacity(n_steps + 1);
    pts.push((x, y, theta));
    for step in 0..n_steps {
        let s = step as f64 * h;
        let (nx, ny, nt) = rk4_step(x, y, theta, ks, ke, length, s, h);
        x = nx;
        y = ny;
        theta = nt;
        pts.push((x, y, theta));
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
pub fn eval_path(
    params: &[f64],
    n_clothoids: usize,
    start: &Pose,
    n_steps: usize,
) -> Vec<(f64, f64, f64)> {
    let mut x = start.x;
    let mut y = start.y;
    let mut theta = start.angle;
    let mut all: Vec<(f64, f64, f64)> = vec![(x, y, theta)];

    for i in 0..n_clothoids {
        let base = 4 * i;
        let l = params[base].max(0.0);
        let ks = params[base + 1];
        let ke = params[base + 2];
        let clen = params[base + 3].max(1e-6);

        if l > 1e-10 {
            x += l * theta.cos();
            y += l * theta.sin();
            all.push((x, y, theta));
        }

        let pts = integrate_clothoid(x, y, theta, ks, ke, clen, n_steps);
        if let Some(&(lx, ly, lt)) = pts.last() {
            x = lx;
            y = ly;
            theta = lt;
        }
        if pts.len() > 1 {
            all.extend_from_slice(&pts[1..]);
        }
    }

    let l_final = params[4 * n_clothoids].max(0.0);
    if l_final > 1e-10 {
        x += l_final * theta.cos();
        y += l_final * theta.sin();
        all.push((x, y, theta));
    }

    all
}

/// Returns the signed, wrapped difference between two angles (in radians), in `(-π, π]`.
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

/// Computes the scalar optimization objective for a path described by `params`.
///
/// The objective penalises:
/// - Distance² from the path endpoint to `end.x/y` (weight 10)
/// - Angle error² at the endpoint (weight 5)
/// - Negative segment lengths (soft constraint, weight 100)
/// - Very short clothoid arcs (soft constraint, weight 10)
/// - Total path length (regularisation, weight 0.001)
pub fn compute_error(params: &[f64], n_clothoids: usize, start: &Pose, end: &Pose) -> f64 {
    let mut neg_penalty = 0.0f64;
    let mut len_penalty = 0.0f64;
    let mut total_length = 0.0f64;

    for i in 0..n_clothoids {
        let base = 4 * i;
        let l = params[base];
        let clen = params[base + 3];

        if l < 0.0 {
            neg_penalty += 100.0 * l * l;
        }
        if clen < 0.0 {
            neg_penalty += 100.0 * clen * clen;
        }
        if clen >= 0.0 && clen < 0.1 {
            len_penalty += 10.0 * (0.1 - clen).powi(2);
        }
        total_length += l.max(0.0) + clen.max(0.0);
    }
    let l_final = params[4 * n_clothoids];
    if l_final < 0.0 {
        neg_penalty += 100.0 * l_final * l_final;
    }
    total_length += l_final.max(0.0);

    let pts = eval_path(params, n_clothoids, start, 20);
    let (xe, ye, te) = *pts.last().unwrap_or(&(start.x, start.y, start.angle));

    if xe.is_nan() || ye.is_nan() || te.is_nan() {
        return 1e10;
    }

    let dx = xe - end.x;
    let dy = ye - end.y;
    let dist_sq = dx * dx + dy * dy;
    let ad = angle_diff(te, end.angle);

    10.0 * dist_sq + 5.0 * ad * ad + neg_penalty + len_penalty + 0.001 * total_length
}

/// Returns the un-weighted `(position_error, |angle_error|)` for a candidate path.
///
/// * `position_error` — Euclidean distance from path endpoint to `end.x/y`
/// * `angle_error` — absolute wrapped angle difference at the endpoint
pub fn compute_end_errors(
    params: &[f64],
    n_clothoids: usize,
    start: &Pose,
    end: &Pose,
) -> (f64, f64) {
    let pts = eval_path(params, n_clothoids, start, 20);
    let (xe, ye, te) = *pts.last().unwrap_or(&(start.x, start.y, start.angle));
    let dist = ((xe - end.x).powi(2) + (ye - end.y).powi(2)).sqrt();
    let ad = angle_diff(te, end.angle).abs();
    (dist, ad)
}

// ============================================================================
// Nelder-Mead optimizer
// ============================================================================

/// Minimises `f` starting from `x0` using the Nelder-Mead simplex method.
///
/// Standard coefficients: reflection α=1, expansion γ=2, contraction ρ=0.5,
/// shrink σ=0.5.  NaN / ±∞ function values are replaced with `1e10`.
///
/// Returns the best parameter vector found within `max_iter` iterations.
/// Returns an empty vector when `x0` is empty.
pub fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
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

    let sanitize = |v: f64| if v.is_nan() || v.is_infinite() { 1e10 } else { v };
    let mut fvals: Vec<f64> = simplex.iter().map(|p| sanitize(f(p))).collect();

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal)
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
        for c in centroid.iter_mut() {
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
                for i in 1..=n {
                    let idx = order[i];
                    for j in 0..n {
                        simplex[idx][j] =
                            simplex[best][j] + sigma * (simplex[idx][j] - simplex[best][j]);
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
        .map(|(i, _)| i)
        .unwrap_or(0);
    simplex[best_idx].clone()
}

// ============================================================================
// LCG random number generator
// ============================================================================

/// A fast, deterministic linear congruential generator (LCG).
///
/// Produces values in `[0, 1)` suitable for random initialisation of optimizer
/// parameters.  Use a fixed seed for reproducibility.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Creates a new [`Lcg`] with the given `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the generator and returns the next value in `[0, 1)`.
    pub fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as f64 / 2_147_483_648.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ------------------------------------------------------------------
    // rk4_step
    // ------------------------------------------------------------------

    /// Zero curvature, θ=0: state should advance purely in +x.
    #[test]
    fn rk4_step_straight_x() {
        let (nx, ny, nt) = rk4_step(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        assert!((nx - 1.0).abs() < 1e-10, "x should advance by step h=1");
        assert!(ny.abs() < 1e-10, "y should not change");
        assert!(nt.abs() < 1e-10, "θ should not change");
    }

    /// Zero curvature, θ=π/2: state should advance purely in +y.
    #[test]
    fn rk4_step_straight_y() {
        let (nx, ny, nt) = rk4_step(0.0, 0.0, PI / 2.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(nx.abs() < 1e-10, "x should not change");
        assert!((ny - 1.0).abs() < 1e-10, "y should advance by h=1");
        assert!((nt - PI / 2.0).abs() < 1e-10, "θ should not change");
    }

    /// Non-zero constant curvature: θ changes by κ·h.
    #[test]
    fn rk4_step_constant_curvature_angle() {
        let kappa = 1.0;
        let h = 0.1;
        // With constant κ=1 (ks=ke=1), angle change ≈ κ·h for small h.
        let (_, _, nt) = rk4_step(0.0, 0.0, 0.0, kappa, kappa, 1.0, 0.0, h);
        // For constant κ the exact θ change is κ·h (straight-forward integration)
        assert!((nt - kappa * h).abs() < 1e-6);
    }

    /// Step from a non-origin position with θ=0 advances x by h.
    #[test]
    fn rk4_step_non_origin_start() {
        let (nx, ny, nt) = rk4_step(3.0, -2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0);
        assert!((nx - 5.0).abs() < 1e-10);
        assert!((ny + 2.0).abs() < 1e-10);
        assert!(nt.abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // integrate_clothoid
    // ------------------------------------------------------------------

    /// Zero-length arc returns only the start state.
    #[test]
    fn integrate_clothoid_zero_length() {
        let pts = integrate_clothoid(1.0, 2.0, 0.5, 1.0, -1.0, 0.0, 100);
        assert_eq!(pts.len(), 1);
        let (x, y, theta) = pts[0];
        assert_eq!(x, 1.0);
        assert_eq!(y, 2.0);
        assert_eq!(theta, 0.5);
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
        let &(x, y, theta) = pts.last().unwrap();
        assert!((x - length).abs() < 1e-6, "x should equal arc length");
        assert!(y.abs() < 1e-6, "y should be zero");
        assert!(theta.abs() < 1e-6, "heading should be unchanged");
    }

    /// Zero curvature, θ=π/2: should trace a straight line in +y.
    #[test]
    fn integrate_clothoid_straight_line_y() {
        let length = 3.0;
        let pts = integrate_clothoid(0.0, 0.0, PI / 2.0, 0.0, 0.0, length, 200);
        let &(x, y, theta) = pts.last().unwrap();
        assert!(x.abs() < 1e-6);
        assert!((y - length).abs() < 1e-6);
        assert!((theta - PI / 2.0).abs() < 1e-6);
    }

    /// Constant curvature κ=1, arc=π: heading should change by π.
    #[test]
    fn integrate_clothoid_constant_kappa_angle_change() {
        let kappa = 1.0;
        let arc = PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 1000);
        let &(_, _, theta) = pts.last().unwrap();
        // θ should change by κ * arc = π
        assert!((theta - PI).abs() < 1e-3, "heading should rotate by π rad");
    }

    /// Constant curvature κ=1, arc=π: traces a half-circle of radius 1.
    /// Starting at (0,0) heading +x, the centre is at (0,1) and the endpoint is (0,2).
    #[test]
    fn integrate_clothoid_half_circle_position() {
        let kappa = 1.0; // radius = 1
        let arc = PI; // half circle
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 2000);
        let &(x, y, _) = pts.last().unwrap();
        assert!(x.abs() < 1e-3, "x should return to ~0 after half circle");
        assert!((y - 2.0).abs() < 1e-3, "y should be 2*radius = 2");
    }

    /// Constant curvature κ=1, arc=2π: full circle — should return to near (0,0)
    /// with heading equal to 0 (modulo 2π).
    #[test]
    fn integrate_clothoid_full_circle() {
        let kappa = 1.0;
        let arc = 2.0 * PI;
        let pts = integrate_clothoid(0.0, 0.0, 0.0, kappa, kappa, arc, 4000);
        let &(x, y, theta) = pts.last().unwrap();
        assert!(x.abs() < 1e-2, "x should return near 0");
        assert!(y.abs() < 1e-2, "y should return near 0");
        // θ should be ≈ 2π (i.e. one full rotation)
        assert!((theta - 2.0 * PI).abs() < 1e-2);
    }

    /// Linearly varying curvature (clothoid spiral): heading change = integral of κ(s) ds
    /// = (ks + ke) / 2 * length.
    #[test]
    fn integrate_clothoid_linearly_varying_kappa_angle() {
        let ks = 0.0;
        let ke = 2.0;
        let length = 1.0;
        // Expected total angle change = ∫₀¹ (0 + 2t) dt = [t²]₀¹ = 1.0 rad
        let pts = integrate_clothoid(0.0, 0.0, 0.0, ks, ke, length, 1000);
        let &(_, _, theta) = pts.last().unwrap();
        let expected = (ks + ke) / 2.0 * length; // = 1.0
        assert!((theta - expected).abs() < 1e-4);
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
        // params: [l=0, ks=0, ke=0, clen=5, l_final=0]
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let &(x, y, theta) = pts.last().unwrap();
        assert!((x - 5.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
        assert!(theta.abs() < 1e-6);
    }

    /// Initial straight segment + zero-curvature clothoid: total length is sum.
    #[test]
    fn eval_path_initial_straight_plus_clothoid() {
        // params: [l=2, ks=0, ke=0, clen=3, l_final=0] — total 5 in +x
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [2.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let &(x, y, _) = pts.last().unwrap();
        assert!((x - 5.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    /// Final trailing straight segment advances position.
    #[test]
    fn eval_path_trailing_straight() {
        // params: [l=0, ks=0, ke=0, clen=2, l_final=3] — total 5 in +x
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 3.0];
        let pts = eval_path(&params, 1, &start, 100);
        let &(x, y, _) = pts.last().unwrap();
        assert!((x - 5.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    /// Non-origin start pose: position offset is respected.
    #[test]
    fn eval_path_non_origin_start() {
        let start = Pose::new(1.0, 2.0, 0.0);
        let params = [0.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 1, &start, 100);
        let &(x, y, _) = pts.last().unwrap();
        assert!((x - 4.0).abs() < 1e-6);
        assert!((y - 2.0).abs() < 1e-6);
    }

    /// Two clothoid segments, both zero-curvature: lengths add up correctly.
    #[test]
    fn eval_path_two_clothoids_straight() {
        // params: [l=0, ks=0, ke=0, clen=2, l=0, ks=0, ke=0, clen=3, l_final=0]
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let pts = eval_path(&params, 2, &start, 100);
        let &(x, y, _) = pts.last().unwrap();
        assert!((x - 5.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    /// Negative lengths are clamped to zero.
    #[test]
    fn eval_path_negative_lengths_clamped() {
        let start = Pose::new(0.0, 0.0, 0.0);
        // Negative l is clamped to 0, negative clen is clamped to 1e-6 (near zero)
        let params = [-1.0, 0.0, 0.0, 3.0, -0.5];
        let pts = eval_path(&params, 1, &start, 50);
        // The clothoid should still integrate for ~3 units; final negative straight ignored
        let &(x, y, _) = pts.last().unwrap();
        assert!((x - 3.0).abs() < 1e-5);
        assert!(y.abs() < 1e-5);
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
        let result = nelder_mead(
            &|x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[5.0, -3.0],
            2000,
        );
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
            assert_eq!(a.next().to_bits(), b.next().to_bits());
        }
    }

    /// All values lie in [0, 1).
    #[test]
    fn lcg_range() {
        let mut lcg = Lcg::new(12345);
        for _ in 0..10_000 {
            let v = lcg.next();
            assert!(v >= 0.0, "value {v} is negative");
            assert!(v < 1.0, "value {v} is ≥ 1.0");
        }
    }

    /// Different seeds produce different sequences.
    #[test]
    fn lcg_different_seeds() {
        let mut a = Lcg::new(1);
        let mut b = Lcg::new(2);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next().to_bits()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next().to_bits()).collect();
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
        let params = nelder_mead(
            &|p: &[f64]| compute_error(p, n, &start, &end),
            &x0,
            1000,
        );
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
        let params = nelder_mead(
            &|p: &[f64]| compute_error(p, n, &start, &end),
            &x0,
            2000,
        );
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);
        assert!(pos_err < 0.1, "pos_err = {pos_err}");
        assert!(angle_err < 0.1, "angle_err = {angle_err}");
    }
}
