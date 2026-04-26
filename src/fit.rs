//! Incremental path fitting using clothoid segments.
//!
//! This module provides a stateful fitter that incrementally optimizes a path
//! between two 2D poses using the Nelder-Mead optimizer. It supports multi-segment
//! paths with automatic restart and complexity escalation.

use crate::optimizer::{
    compute_end_errors, compute_error, eval_path_segmented, nelder_mead, Lcg, PathSegment, Pose,
    SegmentKind,
};

/// A render-ready segment of a fitted path.
///
/// Contains the segment type, 2D points (as `f32` for rendering), and the
/// boundary heading angle.
#[derive(Clone)]
pub struct RenderSegment {
    /// The kind of this segment (linear or curve).
    pub kind: SegmentKind,
    /// Sampled 2D points `(x, y)` as 32-bit floats for rendering.
    pub points: Vec<(f32, f32)>,
    /// The heading angle at the segment boundary (as `f32`).
    pub boundary_theta: f32,
}

impl From<&PathSegment> for RenderSegment {
    fn from(s: &PathSegment) -> Self {
        RenderSegment {
            kind: s.kind,
            points: s.points.iter().map(|p| (p.x as f32, p.y as f32)).collect(),
            boundary_theta: s.boundary_theta as f32,
        }
    }
}

/// A complete fit result describing a path between two poses.
#[derive(Clone)]
pub struct PathFit {
    /// The flat parameter vector describing the path (see [`crate::optimizer::eval_path`]).
    pub params: Vec<f64>,
    /// The number of clothoid segments in this path.
    pub n_clothoids: usize,
    /// The total weighted error score from the optimization objective.
    pub total_error: f64,
    /// The unweighted Euclidean distance error at the endpoint.
    pub pos_error: f64,
    /// The unweighted absolute angle error at the endpoint.
    pub angle_error: f64,
    /// The renderable segments of this path.
    pub segments: Vec<RenderSegment>,
}

/// Configuration for the path fitter.
#[derive(Clone)]
pub struct FitConfig {
    /// Maximum number of clothoid segments to try before restarting.
    pub max_segments: usize,
    /// Maximum absolute curvature allowed for any segment.
    pub max_kappa: f64,
    /// Tolerance for position error to consider a fit successful.
    pub tol_pos: f64,
    /// Tolerance for angle error to consider a fit successful.
    pub tol_angle: f64,
}

/// Stateful incremental clothoid path fitter.
///
/// Maintains exploration and best-fit solutions across generations.
/// Automatically increases segment count if the current complexity fails
/// to produce a solution within tolerance after 200 iterations.
pub struct FitState {
    /// The best fit found so far that meets the tolerance criteria.
    pub best_fit: Option<PathFit>,
    /// The most recent exploration result (may not meet tolerances).
    pub exploration: Option<PathFit>,
    /// A rolling log of recent fit messages (capped at 20 entries).
    pub log: Vec<String>,
    /// The current generation counter.
    generation: u64,
    /// The generation at which the last reset occurred.
    last_gen: u64,
    /// The current number of clothoid segments being tried.
    n_clothoids: usize,
    /// The number of iterations since the last segment count change.
    restart_count: usize,
    /// The random number generator for parameter initialization.
    lcg: Lcg,
}

impl Default for FitState {
    fn default() -> Self {
        Self {
            best_fit: None,
            exploration: None,
            log: Vec::new(),
            generation: 0,
            last_gen: u64::MAX,
            n_clothoids: 1,
            restart_count: 0,
            lcg: Lcg::new(crate::optimizer::DEFAULT_RNG_SEED),
        }
    }
}

impl FitState {
    /// Creates a new [`FitState`] with default settings.
    ///
    /// The fitter starts with 1 clothoid segments, generation 0, and uses
    /// the default RNG seed for reproducible parameter initialization.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Increments the generation counter.
    ///
    /// On the next [`step`](FitState::step) call, if the generation has changed,
    /// the fitter will reset to 1 clothoid segment and clear previous results.
    pub fn bump_generation(&mut self) {
        self.generation += 1;
    }

    /// Runs one optimization iteration.
    ///
    /// Generates random initial parameters, runs the Nelder-Mead optimizer
    /// for 500 iterations, and evaluates the result. If the result meets the
    /// tolerance criteria and is better than the current best, it is stored.
    ///
    /// After 200 iterations without success, the segment count is increased
    /// (up to `max_segments`), or reset to 1 if already at maximum.
    ///
    /// # Arguments
    ///
    /// * `start` — The starting pose.
    /// * `end` — The target pose.
    /// * `config` — Fitting configuration (tolerances, limits).
    ///
    /// # Returns
    ///
    /// A tuple of `(exploration, best_fit)`, where `exploration` is the most
    /// recent result and `best_fit` is the best valid fit found so far.
    pub fn step(
        &mut self,
        start: &Pose,
        end: &Pose,
        config: &FitConfig,
    ) -> (Option<PathFit>, Option<PathFit>) {
        if self.generation != self.last_gen {
            self.last_gen = self.generation;
            self.n_clothoids = 1;
            self.restart_count = 0;
            self.best_fit = None;
            self.exploration = None;
        }

        let n = self.n_clothoids;
        let n_params = 4 * n + 1;

        let dist_est = {
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            (dx * dx + dy * dy).sqrt().max(0.1)
        };

        let mut x0 = vec![0.0f64; n_params];
        for (idx, v) in x0.iter_mut().enumerate() {
            let r = self.lcg.next_val();
            *v = match idx % 4 {
                0 => r * dist_est * 0.5,
                1 | 2 => (r - 0.5) * 2.0 * config.max_kappa,
                3 => r * dist_est * 1.5 + 0.1,
                _ => 0.0,
            };
        }
        let r = self.lcg.next_val();
        x0[n_params - 1] = r * dist_est * 0.5;

        let start_c = *start;
        let end_c = *end;
        let params = nelder_mead(&|p: &[f64]| compute_error(p, n, &start_c, &end_c), &x0, 500);

        let total_err = compute_error(&params, n, start, end);
        let (pos_err, angle_err) = compute_end_errors(&params, n, start, end);

        let path_segs = eval_path_segmented(&params, n, start, 40);
        let render_segs: Vec<RenderSegment> = path_segs.iter().map(RenderSegment::from).collect();

        let fit = PathFit {
            params,
            n_clothoids: n,
            total_error: total_err,
            pos_error: pos_err,
            angle_error: angle_err,
            segments: render_segs,
        };

        if pos_err < config.tol_pos && angle_err < config.tol_angle {
            let is_better = match &self.best_fit {
                None => true,
                Some(best) => total_err < best.total_error,
            };
            if is_better {
                self.log.push(format!(
                    "Found fit: n={} pos={:.3} ang={:.3}",
                    n, pos_err, angle_err
                ));
                if self.log.len() > 20 {
                    self.log.remove(0);
                }
                self.best_fit = Some(fit.clone());
            }
        }

        self.exploration = Some(fit);

        self.restart_count += 1;
        if self.restart_count >= 200 {
            self.restart_count = 0;
            self.n_clothoids = if self.n_clothoids >= config.max_segments {
                1
            } else {
                self.n_clothoids + 1
            };
        }

        (self.exploration.clone(), self.best_fit.clone())
    }
}
