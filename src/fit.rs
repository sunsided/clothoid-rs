//! Incremental path fitting using clothoid segments.
//!
//! This module provides a stateful fitter that incrementally optimizes a path
//! between two 2D poses using a pluggable [`Planner`] strategy.

use crate::optimizer::{
    compute_end_errors, compute_error, eval_path_segmented, CmaEs, Lcg, NelderMead, Optimizer,
    PathSegment, Pose, SegmentKind, DEFAULT_RNG_SEED,
};

/// A render-ready segment of a fitted path.
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
    /// The flat parameter vector describing the path.
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

/// High-level planner for the per-iteration fit strategy.
pub trait Planner: Send {
    /// One outer fit iteration. Returns `(exploration, best_fit)`.
    fn step(
        &mut self,
        start: &Pose,
        end: &Pose,
        config: &FitConfig,
    ) -> (Option<PathFit>, Option<PathFit>);

    fn bump_generation(&mut self);
    fn generation(&self) -> u64;
    fn log(&self) -> &[String];
    fn best(&self) -> Option<&PathFit>;
    fn exploration(&self) -> Option<&PathFit>;
    fn name(&self) -> &'static str;
}

/// Default planner implementation that wraps any [`Optimizer`].
///
/// Carries the restart + segment-escalation heuristic and delegates the inner
/// minimization to the given optimizer.
pub struct DefaultPlanner<O: Optimizer> {
    optimizer: O,
    best_fit: Option<PathFit>,
    exploration: Option<PathFit>,
    log: Vec<String>,
    generation: u64,
    last_gen: u64,
    n_clothoids: usize,
    restart_count: usize,
    lcg: Lcg,
}

impl DefaultPlanner<NelderMead> {
    pub fn new() -> Self {
        Self {
            optimizer: NelderMead,
            best_fit: None,
            exploration: None,
            log: Vec::new(),
            generation: 0,
            last_gen: u64::MAX,
            n_clothoids: 1,
            restart_count: 0,
            lcg: Lcg::new(DEFAULT_RNG_SEED),
        }
    }
}

impl Default for DefaultPlanner<NelderMead> {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultPlanner<CmaEs> {
    pub fn new_cma() -> Self {
        Self {
            optimizer: CmaEs::new(DEFAULT_RNG_SEED),
            best_fit: None,
            exploration: None,
            log: Vec::new(),
            generation: 0,
            last_gen: u64::MAX,
            n_clothoids: 1,
            restart_count: 0,
            lcg: Lcg::new(DEFAULT_RNG_SEED),
        }
    }
}

impl<O: Optimizer> Planner for DefaultPlanner<O> {
    fn step(
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
        let params =
            self.optimizer
                .minimize(&|p: &[f64]| compute_error(p, n, &start_c, &end_c), &x0, 500);

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

    fn bump_generation(&mut self) {
        self.generation += 1;
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn log(&self) -> &[String] {
        &self.log
    }

    fn best(&self) -> Option<&PathFit> {
        self.best_fit.as_ref()
    }

    fn exploration(&self) -> Option<&PathFit> {
        self.exploration.as_ref()
    }

    fn name(&self) -> &'static str {
        self.optimizer.name()
    }
}

/// Stateful incremental clothoid path fitter.
///
/// Maintains exploration and best-fit solutions across generations.
pub struct FitState {
    planner: Box<dyn Planner>,
}

impl Default for FitState {
    fn default() -> Self {
        Self::new()
    }
}

impl FitState {
    /// Creates a new [`FitState`] with the default Nelder-Mead planner.
    pub fn new() -> Self {
        Self {
            planner: Box::new(DefaultPlanner::<NelderMead>::new()),
        }
    }

    /// Creates a new [`FitState`] using the CMA-ES planner.
    pub fn cma_es() -> Self {
        Self {
            planner: Box::new(DefaultPlanner::<CmaEs>::new_cma()),
        }
    }

    /// Creates a new [`FitState`] with an arbitrary planner.
    pub fn with_planner<P: Planner + 'static>(planner: P) -> Self {
        Self {
            planner: Box::new(planner),
        }
    }

    pub fn step(
        &mut self,
        start: &Pose,
        end: &Pose,
        config: &FitConfig,
    ) -> (Option<PathFit>, Option<PathFit>) {
        self.planner.step(start, end, config)
    }

    pub fn best_fit(&self) -> Option<&PathFit> {
        self.planner.best()
    }

    pub fn exploration(&self) -> Option<&PathFit> {
        self.planner.exploration()
    }

    pub fn log(&self) -> &[String] {
        self.planner.log()
    }

    pub fn generation(&self) -> u64 {
        self.planner.generation()
    }

    pub fn bump_generation(&mut self) {
        self.planner.bump_generation()
    }

    pub fn name(&self) -> &'static str {
        self.planner.name()
    }
}
