use crate::optimizer::{
    compute_end_errors, compute_error, eval_path_segmented, nelder_mead, Lcg, Pose, SegmentKind,
};

#[derive(Clone)]
pub struct RenderSegment {
    pub kind: SegmentKind,
    pub points: Vec<(f32, f32)>,
    pub boundary_theta: f32,
}

#[derive(Clone)]
pub struct PathFit {
    pub params: Vec<f64>,
    pub n_clothoids: usize,
    pub total_error: f64,
    pub pos_error: f64,
    pub angle_error: f64,
    pub segments: Vec<RenderSegment>,
}

#[derive(Clone)]
pub struct FitConfig {
    pub max_segments: usize,
    pub max_kappa: f64,
    pub tol_pos: f64,
    pub tol_angle: f64,
}

pub struct FitState {
    pub best_fit: Option<PathFit>,
    pub exploration: Option<PathFit>,
    pub log: Vec<String>,
    generation: u64,
    last_gen: u64,
    n_clothoids: usize,
    restart_count: usize,
    lcg: Lcg,
}

impl FitState {
    pub fn new() -> Self {
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

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn bump_generation(&mut self) {
        self.generation += 1;
    }

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
            let r = self.lcg.next();
            *v = match idx % 4 {
                0 => r * dist_est * 0.5,
                1 | 2 => (r - 0.5) * 2.0 * config.max_kappa,
                3 => r * dist_est * 1.5 + 0.1,
                _ => 0.0,
            };
        }
        let r = self.lcg.next();
        x0[n_params - 1] = r * dist_est * 0.5;

        let start_c = start.clone();
        let end_c = end.clone();
        let params = nelder_mead(&|p: &[f64]| compute_error(p, n, &start_c, &end_c), &x0, 500);

        let total_err = compute_error(&params, n, start, end);
        let (pos_err, angle_err) = compute_end_errors(&params, n, start, end);

        let path_segs = eval_path_segmented(&params, n, start, 40);
        let render_segs: Vec<RenderSegment> = path_segs
            .into_iter()
            .map(|s| RenderSegment {
                kind: s.kind,
                points: s
                    .points
                    .iter()
                    .map(|&(x, y, _)| (x as f32, y as f32))
                    .collect(),
                boundary_theta: s.boundary_theta as f32,
            })
            .collect();

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
