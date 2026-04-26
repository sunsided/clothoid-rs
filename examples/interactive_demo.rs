#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use softbuffer::{Context, Surface};
use tiny_skia::{
    Color, FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Seed for the LCG random number generator used by the optimizer.
const DEFAULT_RNG_SEED: u64 = 42;

// ============================================================================
// Pose
// ============================================================================

#[derive(Clone, Debug)]
struct Pose {
    x: f64,
    y: f64,
    angle: f64,
}

// ============================================================================
// Clothoid math (RK4)
// ============================================================================

/// Single RK4 step over arc-length h for a clothoid with linearly varying κ.
fn rk4_step(
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

fn integrate_clothoid(
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

/// Evaluate the full path.
/// Params layout: [l0, ks0, ke0, clen0, l1, ks1, ke1, clen1, ..., lN]
/// Length: 4*N + 1.
fn eval_path(
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

fn angle_diff(a: f64, b: f64) -> f64 {
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

fn compute_error(params: &[f64], n_clothoids: usize, start: &Pose, end: &Pose) -> f64 {
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

fn compute_end_errors(params: &[f64], n_clothoids: usize, start: &Pose, end: &Pose) -> (f64, f64) {
    let pts = eval_path(params, n_clothoids, start, 20);
    let (xe, ye, te) = *pts.last().unwrap_or(&(start.x, start.y, start.angle));
    let dist = ((xe - end.x).powi(2) + (ye - end.y).powi(2)).sqrt();
    let ad = angle_diff(te, end.angle).abs();
    (dist, ad)
}

// ============================================================================
// Nelder-Mead optimizer
// ============================================================================

fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_iter: usize) -> Vec<f64> {
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
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));

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
// LCG RNG
// ============================================================================

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as f64 / 2_147_483_648.0
    }
}

// ============================================================================
// Shared state
// ============================================================================

#[derive(Clone)]
struct PathFit {
    params: Vec<f64>,
    n_clothoids: usize,
    total_error: f64,
    pos_error: f64,
    angle_error: f64,
    points: Vec<(f32, f32)>,
}

struct SharedState {
    start: Pose,
    end: Pose,
    generation: u64,
    best_fit: Option<PathFit>,
    exploration: Option<PathFit>,
    max_segments: usize,
    max_kappa: f64,
    tol_pos: f64,
    tol_angle: f64,
    paused: bool,
    log: Vec<String>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            start: Pose { x: -3.0, y: 0.0, angle: 0.0 },
            end: Pose { x: 3.0, y: 0.0, angle: 0.0 },
            generation: 0,
            best_fit: None,
            exploration: None,
            max_segments: 2,
            max_kappa: 2.0,
            tol_pos: 0.05,
            tol_angle: 0.05,
            paused: false,
            log: Vec::new(),
        }
    }
}

// ============================================================================
// Optimizer thread
// ============================================================================

fn run_optimizer(shared: Arc<Mutex<SharedState>>) {
    let mut lcg = Lcg::new(DEFAULT_RNG_SEED);
    let mut last_gen: u64 = u64::MAX;
    let mut n_clothoids: usize = 1;
    let mut restart_count: usize = 0;

    loop {
        let (start, end, max_seg, max_kappa, tol_pos, tol_angle, paused, gen) = {
            let st = shared.lock().unwrap();
            (
                st.start.clone(),
                st.end.clone(),
                st.max_segments,
                st.max_kappa,
                st.tol_pos,
                st.tol_angle,
                st.paused,
                st.generation,
            )
        };

        if paused {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        if gen != last_gen {
            last_gen = gen;
            n_clothoids = 1;
            restart_count = 0;
            let mut st = shared.lock().unwrap();
            st.best_fit = None;
            st.exploration = None;
        }

        let n = n_clothoids;
        let n_params = 4 * n + 1;

        let dist_est = {
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            (dx * dx + dy * dy).sqrt().max(0.1)
        };

        let mut x0 = vec![0.0f64; n_params];
        for (idx, v) in x0.iter_mut().enumerate() {
            let r = lcg.next();
            *v = match idx % 4 {
                0 => r * dist_est * 0.5,
                1 | 2 => (r - 0.5) * 2.0 * max_kappa,
                3 => r * dist_est * 1.5 + 0.1,
                _ => 0.0,
            };
        }
        // Override last param (final straight segment)
        let r = lcg.next();
        x0[n_params - 1] = r * dist_est * 0.5;

        let start_c = start.clone();
        let end_c = end.clone();
        let params = nelder_mead(
            &|p: &[f64]| compute_error(p, n, &start_c, &end_c),
            &x0,
            500,
        );

        let total_err = compute_error(&params, n, &start, &end);
        let (pos_err, angle_err) = compute_end_errors(&params, n, &start, &end);

        let render_pts: Vec<(f32, f32)> = eval_path(&params, n, &start, 100)
            .iter()
            .map(|&(x, y, _)| (x as f32, y as f32))
            .collect();

        let fit = PathFit {
            params,
            n_clothoids: n,
            total_error: total_err,
            pos_error: pos_err,
            angle_error: angle_err,
            points: render_pts,
        };

        {
            let mut st = shared.lock().unwrap();
            if st.generation == last_gen {
                st.exploration = Some(fit.clone());
                if pos_err < tol_pos && angle_err < tol_angle {
                    let is_better = match &st.best_fit {
                        None => true,
                        Some(best) => total_err < best.total_error,
                    };
                    if is_better {
                        st.log.push(format!(
                            "Found fit: n={} pos={:.3} ang={:.3}",
                            n, pos_err, angle_err
                        ));
                        if st.log.len() > 20 {
                            st.log.remove(0);
                        }
                        st.best_fit = Some(fit);
                    }
                }
            }
        }

        restart_count += 1;
        if restart_count >= 200 {
            restart_count = 0;
            n_clothoids = if n_clothoids >= max_seg { 1 } else { n_clothoids + 1 };
        }

        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}

// ============================================================================
// Camera
// ============================================================================

struct Camera {
    pan_x: f64,
    pan_y: f64,
    zoom: f64,
}

impl Camera {
    fn new() -> Self {
        Self { pan_x: 0.0, pan_y: 0.0, zoom: 80.0 }
    }

    fn world_to_screen(&self, wx: f64, wy: f64, w: f32, h: f32) -> (f32, f32) {
        let sx = ((wx - self.pan_x) * self.zoom + w as f64 / 2.0) as f32;
        let sy = ((wy - self.pan_y) * self.zoom + h as f64 / 2.0) as f32;
        (sx, sy)
    }

    fn screen_to_world(&self, sx: f64, sy: f64, w: f32, h: f32) -> (f64, f64) {
        let wx = (sx - w as f64 / 2.0) / self.zoom + self.pan_x;
        let wy = (sy - h as f64 / 2.0) / self.zoom + self.pan_y;
        (wx, wy)
    }
}

// ============================================================================
// Drag
// ============================================================================

#[derive(Clone, Copy)]
enum DragTarget {
    StartPos,
    StartDir,
    EndPos,
    EndDir,
}

// ============================================================================
// Rendering helpers
// ============================================================================

fn fill_rect_solid(pixmap: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, color: Color) {
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    if let Some(rect) = tiny_skia::Rect::from_xywh(x, y, w, h) {
        let mut paint = Paint::default();
        paint.set_color(color);
        pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    }
}

fn draw_line(
    pixmap: &mut Pixmap,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: Color,
    width: f32,
) {
    let mut pb = PathBuilder::new();
    pb.move_to(x0, y0);
    pb.line_to(x1, y1);
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let mut stroke = Stroke::default();
        stroke.width = width;
        stroke.line_cap = LineCap::Round;
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
}

fn draw_circle_stroke(pixmap: &mut Pixmap, cx: f32, cy: f32, r: f32, color: Color, width: f32) {
    let mut pb = PathBuilder::new();
    pb.push_circle(cx, cy, r);
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let mut stroke = Stroke::default();
        stroke.width = width;
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
}

fn draw_filled_triangle(
    pixmap: &mut Pixmap,
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    color: Color,
) {
    let mut pb = PathBuilder::new();
    pb.move_to(p0.0, p0.1);
    pb.line_to(p1.0, p1.1);
    pb.line_to(p2.0, p2.1);
    pb.close();
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }
}

fn draw_gizmo(pixmap: &mut Pixmap, pose: &Pose, color: Color, camera: &Camera) {
    let w = pixmap.width() as f32;
    let h = pixmap.height() as f32;
    let (cx, cy) = camera.world_to_screen(pose.x, pose.y, w, h);

    draw_circle_stroke(pixmap, cx, cy, 12.0, color, 2.0);

    let tip_wx = pose.x + 1.5 * pose.angle.cos();
    let tip_wy = pose.y + 1.5 * pose.angle.sin();
    let (tx, ty) = camera.world_to_screen(tip_wx, tip_wy, w, h);

    draw_line(pixmap, cx, cy, tx, ty, color, 2.0);

    let angle = pose.angle as f32;
    let head_len = 10.0f32;
    let head_w = 5.0f32;
    let fwd_x = angle.cos();
    let fwd_y = angle.sin();
    let lft_x = -fwd_y;
    let lft_y = fwd_x;

    let tip = (tx, ty);
    let bl = (tx - head_len * fwd_x + head_w * lft_x, ty - head_len * fwd_y + head_w * lft_y);
    let br = (tx - head_len * fwd_x - head_w * lft_x, ty - head_len * fwd_y - head_w * lft_y);
    draw_filled_triangle(pixmap, tip, bl, br, color);
}

fn draw_world_path(
    pixmap: &mut Pixmap,
    pts: &[(f32, f32)],
    color: Color,
    width: f32,
    camera: &Camera,
) {
    if pts.len() < 2 {
        return;
    }
    let w = pixmap.width() as f32;
    let h = pixmap.height() as f32;

    let mut pb = PathBuilder::new();
    let (sx, sy) = camera.world_to_screen(pts[0].0 as f64, pts[0].1 as f64, w, h);
    pb.move_to(sx, sy);
    for &(wx, wy) in &pts[1..] {
        let (sx, sy) = camera.world_to_screen(wx as f64, wy as f64, w, h);
        pb.line_to(sx, sy);
    }
    if let Some(path) = pb.finish() {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;
        let mut stroke = Stroke::default();
        stroke.width = width;
        stroke.line_cap = LineCap::Round;
        stroke.line_join = LineJoin::Round;
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
}

fn draw_grid(pixmap: &mut Pixmap, camera: &Camera) {
    let pw = pixmap.width() as f64;
    let ph = pixmap.height() as f64;
    let w = pw as f32;
    let h = ph as f32;

    let (x_min, y_min) = camera.screen_to_world(0.0, 0.0, w, h);
    let (x_max, y_max) = camera.screen_to_world(pw, ph, w, h);

    let grid_color = Color::from_rgba8(60, 60, 60, 255);

    let x0 = x_min.floor() as i32;
    let x1 = x_max.ceil() as i32;
    let y0 = y_min.floor() as i32;
    let y1 = y_max.ceil() as i32;

    // Guard against too many lines (extreme zoom-out).
    if (x1 - x0).abs() > 200 || (y1 - y0).abs() > 200 {
        return;
    }

    for xi in x0..=x1 {
        let (sx, _) = camera.world_to_screen(xi as f64, 0.0, w, h);
        draw_line(pixmap, sx, 0.0, sx, h, grid_color, 1.0);
    }
    for yi in y0..=y1 {
        let (_, sy) = camera.world_to_screen(0.0, yi as f64, w, h);
        draw_line(pixmap, 0.0, sy, w, sy, grid_color, 1.0);
    }
}

fn draw_hud(pixmap: &mut Pixmap, state: &SharedState) {
    let w = pixmap.width() as f32;
    let h = pixmap.height() as f32;

    // Segment count squares
    let sq = 12.0f32;
    let gap = 3.0f32;
    let sy = h - 35.0;
    for i in 0..8usize {
        let color = if i < state.max_segments {
            Color::from_rgba8(100, 200, 100, 255)
        } else {
            Color::from_rgba8(40, 40, 40, 255)
        };
        let sx = 10.0 + i as f32 * (sq + gap);
        fill_rect_solid(pixmap, sx, sy, sq, sq, color);
    }

    // Max kappa bar
    let bar_x = 10.0f32;
    let bar_y = h - 55.0;
    let bar_w = 120.0f32;
    let bar_h = 8.0f32;
    fill_rect_solid(pixmap, bar_x, bar_y, bar_w, bar_h, Color::from_rgba8(40, 40, 40, 255));
    let frac = ((state.max_kappa - 0.05) / (20.0 - 0.05)).clamp(0.0, 1.0) as f32;
    fill_rect_solid(
        pixmap,
        bar_x,
        bar_y,
        bar_w * frac,
        bar_h,
        Color::from_rgba8(200, 150, 50, 255),
    );

    // Fit found indicator (top-right)
    let fit_color = if state.best_fit.is_some() {
        Color::from_rgba8(50, 200, 50, 255)
    } else {
        Color::from_rgba8(50, 50, 50, 255)
    };
    fill_rect_solid(pixmap, w - 35.0, 10.0, 25.0, 25.0, fit_color);

    // Paused indicator
    if state.paused {
        fill_rect_solid(pixmap, w - 35.0, 45.0, 25.0, 25.0, Color::from_rgba8(200, 50, 50, 255));
    }
}

// ============================================================================
// Application
// ============================================================================

struct App {
    window: Option<Arc<Window>>,
    context: Option<Context<Arc<Window>>>,
    surface: Option<Surface<Arc<Window>, Arc<Window>>>,
    shared: Arc<Mutex<SharedState>>,
    camera: Camera,
    drag_target: Option<DragTarget>,
    last_mouse: (f64, f64),
    is_panning: bool,
    pan_start_mouse: (f64, f64),
    pan_start_world: (f64, f64),
    width: u32,
    height: u32,
}

impl App {
    fn new(shared: Arc<Mutex<SharedState>>) -> Self {
        Self {
            window: None,
            context: None,
            surface: None,
            shared,
            camera: Camera::new(),
            drag_target: None,
            last_mouse: (0.0, 0.0),
            is_panning: false,
            pan_start_mouse: (0.0, 0.0),
            pan_start_world: (0.0, 0.0),
            width: 800,
            height: 600,
        }
    }

    fn render(&mut self) {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return;
        }

        let mut pixmap = match Pixmap::new(w, h) {
            Some(p) => p,
            None => return,
        };

        pixmap.fill(Color::from_rgba8(40, 40, 40, 255));
        draw_grid(&mut pixmap, &self.camera);

        let state = self.shared.lock().unwrap();

        if let Some(ref expl) = state.exploration {
            draw_world_path(
                &mut pixmap,
                &expl.points,
                Color::from_rgba8(50, 80, 180, 128),
                1.5,
                &self.camera,
            );
        }
        if let Some(ref best) = state.best_fit {
            draw_world_path(
                &mut pixmap,
                &best.points,
                Color::from_rgba8(50, 230, 200, 255),
                3.0,
                &self.camera,
            );
        }

        let start = state.start.clone();
        let end = state.end.clone();
        draw_hud(&mut pixmap, &state);
        drop(state);

        draw_gizmo(&mut pixmap, &start, Color::from_rgba8(50, 220, 50, 255), &self.camera);
        draw_gizmo(&mut pixmap, &end, Color::from_rgba8(220, 50, 50, 255), &self.camera);

        if let Some(surface) = &mut self.surface {
            if surface.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap()).is_err() {
                return;
            }
            let mut buf = match surface.buffer_mut() {
                Ok(b) => b,
                Err(_) => return,
            };
            let pixels = pixmap.pixels();
            for (i, px) in pixels.iter().enumerate() {
                let a = px.alpha() as u32;
                let (r, g, b) = if a == 0 {
                    (0, 0, 0)
                } else {
                    let r = ((px.red() as u32 * 255 + a / 2) / a).min(255);
                    let g = ((px.green() as u32 * 255 + a / 2) / a).min(255);
                    let b = ((px.blue() as u32 * 255 + a / 2) / a).min(255);
                    (r, g, b)
                };
                if i < buf.len() {
                    buf[i] = (r << 16) | (g << 8) | b;
                }
            }
            let _ = buf.present();
        }
    }

    fn hit_test_drag(&mut self) {
        let (mx, my) = self.last_mouse;
        let w = self.width as f32;
        let h = self.height as f32;

        let (scx, scy, stx, sty, ecx, ecy, etx, ety) = {
            let st = self.shared.lock().unwrap();
            let s = &st.start;
            let e = &st.end;
            let (scx, scy) = self.camera.world_to_screen(s.x, s.y, w, h);
            let (stx, sty) = self.camera.world_to_screen(
                s.x + 1.5 * s.angle.cos(),
                s.y + 1.5 * s.angle.sin(),
                w,
                h,
            );
            let (ecx, ecy) = self.camera.world_to_screen(e.x, e.y, w, h);
            let (etx, ety) = self.camera.world_to_screen(
                e.x + 1.5 * e.angle.cos(),
                e.y + 1.5 * e.angle.sin(),
                w,
                h,
            );
            (scx, scy, stx, sty, ecx, ecy, etx, ety)
        };

        let dist = |ax: f64, ay: f64| ((mx - ax).powi(2) + (my - ay).powi(2)).sqrt();

        let candidates: [(f64, DragTarget); 4] = [
            (dist(scx as f64, scy as f64), DragTarget::StartPos),
            (dist(stx as f64, sty as f64), DragTarget::StartDir),
            (dist(ecx as f64, ecy as f64), DragTarget::EndPos),
            (dist(etx as f64, ety as f64), DragTarget::EndDir),
        ];

        self.drag_target = candidates
            .iter()
            .filter(|(d, _)| *d < 20.0)
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, t)| *t);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Clothoid Fitting Demo")
            .with_inner_size(winit::dpi::LogicalSize::new(800u32, 600u32));

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("Failed to create window: {e:?}");
                event_loop.exit();
                return;
            }
        };

        let ctx = match Context::new(window.clone()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to create softbuffer context: {e:?}");
                event_loop.exit();
                return;
            }
        };

        let surface = match Surface::new(&ctx, window.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to create surface: {e:?}");
                event_loop.exit();
                return;
            }
        };

        event_loop.set_control_flow(ControlFlow::Poll);
        window.request_redraw();

        self.context = Some(ctx);
        self.surface = Some(surface);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) | PhysicalKey::Code(KeyCode::KeyQ) => {
                            event_loop.exit();
                        }
                        PhysicalKey::Code(KeyCode::Space) => {
                            let mut st = self.shared.lock().unwrap();
                            st.paused = !st.paused;
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.camera = Camera::new();
                        }
                        PhysicalKey::Code(KeyCode::Equal) => {
                            let mut st = self.shared.lock().unwrap();
                            if st.max_segments < 8 {
                                st.max_segments += 1;
                            }
                        }
                        PhysicalKey::Code(KeyCode::Minus) => {
                            let mut st = self.shared.lock().unwrap();
                            if st.max_segments > 1 {
                                st.max_segments -= 1;
                            }
                        }
                        PhysicalKey::Code(KeyCode::BracketLeft) => {
                            let mut st = self.shared.lock().unwrap();
                            st.max_kappa = (st.max_kappa * 0.8).max(0.05);
                        }
                        PhysicalKey::Code(KeyCode::BracketRight) => {
                            let mut st = self.shared.lock().unwrap();
                            st.max_kappa = (st.max_kappa * 1.25).min(20.0);
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::Resized(size) => {
                self.width = size.width;
                self.height = size.height;
            }

            WindowEvent::CursorMoved { position, .. } => {
                let mx = position.x;
                let my = position.y;

                if self.is_panning {
                    let dx = mx - self.pan_start_mouse.0;
                    let dy = my - self.pan_start_mouse.1;
                    self.camera.pan_x = self.pan_start_world.0 - dx / self.camera.zoom;
                    self.camera.pan_y = self.pan_start_world.1 - dy / self.camera.zoom;
                }

                if let Some(target) = self.drag_target {
                    let w = self.width as f32;
                    let h = self.height as f32;
                    let (wx, wy) = self.camera.screen_to_world(mx, my, w, h);
                    let mut st = self.shared.lock().unwrap();
                    match target {
                        DragTarget::StartPos => {
                            st.start.x = wx;
                            st.start.y = wy;
                            st.generation += 1;
                        }
                        DragTarget::StartDir => {
                            let dx = wx - st.start.x;
                            let dy = wy - st.start.y;
                            st.start.angle = dy.atan2(dx);
                            st.generation += 1;
                        }
                        DragTarget::EndPos => {
                            st.end.x = wx;
                            st.end.y = wy;
                            st.generation += 1;
                        }
                        DragTarget::EndDir => {
                            let dx = wx - st.end.x;
                            let dy = wy - st.end.y;
                            st.end.angle = dy.atan2(dx);
                            st.generation += 1;
                        }
                    }
                }

                self.last_mouse = (mx, my);
            }

            WindowEvent::MouseInput { state, button, .. } => match (button, state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    self.hit_test_drag();
                }
                (MouseButton::Left, ElementState::Released) => {
                    self.drag_target = None;
                }
                (MouseButton::Middle, ElementState::Pressed) => {
                    self.is_panning = true;
                    self.pan_start_mouse = self.last_mouse;
                    self.pan_start_world = (self.camera.pan_x, self.camera.pan_y);
                }
                (MouseButton::Middle, ElementState::Released) => {
                    self.is_panning = false;
                }
                _ => {}
            },

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 50.0,
                };
                let factor =
                    if scroll_y > 0.0 { 1.1f64.powf(scroll_y) } else { 1.0 / 1.1f64.powf(-scroll_y) };

                let (mx, my) = self.last_mouse;
                let w = self.width as f32;
                let h = self.height as f32;
                let (wx, wy) = self.camera.screen_to_world(mx, my, w, h);
                self.camera.zoom *= factor;
                let (nsx, nsy) = self.camera.world_to_screen(wx, wy, w, h);
                self.camera.pan_x += (nsx as f64 - mx) / self.camera.zoom;
                self.camera.pan_y += (nsy as f64 - my) / self.camera.zoom;
            }

            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    let shared = Arc::new(Mutex::new(SharedState::new()));

    let shared_opt = shared.clone();
    std::thread::spawn(move || {
        run_optimizer(shared_opt);
    });

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(shared);
    event_loop.run_app(&mut app).unwrap();
}
