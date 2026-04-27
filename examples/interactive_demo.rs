#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use clothoid::fit::{FitConfig, FitState, RenderSegment};
use clothoid::optimizer::{Pose, SegmentKind};
use softbuffer::{Context, Surface};
use tiny_skia::{
    Color, FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Rect, Stroke, StrokeDash,
    Transform,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

enum OptimizerMode {
    NelderMead,
    CmaEs,
}

// ============================================================================
// Shared state
// ============================================================================

struct SharedState {
    start: Pose,
    end: Pose,
    fit_state: FitState,
    config: FitConfig,
    paused: bool,
    optimizer_mode: OptimizerMode,
}

impl SharedState {
    fn new() -> Self {
        Self {
            start: Pose {
                x: -3.0,
                y: 0.0,
                angle: 0.0,
            },
            end: Pose {
                x: 3.0,
                y: 0.0,
                angle: 0.0,
            },
            fit_state: FitState::new(),
            config: FitConfig {
                max_segments: 2,
                max_kappa: 2.0,
                tol_pos: 0.05,
                tol_angle: 0.05,
            },
            paused: false,
            optimizer_mode: OptimizerMode::NelderMead,
        }
    }

    fn optimizer_name(&self) -> &'static str {
        match self.optimizer_mode {
            OptimizerMode::NelderMead => "Nelder-Mead",
            OptimizerMode::CmaEs => "CMA-ES",
        }
    }

    fn toggle_optimizer(&mut self) {
        self.optimizer_mode = match self.optimizer_mode {
            OptimizerMode::NelderMead => OptimizerMode::CmaEs,
            OptimizerMode::CmaEs => OptimizerMode::NelderMead,
        };
        self.fit_state = match self.optimizer_mode {
            OptimizerMode::NelderMead => FitState::new(),
            OptimizerMode::CmaEs => FitState::cma_es(),
        };
    }
}

// ============================================================================
// Optimizer thread
// ============================================================================

fn run_optimizer(shared: Arc<Mutex<SharedState>>) {
    loop {
        let (start, end, config, paused) = {
            let st = shared.lock().unwrap();
            (
                st.start.clone(),
                st.end.clone(),
                st.config.clone(),
                st.paused,
            )
        };

        if paused {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        {
            let mut st = shared.lock().unwrap();
            st.fit_state.step(&start, &end, &config);
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
        Self {
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 80.0,
        }
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
    if let Some(rect) = Rect::from_xywh(x, y, w, h) {
        let mut paint = Paint::default();
        paint.set_color(color);
        pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    }
}

fn draw_line(pixmap: &mut Pixmap, x0: f32, y0: f32, x1: f32, y1: f32, color: Color, width: f32) {
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
        pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
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
    let bl = (
        tx - head_len * fwd_x + head_w * lft_x,
        ty - head_len * fwd_y + head_w * lft_y,
    );
    let br = (
        tx - head_len * fwd_x - head_w * lft_x,
        ty - head_len * fwd_y - head_w * lft_y,
    );
    draw_filled_triangle(pixmap, tip, bl, br, color);
}

fn segment_color(base: Color, idx: usize) -> Color {
    let shift = (idx as f32 * 0.7).sin() * 0.1;

    let nr = (base.red() + shift).clamp(0.0, 1.0);
    let ng = (base.green() - shift * 0.6).clamp(0.0, 1.0);
    let nb = (base.blue() + shift * 0.4).clamp(0.0, 1.0);

    Color::from_rgba(nr, ng, nb, base.alpha()).unwrap_or(base)
}

fn draw_world_segments(
    pixmap: &mut Pixmap,
    segments: &[RenderSegment],
    color: Color,
    width: f32,
    camera: &Camera,
) {
    if segments.is_empty() {
        return;
    }
    let w = pixmap.width() as f32;
    let h = pixmap.height() as f32;

    let dash_intervals = vec![width * 2.0, width * 3.0];

    for (idx, seg) in segments.iter().enumerate() {
        if seg.points.len() < 2 {
            continue;
        }
        let seg_color = segment_color(color, idx);
        let mut pb = PathBuilder::new();
        let (sx, sy) = camera.world_to_screen(seg.points[0].0 as f64, seg.points[0].1 as f64, w, h);
        pb.move_to(sx, sy);
        for &(wx, wy) in &seg.points[1..] {
            let (sx, sy) = camera.world_to_screen(wx as f64, wy as f64, w, h);
            pb.line_to(sx, sy);
        }
        if let Some(path) = pb.finish() {
            let mut paint = Paint::default();
            paint.set_color(seg_color);
            paint.anti_alias = true;
            let mut stroke = Stroke::default();
            stroke.width = width;
            stroke.line_cap = LineCap::Round;
            stroke.line_join = LineJoin::Round;
            if seg.kind == SegmentKind::Linear {
                stroke.dash = StrokeDash::new(dash_intervals.clone(), 0.0);
            }
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }

    let tick_width = 1.0f32;
    let tick_half_len = 4.0f32;
    for i in 0..segments.len() - 1 {
        let seg = &segments[i];
        if seg.points.is_empty() {
            continue;
        }
        let tick_c = segment_color(color, i);
        let (lx, ly) = seg.points.last().unwrap();
        let theta = seg.boundary_theta;
        let (sx, sy) = camera.world_to_screen(*lx as f64, *ly as f64, w, h);
        let perp_x = -theta.sin();
        let perp_y = theta.cos();
        let x0 = sx - tick_half_len * perp_x;
        let y0 = sy - tick_half_len * perp_y;
        let x1 = sx + tick_half_len * perp_x;
        let y1 = sy + tick_half_len * perp_y;
        draw_line(pixmap, x0, y0, x1, y1, tick_c, tick_width);
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

fn draw_hud(
    pixmap: &mut Pixmap,
    max_segments: usize,
    max_kappa: f64,
    best_fit_exists: bool,
    paused: bool,
    optimizer_name: &str,
) {
    let w = pixmap.width() as f32;
    let h = pixmap.height() as f32;

    let sq = 12.0f32;
    let gap = 3.0f32;
    let sy = h - 35.0;
    for i in 0..8usize {
        let color = if i < max_segments {
            Color::from_rgba8(100, 200, 100, 255)
        } else {
            Color::from_rgba8(40, 40, 40, 255)
        };
        let sx = 10.0 + i as f32 * (sq + gap);
        fill_rect_solid(pixmap, sx, sy, sq, sq, color);
    }

    let bar_x = 10.0f32;
    let bar_y = h - 55.0;
    let bar_w = 120.0f32;
    let bar_h = 8.0f32;
    fill_rect_solid(
        pixmap,
        bar_x,
        bar_y,
        bar_w,
        bar_h,
        Color::from_rgba8(40, 40, 40, 255),
    );
    let frac = ((max_kappa - 0.05) / (20.0 - 0.05)).clamp(0.0, 1.0) as f32;
    fill_rect_solid(
        pixmap,
        bar_x,
        bar_y,
        bar_w * frac,
        bar_h,
        Color::from_rgba8(200, 150, 50, 255),
    );

    let fit_color = if best_fit_exists {
        Color::from_rgba8(50, 200, 50, 255)
    } else {
        Color::from_rgba8(50, 50, 50, 255)
    };
    fill_rect_solid(pixmap, w - 35.0, 10.0, 25.0, 25.0, fit_color);

    if paused {
        fill_rect_solid(
            pixmap,
            w - 35.0,
            45.0,
            25.0,
            25.0,
            Color::from_rgba8(200, 50, 50, 255),
        );
    }

    let opt_color = if optimizer_name == "CMA-ES" {
        Color::from_rgba8(180, 100, 220, 255)
    } else {
        Color::from_rgba8(100, 180, 220, 255)
    };
    fill_rect_solid(pixmap, w - 35.0, 80.0, 25.0, 25.0, opt_color);
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

        if let Some(ref expl) = state.fit_state.exploration() {
            draw_world_segments(
                &mut pixmap,
                &expl.segments,
                Color::from_rgba8(50, 80, 180, 128),
                1.5,
                &self.camera,
            );
        }
        if let Some(ref best) = state.fit_state.best_fit() {
            draw_world_segments(
                &mut pixmap,
                &best.segments,
                Color::from_rgba8(50, 230, 200, 255),
                3.0,
                &self.camera,
            );
        }

        let start = state.start.clone();
        let end = state.end.clone();
        draw_hud(
            &mut pixmap,
            state.config.max_segments,
            state.config.max_kappa,
            state.fit_state.best_fit().is_some(),
            state.paused,
            state.optimizer_name(),
        );
        drop(state);

        draw_gizmo(
            &mut pixmap,
            &start,
            Color::from_rgba8(50, 220, 50, 255),
            &self.camera,
        );
        draw_gizmo(
            &mut pixmap,
            &end,
            Color::from_rgba8(220, 50, 50, 255),
            &self.camera,
        );

        if let Some(surface) = &mut self.surface {
            if surface
                .resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap())
                .is_err()
            {
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                        PhysicalKey::Code(KeyCode::KeyO) => {
                            let mut st = self.shared.lock().unwrap();
                            st.toggle_optimizer();
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.camera = Camera::new();
                        }
                        PhysicalKey::Code(KeyCode::Equal) => {
                            let mut st = self.shared.lock().unwrap();
                            if st.config.max_segments < 8 {
                                st.config.max_segments += 1;
                            }
                        }
                        PhysicalKey::Code(KeyCode::Minus) => {
                            let mut st = self.shared.lock().unwrap();
                            if st.config.max_segments > 1 {
                                st.config.max_segments -= 1;
                            }
                        }
                        PhysicalKey::Code(KeyCode::BracketLeft) => {
                            let mut st = self.shared.lock().unwrap();
                            st.config.max_kappa = (st.config.max_kappa * 0.8).max(0.05);
                        }
                        PhysicalKey::Code(KeyCode::BracketRight) => {
                            let mut st = self.shared.lock().unwrap();
                            st.config.max_kappa = (st.config.max_kappa * 1.25).min(20.0);
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
                            st.fit_state.bump_generation();
                        }
                        DragTarget::StartDir => {
                            let dx = wx - st.start.x;
                            let dy = wy - st.start.y;
                            st.start.angle = dy.atan2(dx);
                            st.fit_state.bump_generation();
                        }
                        DragTarget::EndPos => {
                            st.end.x = wx;
                            st.end.y = wy;
                            st.fit_state.bump_generation();
                        }
                        DragTarget::EndDir => {
                            let dx = wx - st.end.x;
                            let dy = wy - st.end.y;
                            st.end.angle = dy.atan2(dx);
                            st.fit_state.bump_generation();
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
                let factor = if scroll_y > 0.0 {
                    1.1f64.powf(scroll_y)
                } else {
                    1.0 / 1.1f64.powf(-scroll_y)
                };

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
