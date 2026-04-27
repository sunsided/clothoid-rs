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
// Bitmap font (5x7 uppercase)
// ============================================================================

mod font {
    /// Glyph data for printable chars 0x20–0x7E.
    /// Each glyph is 5 bytes, one per column, low 7 bits = rows (bit 0 = top).
    const GLYPHS: [[u8; 5]; 95] = [
        [0x00, 0x00, 0x00, 0x00, 0x00], //   0x20
        [0x00, 0x00, 0x5f, 0x00, 0x00], // ! 0x21
        [0x00, 0x07, 0x00, 0x07, 0x00], // " 0x22
        [0x14, 0x7f, 0x14, 0x7f, 0x14], // # 0x23
        [0x24, 0x2a, 0x7f, 0x2a, 0x12], // $ 0x24
        [0x23, 0x13, 0x08, 0x64, 0x62], // % 0x25
        [0x36, 0x49, 0x55, 0x22, 0x50], // & 0x26
        [0x00, 0x05, 0x03, 0x00, 0x00], // ' 0x27
        [0x00, 0x1c, 0x22, 0x41, 0x00], // ( 0x28
        [0x00, 0x41, 0x22, 0x1c, 0x00], // ) 0x29
        [0x14, 0x08, 0x3e, 0x08, 0x14], // * 0x2a
        [0x08, 0x08, 0x3e, 0x08, 0x08], // + 0x2b
        [0x00, 0x50, 0x30, 0x00, 0x00], // , 0x2c
        [0x08, 0x08, 0x08, 0x08, 0x08], // - 0x2d
        [0x00, 0x60, 0x60, 0x00, 0x00], // . 0x2e
        [0x20, 0x10, 0x08, 0x04, 0x02], // / 0x2f
        [0x3e, 0x51, 0x49, 0x45, 0x3e], // 0 0x30
        [0x00, 0x42, 0x7f, 0x40, 0x00], // 1 0x31
        [0x42, 0x61, 0x51, 0x49, 0x46], // 2 0x32
        [0x21, 0x41, 0x45, 0x4b, 0x31], // 3 0x33
        [0x18, 0x14, 0x12, 0x7f, 0x10], // 4 0x34
        [0x27, 0x45, 0x45, 0x45, 0x39], // 5 0x35
        [0x3c, 0x4a, 0x49, 0x49, 0x30], // 6 0x36
        [0x01, 0x71, 0x09, 0x05, 0x03], // 7 0x37
        [0x36, 0x49, 0x49, 0x49, 0x36], // 8 0x38
        [0x06, 0x49, 0x49, 0x29, 0x1e], // 9 0x39
        [0x00, 0x36, 0x36, 0x00, 0x00], // : 0x3a
        [0x00, 0x56, 0x36, 0x00, 0x00], // ; 0x3b
        [0x08, 0x14, 0x22, 0x41, 0x00], // < 0x3c
        [0x14, 0x14, 0x14, 0x14, 0x14], // = 0x3d
        [0x00, 0x41, 0x22, 0x14, 0x08], // > 0x3e
        [0x02, 0x01, 0x51, 0x09, 0x06], // ? 0x3f
        [0x32, 0x49, 0x79, 0x41, 0x3e], // @ 0x40
        [0x7e, 0x11, 0x11, 0x11, 0x7e], // A 0x41
        [0x7f, 0x49, 0x49, 0x49, 0x36], // B 0x42
        [0x3e, 0x41, 0x41, 0x41, 0x22], // C 0x43
        [0x7f, 0x41, 0x41, 0x22, 0x1c], // D 0x44
        [0x7f, 0x49, 0x49, 0x49, 0x41], // E 0x45
        [0x7f, 0x09, 0x09, 0x09, 0x01], // F 0x46
        [0x3e, 0x41, 0x49, 0x49, 0x7a], // G 0x47
        [0x7f, 0x08, 0x08, 0x08, 0x7f], // H 0x48
        [0x00, 0x41, 0x7f, 0x41, 0x00], // I 0x49
        [0x20, 0x40, 0x41, 0x3f, 0x01], // J 0x4a
        [0x7f, 0x08, 0x14, 0x22, 0x41], // K 0x4b
        [0x7f, 0x40, 0x40, 0x40, 0x40], // L 0x4c
        [0x7f, 0x02, 0x0c, 0x02, 0x7f], // M 0x4d
        [0x7f, 0x04, 0x08, 0x10, 0x7f], // N 0x4e
        [0x3e, 0x41, 0x41, 0x41, 0x3e], // O 0x4f
        [0x7f, 0x09, 0x09, 0x09, 0x06], // P 0x50
        [0x3e, 0x41, 0x51, 0x21, 0x5e], // Q 0x51
        [0x7f, 0x09, 0x19, 0x29, 0x46], // R 0x52
        [0x46, 0x49, 0x49, 0x49, 0x31], // S 0x53
        [0x01, 0x01, 0x7f, 0x01, 0x01], // T 0x54
        [0x3f, 0x40, 0x40, 0x40, 0x3f], // U 0x55
        [0x1f, 0x20, 0x40, 0x20, 0x1f], // V 0x56
        [0x3f, 0x40, 0x38, 0x40, 0x3f], // W 0x57
        [0x63, 0x14, 0x08, 0x14, 0x63], // X 0x58
        [0x07, 0x08, 0x70, 0x08, 0x07], // Y 0x59
        [0x61, 0x51, 0x49, 0x45, 0x43], // Z 0x5a
        [0x00, 0x7f, 0x41, 0x41, 0x00], // [ 0x5b
        [0x02, 0x04, 0x08, 0x10, 0x20], // \ 0x5c
        [0x00, 0x41, 0x41, 0x7f, 0x00], // ] 0x5d
        [0x04, 0x02, 0x01, 0x02, 0x04], // ^ 0x5e
        [0x40, 0x40, 0x40, 0x40, 0x40], // _ 0x5f
        [0x00, 0x01, 0x02, 0x04, 0x00], // ` 0x60
        [0x20, 0x54, 0x54, 0x54, 0x78], // a 0x61
        [0x7f, 0x48, 0x44, 0x44, 0x38], // b 0x62
        [0x38, 0x44, 0x44, 0x44, 0x20], // c 0x63
        [0x38, 0x44, 0x44, 0x48, 0x7f], // d 0x64
        [0x38, 0x54, 0x54, 0x54, 0x18], // e 0x65
        [0x08, 0x7e, 0x09, 0x01, 0x02], // f 0x66
        [0x0c, 0x52, 0x52, 0x52, 0x3e], // g 0x67
        [0x7f, 0x08, 0x04, 0x04, 0x78], // h 0x68
        [0x00, 0x44, 0x7d, 0x40, 0x00], // i 0x69
        [0x20, 0x40, 0x44, 0x3d, 0x00], // j 0x6a
        [0x7f, 0x10, 0x28, 0x44, 0x00], // k 0x6b
        [0x00, 0x41, 0x7f, 0x40, 0x00], // l 0x6c
        [0x7c, 0x04, 0x18, 0x04, 0x78], // m 0x6d
        [0x7c, 0x08, 0x04, 0x04, 0x78], // n 0x6e
        [0x38, 0x44, 0x44, 0x44, 0x38], // o 0x6f
        [0x7c, 0x14, 0x14, 0x14, 0x08], // p 0x70
        [0x08, 0x14, 0x14, 0x18, 0x7c], // q 0x71
        [0x7c, 0x08, 0x04, 0x04, 0x08], // r 0x72
        [0x48, 0x54, 0x54, 0x54, 0x20], // s 0x73
        [0x04, 0x3f, 0x44, 0x40, 0x20], // t 0x74
        [0x3c, 0x40, 0x40, 0x20, 0x7c], // u 0x75
        [0x1c, 0x20, 0x40, 0x20, 0x1c], // v 0x76
        [0x3c, 0x40, 0x30, 0x40, 0x3c], // w 0x77
        [0x44, 0x28, 0x10, 0x28, 0x44], // x 0x78
        [0x0c, 0x50, 0x50, 0x50, 0x3c], // y 0x79
        [0x44, 0x64, 0x54, 0x4c, 0x44], // z 0x7a
        [0x00, 0x08, 0x36, 0x41, 0x00], // { 0x7b
        [0x00, 0x00, 0x7f, 0x00, 0x00], // | 0x7c
        [0x00, 0x41, 0x36, 0x08, 0x00], // } 0x7d
        [0x10, 0x08, 0x08, 0x10, 0x08], // ~ 0x7e
    ];

    pub fn glyph(c: char) -> &'static [u8; 5] {
        let idx = c as usize;
        if (0x20..0x7f).contains(&idx) {
            &GLYPHS[idx - 0x20]
        } else {
            &GLYPHS['?' as usize - 0x20]
        }
    }

    pub fn text_width(s: &str, scale: u32) -> u32 {
        s.len() as u32 * 6 * scale
    }
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

fn draw_text(pixmap: &mut Pixmap, x: f32, y: f32, text: &str, color: Color, scale: f32) {
    let s = scale;
    let mut cx = x;
    for ch in text.chars() {
        let glyph = font::glyph(ch.to_ascii_uppercase());
        for col in 0..5 {
            let bits = glyph[col];
            for row in 0..7 {
                if bits & (1 << row) != 0 {
                    fill_rect_solid(pixmap, cx + col as f32 * s, y + row as f32 * s, s, s, color);
                }
            }
        }
        cx += 6.0 * s;
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
    best_fit: Option<(f64, f64, f64)>,
    paused: bool,
    optimizer_name: &str,
    generation: u64,
    log_tail: &[String],
) {
    let w = pixmap.width() as f32;
    let h = pixmap.height() as f32;
    let scale = 2.0f32;
    let line_h = 7.0 * scale + 2.0;
    let grey = Color::from_rgba8(200, 200, 200, 255);
    let dim_grey = Color::from_rgba8(150, 150, 150, 255);
    let green = Color::from_rgba8(50, 200, 50, 255);
    let red = Color::from_rgba8(220, 50, 50, 255);

    // Bottom-left: controls/status stack
    let margin = 10.0f32;
    let bl_x = margin;
    let mut bl_y = h - margin - line_h;

    let running_text = if paused { "PAUSED" } else { "RUNNING" };
    let running_color = if paused { red } else { green };
    draw_text(
        pixmap,
        bl_x,
        bl_y,
        &format!("[SPACE] {}", running_text),
        running_color,
        scale,
    );
    bl_y -= line_h;

    draw_text(
        pixmap,
        bl_x,
        bl_y,
        &format!("[+/-]   SEGMENTS: {}/{}", max_segments, 8),
        grey,
        scale,
    );
    bl_y -= line_h;

    draw_text(
        pixmap,
        bl_x,
        bl_y,
        &format!("[[/]]   MAX KAPPA: {:.2}", max_kappa),
        grey,
        scale,
    );
    bl_y -= line_h;

    draw_text(
        pixmap,
        bl_x,
        bl_y,
        &format!("[O]     OPTIMIZER: {}", optimizer_name.to_uppercase()),
        grey,
        scale,
    );
    bl_y -= line_h;

    draw_text(pixmap, bl_x, bl_y, "[R]     RESET VIEW", grey, scale);

    // Top-right: generation and best fit
    let mut tr_y = margin;

    let gen_text = format!("GEN: {}", generation);
    let gen_w = font::text_width(&gen_text, scale as u32);
    let gen_x = w - margin - gen_w as f32;
    draw_text(pixmap, gen_x, tr_y, &gen_text, grey, scale);
    tr_y += line_h;

    let fit_text = match best_fit {
        Some((pos_err, ang_err, total_err)) => {
            format!(
                "BEST FIT: POS={:.3} ANG={:.3} ERR={:.4}",
                pos_err, ang_err, total_err
            )
        }
        None => "BEST FIT: -".to_string(),
    };
    let fit_w = font::text_width(&fit_text, scale as u32);
    let fit_x = w - margin - fit_w as f32;
    draw_text(pixmap, fit_x, tr_y, &fit_text, grey, scale);

    // Bottom-right: log tail (last 3 entries)
    let log_count = log_tail.len().min(3);
    for i in 0..log_count {
        let entry = &log_tail[log_tail.len() - log_count + i];
        let log_w = font::text_width(entry, scale as u32);
        let log_x = w - margin - log_w as f32;
        let log_y = h - margin - (log_count as f32 - i as f32) * line_h;
        draw_text(pixmap, log_x, log_y, entry, dim_grey, scale);
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
        let best_fit = state
            .fit_state
            .best_fit()
            .map(|b| (b.pos_error, b.angle_error, b.total_error));
        let generation = state.fit_state.generation();
        let log_tail: Vec<String> = state
            .fit_state
            .log()
            .iter()
            .rev()
            .take(3)
            .rev()
            .cloned()
            .collect();
        draw_hud(
            &mut pixmap,
            state.config.max_segments,
            state.config.max_kappa,
            best_fit,
            state.paused,
            state.optimizer_name(),
            generation,
            &log_tail,
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
