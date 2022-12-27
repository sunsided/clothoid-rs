use clothoid::Clothoid;
use piston_window::{
    clear, Ellipse, EventLoop, OpenGL, PistonWindow, ResizeEvent, Transformed, WindowSettings,
};

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn main() {
    let mut window: PistonWindow = WindowSettings::new("Clothoids", [800, 800])
        .exit_on_esc(true)
        .vsync(true)
        .graphics_api(OpenGL::V4_5)
        .build()
        .unwrap();

    window.set_ups(30);
    window.set_swap_buffers(true);

    let ellipse = Ellipse::new(BLACK).resolution(16).border(
        Ellipse::new_border(BLACK, 1.0)
            .resolution(16)
            .border
            .unwrap(),
    );

    let mut window_size = [0.0; 2];

    while let Some(e) = window.next() {
        if let Some(args) = e.resize_args() {
            window_size = args.window_size;
        }

        window.draw_2d(&e, |c, g, _| {
            clear([1.0; 4], g);

            let half_window_size = [window_size[0] * 0.5, window_size[1] * 0.5];

            {
                let c = c.trans(half_window_size[0], half_window_size[1]);

                let clothoid = Clothoid::new(200.);

                for i in 0..100 {
                    let pt = clothoid.calculate((i as f64) * 0.05);

                    let rect = [-0.1 + pt.x, -0.1 + pt.y, 0.2, 0.2];
                    ellipse
                        .color(BLACK)
                        .draw(rect, &c.draw_state, c.transform, g);
                }
            }
        });
    }
}
