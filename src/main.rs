
mod app;
mod clipmap;
mod render;
mod streaming;
mod svo;
mod world;
mod physics;

use std::sync::Arc;
use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Fullscreen, WindowBuilder},
};

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("SVO MVP")
            .with_inner_size(PhysicalSize::new(1280, 720))
            .build(&event_loop)
            .unwrap(),
    );
    window.set_fullscreen(Some(Fullscreen::Borderless(None)));

    pollster::block_on(app::run(event_loop, window));
}
