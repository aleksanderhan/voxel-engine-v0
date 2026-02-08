// src/input.rs
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent, MouseScrollDelta, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

#[derive(Default, Clone, Copy)]
pub struct KeyState {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub space: bool,
    pub alt: bool,
    pub shift: bool,
    pub c: bool,
    pub p: bool,
}

impl KeyState {
    pub fn set(&mut self, code: KeyCode, down: bool) {
        match code {
            KeyCode::KeyW => self.w = down,
            KeyCode::KeyA => self.a = down,
            KeyCode::KeyS => self.s = down,
            KeyCode::KeyD => self.d = down,
            KeyCode::Space => self.space = down,
            KeyCode::AltLeft | KeyCode::AltRight => self.alt = down,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.shift = down,
            KeyCode::KeyC => self.c = down,
            KeyCode::KeyP => self.p = down,
            _ => {}
        }
    }
}

#[derive(Default)]
pub struct InputState {
    pub keys: KeyState,
    pub focused: bool,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub c_pressed: bool,
    pub p_pressed: bool,
    pub wheel_steps: i32,
    pub lmb_pressed: bool,
    pub lmb_down: bool,
}

impl InputState {
    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if !self.focused {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_dx += delta.0 as f32;
            self.mouse_dy += delta.1 as f32;
        }
    }

    /// Returns true if event is fully handled/consumed.
    pub fn on_window_event(&mut self, event: &WindowEvent, window: &Window) -> bool {
        match event {
            WindowEvent::Focused(f) => {
                self.focused = *f;
                if self.focused {
                    let _ = window
                        .set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                    window.set_cursor_visible(false);
                } else {
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
                true
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state,
                    ..
                } = event
                {
                    let down = *state == ElementState::Pressed;

                    if down && *code == KeyCode::KeyC && !self.keys.c {
                        self.c_pressed = true;
                    }
                    if down && *code == KeyCode::KeyP && !self.keys.p {
                        self.p_pressed = true;
                    }

                    self.keys.set(*code, down);

                    if down && *code == KeyCode::Escape {
                        self.focused = false;
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        return true;
                    }
                }
                false
            }

            WindowEvent::MouseInput { button, state, .. } => {
                if *button == MouseButton::Left {
                    let down = *state == ElementState::Pressed;
                    if down && !self.lmb_down {
                        self.lmb_pressed = true; // rising edge
                    }
                    self.lmb_down = down;
                    return true;
                }
                false
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // convert to "notches"
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => (*y).round() as i32,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 120.0).round() as i32, // common Windows scale
                };
                self.wheel_steps += steps;
                true
            }

            _ => false,
        }
    }

    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        let dx = self.mouse_dx;
        let dy = self.mouse_dy;
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        (dx, dy)
    }

    pub fn take_c_pressed(&mut self) -> bool {
        let v = self.c_pressed;
        self.c_pressed = false;
        v
    }

    pub fn take_p_pressed(&mut self) -> bool {
        let v = self.p_pressed;
        self.p_pressed = false;
        v
    }

    pub fn take_wheel_steps(&mut self) -> i32 {
        let v = self.wheel_steps;
        self.wheel_steps = 0;
        v
    }

    pub fn take_lmb_pressed(&mut self) -> bool {
        let v = self.lmb_pressed;
        self.lmb_pressed = false;
        v
    }
}
