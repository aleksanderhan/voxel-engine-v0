// src/app/input.rs
// ----------------
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent, MouseButton},
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
            _ => {}
        }
    }
}

#[derive(Default)]
pub struct InputState {
    pub keys: KeyState,

    /// "Captured mouse" mode (cursor hidden + grabbed) used for FPS-style look.
    pub focused: bool,

    pub mouse_dx: f32,
    pub mouse_dy: f32,

    c_pressed: bool,
    lmb_down: bool,
    lmb_pressed: bool,
}

impl InputState {
    #[inline]
    fn capture_mouse(&mut self, window: &Window) {
        self.focused = true;

        let _ = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        window.set_cursor_visible(false);

        // avoid a big jump on capture
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
    }

    #[inline]
    fn release_mouse(&mut self, window: &Window) {
        self.focused = false;

        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);

        self.lmb_down = false;
        self.lmb_pressed = false;

        // optional: clear deltas
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
    }

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
            // OS focus change: if we gain focus, capture; if we lose it, release.
            WindowEvent::Focused(f) => {
                if *f {
                    self.capture_mouse(window);
                } else {
                    self.release_mouse(window);
                }
                true
            }

            // Click-to-capture when not focused/captured.
            WindowEvent::MouseInput { state, button, .. } => {
                // If not in captured mode, allow left click to enter captured mode.
                if !self.focused {
                    if *button == MouseButton::Left && *state == ElementState::Pressed {
                        self.capture_mouse(window);

                        // Consume this click (don't treat it as a gameplay click).
                        self.lmb_down = false;
                        self.lmb_pressed = false;
                        return true;
                    }
                    return false;
                }

                // Normal mouse handling while captured.
                if *button == MouseButton::Left {
                    let down = *state == ElementState::Pressed;
                    if down && !self.lmb_down {
                        self.lmb_pressed = true; // edge-trigger
                    }
                    self.lmb_down = down;
                }
                false
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

                    self.keys.set(*code, down);

                    // Escape releases capture
                    if down && *code == KeyCode::Escape {
                        self.release_mouse(window);
                        return true;
                    }
                }
                false
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

    pub fn take_lmb_pressed(&mut self) -> bool {
        let v = self.lmb_pressed;
        self.lmb_pressed = false;
        v
    }
}
