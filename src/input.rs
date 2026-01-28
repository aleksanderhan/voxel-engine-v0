// src/input.rs
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
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

    pub lmb_down: bool,
    pub rmb_down: bool,
    pub lmb_pressed: bool, // edge
    pub rmb_pressed: bool, // edge
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

            WindowEvent::MouseInput { state, button, .. } => {
                let down = *state == ElementState::Pressed;

                // click-to-capture
                if down && !self.focused {
                    self.focused = true;
                    let _ = window
                        .set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                    window.set_cursor_visible(false);
                    return true; // consume this click (so first click just captures)
                }

                match button {
                    winit::event::MouseButton::Left => {
                        self.lmb_pressed = down && !self.lmb_down;
                        self.lmb_down = down;
                    }
                    winit::event::MouseButton::Right => {
                        self.rmb_pressed = down && !self.rmb_down;
                        self.rmb_down = down;
                    }
                    _ => {}
                }
                false
            }


            _ => false,
        }
    }

    pub fn take_mouse_clicks(&mut self) -> (bool, bool) {
        let l = self.lmb_pressed;
        let r = self.rmb_pressed;
        self.lmb_pressed = false;
        self.rmb_pressed = false;
        (l, r)
    }

    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        let dx = self.mouse_dx;
        let dy = self.mouse_dy;
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        (dx, dy)
    }
}
