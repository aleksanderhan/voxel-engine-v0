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
}
