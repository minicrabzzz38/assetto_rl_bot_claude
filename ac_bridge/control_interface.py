"""
Control interface — sends steering / throttle / brake to Assetto Corsa.

Primary:  VGamepadControl  — virtual Xbox 360 via ViGEm (install ViGEm Bus Driver
                             then: pip install vgamepad)
Fallback: KeyboardControl  — maps analog values to key presses (coarse but zero deps)

AC must be configured to accept the virtual controller as its input device.
In AC options → Controls → select the ViGEm Xbox controller and map axes.
"""

import logging
import time
from abc import ABC, abstractmethod

import pyautogui

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False   # prevent corner-of-screen abort during training
pyautogui.PAUSE = 0.0        # no inter-call pause


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ControlInterface(ABC):
    """Abstract base — all control backends implement this interface."""

    @abstractmethod
    def apply_action(self, steering: float, throttle: float, brake: float) -> None:
        """
        Apply driving inputs.
          steering : float in [-1.0, 1.0]   left = -1, right = +1
          throttle : float in [ 0.0, 1.0]
          brake    : float in [ 0.0, 1.0]
        """

    @abstractmethod
    def reset_inputs(self) -> None:
        """Zero all inputs (neutral steering, no gas, no brake)."""

    def close(self) -> None:
        """Release any resources."""
        self.reset_inputs()


# ---------------------------------------------------------------------------
# vgamepad — virtual Xbox 360 controller via ViGEm Bus Driver
# ---------------------------------------------------------------------------

class VGamepadControl(ControlInterface):
    """
    Uses the `vgamepad` library to create a virtual Xbox 360 controller.

    Prerequisites:
      1. Install ViGEm Bus Driver: https://github.com/ViGEm/ViGEmBus/releases
      2. pip install vgamepad
      3. In AC: Options → Controls → select the virtual controller, map:
             - Left stick X  → Steering
             - Right trigger → Gas
             - Left trigger  → Brake
    """

    def __init__(self):
        try:
            import vgamepad as vg
            self._gamepad = vg.VX360Gamepad()
            self._gamepad.reset()
            self._gamepad.update()
            logger.info("VGamepadControl: virtual Xbox 360 controller ready")
        except ImportError:
            raise RuntimeError(
                "vgamepad not installed. Run: pip install vgamepad\n"
                "Also install ViGEm Bus Driver from https://github.com/ViGEm/ViGEmBus/releases"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create virtual gamepad: {exc}\n"
                "Make sure ViGEm Bus Driver is installed."
            ) from exc

    def apply_action(self, steering: float, throttle: float, brake: float) -> None:
        # Clamp inputs
        steering = max(-1.0, min(1.0, float(steering)))
        throttle = max(0.0,  min(1.0, float(throttle)))
        brake    = max(0.0,  min(1.0, float(brake)))

        # Xbox 360 mapping:
        #   Left stick X  → steering  (range -1..1)
        #   Right trigger → throttle  (range  0..1)
        #   Left trigger  → brake     (range  0..1)
        self._gamepad.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        self._gamepad.right_trigger_float(value_float=throttle)
        self._gamepad.left_trigger_float(value_float=brake)
        self._gamepad.update()

    def reset_inputs(self) -> None:
        self._gamepad.reset()
        self._gamepad.update()

    def close(self) -> None:
        self.reset_inputs()
        logger.debug("VGamepadControl closed")


# ---------------------------------------------------------------------------
# Keyboard fallback — analog → key presses (coarse control, for testing only)
# ---------------------------------------------------------------------------

class KeyboardControl(ControlInterface):
    """
    Maps continuous action values to keyboard key presses.
    This is COARSE — analog precision is not possible with keys.
    Use only for quick smoke-testing when vgamepad is unavailable.

    Default mapping (configurable via constructor kwargs):
      steer left  → left arrow
      steer right → right arrow
      throttle    → up arrow
      brake       → down arrow
    """

    THRESHOLD = 0.15  # dead-zone: ignore inputs below this magnitude

    def __init__(
        self,
        key_left:  str = "left",
        key_right: str = "right",
        key_accel: str = "up",
        key_brake: str = "down",
    ):
        import pydirectinput
        self._di = pydirectinput
        self._di.FAILSAFE = False
        self._key_left  = key_left
        self._key_right = key_right
        self._key_accel = key_accel
        self._key_brake = key_brake
        self._held: set = set()
        logger.warning(
            "KeyboardControl is a coarse fallback. "
            "Install vgamepad for proper analog control."
        )

    def _press(self, key: str):
        if key not in self._held:
            self._di.keyDown(key)
            self._held.add(key)

    def _release(self, key: str):
        if key in self._held:
            self._di.keyUp(key)
            self._held.discard(key)

    def apply_action(self, steering: float, throttle: float, brake: float) -> None:
        steering = float(steering)
        throttle = float(throttle)
        brake    = float(brake)

        # Steering
        if steering < -self.THRESHOLD:
            self._press(self._key_left)
            self._release(self._key_right)
        elif steering > self.THRESHOLD:
            self._press(self._key_right)
            self._release(self._key_left)
        else:
            self._release(self._key_left)
            self._release(self._key_right)

        # Throttle
        if throttle > self.THRESHOLD:
            self._press(self._key_accel)
        else:
            self._release(self._key_accel)

        # Brake
        if brake > self.THRESHOLD:
            self._press(self._key_brake)
        else:
            self._release(self._key_brake)

    def reset_inputs(self) -> None:
        for key in list(self._held):
            self._release(key)

    def close(self) -> None:
        self.reset_inputs()
        logger.debug("KeyboardControl closed")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_control_interface(backend: str = "vgamepad", **kwargs) -> ControlInterface:
    """
    Factory function.  backend: 'vgamepad' | 'keyboard'
    Falls back to keyboard if vgamepad fails to initialize.
    """
    if backend == "vgamepad":
        try:
            return VGamepadControl(**kwargs)
        except RuntimeError as exc:
            logger.warning("VGamepadControl failed (%s) — falling back to keyboard", exc)
            return KeyboardControl(**kwargs)
    elif backend == "keyboard":
        return KeyboardControl(**kwargs)
    else:
        raise ValueError(f"Unknown control backend: {backend!r}. Use 'vgamepad' or 'keyboard'.")


# ---------------------------------------------------------------------------
# Game reset helper (shared by both backends)
# ---------------------------------------------------------------------------

class GameResetHelper:
    """
    Sends keyboard inputs to restart the AC session.

    In practice mode, AC's pause menu has a Restart button.
    The sequence:  ESC → wait → navigate to Restart → Enter
    This sequence is configurable because menu layouts vary.
    """

    def __init__(self, esc_wait: float = 1.5, confirm_key: str = "r", confirm_wait: float = 3.0):
        self._esc_wait     = esc_wait
        self._confirm_key  = confirm_key
        self._confirm_wait = confirm_wait

    def reset(self) -> None:
        """Press ESC to open AC pause menu then press the configured restart key."""
        logger.info("Triggering AC session reset via keyboard (ESC → %s)", self._confirm_key)
        pyautogui.press("escape")
        time.sleep(self._esc_wait)
        pyautogui.press(self._confirm_key)
        time.sleep(self._confirm_wait)
        logger.info("Reset command sent — waiting for session to reload")
