"""
Episode management: detects termination conditions and triggers game resets.

Termination triggers:
  - Heavy crash       : total damage > damage_threshold
  - Off-track too long: tyres_out >= 3 for > off_track_timeout seconds
  - Stuck             : speed < stuck_speed for > stuck_timeout seconds
  - Going backwards   : normalizedCarPosition drops by > backwards_threshold
  - Invalid lap done  : completed a lap but isValidLap == 0
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ac_bridge.control_interface import GameResetHelper
from ac_bridge.data_types import SPageFilePhysics, SPageFileGraphics

logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    damage_threshold:     float = 80.0   # sum of carDamage[0..4]
    off_track_timeout:    float = 4.0    # seconds with >= 3 tyres out
    stuck_speed_kmh:      float = 2.0    # below this = potentially stuck
    stuck_timeout:        float = 8.0    # seconds below stuck_speed
    backwards_threshold:  float = 0.05  # lap progress drop = going backwards
    reset_esc_wait:       float = 1.5    # seconds after ESC before confirm
    reset_confirm_key:    str   = "r"    # key to press for restart in menu
    reset_confirm_wait:   float = 4.0    # seconds to wait after confirm


@dataclass
class EpisodeState:
    """Mutable state tracked across steps within one episode."""
    off_track_since:    Optional[float] = None
    stuck_since:        Optional[float] = None
    last_lap_progress:  float = 0.0
    laps_completed:     int = 0
    episode_start_time: float = field(default_factory=time.time)
    total_reward:       float = 0.0
    step_count:         int = 0
    terminal_reason:    str = ""


class EpisodeManager:
    """
    Evaluates whether the current episode should end and triggers a reset.

    Call `check()` every environment step; it returns (done, info_dict).
    Call `reset()` to perform the actual game restart.
    """

    def __init__(self, cfg: EpisodeConfig = EpisodeConfig()):
        self._cfg = cfg
        self._state = EpisodeState()
        self._resetter = GameResetHelper(
            esc_wait=cfg.reset_esc_wait,
            confirm_key=cfg.reset_confirm_key,
            confirm_wait=cfg.reset_confirm_wait,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Call at the start of each new episode."""
        self._state = EpisodeState()
        logger.debug("EpisodeManager: state reset")

    def check(
        self,
        phys: Optional[SPageFilePhysics],
        gfx:  Optional[SPageFileGraphics],
    ) -> tuple[bool, dict]:
        """
        Returns (done: bool, info: dict).
        Updates internal timers every call.
        """
        if phys is None or gfx is None:
            return False, {}

        s = self._state
        s.step_count += 1
        now = time.time()

        info: dict = {}

        # 1. Heavy crash
        total_damage = sum(phys.carDamage[i] for i in range(5))
        if total_damage > self._cfg.damage_threshold:
            s.terminal_reason = f"crash (damage={total_damage:.1f})"
            info["terminal"] = s.terminal_reason
            return True, info

        # 2. Off-track too long
        tyres_out = phys.numberOfTyresOut
        if tyres_out >= 3:
            if s.off_track_since is None:
                s.off_track_since = now
            elif now - s.off_track_since > self._cfg.off_track_timeout:
                s.terminal_reason = f"off_track (tyres_out={tyres_out} for {now-s.off_track_since:.1f}s)"
                info["terminal"] = s.terminal_reason
                return True, info
        else:
            s.off_track_since = None

        # 3. Stuck
        if phys.speedKmh < self._cfg.stuck_speed_kmh:
            if s.stuck_since is None:
                s.stuck_since = now
            elif now - s.stuck_since > self._cfg.stuck_timeout:
                s.terminal_reason = f"stuck (speed={phys.speedKmh:.1f} km/h)"
                info["terminal"] = s.terminal_reason
                return True, info
        else:
            s.stuck_since = None

        # 4. Going backwards (lap progress dropped significantly)
        pos = gfx.normalizedCarPosition
        progress_delta = pos - s.last_lap_progress
        # Handle lap wrap-around: near 0→1 is a lap completion, not backwards
        if progress_delta < -self._cfg.backwards_threshold and pos > 0.1:
            s.terminal_reason = f"backwards (pos dropped {progress_delta:.3f})"
            info["terminal"] = s.terminal_reason
            return True, info

        # 5. Lap completion with invalid lap
        laps = gfx.completedLaps
        if laps > s.laps_completed:
            s.laps_completed = laps
            if not gfx.isValidLap:
                s.terminal_reason = "invalid_lap_completed"
                info["terminal"] = s.terminal_reason
                info["lap_completed"] = True
                info["valid_lap"] = False
                return True, info
            else:
                info["lap_completed"] = True
                info["valid_lap"] = True
                info["lap_time_ms"] = gfx.iLastTime
                logger.info(
                    "Valid lap completed! Time: %s  (%.3f s)",
                    gfx.lastTime, gfx.iLastTime / 1000.0,
                )

        # Track progress (handle wrap-around)
        if pos < 0.1 and s.last_lap_progress > 0.9:
            pass  # lap wrap-around — don't update backwards detection
        else:
            s.last_lap_progress = pos

        info["tyres_out"] = tyres_out
        info["damage"] = total_damage
        info["speed_kmh"] = phys.speedKmh
        info["lap_progress"] = pos
        return False, info

    def trigger_reset(self, control_interface=None) -> None:
        """
        Zero controls then send the keyboard reset sequence to AC.
        Blocks until the session has likely reloaded.
        """
        if control_interface is not None:
            control_interface.reset_inputs()
        logger.info("Episode ended: %s — resetting...", self._state.terminal_reason)
        self._resetter.reset()

    @property
    def state(self) -> EpisodeState:
        return self._state
