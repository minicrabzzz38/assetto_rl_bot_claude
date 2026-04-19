"""
Reward function for the AC autonomous driving agent.

Design principles:
  - Reward forward progress above all else
  - Big bonus for completing a clean, valid lap
  - Heavy instant penalty for off-track wheels
  - Penalty for damage (wall hits)
  - Penalty for spinning / erratic behaviour
  - Penalty for being stuck
  - Smoothness bonus to discourage twitchy inputs
  - Penalty at episode termination for non-lap-completion endings

All weights are defined as class-level constants and can be overridden
via the RewardConfig dataclass for easy hyperparameter tuning.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ac_bridge.data_types import SPageFilePhysics, SPageFileGraphics


@dataclass
class RewardConfig:
    # Positive signals
    speed_weight:          float = 0.003   # per km/h per step
    progress_weight:       float = 5.0     # per unit of normalizedCarPosition gained
    valid_lap_bonus:       float = 200.0   # completing a valid lap
    smoothness_weight:     float = 0.05    # reward for stable steering

    # Negative signals
    tyre_out_weight:       float = 2.0     # per tyre off track per step
    damage_weight:         float = 0.3     # per total damage point per step
    yaw_rate_weight:       float = 0.08    # per rad/s of yaw rate
    stuck_penalty:         float = 0.5     # per step when speed < stuck_threshold
    invalid_lap_penalty:   float = 5.0     # per step when isValidLap == 0
    terminal_crash_penalty: float = 50.0   # one-time at crash termination
    terminal_stuck_penalty: float = 20.0   # one-time at stuck termination
    backwards_penalty:     float = 1.5     # per step when going backwards

    # Thresholds
    stuck_speed_kmh:       float = 2.0
    progress_backwards_threshold: float = -0.005  # below this = backwards


class RewardFunction:
    """
    Computes the per-step reward from raw telemetry snapshots.

    Usage:
        rf = RewardFunction(cfg)
        reward = rf.compute(phys, gfx, prev_action, action, done, done_reason)
    """

    def __init__(self, cfg: Optional[RewardConfig] = None):
        self._cfg = cfg or RewardConfig()
        self._prev_lap_progress: float = 0.0
        self._prev_completed_laps: int = 0
        self._prev_damage_total: float = 0.0

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._prev_lap_progress = 0.0
        self._prev_completed_laps = 0
        self._prev_damage_total = 0.0

    def compute(
        self,
        phys:        Optional[SPageFilePhysics],
        gfx:         Optional[SPageFileGraphics],
        prev_action: np.ndarray,
        action:      np.ndarray,
        done:        bool,
        done_reason: str = "",
    ) -> float:
        if phys is None or gfx is None:
            return 0.0

        cfg = self._cfg
        reward = 0.0

        # ----------------------------------------------------------------
        # 1. Speed reward — encourage going fast
        # ----------------------------------------------------------------
        reward += cfg.speed_weight * phys.speedKmh

        # ----------------------------------------------------------------
        # 2. Lap progress reward — encourage moving forward
        # ----------------------------------------------------------------
        pos = gfx.normalizedCarPosition
        laps = gfx.completedLaps

        # Compute actual progress delta, handling lap wrap-around
        if laps > self._prev_completed_laps:
            # Lap just completed — progress went 0.9+ → 0.0
            progress_delta = (1.0 - self._prev_lap_progress) + pos
        else:
            progress_delta = pos - self._prev_lap_progress

        if progress_delta > cfg.progress_backwards_threshold:
            reward += cfg.progress_weight * max(0.0, progress_delta)
        else:
            # Going backwards
            reward -= cfg.backwards_penalty

        self._prev_lap_progress  = pos
        self._prev_completed_laps = laps

        # ----------------------------------------------------------------
        # 3. Valid lap completion bonus
        # ----------------------------------------------------------------
        if laps > self._prev_completed_laps - (1 if laps > self._prev_completed_laps else 0):
            pass  # handled above — but we check gfx.isValidLap for bonus
        # Check separately: if last time just updated and lap is valid
        if laps > 0 and gfx.iLastTime > 0 and gfx.isValidLap:
            # Only award once per lap
            if laps > getattr(self, "_bonus_lap_awarded", -1):
                reward += cfg.valid_lap_bonus
                self._bonus_lap_awarded = laps

        # ----------------------------------------------------------------
        # 4. Off-track penalty — per tyre over the limit
        # ----------------------------------------------------------------
        tyres_out = phys.numberOfTyresOut
        reward -= cfg.tyre_out_weight * tyres_out

        # ----------------------------------------------------------------
        # 5. Damage penalty (incremental — punish new damage each step)
        # ----------------------------------------------------------------
        total_damage = sum(phys.carDamage[i] for i in range(5))
        damage_delta = max(0.0, total_damage - self._prev_damage_total)
        reward -= cfg.damage_weight * damage_delta
        self._prev_damage_total = total_damage

        # ----------------------------------------------------------------
        # 6. Yaw rate penalty — prevent spinning
        # ----------------------------------------------------------------
        yaw_rate = abs(phys.localAngularVel[1])
        reward -= cfg.yaw_rate_weight * yaw_rate

        # ----------------------------------------------------------------
        # 7. Stuck penalty
        # ----------------------------------------------------------------
        if phys.speedKmh < cfg.stuck_speed_kmh:
            reward -= cfg.stuck_penalty

        # ----------------------------------------------------------------
        # 8. Invalid lap penalty (per step while flag is 0)
        # ----------------------------------------------------------------
        if not gfx.isValidLap and laps > 0:
            reward -= cfg.invalid_lap_penalty

        # ----------------------------------------------------------------
        # 9. Smoothness bonus — penalise large steering changes
        # ----------------------------------------------------------------
        if prev_action is not None and len(prev_action) >= 1:
            steering_change = abs(float(action[0]) - float(prev_action[0]))
            reward += cfg.smoothness_weight * (1.0 - steering_change)

        # ----------------------------------------------------------------
        # 10. Terminal penalties (one-shot at episode end)
        # ----------------------------------------------------------------
        if done:
            if "crash" in done_reason:
                reward -= cfg.terminal_crash_penalty
            elif "stuck" in done_reason:
                reward -= cfg.terminal_stuck_penalty

        return float(reward)
