"""
Builds the normalized 30-dimensional observation vector from raw AC telemetry.

Index map:
  [0]  speed_kmh         / 300.0
  [1]  local_vel_x       / 100.0
  [2]  local_vel_y       / 100.0
  [3]  local_vel_z       / 100.0
  [4]  heading           / pi
  [5]  normalized_pos    (lap progress 0..1)
  [6]  steer_angle       (already -1..1 in AC)
  [7]  throttle          (0..1)
  [8]  brake             (0..1)
  [9]  yaw_rate          / 5.0
  [10] acc_x             / 5.0
  [11] acc_y             / 5.0
  [12] acc_z             / 5.0
  [13] gear              / 8.0
  [14] rpms              / max_rpm (default 8000)
  [15] wheel_slip_fl     / 10.0
  [16] wheel_slip_fr     / 10.0
  [17] wheel_slip_rl     / 10.0
  [18] wheel_slip_rr     / 10.0
  [19] tyres_out         / 4.0
  [20] damage_front      / 100.0
  [21] damage_rear       / 100.0
  [22] damage_left       / 100.0
  [23] damage_right      / 100.0
  [24] prev_steering
  [25] prev_throttle
  [26] prev_brake
  [27] is_valid_lap      (0 or 1)
  [28] pitch             / pi
  [29] roll              / pi
"""

import math
from typing import Optional

import numpy as np

from ac_bridge.data_types import SPageFilePhysics, SPageFileGraphics, SPageFileStatic

OBS_DIM = 30


class ObservationBuilder:
    """
    Converts raw AC shared memory snapshots into a normalized numpy array.
    Clamps all values to [-1, 1] after normalization.
    """

    def __init__(self, max_speed_kmh: float = 300.0, max_rpm: float = 8000.0):
        self._max_speed = max_speed_kmh
        self._max_rpm   = max_rpm
        self._prev_action = np.zeros(3, dtype=np.float32)  # steering, throttle, brake

    def update_prev_action(self, action: np.ndarray) -> None:
        self._prev_action = np.array(action, dtype=np.float32)

    def build(
        self,
        phys:   Optional[SPageFilePhysics],
        gfx:    Optional[SPageFileGraphics],
        static: Optional[SPageFileStatic],
    ) -> np.ndarray:
        """Return a float32 array of shape (OBS_DIM,). Returns zeros if telemetry is None."""
        if phys is None or gfx is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        # Prefer max_rpm from static if available
        max_rpm = float(static.maxRpm) if (static and static.maxRpm > 0) else self._max_rpm

        obs = np.array([
            # --- kinematics ---
            phys.speedKmh                / self._max_speed,
            phys.localVelocity[0]        / 100.0,
            phys.localVelocity[1]        / 100.0,
            phys.localVelocity[2]        / 100.0,
            phys.heading                 / math.pi,
            gfx.normalizedCarPosition,                       # already 0..1
            # --- driver inputs ---
            phys.steerAngle,                                 # AC: -1..1
            phys.gas,                                        # 0..1
            phys.brake,                                      # 0..1
            # --- angular & linear accel ---
            phys.localAngularVel[1]      / 5.0,              # yaw rate
            phys.accG[0]                 / 5.0,
            phys.accG[1]                 / 5.0,
            phys.accG[2]                 / 5.0,
            # --- powertrain ---
            float(phys.gear)             / 8.0,
            float(phys.rpms)             / max_rpm,
            # --- tyre slip ---
            phys.wheelSlip[0]            / 10.0,
            phys.wheelSlip[1]            / 10.0,
            phys.wheelSlip[2]            / 10.0,
            phys.wheelSlip[3]            / 10.0,
            # --- track limits ---
            float(phys.numberOfTyresOut) / 4.0,
            # --- damage ---
            phys.carDamage[0]            / 100.0,  # front
            phys.carDamage[1]            / 100.0,  # rear
            phys.carDamage[2]            / 100.0,  # left
            phys.carDamage[3]            / 100.0,  # right
            # --- previous action ---
            float(self._prev_action[0]),
            float(self._prev_action[1]),
            float(self._prev_action[2]),
            # --- lap validity ---
            float(gfx.isValidLap),
            # --- orientation ---
            phys.pitch                   / math.pi,
            phys.roll                    / math.pi,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    @staticmethod
    def observation_space():
        """Returns a gymnasium.spaces.Box for this observation space."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
