"""
Gymnasium-compatible environment wrapping Assetto Corsa.

Observation space : Box(-1, 1, shape=(30,), float32)
Action space      : Box([-1,0,0], [1,1,1], shape=(3,), float32)
                    [steering, throttle, brake]

The environment:
  1. Connects to AC shared memory on reset()
  2. Reads telemetry each step()
  3. Sends actions via the configured control interface
  4. Computes rewards and detects episode termination
  5. Triggers game restart when an episode ends
"""

import logging
import time
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ac_bridge.shared_memory import ACSharedMemoryReader
from ac_bridge.control_interface import ControlInterface, make_control_interface, GameResetHelper
from ac_bridge.data_types import ACStatus
from env.observation import ObservationBuilder, OBS_DIM
from env.episode_manager import EpisodeManager, EpisodeConfig
from rewards.reward_function import RewardFunction, RewardConfig

logger = logging.getLogger(__name__)


class ACEnv(gym.Env):
    """
    Assetto Corsa reinforcement learning environment.

    Parameters
    ----------
    control_backend : str
        'vgamepad' (default) or 'keyboard'
    step_delay : float
        Seconds to wait between steps (controls effective Hz).
        Default 0.05 → ~20 Hz.
    connect_timeout : float
        Seconds to wait for AC to appear on startup.
    episode_cfg : EpisodeConfig
        Tunable episode termination parameters.
    reward_cfg : RewardConfig
        Tunable reward weights.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_backend:  str = "vgamepad",
        step_delay:       float = 0.05,
        connect_timeout:  float = 60.0,
        episode_cfg:      Optional[EpisodeConfig] = None,
        reward_cfg:       Optional[RewardConfig] = None,
    ):
        super().__init__()

        self._step_delay      = step_delay
        self._connect_timeout = connect_timeout

        # Spaces
        self.observation_space = ObservationBuilder.observation_space()
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Components
        self._reader    = ACSharedMemoryReader()
        self._control   = make_control_interface(control_backend)
        self._obs_builder = ObservationBuilder()
        self._ep_manager  = EpisodeManager(episode_cfg or EpisodeConfig())
        self._reward_fn   = RewardFunction(reward_cfg or RewardConfig())

        # Episode tracking
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._static_cache = None  # SPageFileStatic (read once)
        self._connected = False
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._episode_count += 1
        logger.info("=== Episode %d — resetting ===", self._episode_count)

        # Connect to AC shared memory (only first time or after disconnect)
        if not self._connected:
            ok = self._reader.connect(timeout=self._connect_timeout)
            if not ok:
                raise RuntimeError(
                    "Cannot connect to Assetto Corsa shared memory. "
                    "Make sure AC is running in practice mode."
                )
            self._connected = True

        # Read static once per session (car/track metadata)
        if self._static_cache is None:
            self._static_cache = self._reader.read_static()

        # Trigger game restart (episodes 2+)
        if self._episode_count > 1:
            self._ep_manager.trigger_reset(self._control)
            self._wait_for_session_ready()

        # Reset internal states
        self._ep_manager.reset_state()
        self._reward_fn.reset()
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._obs_builder.update_prev_action(self._prev_action)

        # Initial observation
        phys, gfx, _ = self._reader.read_all()
        obs = self._obs_builder.build(phys, gfx, self._static_cache)
        return obs, {}

    def step(self, action: np.ndarray):
        # Clamp action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        steering, throttle, brake = float(action[0]), float(action[1]), float(action[2])

        # Send control to AC
        self._control.apply_action(steering, throttle, brake)

        # Wait one step
        time.sleep(self._step_delay)

        # Read telemetry
        phys, gfx, _ = self._reader.read_all()

        # Build observation
        self._obs_builder.update_prev_action(action)
        obs = self._obs_builder.build(phys, gfx, self._static_cache)

        # Check episode termination
        done, info = self._ep_manager.check(phys, gfx)

        # Compute reward
        reward = self._reward_fn.compute(
            phys, gfx,
            self._prev_action, action,
            done, info.get("terminal", ""),
        )

        # Update episode state
        self._ep_manager.state.total_reward += reward
        self._prev_action = action.copy()

        if done:
            s = self._ep_manager.state
            logger.info(
                "Episode %d done | reason=%s | steps=%d | reward=%.2f | laps=%d",
                self._episode_count,
                s.terminal_reason,
                s.step_count,
                s.total_reward,
                s.laps_completed,
            )

        truncated = False
        return obs, reward, done, truncated, info

    def render(self):
        pass  # No visual rendering — we use the game window itself

    def close(self):
        self._control.close()
        self._reader.disconnect()
        self._connected = False
        logger.info("ACEnv closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_session_ready(self, timeout: float = 30.0, poll: float = 0.25) -> bool:
        """Wait until AC is back in LIVE state and speed is near 0 (at start line)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            phys, gfx, _ = self._reader.read_all()
            if (
                gfx is not None
                and phys is not None
                and gfx.status == ACStatus.LIVE
                and phys.speedKmh < 5.0
                and gfx.normalizedCarPosition < 0.1
            ):
                logger.debug("Session ready (speed=%.1f, pos=%.3f)", phys.speedKmh, gfx.normalizedCarPosition)
                return True
            time.sleep(poll)
        logger.warning("Timed out waiting for session ready — proceeding anyway")
        return False
