"""
SAC agent wrapper around stable-baselines3.

Why SAC over PPO?
  - SAC is off-policy → reuses past experience via replay buffer → more sample-efficient
  - Native continuous action support (no need to discretize steering/throttle/brake)
  - Entropy regularisation encourages exploration without manual tuning
  - PPO is available as a drop-in swap via the `algorithm` param if needed

This module is a thin wrapper so the training script stays clean.
"""

import logging
import os
from typing import Optional, Type

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

ALGORITHMS = {"sac": SAC, "ppo": PPO}


class SACAgent:
    """
    Wraps a stable-baselines3 SAC (or PPO) model with helpers
    for training, saving, loading, and prediction.
    """

    def __init__(
        self,
        env,
        algorithm:       str   = "sac",
        learning_rate:   float = 3e-4,
        buffer_size:     int   = 300_000,
        batch_size:      int   = 256,
        tau:             float = 0.005,
        gamma:           float = 0.99,
        ent_coef:        str   = "auto",
        policy:          str   = "MlpPolicy",
        tensorboard_log: Optional[str] = None,
        device:          str   = "auto",
        verbose:         int   = 1,
    ):
        algo_cls = ALGORITHMS.get(algorithm.lower())
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm: {algorithm!r}. Use 'sac' or 'ppo'.")

        self._algorithm = algorithm.lower()
        self._env = env

        kwargs = dict(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
        )

        if self._algorithm == "sac":
            kwargs.update(
                buffer_size=buffer_size,
                batch_size=batch_size,
                tau=tau,
                ent_coef=ent_coef,
                learning_starts=1000,
            )

        self._model = algo_cls(**kwargs)
        logger.info("Initialized %s agent | device=%s", algorithm.upper(), self._model.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        checkpoint_dir:  str = "models",
        checkpoint_freq: int = 10_000,
        log_interval:    int = 1,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{self._algorithm}_ac_bot",
            verbose=1,
        )

        logger.info("Starting training for %d timesteps", total_timesteps)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            log_interval=log_interval,
            reset_num_timesteps=False,
        )
        logger.info("Training complete")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        action, state = self._model.predict(observation, deterministic=deterministic)
        return action, state

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._model.save(path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, env, algorithm: str = "sac", device: str = "auto") -> "SACAgent":
        algo_cls = ALGORITHMS.get(algorithm.lower())
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")
        agent = cls.__new__(cls)
        agent._algorithm = algorithm.lower()
        agent._env = env
        agent._model = algo_cls.load(path, env=env, device=device)
        logger.info("Model loaded from %s", path)
        return agent

    def set_env(self, env) -> None:
        self._model.set_env(env)

    @property
    def num_timesteps(self) -> int:
        return self._model.num_timesteps

    @property
    def model(self):
        return self._model
