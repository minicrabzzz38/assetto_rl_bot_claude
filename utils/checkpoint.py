"""
Checkpoint manager — wraps model save/load with metadata tracking.

Metadata (JSON sidecar file) stores:
  timesteps, episode, best_reward, lap_times, saved_at
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints in a given directory.

    Usage:
        cm = CheckpointManager("models/run_01")
        cm.save(agent, timesteps=10000, meta={"best_reward": 42.3})
        agent = cm.load_best(agent_cls, env)
    """

    BEST_TAG    = "best"
    LATEST_TAG  = "latest"
    META_SUFFIX = "_meta.json"

    def __init__(self, checkpoint_dir: str = "models"):
        self._dir = checkpoint_dir
        os.makedirs(self._dir, exist_ok=True)

    def _meta_path(self, tag: str) -> str:
        return os.path.join(self._dir, f"{tag}{self.META_SUFFIX}")

    def _model_path(self, tag: str) -> str:
        return os.path.join(self._dir, tag)

    def save(
        self,
        agent,
        timesteps: int,
        tag: str = LATEST_TAG,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = self._model_path(tag)
        agent.save(path)
        metadata = {
            "timesteps": timesteps,
            "saved_at":  datetime.now().isoformat(),
            **(meta or {}),
        }
        with open(self._meta_path(tag), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Checkpoint saved: %s (step %d)", path, timesteps)

    def save_best(self, agent, timesteps: int, reward: float) -> None:
        meta_path = self._meta_path(self.BEST_TAG)
        prev_best = -float("inf")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    prev_meta = json.load(f)
                prev_best = prev_meta.get("best_reward", -float("inf"))
            except Exception:
                pass

        if reward > prev_best:
            self.save(agent, timesteps, tag=self.BEST_TAG, meta={"best_reward": reward})
            logger.info("New best model! reward=%.3f (was %.3f)", reward, prev_best)

    def load_meta(self, tag: str = LATEST_TAG) -> Optional[Dict[str, Any]]:
        path = self._meta_path(tag)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def exists(self, tag: str = LATEST_TAG) -> bool:
        # SB3 saves with .zip extension
        return os.path.isfile(self._model_path(tag) + ".zip")

    def latest_path(self) -> Optional[str]:
        p = self._model_path(self.LATEST_TAG)
        return p if os.path.isfile(p + ".zip") else None

    def best_path(self) -> Optional[str]:
        p = self._model_path(self.BEST_TAG)
        return p if os.path.isfile(p + ".zip") else None
