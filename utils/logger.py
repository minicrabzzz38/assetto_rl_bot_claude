"""
Logging utilities.

get_logger()   — returns a colorized console logger
EpisodeLogger  — writes per-episode stats to a CSV file
"""

import csv
import logging
import os
import sys
from datetime import datetime
from typing import Optional

try:
    import colorlog
    _COLORLOG = True
except ImportError:
    _COLORLOG = False


def get_logger(name: str = "ac_bot", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with color formatting if colorlog is installed."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if _COLORLOG:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(cyan)s%(name)s%(reset)s — %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "white",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class EpisodeLogger:
    """
    Appends per-episode rows to a CSV file for offline analysis.

    Columns:
        episode, timestamp, steps, total_reward, laps_completed,
        best_lap_ms, terminal_reason, mean_speed_kmh
    """

    FIELDNAMES = [
        "episode",
        "timestamp",
        "steps",
        "total_reward",
        "laps_completed",
        "best_lap_ms",
        "terminal_reason",
    ]

    def __init__(self, log_dir: str = "logs", run_name: Optional[str] = None):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{run_name or 'training'}_{ts}.csv"
        self._path = os.path.join(log_dir, fname)
        self._episode = 0

        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

        logging.getLogger("ac_bot").info("EpisodeLogger writing to %s", self._path)

    def log(
        self,
        steps:          int,
        total_reward:   float,
        laps_completed: int,
        best_lap_ms:    int = 0,
        terminal_reason: str = "",
    ) -> None:
        self._episode += 1
        row = {
            "episode":        self._episode,
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "steps":          steps,
            "total_reward":   f"{total_reward:.3f}",
            "laps_completed": laps_completed,
            "best_lap_ms":    best_lap_ms,
            "terminal_reason": terminal_reason,
        }
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)
