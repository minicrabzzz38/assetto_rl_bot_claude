"""
Evaluation script — loads a trained model and runs deterministic laps.

Records per-lap and per-episode stats to a CSV file.

Usage:
    python scripts/evaluate.py --model models/best
    python scripts/evaluate.py --model models/best --episodes 20
    python scripts/evaluate.py --model models/latest --no-deterministic
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger
from utils.config_loader import load_config, get
from env.ac_env import ACEnv
from env.episode_manager import EpisodeConfig
from rewards.reward_function import RewardConfig
from agents.sac_agent import SACAgent

logger = get_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained AC RL agent")
    parser.add_argument("--model",         required=True,             help="Path to saved model (without .zip)")
    parser.add_argument("--config", nargs="+",
                        default=["configs/default.yaml", "configs/sac_config.yaml", "configs/env_config.yaml"])
    parser.add_argument("--episodes",      type=int,  default=10,     help="Number of evaluation episodes")
    parser.add_argument("--no-deterministic", action="store_true",    help="Use stochastic policy")
    parser.add_argument("--output",        type=str,  default=None,   help="Output CSV path (auto if omitted)")
    parser.add_argument("--algorithm",     type=str,  default="sac",  help="Algorithm used for training")
    return parser.parse_args()


RESULT_FIELDS = [
    "episode", "steps", "total_reward", "laps_completed",
    "valid_laps", "invalid_laps", "best_lap_ms", "best_lap_s",
    "off_track_events", "terminal_reason",
]


def main():
    args = parse_args()

    try:
        cfg = load_config(*args.config)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    deterministic = not args.no_deterministic
    logger.info("Evaluating model: %s | episodes=%d | deterministic=%s",
                args.model, args.episodes, deterministic)

    # Build env
    ep_cfg = EpisodeConfig(
        reset_esc_wait=  get(cfg, "reset", "esc_wait",       default=1.5),
        reset_confirm_key=get(cfg, "reset", "confirm_key",   default="r"),
        reset_confirm_wait=get(cfg, "reset", "confirm_wait", default=4.0),
    )
    env = ACEnv(
        control_backend=get(cfg, "env", "control_backend", default="vgamepad"),
        step_delay=     get(cfg, "env", "step_delay",      default=0.05),
        connect_timeout=get(cfg, "env", "connect_timeout", default=60.0),
        episode_cfg=ep_cfg,
    )

    # Load model
    try:
        agent = SACAgent.load(args.model, env=env, algorithm=args.algorithm)
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", args.model, exc)
        env.close()
        sys.exit(1)

    # Prepare output CSV
    logs_dir = get(cfg, "paths", "logs_dir", default="logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or os.path.join(logs_dir, f"eval_{ts}.csv")

    results = []

    logger.info("Starting evaluation...")
    logger.info("Make sure Assetto Corsa is running in Practice mode.")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        laps = 0
        valid_laps = 0
        invalid_laps = 0
        best_lap_ms = 0
        off_track_events = 0
        terminal_reason = ""
        prev_tyres_out = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            steps += 1

            # Track off-track events
            tyres_out = info.get("tyres_out", 0)
            if tyres_out >= 2 and prev_tyres_out < 2:
                off_track_events += 1
            prev_tyres_out = tyres_out

            # Track laps
            if info.get("lap_completed"):
                laps += 1
                if info.get("valid_lap"):
                    valid_laps += 1
                    lap_ms = info.get("lap_time_ms", 0)
                    if best_lap_ms == 0 or lap_ms < best_lap_ms:
                        best_lap_ms = lap_ms
                else:
                    invalid_laps += 1

            if done:
                terminal_reason = info.get("terminal", "")

        best_lap_s = best_lap_ms / 1000.0 if best_lap_ms > 0 else 0.0
        row = {
            "episode":         ep,
            "steps":           steps,
            "total_reward":    round(ep_reward, 3),
            "laps_completed":  laps,
            "valid_laps":      valid_laps,
            "invalid_laps":    invalid_laps,
            "best_lap_ms":     best_lap_ms,
            "best_lap_s":      round(best_lap_s, 3),
            "off_track_events": off_track_events,
            "terminal_reason": terminal_reason,
        }
        results.append(row)

        logger.info(
            "Ep %2d/%d | steps=%4d | reward=%8.2f | laps=%d (valid=%d) | best=%.3fs | off_track=%d | end=%s",
            ep, args.episodes, steps, ep_reward, laps, valid_laps,
            best_lap_s, off_track_events, terminal_reason or "ok",
        )

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_reward  = sum(r["total_reward"]   for r in results) / len(results)
    avg_valid   = sum(r["valid_laps"]     for r in results) / len(results)
    best_lap    = min((r["best_lap_ms"]   for r in results if r["best_lap_ms"] > 0), default=0)
    total_valid = sum(r["valid_laps"]     for r in results)
    total_inv   = sum(r["invalid_laps"]   for r in results)

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY (%d episodes)", args.episodes)
    logger.info("  Avg reward      : %.3f", avg_reward)
    logger.info("  Avg valid laps  : %.2f", avg_valid)
    logger.info("  Total valid laps: %d", total_valid)
    logger.info("  Total invalid   : %d", total_inv)
    logger.info("  Best lap time   : %.3f s", best_lap / 1000.0 if best_lap else 0)
    logger.info("  Results written : %s", out_path)
    logger.info("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
