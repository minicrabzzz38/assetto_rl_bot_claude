"""
Training script — launches the full SAC training loop against Assetto Corsa.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml configs/sac_config.yaml configs/env_config.yaml
    python scripts/train.py --resume        # resume from latest checkpoint
    python scripts/train.py --timesteps 500000
"""

import argparse
import os
import sys
import signal

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger, EpisodeLogger
from utils.config_loader import load_config, get
from utils.checkpoint import CheckpointManager
from env.ac_env import ACEnv
from env.episode_manager import EpisodeConfig
from rewards.reward_function import RewardConfig
from agents.sac_agent import SACAgent

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AC autonomous driving agent")
    parser.add_argument(
        "--config", nargs="+",
        default=["configs/default.yaml", "configs/sac_config.yaml", "configs/env_config.yaml"],
        help="YAML config files (merged left-to-right)",
    )
    parser.add_argument("--resume",     action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--timesteps",  type=int,   default=None, help="Override total_timesteps")
    parser.add_argument("--device",     type=str,   default=None, help="Override device (cpu/cuda/auto)")
    parser.add_argument("--run-name",   type=str,   default=None, help="Override run name")
    return parser.parse_args()


def build_env(cfg: dict) -> ACEnv:
    ep_cfg = EpisodeConfig(
        damage_threshold=   get(cfg, "episode", "damage_threshold",   default=80.0),
        off_track_timeout=  get(cfg, "episode", "off_track_timeout",  default=4.0),
        stuck_speed_kmh=    get(cfg, "episode", "stuck_speed_kmh",    default=2.0),
        stuck_timeout=      get(cfg, "episode", "stuck_timeout",      default=8.0),
        backwards_threshold=get(cfg, "episode", "backwards_threshold",default=0.05),
        reset_esc_wait=     get(cfg, "reset",   "esc_wait",           default=1.5),
        reset_confirm_key=  get(cfg, "reset",   "confirm_key",        default="r"),
        reset_confirm_wait= get(cfg, "reset",   "confirm_wait",       default=4.0),
    )
    rw_cfg = RewardConfig(
        speed_weight=          get(cfg, "reward", "speed_weight",          default=0.003),
        progress_weight=       get(cfg, "reward", "progress_weight",       default=5.0),
        valid_lap_bonus=       get(cfg, "reward", "valid_lap_bonus",       default=200.0),
        smoothness_weight=     get(cfg, "reward", "smoothness_weight",     default=0.05),
        tyre_out_weight=       get(cfg, "reward", "tyre_out_weight",       default=2.0),
        damage_weight=         get(cfg, "reward", "damage_weight",         default=0.3),
        yaw_rate_weight=       get(cfg, "reward", "yaw_rate_weight",       default=0.08),
        stuck_penalty=         get(cfg, "reward", "stuck_penalty",         default=0.5),
        invalid_lap_penalty=   get(cfg, "reward", "invalid_lap_penalty",   default=5.0),
        terminal_crash_penalty=get(cfg, "reward", "terminal_crash_penalty",default=50.0),
        terminal_stuck_penalty=get(cfg, "reward", "terminal_stuck_penalty",default=20.0),
        backwards_penalty=     get(cfg, "reward", "backwards_penalty",     default=1.5),
    )
    return ACEnv(
        control_backend=get(cfg, "env", "control_backend", default="vgamepad"),
        step_delay=     get(cfg, "env", "step_delay",      default=0.05),
        connect_timeout=get(cfg, "env", "connect_timeout", default=60.0),
        episode_cfg=ep_cfg,
        reward_cfg=rw_cfg,
    )


def main():
    args = parse_args()

    # --- Load config ---
    try:
        cfg = load_config(*args.config)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    # CLI overrides
    total_timesteps = args.timesteps or get(cfg, "training", "total_timesteps", default=1_000_000)
    device          = args.device    or get(cfg, "training", "device",           default="auto")
    run_name        = args.run_name  or get(cfg, "training", "run_name",          default="run_01")

    models_dir  = get(cfg, "paths", "models_dir",  default="models")
    logs_dir    = get(cfg, "paths", "logs_dir",    default="logs")
    tb_log      = get(cfg, "training", "tensorboard_log", default=None)
    ckpt_freq   = get(cfg, "training", "checkpoint_freq", default=10_000)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ep_logger = EpisodeLogger(log_dir=logs_dir, run_name=run_name)
    ckpt_mgr  = CheckpointManager(checkpoint_dir=models_dir)

    # --- Build environment ---
    logger.info("Building AC environment...")
    env = build_env(cfg)

    # --- Build or load agent ---
    sac_cfg = get(cfg, "sac") or {}

    if args.resume and ckpt_mgr.exists("latest"):
        logger.info("Resuming from latest checkpoint")
        agent = SACAgent.load(
            ckpt_mgr.latest_path(),
            env=env,
            algorithm=get(cfg, "training", "algorithm", default="sac"),
            device=device,
        )
        meta = ckpt_mgr.load_meta("latest") or {}
        steps_done = meta.get("timesteps", 0)
        logger.info("Resuming from step %d", steps_done)
    else:
        logger.info("Starting fresh training run: %s", run_name)
        agent = SACAgent(
            env=env,
            algorithm=    get(cfg, "training", "algorithm",   default="sac"),
            learning_rate=sac_cfg.get("learning_rate", 3e-4),
            buffer_size=  sac_cfg.get("buffer_size",   300_000),
            batch_size=   sac_cfg.get("batch_size",    256),
            tau=          sac_cfg.get("tau",            0.005),
            gamma=        sac_cfg.get("gamma",          0.99),
            ent_coef=     sac_cfg.get("ent_coef",       "auto"),
            policy=       sac_cfg.get("policy",         "MlpPolicy"),
            tensorboard_log=tb_log,
            device=device,
            verbose=1,
        )

    # --- Graceful shutdown handler ---
    _shutdown = {"flag": False}
    def _handle_signal(sig, frame):
        logger.warning("Interrupt received — saving checkpoint and shutting down...")
        _shutdown["flag"] = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Training loop ---
    logger.info("Starting training | total_timesteps=%d | device=%s", total_timesteps, device)
    logger.info(
        "Make sure Assetto Corsa is:\n"
        "  → Running in Practice mode\n"
        "  → No opponents\n"
        "  → Automatic gearbox enabled\n"
        "  → Virtual controller configured (if using vgamepad)\n"
    )

    try:
        agent.train(
            total_timesteps=total_timesteps,
            checkpoint_dir=models_dir,
            checkpoint_freq=ckpt_freq,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("Training error: %s", exc, exc_info=True)
    finally:
        logger.info("Saving final checkpoint...")
        ckpt_mgr.save(agent, timesteps=agent.num_timesteps, tag="latest")
        env.close()
        logger.info("Done. Model saved to %s/latest.zip", models_dir)


if __name__ == "__main__":
    main()
