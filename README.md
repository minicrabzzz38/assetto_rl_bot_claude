# assetto_rl_bot_claude

Autonomous driving RL agent for **Assetto Corsa** — built with SAC (Soft Actor-Critic) and a direct telemetry bridge via Windows shared memory.

> **Claude build** — compare with the Codex build in `assetto_rl_bot_codex/`

---

## 1. Project Overview

The agent learns to drive clean, fast laps by:
- Reading live telemetry from AC (speed, damage, tyre state, lap progress, etc.)
- Sending continuous steering / throttle / brake commands via a virtual Xbox controller
- Receiving rewards for forward progress, lap completion, and smooth driving
- Being penalised hard for going off-track, hitting walls, spinning, or driving invalid laps

**No vision** is used. The full observation is a normalised 30-dimensional vector of physics telemetry.

---

## 2. Architecture

```
assetto_rl_bot_claude/
├── ac_bridge/
│   ├── data_types.py        ctypes AC shared memory structures (physics/graphics/static)
│   ├── shared_memory.py     Connects to AC via Windows named shared memory
│   └── control_interface.py Sends inputs via vgamepad (Xbox 360) or keyboard
├── env/
│   ├── ac_env.py            Gymnasium environment (step / reset / close)
│   ├── observation.py       Builds & normalises 30-dim observation vector
│   └── episode_manager.py   Detects termination, triggers game restart
├── rewards/
│   └── reward_function.py   Reward computation (speed + progress + penalties)
├── agents/
│   └── sac_agent.py         SB3 SAC wrapper (train / save / load / predict)
├── utils/
│   ├── logger.py            Coloured console + CSV episode logger
│   ├── config_loader.py     YAML deep-merge loader
│   └── checkpoint.py        Save/load with JSON metadata sidecar
├── configs/
│   ├── default.yaml         Master config
│   ├── sac_config.yaml      SAC hyperparameters
│   └── env_config.yaml      Environment & reward weights
├── scripts/
│   ├── train.py             Training entry point
│   ├── evaluate.py          Deterministic evaluation + CSV export
│   └── test_bridge.py       Smoke test for shared memory + controls
├── models/                  Saved checkpoints (.zip)
└── logs/                    CSV logs + TensorBoard events
```

**Algorithm: SAC** — chosen over PPO because:
- Off-policy replay buffer → far more sample-efficient than PPO
- Native continuous actions → no discretisation of steering/throttle/brake
- Entropy regularisation → automatic exploration without manual tuning

---

## 3. Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Assetto Corsa | Steam version with shared memory plugin (enabled by default) |
| Python 3.11+ | |
| ViGEm Bus Driver | Required for virtual gamepad — [download here](https://github.com/ViGEm/ViGEmBus/releases) |
| CUDA (optional) | Speeds up training significantly |

### Steps

```batch
cd C:\assetto_rl_bot_claude

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Install ViGEm Bus Driver **before** running (needed for `vgamepad`):
https://github.com/ViGEm/ViGEmBus/releases → download and run the installer.

---

## 4. How to Connect to Assetto Corsa

AC's shared memory is **enabled by default** — no plugin needed.

However, you must:

1. **Launch AC** from Steam
2. **Start a practice session** (Practice mode, any circuit, one car, no opponents)
3. **Enable auto-gearbox** in the car setup (In-game: Garage → Setup → Gearbox → Auto)
4. **Configure the virtual controller** (first time only):
   - AC Options → Controls → select the ViGEm Xbox 360 Controller
   - Map: Left stick X → Steering, Right Trigger → Gas, Left Trigger → Brake
   - Set steering range to match your car (usually 900° or 540°)
5. **Disable assists** that fight the agent: traction control, ABS, stability control

---

## 5. How to Start a Session

Before training, verify everything works:

```batch
cd C:\assetto_rl_bot_claude
venv\Scripts\activate

# With AC running in practice mode:
python scripts/test_bridge.py
```

Expected output:
```
PASS  Shared memory: speed/throttle/brake values printed every 0.25s
PASS  Control: AC car should twitch left/right as test inputs fire
```

---

## 6. How to Train

```batch
# Basic training (1M steps, SAC)
python scripts/train.py

# Resume interrupted training
python scripts/train.py --resume

# Custom timesteps
python scripts/train.py --timesteps 500000

# Use CPU explicitly
python scripts/train.py --device cpu

# Custom run name (for organised logging)
python scripts/train.py --run-name my_lap_attempt
```

**During training**, AC must remain in focus (or at least not minimised to tray — keep it in the background with the window visible).

**TensorBoard** (optional):
```batch
tensorboard --logdir logs/tb
# Open http://localhost:6006
```

Checkpoints are saved every 10,000 steps (configurable in `configs/default.yaml`) to `models/`.

---

## 7. How to Evaluate

```batch
# Evaluate the best saved model (10 episodes by default)
python scripts/evaluate.py --model models/best

# More episodes
python scripts/evaluate.py --model models/best --episodes 20

# Latest checkpoint
python scripts/evaluate.py --model models/latest

# Stochastic (non-deterministic) policy
python scripts/evaluate.py --model models/best --no-deterministic
```

Results are written to `logs/eval_YYYYMMDD_HHMMSS.csv` with columns:

| Column | Description |
|--------|-------------|
| episode | Episode number |
| steps | Steps taken |
| total_reward | Cumulative reward |
| valid_laps | Clean laps completed |
| invalid_laps | Laps with track cuts / off-track |
| best_lap_ms | Fastest valid lap in milliseconds |
| off_track_events | Times ≥2 wheels left track |
| terminal_reason | Why the episode ended |

---

## 8. How to Debug

### AC not connecting
```
FAILED: Cannot connect to AC shared memory
```
→ Make sure AC is running and you are **in an active session** (not the main menu).
→ Shared memory only activates once a session is loaded.

### vgamepad fails
```
RuntimeError: Failed to create virtual gamepad
```
→ ViGEm Bus Driver is not installed. Download from https://github.com/ViGEm/ViGEmBus/releases

### Car not moving
→ Check that the virtual Xbox controller is selected as the **primary input** in AC Controls.
→ Verify axes are mapped correctly (Left stick X = steering, triggers = gas/brake).
→ Try the keyboard fallback: edit `configs/default.yaml` → `control_backend: keyboard`

### Reset not working (car stays crashed)
→ The reset sends `ESC` then `R`. If your AC menu layout differs, edit `configs/default.yaml`:
```yaml
reset:
  confirm_key: "escape"    # try different keys
  esc_wait: 2.5            # increase if menu is slow to open
```
→ Alternative: configure a custom hotkey in AC for session restart.

### Training is very slow
→ Reduce `step_delay` in `configs/default.yaml` (e.g. `0.02` for ~50 Hz).
→ Use `device: cuda` if you have a GPU.
→ The agent's first 1,000 steps use random actions (learning_starts) — this is normal.

### Import errors
```
ModuleNotFoundError: No module named 'ac_bridge'
```
→ Run scripts from the project root: `cd C:\assetto_rl_bot_claude && python scripts/train.py`

---

## 9. Limitations

| Limitation | Impact |
|-----------|--------|
| Windows only | AC shared memory is a Windows API — cannot run on Linux/Mac |
| No vision | Agent cannot see the track visually; relies purely on telemetry |
| Reset is fragile | Keyboard macro for restart can fail if menu layout changes |
| Single track/car | Training is not transferable between different tracks/cars |
| No wet weather | Fixed conditions only (no rain, no dynamic grip changes) |
| No opponents | Multi-agent racing not supported in V1 |
| Telemetry latency | ~50ms step delay means the agent runs at ~20 Hz |
| No spline data | Track centre-line is not used directly (lateral offset is approximate) |

---

## 10. V2 Improvements

### Reliability
- [ ] LUA AC plugin for reliable session restart (no keyboard macro)
- [ ] UDP bridge for lower-latency telemetry (~5ms vs ~50ms)
- [ ] Automatic AC focus management

### Features
- [ ] Track centre-line data for accurate lateral offset measurement
- [ ] Multi-track generalisation (domain randomisation)
- [ ] Curriculum learning: start slow, increase difficulty
- [ ] Opponent awareness (multi-car sessions)
- [ ] Weather & grip variation

### ML
- [ ] Recurrent policy (LSTM/GRU) to handle partial observability
- [ ] World model (Dreamer-style) for better sample efficiency
- [ ] Image augmentation for lookahead from minimap
- [ ] Multi-objective reward (laptime vs tyre wear vs fuel)

### Tooling
- [ ] Live telemetry dashboard (Streamlit / Dash)
- [ ] Automatic hyperparameter tuning (Optuna)
- [ ] Docker container for reproducible training setup
- [ ] CI pipeline for regression testing

---

## Quick Reference

```batch
# Install
pip install -r requirements.txt

# Test (AC must be running in practice)
python scripts/test_bridge.py

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py --model models/best

# Resume training
python scripts/train.py --resume

# TensorBoard
tensorboard --logdir logs/tb
```
