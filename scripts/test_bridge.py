"""
Telemetry bridge smoke test.

Run this FIRST before training to verify:
  1. AC shared memory is accessible
  2. Telemetry values look sane
  3. Virtual gamepad fires without errors (if using vgamepad)

Usage:
    python scripts/test_bridge.py
    python scripts/test_bridge.py --no-control   # skip control test
    python scripts/test_bridge.py --backend keyboard
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger
from ac_bridge.shared_memory import ACSharedMemoryReader
from ac_bridge.control_interface import make_control_interface
from ac_bridge.data_types import ACStatus

logger = get_logger("test_bridge")


def parse_args():
    parser = argparse.ArgumentParser(description="Test AC telemetry bridge and controls")
    parser.add_argument("--no-control", action="store_true", help="Skip control interface test")
    parser.add_argument("--backend",    default="vgamepad",  help="Control backend: vgamepad | keyboard")
    parser.add_argument("--duration",   type=float, default=5.0, help="How long to print live telemetry (seconds)")
    return parser.parse_args()


STATUS_NAMES = {
    ACStatus.OFF:    "OFF",
    ACStatus.REPLAY: "REPLAY",
    ACStatus.LIVE:   "LIVE",
    ACStatus.PAUSE:  "PAUSE",
}


def test_shared_memory(duration: float):
    logger.info("=" * 60)
    logger.info("TEST 1: AC Shared Memory")
    logger.info("=" * 60)

    reader = ACSharedMemoryReader()
    logger.info("Connecting to AC shared memory (timeout=10s)...")
    ok = reader.connect(timeout=10.0)

    if not ok:
        logger.error(
            "FAILED: Cannot connect to AC shared memory.\n"
            "  → Make sure Assetto Corsa is running.\n"
            "  → Start a practice session first.\n"
            "  → AC must have the shared memory plugin loaded (it is by default)."
        )
        return False

    logger.info("SUCCESS: Connected to all three shared memory pages.")

    static = reader.read_static()
    if static:
        logger.info("  Car   : %s", static.carModel or "(unknown)")
        logger.info("  Track : %s", static.track    or "(unknown)")
        logger.info("  MaxRPM: %d", static.maxRpm)

    logger.info("Live telemetry for %.0f seconds:", duration)
    logger.info("  %-12s %-10s %-10s %-10s %-8s %-8s %-12s",
                "Status", "Speed", "Throttle", "Brake", "Gear", "TyresOut", "LapProgress")

    deadline = time.time() + duration
    while time.time() < deadline:
        phys, gfx, _ = reader.read_all()
        if phys is None or gfx is None:
            logger.warning("Telemetry read returned None — is AC still running?")
            time.sleep(0.5)
            continue

        status_name = STATUS_NAMES.get(gfx.status, str(gfx.status))
        logger.info(
            "  %-12s %-10.1f %-10.2f %-10.2f %-8d %-8d %-12.4f",
            status_name,
            phys.speedKmh,
            phys.gas,
            phys.brake,
            phys.gear,
            phys.numberOfTyresOut,
            gfx.normalizedCarPosition,
        )
        time.sleep(0.25)

    reader.disconnect()
    logger.info("Shared memory test PASSED.")
    return True


def test_control(backend: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Control Interface (%s)", backend)
    logger.info("=" * 60)

    try:
        ctrl = make_control_interface(backend)
    except Exception as exc:
        logger.error("FAILED to initialize %s control: %s", backend, exc)
        return False

    logger.info("Sending test inputs (you should see steering/throttle in AC):")

    sequences = [
        ("Neutral",           0.0,  0.0, 0.0),
        ("Full throttle",     0.0,  1.0, 0.0),
        ("Steer left",       -1.0,  0.3, 0.0),
        ("Steer right",       1.0,  0.3, 0.0),
        ("Full brake",        0.0,  0.0, 1.0),
        ("Neutral (release)", 0.0,  0.0, 0.0),
    ]

    for label, steer, throttle, brake in sequences:
        logger.info("  → %s (steer=%.1f throttle=%.1f brake=%.1f)", label, steer, throttle, brake)
        ctrl.apply_action(steer, throttle, brake)
        time.sleep(0.8)

    ctrl.reset_inputs()
    ctrl.close()
    logger.info("Control test PASSED.")
    return True


def main():
    args = parse_args()

    logger.info("assetto_rl_bot_claude — Bridge Test")
    logger.info("Make sure Assetto Corsa is running in Practice mode.")
    logger.info("")

    sm_ok = test_shared_memory(duration=args.duration)

    ctrl_ok = True
    if not args.no_control:
        ctrl_ok = test_control(backend=args.backend)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULT SUMMARY")
    logger.info("  Shared memory : %s", "PASS" if sm_ok   else "FAIL")
    logger.info("  Control       : %s", "PASS" if ctrl_ok else "FAIL (or skipped)")
    logger.info("=" * 60)

    if sm_ok and ctrl_ok:
        logger.info("All tests passed. You can now run: python scripts/train.py")
    else:
        logger.warning("Some tests failed. Fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
