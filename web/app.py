"""
Flask web server — assetto_rl_bot_claude dashboard.
Launched via launch.bat from the project root.
"""

import csv
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

ROOT = Path(__file__).parent.parent.resolve()
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

_procs: dict[str, subprocess.Popen] = {}
_queues: dict[str, queue.Queue] = {}
_proc_lock = threading.Lock()


def _run_process(name: str, cmd: list[str]):
    """Start a subprocess, stream stdout+stderr into a queue."""
    with _proc_lock:
        # Kill existing
        if name in _procs and _procs[name].poll() is None:
            _procs[name].terminate()
            time.sleep(0.3)
        q: queue.Queue = queue.Queue(maxsize=2000)
        _queues[name] = q
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        _procs[name] = proc

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                q.put_nowait(line.decode("utf-8", errors="replace").rstrip())
            except queue.Full:
                pass
        proc.wait()
        try:
            q.put_nowait(f"\n[Process '{name}' exited — code {proc.returncode}]")
            q.put_nowait(None)
        except queue.Full:
            pass

    threading.Thread(target=reader, daemon=True).start()


def _stop_process(name: str):
    with _proc_lock:
        proc = _procs.get(name)
        if proc and proc.poll() is None:
            proc.terminate()


def _is_running(name: str) -> bool:
    proc = _procs.get(name)
    return proc is not None and proc.poll() is None


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — status
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    return jsonify({
        "training":   _is_running("training"),
        "evaluating": _is_running("eval"),
        "testing":    _is_running("test"),
    })


# ---------------------------------------------------------------------------
# Routes — training
# ---------------------------------------------------------------------------

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    data = request.get_json(force=True, silent=True) or {}
    cmd = [str(VENV_PY), "scripts/train.py"]
    if data.get("resume"):
        cmd.append("--resume")
    if data.get("timesteps"):
        cmd += ["--timesteps", str(int(data["timesteps"]))]
    _run_process("training", cmd)
    return jsonify({"ok": True})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    _stop_process("training")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — evaluate
# ---------------------------------------------------------------------------

@app.route("/api/eval/start", methods=["POST"])
def api_eval_start():
    data = request.get_json(force=True, silent=True) or {}
    model = data.get("model", "models/best")
    episodes = int(data.get("episodes", 10))
    cmd = [str(VENV_PY), "scripts/evaluate.py",
           "--model", model, "--episodes", str(episodes)]
    _run_process("eval", cmd)
    return jsonify({"ok": True})


@app.route("/api/eval/stop", methods=["POST"])
def api_eval_stop():
    _stop_process("eval")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — bridge test
# ---------------------------------------------------------------------------

@app.route("/api/test/start", methods=["POST"])
def api_test_start():
    data = request.get_json(force=True, silent=True) or {}
    backend = data.get("backend", "vgamepad")
    no_ctrl = data.get("no_control", False)
    cmd = [str(VENV_PY), "scripts/test_bridge.py", "--backend", backend]
    if no_ctrl:
        cmd.append("--no-control")
    _run_process("test", cmd)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — SSE log stream
# ---------------------------------------------------------------------------

@app.route("/api/stream/<name>")
def api_stream(name: str):
    """Server-Sent Events stream for process stdout."""
    if name not in _queues:
        return Response("data: [no process started yet]\n\n",
                        content_type="text/event-stream")

    q = _queues[name]

    def generate():
        while True:
            try:
                line = q.get(timeout=15)
            except queue.Empty:
                yield "data: [waiting...]\n\n"
                continue
            if line is None:
                yield "data: [done]\n\n"
                break
            safe = line.replace("\n", " ")
            yield f"data: {safe}\n\n"

    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Routes — config
# ---------------------------------------------------------------------------

CONFIG_FILES = {
    "default":  ROOT / "configs" / "default.yaml",
    "sac":      ROOT / "configs" / "sac_config.yaml",
    "env":      ROOT / "configs" / "env_config.yaml",
}


@app.route("/api/config/<name>")
def api_config_get(name: str):
    path = CONFIG_FILES.get(name)
    if not path or not path.exists():
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(yaml.safe_load(f) or {})


@app.route("/api/config/<name>", methods=["POST"])
def api_config_save(name: str):
    path = CONFIG_FILES.get(name)
    if not path:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(force=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return jsonify({"ok": True})


@app.route("/api/config/<name>/raw")
def api_config_raw(name: str):
    path = CONFIG_FILES.get(name)
    if not path or not path.exists():
        return jsonify({"error": "not found"}), 404
    return path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain"}


@app.route("/api/config/<name>/raw", methods=["POST"])
def api_config_raw_save(name: str):
    path = CONFIG_FILES.get(name)
    if not path:
        return jsonify({"error": "not found"}), 404
    text = request.get_data(as_text=True)
    try:
        yaml.safe_load(text)  # validate
    except yaml.YAMLError as e:
        return jsonify({"error": str(e)}), 400
    path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — results / logs
# ---------------------------------------------------------------------------

@app.route("/api/results")
def api_results():
    """Return latest eval CSV as JSON rows."""
    pattern = str(ROOT / "logs" / "eval_*.csv")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return jsonify([])
    rows = []
    with open(files[0], newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return jsonify(rows)


@app.route("/api/training_log")
def api_training_log():
    """Return latest training CSV as JSON for charting."""
    pattern = str(ROOT / "logs" / "*.csv")
    files = [f for f in sorted(glob.glob(pattern), reverse=True)
             if "eval_" not in os.path.basename(f)]
    if not files:
        return jsonify([])
    rows = []
    with open(files[0], newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return jsonify(rows)


@app.route("/api/models")
def api_models():
    """List available saved models."""
    pattern = str(ROOT / "models" / "*.zip")
    files = [os.path.splitext(os.path.basename(f))[0]
             for f in sorted(glob.glob(pattern))]
    return jsonify(files)


@app.route("/api/logfiles")
def api_logfiles():
    pattern = str(ROOT / "logs" / "*.csv")
    files = [os.path.basename(f) for f in sorted(glob.glob(pattern), reverse=True)]
    return jsonify(files)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  assetto_rl_bot_claude — Web Dashboard")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
