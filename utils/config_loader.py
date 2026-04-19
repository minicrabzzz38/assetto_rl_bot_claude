"""
YAML configuration loader with deep-merge support.

Usage:
    cfg = load_config("configs/default.yaml", "configs/sac_config.yaml")
    lr  = cfg["sac"]["learning_rate"]
"""

import os
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(*paths: str) -> Dict[str, Any]:
    """
    Load one or more YAML files and deep-merge them left-to-right
    (rightmost file wins on conflicts).

    Raises FileNotFoundError if any path doesn't exist.
    """
    merged: dict = {}
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return merged


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested key access: get(cfg, 'sac', 'learning_rate', default=3e-4)."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
