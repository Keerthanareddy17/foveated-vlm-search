"""Load configs/*.yaml with env-var overrides for paths and clear errors on missing keys."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"

# env var -> dotted config key
ENV_OVERRIDES = {
    "PROJECT_DRIVE_DIR": "paths.drive_project_dir",
    "FVS_DATA_DIR": "paths.data_dir",
    "FVS_CHECKPOINTS_DIR": "paths.checkpoints_dir",
    "HF_HUB_CACHE": "paths.hf_hub_cache",
    "FVS_MODELS_DIR": "paths.models_dir",
    "FVS_RESULTS_DIR": "paths.results_dir",
}

REQUIRED_KEYS = [
    "seed",
    "paths.drive_project_dir",
    "paths.data_dir",
    "paths.checkpoints_dir",
    "paths.hf_hub_cache",
    "paths.models_dir",
    "paths.results_dir",
    "model.id",
    "server.host",
    "server.port",
    "server.image_min_pixels",
    "server.image_max_pixels",
    "search.thumbnail_long_side",
    "search.fixation_size",
    "search.max_fixations",
    "eval.num_queries",
    "eval.negatives_share",
    "eval.min_box_area",
]


class ConfigError(KeyError):
    pass


class Config(dict):
    """dict with dotted access: cfg.get_key("search.fixation_size")."""

    def get_key(self, dotted: str) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Missing config key '{dotted}' (failed at '{part}')")
            node = node[part]
        return node

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(v) if isinstance(v, dict) else v


def _set_key(d: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def load_config(path: str | Path | None = None, set_seed: bool = True) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            _set_key(raw, key, os.environ[env])
    cfg = Config(raw)
    for key in REQUIRED_KEYS:
        cfg.get_key(key)  # raises ConfigError with a clear message
    rd = Path(cfg.get_key("paths.results_dir"))
    if not rd.is_absolute():
        cfg["paths"]["results_dir"] = str(REPO_ROOT / rd)
    if set_seed:
        seed_everything(int(cfg["seed"]))
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
