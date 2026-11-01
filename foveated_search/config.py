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
    "FVS_LOCAL_DIR": "paths.local_dir",
    "PROJECT_DRIVE_DIR": "paths.drive_project_dir",
    "FVS_DRIVE_EVAL_IMAGES_DIR": "paths.drive_eval_images_dir",
    "FVS_DRIVE_ADAPTERS_DIR": "paths.drive_adapters_dir",
    "FVS_DATA_DIR": "paths.data_dir",
    "FVS_CHECKPOINTS_DIR": "paths.checkpoints_dir",
    "HF_HUB_CACHE": "paths.hf_hub_cache",
    "FVS_MODELS_DIR": "paths.models_dir",
    "FVS_RESULTS_DIR": "paths.results_dir",
}

REQUIRED_KEYS = [
    "seed",
    "paths.local_dir",
    "paths.drive_project_dir",
    "paths.drive_eval_images_dir",
    "paths.drive_adapters_dir",
    "paths.data_dir",
    "paths.checkpoints_dir",
    "paths.hf_hub_cache",
    "paths.models_dir",
    "paths.results_dir",
    "model.id",
    "model.served_name",
    "server.host",
    "server.port",
    "server.dtype",
    "server.max_model_len",
    "server.gpu_memory_utilization",
    "server.max_images_per_prompt",
    "server.quantization",
    "server.enforce_eager",
    "server.enable_prefix_caching",
    "server.image_min_pixels",
    "server.image_max_pixels",
    "client.temperature",
    "client.max_tokens",
    "client.timeout_s",
    "client.png_compress_level",
    "client.coord_scale",
    "client.output_format",
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


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    path: str | Path | None = None,
    overlays: list[str | Path] | None = None,
    set_seed: bool = True,
) -> Config:
    """Load base config, deep-merge overlays (args, then $FVS_CONFIG_OVERLAYS, comma-separated)."""
    raw = _read_yaml(Path(path) if path else DEFAULT_CONFIG)
    over = list(overlays or [])
    over += [o for o in os.environ.get("FVS_CONFIG_OVERLAYS", "").split(",") if o.strip()]
    for o in over:
        op = Path(o)
        if not op.is_absolute() and not op.exists():
            op = REPO_ROOT / op
        _deep_merge(raw, _read_yaml(op))
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
