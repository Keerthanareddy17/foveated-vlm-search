import random

import pytest
import yaml

from foveated_search.config import REQUIRED_KEYS, ConfigError, load_config


def test_loads_and_has_required_keys():
    cfg = load_config()
    for k in REQUIRED_KEYS:
        cfg.get_key(k)
    assert cfg.get_key("model.id") == "Qwen/Qwen3-VL-2B-Instruct"
    assert cfg.get_key("search.thumbnail_long_side") == 96
    assert cfg.get_key("search.fixation_size") == 224
    assert cfg.get_key("search.max_fixations") == 3


def test_seed_is_set_and_applied():
    cfg = load_config()
    assert cfg["seed"] == 42
    a = random.random()
    load_config()
    assert random.random() == a


def test_env_override(monkeypatch):
    monkeypatch.setenv("FVS_DATA_DIR", "/tmp/override_data")
    cfg = load_config()
    assert cfg.get_key("paths.data_dir") == "/tmp/override_data"


def test_missing_key_fails_clearly(tmp_path):
    bad = yaml.safe_load(open(load_config.__globals__["DEFAULT_CONFIG"]))
    del bad["search"]["fixation_size"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="search.fixation_size"):
        load_config(p)


def test_overlay_merges(tmp_path, monkeypatch):
    ov = tmp_path / "ov.yaml"
    ov.write_text("server:\n  quantization: fp8\n")
    cfg = load_config(overlays=[ov])
    assert cfg.get_key("server.quantization") == "fp8"
    assert cfg.get_key("server.port") == 8000  # untouched keys survive
    monkeypatch.setenv("FVS_CONFIG_OVERLAYS", str(ov))
    assert load_config().get_key("server.quantization") == "fp8"
