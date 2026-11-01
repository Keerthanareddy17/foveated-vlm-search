"""Step 7 helpers (GPU-free): padding collator, config diff, Hub write-access check, train config."""

import torch

from foveated_search.train.collate import IGNORE
from foveated_search.train.merge import json_diff
from foveated_search.train.push_hub import can_write
from foveated_search.train.train_lora import Collator, train_config


def _item(n, n_patches, grid):
    return {"input_ids": torch.arange(1, n + 1), "attention_mask": torch.ones(n, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([0, 1, 1] + [0] * (n - 3)),
            "labels": torch.tensor([IGNORE] * (n - 2) + [7, 8]), "pixel_values": torch.ones(n_patches, 4),
            "image_grid_thw": torch.tensor([grid])}


def test_collator_pads_right_and_keeps_images():
    b = Collator(pad_id=0)([_item(5, 8, [1, 2, 4]), _item(3, 4, [1, 2, 2])])
    assert b["input_ids"].tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert b["labels"][1].tolist() == [IGNORE, 7, 8, IGNORE, IGNORE]      # padding never trained
    assert b["mm_token_type_ids"][1].tolist() == [0, 1, 1, 0, 0]
    assert b["pixel_values"].shape == (12, 4) and b["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 2]]


def test_json_diff():
    a = {"x": 1, "n": {"a": 1, "b": 2}, "gone": 3}
    b = {"x": 1, "n": {"a": 1, "b": 5}, "new": 4}
    d = json_diff(a, b)
    assert "~ n.b: 2 -> 5" in d and any(s.startswith("- gone") for s in d) and any(s.startswith("+ new") for s in d)
    assert json_diff(a, a) == []


def test_can_write():
    assert can_write({"auth": {"accessToken": {"role": "write"}}}, "me")[0]
    assert not can_write({"auth": {"accessToken": {"role": "read"}}}, "me")[0]
    fg = {"auth": {"accessToken": {"role": "fineGrained", "fineGrained": {"scoped": [
        {"entity": {"name": "me", "type": "user"}, "permissions": ["repo.content.read", "repo.write"]}]}}}}
    assert can_write(fg, "me")[0] and not can_write(fg, "someone_else")[0]
    assert not can_write({}, "me")[0]


def test_train_config_effective_batch_and_lora_targets():
    t = train_config()["train"]
    assert int(t["per_device_batch_size"]) * int(t["grad_accum"]) == 64
    import re

    rx = re.compile(t["lora"]["target_modules_regex"])
    assert rx.fullmatch("model.language_model.layers.3.self_attn.q_proj")
    assert rx.fullmatch("model.language_model.layers.27.mlp.down_proj")
    assert not rx.fullmatch("model.visual.blocks.0.attn.qkv")
    assert not rx.fullmatch("model.visual.merger.linear_fc1") and not rx.fullmatch("lm_head")
