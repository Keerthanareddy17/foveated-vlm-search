"""Turn a Step 6 sample into model inputs + labels (runs in .venv-train; reused by Step 7's trainer).

Only the assistant answer is trained: labels are -100 everywhere except the target JSON and the
closing <|im_end|> of the assistant turn. The image goes through the same processor settings as the
server (server.image_min_pixels / image_max_pixels), so a 96 px thumbnail is 6 vision tokens here too.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

IGNORE = -100


def processor_size(cfg) -> dict:
    """Same resize bounds the vLLM server uses (--mm-processor-kwargs)."""
    return {"shortest_edge": int(cfg.get_key("server.image_min_pixels")),
            "longest_edge": int(cfg.get_key("server.image_max_pixels"))}


def load_processor(cfg):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(cfg.get_key("model.id")))
    size = processor_size(cfg)
    proc.image_processor.size = dict(size)  # applied to every call, like the server's processor kwargs
    return proc


def to_chat(sample: dict, root: Path) -> tuple[list[dict], list[Image.Image]]:
    """Sample -> (messages for apply_chat_template, [PIL image]). Image first, then text (as at inference)."""
    user, assistant = sample["messages"]
    img = Image.open(root / user["content"][0]["image"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user["content"][1]["text"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant["content"][0]["text"]}]}]
    return msgs, [img]


def encode(sample: dict, root: Path, proc) -> dict:
    """input_ids, attention_mask, (mm_token_type_ids), pixel_values, image_grid_thw, labels (assistant-only)."""
    msgs, images = to_chat(sample, root)
    full = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt = proc.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    enc = proc(text=[full], images=images, return_tensors="pt")
    n_prompt = proc(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1]
    ids = enc["input_ids"][0]
    labels = ids.clone()
    labels[:n_prompt] = IGNORE
    # trailing newline after the assistant's <|im_end|> is template padding, not part of the answer
    im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    last = int((ids == im_end).nonzero()[-1])
    labels[last + 1:] = IGNORE
    out = {"input_ids": ids, "attention_mask": enc["attention_mask"][0], "pixel_values": enc["pixel_values"],
           "image_grid_thw": enc["image_grid_thw"], "labels": labels, "n_prompt": n_prompt}
    if "mm_token_type_ids" in enc:  # transformers 5.x Qwen3-VL needs it for multimodal RoPE
        out["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
    return out
