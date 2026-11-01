#!/usr/bin/env python
"""Step 6, section 7: score the BASE model on val samples, per type, before any training.

Same request shape as eye search (image first, then the sample's prompt text; structured output with
the matching schema), same parsers. Success rules:
  where / look_again: positive -> point inside a target box (in the image sent); negative -> absent
  confirm:            correct yes/no; on a "yes" sample the refined point must also be inside a target box
Writes results/step6_base_on_val.json. Needs the BF16 server (make serve).
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict

from PIL import Image

from foveated_search.config import REPO_ROOT
from foveated_search.data.train_data import data_config, out_dir, read_samples
from foveated_search.search.parsing import ParseError
from foveated_search.search.schemas import confirm_schema, parse_confirm, parse_p, point_schema, structured_body
from foveated_search.serving.client import VLMClient


def inside(px: float, py: float, boxes: list[list[float]]) -> bool:
    return any(x <= px <= x + w and y <= py <= y + h for x, y, w, h in boxes)


async def main() -> None:
    cfg = data_config()
    n = int(cfg.get_key("train_data.base_check_per_type"))
    lmax = int(cfg.get_key("search.confirm.label_max_chars"))
    val = read_samples(cfg, "val")
    rng = random.Random(f"{cfg['seed']}:base_check")
    by_type = defaultdict(list)
    for s in val:
        by_type[s["type"]].append(s)
    root = out_dir(cfg)
    client = VLMClient(cfg, counter=False)
    sem = asyncio.Semaphore(int(cfg.get_key("harness.concurrency")))
    rows = []

    async def one(s: dict) -> None:
        m = s["meta"]
        text = s["messages"][0]["content"][1]["text"]
        if s["type"] == "confirm":
            schema = confirm_schema(True, bool(m["label_on"]), lmax)
        else:
            schema = point_schema()
        img = Image.open(root / s["image"]).convert("RGB")
        async with sem:
            r = await client.chat([img], text, max_tokens=48, extra_body=structured_body(schema))
        ok, parsed = False, None
        try:
            if s["type"] == "confirm":
                a = parse_confirm(r.answer)
                parsed = {"y": a.yes, "point": a.point}
                if m["polarity"] == "no":
                    ok = not a.yes
                else:
                    ok = a.yes and a.point is not None and inside(a.point[0] / 1000 * m["sent_w"], a.point[1] / 1000 * m["sent_h"],
                                                                  m["gt_boxes_sent"])
            else:
                p = parse_p(r.answer)
                parsed = {"point": p}
                if m["polarity"] == "neg":
                    ok = p is None
                else:
                    ok = p is not None and inside(p[0] / 1000 * m["sent_w"], p[1] / 1000 * m["sent_h"], m["gt_boxes_sent"])
        except ParseError:
            parsed = "unparsed"
        rows.append({"id": s["id"], "type": s["type"], "polarity": m["polarity"], "query_type": m["query_type"],
                     "thumb_long_side": m.get("thumb_long_side"), "crop_size": m.get("crop_size"),
                     "no_kind": m.get("no_kind"), "ok": ok, "unparsed": parsed == "unparsed", "answer": r.answer, "target": s["messages"][1]["content"][0]["text"],
                     "completion_tokens": r.completion_tokens})

    picked = [s for t in ["where", "look_again", "confirm"] for s in rng.sample(by_type[t], min(n, len(by_type[t])))]
    try:
        await asyncio.gather(*(one(s) for s in picked))
    finally:
        await client.close()
    rows.sort(key=lambda r: r["id"])

    def rate(rs):
        return {"n": len(rs), "success": round(sum(r["ok"] for r in rs) / len(rs), 4) if rs else None}

    summary = {}
    for t in ["where", "look_again", "confirm"]:
        rs = [r for r in rows if r["type"] == t]
        d = {"all": rate(rs)}
        for pol in sorted({r["polarity"] for r in rs}):
            d[pol] = rate([r for r in rs if r["polarity"] == pol])
        d["food"] = rate([r for r in rs if r["query_type"] == "supercategory"])
        if t != "confirm":
            for L in sorted({r["thumb_long_side"] for r in rs}):
                d[f"thumb_{L}"] = rate([r for r in rs if r["thumb_long_side"] == L and r["polarity"] == "pos"])
        else:
            for k in sorted({r["no_kind"] for r in rs if r["no_kind"]}):
                d[f"no_{k}"] = rate([r for r in rs if r["no_kind"] == k])
        d["unparsed"] = sum(r["unparsed"] for r in rs)
        summary[t] = d
    out = {"model": cfg.get_key("model.id"), "precision": cfg["server"]["quantization"] or cfg["server"]["dtype"],
           "split": "val", "per_type": n, "prompts_version": cfg["prompts"]["version"], "summary": summary, "rows": rows}
    (REPO_ROOT / "results/step6_base_on_val.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
