#!/usr/bin/env python
"""Step 4 experiment: which point-answer format grounds best on the full frame?

Run on all 50 dev queries (40 positives, 10 negatives), full frame, structured outputs, temperature 0.
Reports hit rate, negatives correctly null, calibration (slope / offset of predicted vs true box centre,
in 0..1000 units) and output tokens. Writes results/step4_point_format.json.
"""

from __future__ import annotations

import asyncio
import json
import statistics

import numpy as np
from PIL import Image

from foveated_search.data.fetch_images import eval_image_path
from foveated_search.data.manifest import read_jsonl
from foveated_search.search.base import search_config
from foveated_search.search.geometry import in_any_xywh
from foveated_search.search.schemas import structured_body
from foveated_search.serving.client import VLMClient

P = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2}
NULLABLE = {"anyOf": [P, {"type": "null"}]}


def obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


NATIVE_PROMPT = "Locate the {q} in the image with a point, report its point coordinates in JSON format."
VARIANTS = {
    # name: (prompt, schema, extractor -> [x, y] | None)
    "p": ('Point to the {q} in the image. Reply as JSON {{"p":[x,y]}} with x,y from 0 to 1000, '
          'or {{"p":null}} if there is no {q}.', obj({"p": NULLABLE}), lambda o: o["p"]),
    "point_2d object": (NATIVE_PROMPT + ' Use null if there is no {q}.', obj({"point_2d": NULLABLE}),
                        lambda o: o["point_2d"]),
    "native list, no label": (NATIVE_PROMPT, {"type": "array", "items": obj({"point_2d": P}), "minItems": 0, "maxItems": 1},
                              lambda o: o[0]["point_2d"] if o else None),
    "native list + label": (NATIVE_PROMPT, {"type": "array", "items": obj({"point_2d": P, "label": {"type": "string", "maxLength": 24}}),
                                            "minItems": 0, "maxItems": 1},
                            lambda o: o[0]["point_2d"] if o else None),
    # abstention designs that keep the native point_2d key
    "visible first": ('Is there a {q} in the image? If yes, locate it with a point. Report JSON '
                      '{{"visible": true|false, "point_2d": [x, y] or null}}.',
                      obj({"visible": {"type": "boolean"}, "point_2d": NULLABLE}),
                      lambda o: o["point_2d"] if o["visible"] else None),
    "point_2d | absent": (NATIVE_PROMPT + ' If there is no {q}, report {{"absent": true}}.',
                          {"anyOf": [obj({"point_2d": P}), obj({"absent": {"const": True}})]},
                          lambda o: o.get("point_2d")),
}


async def main() -> None:
    cfg = search_config(overlays=["configs/eval.yaml"])
    rows = read_jsonl("results/eval_manifest_dev.jsonl")
    c = VLMClient(cfg, counter=False)
    import os
    only = os.environ.get("ONLY")  # comma-separated subset of variants
    out = json.load(open("results/step4_point_format.json")) if only else {}
    try:
        for name, (tmpl, schema, get) in VARIANTS.items():
            if only and name not in only.split(","):
                continue
            recs = []
            for r in rows:
                f = Image.open(eval_image_path(cfg, r["file_name"])).convert("RGB")
                a = await c.chat([f], tmpl.format(q=r["query"]), max_tokens=48, extra_body=structured_body(schema))
                try:
                    p = get(json.loads(a.answer))
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    p = "unparsed"
                recs.append((r, f.size, p, a.completion_tokens, a.answer))
            pos = [x for x in recs if not x[0]["is_negative"]]
            neg = [x for x in recs if x[0]["is_negative"]]
            hits, cal = [], []
            for r, (w, h), p, _, _ in pos:
                if not isinstance(p, list):
                    hits.append(False)
                    continue
                x, y = p[0] / 1000 * w, p[1] / 1000 * h
                hits.append(in_any_xywh(x, y, r["gt_boxes"]))
                bx, by, bw, bh = min(r["gt_boxes"], key=lambda b: (b[0] + b[2] / 2 - x) ** 2 + (b[1] + b[3] / 2 - y) ** 2)
                cal.append(((bx + bw / 2) / w * 1000, (by + bh / 2) / h * 1000, p[0], p[1]))
            if not cal:
                out[name] = {"hit_rate": 0, "note": "never pointed"}
                continue
            A = np.array(cal, float)
            fit = {ax: [round(v, 2) for v in np.polyfit(A[:, ti], A[:, pi], 1)] for ax, ti, pi in [("x", 0, 2), ("y", 1, 3)]}
            out[name] = {
                "positives": len(pos), "hit_rate": round(sum(hits) / len(pos), 3),
                "pos_said_absent": sum(x[2] is None for x in pos), "unparsed": sum(x[2] == "unparsed" for x in recs),
                "negatives": len(neg), "neg_correct_null": sum(x[2] is None for x in neg),
                "median_offset_x": round(statistics.median(A[:, 2] - A[:, 0])), "median_offset_y": round(statistics.median(A[:, 3] - A[:, 1])),
                "fit_slope_intercept": fit,
                "mean_out_tokens_pos": round(statistics.mean(x[3] for x in pos), 2),
                "mean_out_tokens_neg": round(statistics.mean(x[3] for x in neg), 2),
                "examples": [x[4] for x in pos[:2]] + [x[4] for x in neg[:1]],
            }
            print(name, json.dumps(out[name]), flush=True)
    finally:
        await c.close()
    with open("results/step4_point_format.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/step4_point_format.json")


if __name__ == "__main__":
    asyncio.run(main())
