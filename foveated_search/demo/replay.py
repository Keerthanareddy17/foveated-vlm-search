"""Step 9: build ONE self-contained replay page from the saved demo runs.

  python -m foveated_search.demo.replay [--out results/demo/replay.html] [--runs results/demo/a.json ...]

For every run in results/demo/*.json:
- re-create every image the model was sent (full frame, thumbnail, masked thumbnails, crops) from the cleaned photo
  with the SAME code as the search (search/views.py, strategies.masked_thumbnail, serving/images.encode_png) and
  check its sha256 against the trace: the page shows exactly the bytes the model saw, or the build fails;
- convert each trace into a replay timeline (`to_replay`): call k starts at the sum of the measured call times
  before it and ends total_s later, the same accounting as time_to_answer;
- embed images as data: URIs, the timelines as JSON, and the page template (web/template.html) into one HTML file
  with no external requests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

from foveated_search.config import REPO_ROOT, Config, load_config
from foveated_search.search.coords import model_to_frame
from foveated_search.search.geometry import in_box
from foveated_search.search.strategies import masked_thumbnail
from foveated_search.search.views import make_thumbnail, resize
from foveated_search.serving.images import encode_png

WEB_DIR = Path(__file__).resolve().parent / "web"
TEMPLATE = WEB_DIR / "template.html"
PLACEHOLDER = "/*__REPLAY_DATA__*/null"


def replay_config() -> Config:
    cfg = load_config(overlays=["configs/search.yaml", "configs/prompts.yaml", "configs/demo.yaml"], set_seed=False)
    for k in ["demo.out_dir", "demo.replay_out", "demo.embed_long_side", "demo.embed_jpeg_quality", "demo.queries_file"]:
        cfg.get_key(k)
    return cfg


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def b64(mime: str, data: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


class ImageMismatch(RuntimeError):
    pass


def resolve_image(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def sent_images(rec: dict, frame: Image.Image) -> dict[str, list[tuple[str, bytes]]]:
    """Re-create the PNG of every call of every trace, verified against the logged sha256.
    Returns {strategy: [(sha, png bytes) per call]}. Raises ImageMismatch on any difference."""
    st = rec["settings"]
    level = int(st["png_compress_level"])
    out: dict[str, list[tuple[str, bytes]]] = {}
    for name, tr in rec["traces"].items():
        pngs = []
        if name == "eye_search":
            thumb = make_thumbnail(frame, int(st["thumbnail_long_side"]))
            if list(thumb.size) != list(tr["params"]["thumbnail"]):
                raise ImageMismatch(f"{name}: thumbnail {thumb.size} != trace {tr['params']['thumbnail']}")
        checked: list[list[int]] = []
        for c in tr["calls"]:
            x0, y0, x1, y1 = c["region"]
            full = [x0, y0, x1, y1] == [0, 0, frame.width, frame.height]
            if name == "eye_search" and c["role"] == "point":
                img = masked_thumbnail(thumb, checked, frame.size, tuple(st["look_again_mask_rgb"])) \
                    if (tr["params"].get("look_again_mask") and checked) else thumb
            elif full:
                img = resize(frame, (c["sent_w"], c["sent_h"]))
            else:
                img = frame.crop((x0, y0, x1, y1))
            if name == "eye_search" and c["role"] == "confirm":
                checked.append(c["region"])
            png = encode_png(img, level).png
            if sha256(png) != c["image_sha256"]:
                raise ImageMismatch(f"{rec['run_id']} {name} call {len(pngs)} ({c['role']} {c['sent_w']}x{c['sent_h']}): "
                                    "re-created image differs from what was sent")
            pngs.append((c["image_sha256"], png))
        out[name] = pngs
    return out


def _frame_point(parsed_point, c: dict, scale: float) -> list[float] | None:
    if not parsed_point:
        return None
    x, y = model_to_frame(parsed_point[0], parsed_point[1], c["sent_w"], c["sent_h"], tuple(c["region"]), scale)
    return [round(x, 1), round(y, 1)]


def to_replay(tr: dict, scale: float = 1000.0) -> dict:
    """Trace dict -> replay timeline. Pure function (unit-tested).

    Each call event: t0/t1 (s, cumulative measured call times), role, fixation, image sha, region, sent size,
    pixels, png bytes, vision tokens, answer, and what it meant for the display:
    point calls -> `point` (frame px) or absent, `repeat` (inside an already-checked crop, no new crop);
    confirm calls -> `yes` and `point` (refined, frame px)."""
    t = 0.0
    events = []
    checked: list[list[int]] = []
    for i, c in enumerate(tr["calls"]):
        ev = {"i": i, "role": c["role"], "fixation": c["fixation"], "t0": round(t, 4),
              "t1": round(t + c["total_s"], 4), "ttft": c["ttft_s"], "sha": c["image_sha256"],
              "region": c["region"], "sent": [c["sent_w"], c["sent_h"]], "pixels": c["pixels"],
              "bytes": c["png_bytes"], "vision_tokens": c["vision_tokens"] or 0, "answer": c["raw_answer"],
              "parse_error": c["parse_error"]}
        parsed = c.get("parsed") or {}
        if c["role"] in ("point", "tile"):
            ev["point"] = _frame_point(parsed.get("point"), c, scale)
            ev["masked"] = bool(tr["strategy"] == "eye_search" and tr["params"].get("look_again_mask") and checked)
            ev["checked"] = [list(b) for b in checked] if tr["strategy"] == "eye_search" else []
            ev["repeat"] = bool(ev["point"] and tr["strategy"] == "eye_search"
                                and any(in_box(*ev["point"], b) for b in checked))
        if c["role"] == "confirm":
            ev["yes"] = bool(parsed.get("y"))
            ev["point"] = _frame_point(parsed.get("point"), c, scale)
            ev["label"] = parsed.get("l")
            checked.append(c["region"])
        events.append(ev)
        t += c["total_s"]
    return {"strategy": tr["strategy"], "found": bool(tr["found"]), "final_point": tr["final_point"],
            "t_end": round(t, 4), "events": events, "fixations": tr["fixations"], "crops": tr["crops"],
            "repeats": tr["repeats"], "params": tr["params"], "evidence": tr.get("evidence")}


def embed_frame(frame: Image.Image, long_side: int, quality: int) -> tuple[str, list[int]]:
    s = min(1.0, long_side / max(frame.size))
    img = frame if s == 1.0 else frame.resize((round(frame.width * s), round(frame.height * s)),
                                              Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return b64("image/jpeg", buf.getvalue()), list(img.size)


def build_payload(runs: list[dict], cfg: Config) -> dict:
    d = cfg["demo"]
    scale = float(cfg.get_key("client.coord_scale"))
    images: dict[str, str] = {}   # sha -> data URI (deduplicated across runs)
    frames: dict[str, dict] = {}
    out_runs = []
    n_verified = 0
    for rec in runs:
        path = resolve_image(rec["image"])
        raw = path.read_bytes()
        if sha256(raw) != rec["image_sha256"]:
            raise ImageMismatch(f"{rec['run_id']}: {rec['image']} changed since the run (sha256 differs)")
        frame = Image.open(io.BytesIO(raw)).convert("RGB")
        if rec["image_sha256"] not in frames:
            uri, size = embed_frame(frame, int(d["embed_long_side"]), int(d["embed_jpeg_quality"]))
            frames[rec["image_sha256"]] = {"uri": uri, "embed_size": size}
        pngs = sent_images(rec, frame)
        panels = {}
        for name, tr in rec["traces"].items():
            for sha, png in pngs[name]:
                n_verified += 1
                if name == "eye_search":   # full frames are drawn from the frame JPEG, not re-embedded as PNG
                    images.setdefault(sha, b64("image/png", png))
            panels[name] = to_replay(tr, scale)
        out_runs.append({"run_id": rec["run_id"], "image": Path(rec["image"]).name, "query": rec["query"],
                         "photo": rec.get("photo_name") or Path(rec["image"]).name,
                         "frame": rec["image_sha256"], "frame_size": rec["frame_size"], "model": rec["model"],
                         "precision": rec["precision"], "gpu": rec.get("gpu"), "concurrency": rec["concurrency"],
                         "warmup_rounds": rec["warmup_rounds"], "settings": rec["settings"],
                         "created": rec["created"], "summary": rec["summary"], "panels": panels,
                         "expect": rec.get("expect"), "gt_boxes": rec.get("gt_boxes"), "repeats": rec.get("repeats", 1)})
    return {"runs": out_runs, "frames": frames, "images": images, "verified_images": n_verified,
            "strategies": list(d["strategies"]), "gpu": d.get("gpu_label")}


PAGES_BANNER = """<div class="pages-banner"><b>foveated-vlm-search</b> &middot; a 2B vision-language model that finds
  things in high-resolution photos while barely looking: replays of real measured searches &middot;
  <a href="{repo}">code, write-up and results on GitHub</a></div>"""
PAGES_CSS = """  .pages-banner { margin: 0 0 12px; padding: 8px 10px; border: 1px solid var(--line2); border-radius: 3px;
                  color: var(--dim); font-size: 12px; line-height: 1.5; }
  .pages-banner b { color: #fff; } .pages-banner a { color: var(--ok); }
</style>"""


def render(payload: dict, template: str | None = None, pages_repo: str | None = None) -> str:
    html = template if template is not None else TEMPLATE.read_text()
    if pages_repo:   # GitHub Pages copy: same page plus a header linking back to the repo
        html = html.replace("</style>", PAGES_CSS, 1).replace(
            '<div class="wrap">', '<div class="wrap">\n  ' + PAGES_BANNER.format(repo=pages_repo), 1)
    if PLACEHOLDER not in html:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")   # safe inside <script>
    return html.replace(PLACEHOLDER, data)


def load_runs(paths: list[Path], cfg: Config | None = None) -> list[dict]:
    """Runs in the order of configs/demo_queries.yaml (unlisted runs last, by name), with the answer notes from its
    `notes` regions attached to each strategy summary (applied at build time, so older runs get them too)."""
    from foveated_search.demo.run_demo import answer_note, notes_for, queries_spec

    runs = [json.loads(p.read_text()) for p in paths]
    spec = queries_spec(cfg) if cfg is not None else {}
    order = {(Path(r["image"]).name, str(r["query"])): i for i, r in enumerate(spec.get("runs") or [])}
    for r in runs:
        r["photo_name"] = (spec.get("photo_names") or {}).get(Path(r["image"]).name, Path(r["image"]).name)
        regions = notes_for(spec, r["image"])
        for sm in r["summary"].values():
            sm["note"] = answer_note(sm, r["query"], regions)
    return sorted(runs, key=lambda r: (order.get((Path(r["image"]).name, r["query"]), len(order)),
                                       Path(r["image"]).name, r["query"]))


def main(argv: list[str] | None = None) -> int:
    cfg = replay_config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / cfg.get_key("demo.replay_out"))
    ap.add_argument("--pages", action="store_true", help="also write the GitHub Pages copy (demo.pages_out)")
    a = ap.parse_args(argv)
    paths = a.runs or sorted((REPO_ROOT / cfg.get_key("demo.out_dir")).glob("*.json"))
    if not paths:
        print(f"no runs in {cfg.get_key('demo.out_dir')}/: run `make demo` first")
        return 1
    payload = build_payload(load_runs(paths, cfg), cfg)
    html = render(payload)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    if a.pages:
        pages = REPO_ROOT / cfg.get_key("demo.pages_out")
        pages.parent.mkdir(parents=True, exist_ok=True)
        pages.write_text(render(payload, pages_repo=cfg.get_key("demo.repo_url")))
        print(f"GitHub Pages copy -> {pages.relative_to(REPO_ROOT)} ({pages.stat().st_size / 1e6:.2f} MB)")
    print(f"{len(payload['runs'])} runs, {payload['verified_images']} sent images re-created and verified "
          f"byte-identical (sha256) -> {a.out} ({len(html.encode()) / 1e6:.2f} MB)")
    for r in payload["runs"]:
        print("  " + r["run_id"] + ": " + " | ".join(
            f"{s}: {r['summary'][s]['status']} in {r['summary'][s]['time_to_answer_s']:.2f}s" for s in r["panels"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
