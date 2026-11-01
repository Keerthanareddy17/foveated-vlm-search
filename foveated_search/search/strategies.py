"""The four search strategies. All parameters from configs/search.yaml, all text from prompts.yaml.

full_frame      one point call on the native-resolution frame
downscale       one point call on the frame resized to a pixel budget (grid-aligned)
sliding_window  point calls on fixation-size tiles, raster order, stop at first non-null point
eye_search      thumbnail point -> sharp crop confirm, up to max_fixations, thumbnail bytes reused
"""

from __future__ import annotations

import asyncio

from PIL import Image

from foveated_search.config import Config
from foveated_search.search.base import (CallRecord, Trace, call, confirm_prompt, new_trace, point_prompt, prompt,
                                         register)
from foveated_search.search.coords import model_to_frame
from foveated_search.search.geometry import downscale_size, in_box, thumbnail_size, tiles
from foveated_search.search.views import make_crop, make_thumbnail, resize
from foveated_search.search.parsing import ParseError
from foveated_search.search.schemas import confirm_schema, parse_confirm, parse_p, point_schema, trim_label


def _scfg(cfg: Config) -> dict:
    return cfg["search"]


def _parse_point_into(rec: CallRecord) -> tuple[float, float] | None:
    """Parse a point answer into the record; a malformed answer counts as no point."""
    try:
        p = parse_p(rec.raw_answer)
        rec.parsed = {"point": list(p) if p else None}
        return p
    except ParseError as e:
        rec.parse_error = str(e)
        return None


def _frame_point(p: tuple[float, float], rec: CallRecord, scale: float) -> list[float]:
    x, y = model_to_frame(p[0], p[1], rec.sent_w, rec.sent_h, tuple(rec.region), scale)
    return [round(x, 2), round(y, 2)]


async def _single_point(name: str, frame: Image.Image, size: tuple[int, int], query: str, client, cfg: Config,
                        params: dict) -> Trace:
    s = _scfg(cfg)
    tr = new_trace(name, frame, query, cfg, params)
    img = client.encode(resize(frame, size))
    rec = await call(client, cfg, role="point", fixation=None, image=img, region=(0, 0, frame.width, frame.height),
                     text=prompt(cfg, "point", query=query), schema=point_schema(),
                     max_tokens=int(s["max_tokens_point"]))
    tr.calls.append(rec)
    p = _parse_point_into(rec)
    if p is not None:
        tr.found = True
        tr.final_point = _frame_point(p, rec, float(cfg.get_key("client.coord_scale")))
    return tr


@register("full_frame")
async def full_frame(frame: Image.Image, query: str, client, cfg: Config) -> Trace:
    return await _single_point("full_frame", frame, frame.size, query, client, cfg, {"size": list(frame.size)})


def downscale_budget(frame_w: int, frame_h: int, cfg: Config) -> int:
    """Configured budget, or the nominal cost of one thumbnail + one fixation for this frame."""
    s = _scfg(cfg)
    if s["downscale"]["pixel_budget"] is not None:
        return int(s["downscale"]["pixel_budget"])
    tw, th = thumbnail_size(frame_w, frame_h, int(s["thumbnail_long_side"]))
    F = int(s["fixation_size"])
    return tw * th + min(F, frame_w) * min(F, frame_h)


@register("downscale")
async def downscale(frame: Image.Image, query: str, client, cfg: Config) -> Trace:
    budget = downscale_budget(frame.width, frame.height, cfg)
    size = downscale_size(frame.width, frame.height, budget)
    return await _single_point("downscale", frame, size, query, client, cfg, {"pixel_budget": budget, "size": list(size)})


@register("sliding_window")
async def sliding_window(frame: Image.Image, query: str, client, cfg: Config) -> Trace:
    s = _scfg(cfg)
    w = s["sliding_window"]
    tile = int(w["tile_size"] or s["fixation_size"])
    mode = str(w.get("tile_mode", "point"))
    if mode not in ("point", "confirm"):
        raise ValueError(f"unknown sliding_window.tile_mode {mode!r}")
    boxes = tiles(frame.width, frame.height, tile, float(w["overlap"]), str(w["order"]))
    tr = new_trace("sliding_window", frame, query, cfg,
                   {"tile": tile, "overlap": w["overlap"], "order": w["order"], "parallel": bool(w["parallel"]),
                    "n_tiles": len(boxes), "tile_mode": mode})
    point_text = prompt(cfg, "point", query=query)
    confirm_text = confirm_prompt(cfg, query, refine_point=False, label=False)
    scale = float(cfg.get_key("client.coord_scale"))
    yes_no = confirm_schema(False, False, int(s["confirm"]["label_max_chars"]))

    async def ask(i: int, box) -> CallRecord:
        img = client.encode(frame.crop(box))
        if mode == "point":
            return await call(client, cfg, role="tile", fixation=i + 1, image=img, region=box, text=point_text,
                              schema=point_schema(), max_tokens=int(s["max_tokens_point"]))
        return await call(client, cfg, role="tile", fixation=i + 1, image=img, region=box, text=confirm_text,
                          schema=yes_no, max_tokens=int(s["max_tokens_confirm"]))

    def tile_hit(rec: CallRecord) -> tuple[float, float] | bool | None:
        """point mode: the point (or None); confirm mode: True/False."""
        if mode == "point":
            return _parse_point_into(rec)
        try:
            ans = parse_confirm(rec.raw_answer)
            rec.parsed = {"y": ans.yes}
            return ans.yes
        except ParseError as e:
            rec.parse_error = str(e)
            return False

    async def localize(rec: CallRecord) -> list[float]:
        """confirm mode, on a "yes" tile: point call on the same tile; tile centre if it says absent."""
        box = tuple(rec.region)
        prec = await call(client, cfg, role="point", fixation=rec.fixation, image=client.encode(frame.crop(box)),
                          region=box, text=point_text, schema=point_schema(), max_tokens=int(s["max_tokens_point"]))
        tr.calls.append(prec)
        p = _parse_point_into(prec)
        if p is None:
            return [round((box[0] + box[2]) / 2, 2), round((box[1] + box[3]) / 2, 2)]
        return _frame_point(p, prec, scale)

    async def accept(rec: CallRecord, hit) -> None:
        tr.found = True
        tr.final_point = _frame_point(hit, rec, scale) if mode == "point" else await localize(rec)

    if w["parallel"]:
        recs = await asyncio.gather(*(ask(i, b) for i, b in enumerate(boxes)))
        first = None
        for rec in recs:  # all tiles were sent; decision = first hit in tile order
            tr.calls.append(rec)
            tr.crops.append(rec.region)
            hit = tile_hit(rec)
            if hit and first is None:
                first = (rec, hit)
        if first:
            await accept(*first)
        return tr

    for i, box in enumerate(boxes):
        rec = await ask(i, box)
        tr.calls.append(rec)
        tr.crops.append(rec.region)
        hit = tile_hit(rec)
        if hit:
            await accept(rec, hit)
            break
    return tr


@register("eye_search")
async def eye_search(frame: Image.Image, query: str, client, cfg: Config) -> Trace:
    s = _scfg(cfg)
    c = s["confirm"]
    scale = float(cfg.get_key("client.coord_scale"))
    F = int(s["fixation_size"])
    tsize = thumbnail_size(frame.width, frame.height, int(s["thumbnail_long_side"]))
    tr = new_trace("eye_search", frame, query, cfg,
                   {"thumbnail": list(tsize), "fixation_size": F, "max_fixations": int(s["max_fixations"]),
                    "refine_point": bool(c["refine_point"]), "label": bool(c["label"]),
                    "retry_prompt": str(s["retry_prompt"])})
    thumb = client.encode(make_thumbnail(frame, int(s["thumbnail_long_side"])))  # encoded ONCE: byte-identical
    assert (thumb.width, thumb.height) == tsize
    tr.thumbnail_sha256 = thumb_hash = None
    full = (0, 0, frame.width, frame.height)
    c_schema = confirm_schema(bool(c["refine_point"]), bool(c["label"]), int(c["label_max_chars"]))
    confirm_text = confirm_prompt(cfg, query, bool(c["refine_point"]), bool(c["label"]))
    checked_model: list[tuple[float, float]] = []   # thumbnail points already looked at (0..1000)

    for k in range(1, int(s["max_fixations"]) + 1):
        text = point_prompt(cfg, query, checked_model)
        rec = await call(client, cfg, role="point", fixation=k, image=thumb, region=full, text=text,
                         schema=point_schema(), max_tokens=int(s["max_tokens_point"]))
        tr.calls.append(rec)
        if thumb_hash is None:
            thumb_hash = tr.thumbnail_sha256 = rec.image_sha256
        assert rec.image_sha256 == thumb_hash, "thumbnail bytes changed between fixations"
        p = _parse_point_into(rec)
        if p is None:
            break  # "not visible" (or unparseable): stop, not found
        fx, fy = _frame_point(p, rec, scale)
        checked_model.append(p)
        if any(in_box(fx, fy, b) for b in tr.crops):
            tr.repeats += 1   # points back into an already-checked crop: no new crop, uses an attempt
            continue
        tr.fixations.append([fx, fy])
        crop, box = make_crop(frame, fx, fy, F)
        tr.crops.append(list(box))
        crec = await call(client, cfg, role="confirm", fixation=k, image=client.encode(crop), region=box,
                          text=confirm_text, schema=c_schema,
                          max_tokens=int(s["max_tokens_confirm"]))
        tr.calls.append(crec)
        try:
            ans = parse_confirm(crec.raw_answer)
            crec.parsed = {"y": ans.yes, "point": list(ans.point) if ans.point else None, "l": ans.label}
        except ParseError as e:
            crec.parse_error = str(e)
            continue
        if ans.yes:
            tr.found = True
            tr.final_point = _frame_point(ans.point, crec, scale) if ans.point else [fx, fy]
            tr.evidence = trim_label(ans.label, int(c["label_max_chars"]))
            break
    return tr
