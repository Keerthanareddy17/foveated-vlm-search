"""Trace visualization: one frame with ground truth and what every strategy looked at.

Reused by the demo (Step 9) and README figures (Step 10), so the drawing is deliberately plain:
ground truth green, sliding-window tiles grey (hit tile amber), eye-search fixations numbered
1 -> 2 -> 3 in magenta with their crops (confirmed crop solid, rejected dashed), final points as
labelled markers, and an inset of the low-res thumbnail eye search actually saw.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from foveated_search.search.base import Trace

COL = {"gt": (40, 220, 90), "tile": (170, 170, 170), "tile_hit": (255, 176, 0), "eye": (255, 60, 200),
       "full_frame": (60, 150, 255), "downscale": (170, 110, 255), "sliding_window": (255, 176, 0),
       "eye_search": (255, 60, 200), "text": (240, 240, 240), "bg": (18, 18, 22), "bad": (255, 80, 80)}
SHORT = {"full_frame": "F", "downscale": "D", "sliding_window": "S", "eye_search": "E"}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        import matplotlib

        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def _dashed_rect(d: ImageDraw.ImageDraw, box, color, width=2, dash=8):
    x0, y0, x1, y1 = box
    for a, b, horiz, c in [(x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)]:
        t = a
        while t < b:
            e = min(t + dash, b)
            d.line((t, c, e, c) if horiz else (c, t, c, e), fill=color, width=width)
            t += 2 * dash


def _marker(d, x, y, color, label, font, r=9):
    d.ellipse((x - r, y - r, x + r, y + r), outline=(0, 0, 0), width=5)
    d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    d.line((x - r - 6, y, x + r + 6, y), fill=color, width=2)
    d.line((x, y - r - 6, x, y + r + 6), fill=color, width=2)
    tw = d.textlength(label, font=font)
    d.rectangle((x + r + 3, y - r - 16, x + r + 9 + tw, y - r + 2), fill=(0, 0, 0))
    d.text((x + r + 6, y - r - 16), label, fill=color, font=font)


def render(frame: Image.Image, traces: dict[str, Trace], meta: dict, thumb: Image.Image | None = None,
           scale: float = 1.5, summary_lines: list[str] | None = None) -> Image.Image:
    f_small, f_med, f_big = _font(13), _font(16), _font(20)
    W, H = round(frame.width * scale), round(frame.height * scale)
    img = frame.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(img)
    S = lambda v: v * scale  # noqa: E731

    for x, y, w, h in meta.get("gt_boxes", []):
        d.rectangle((S(x), S(y), S(x + w), S(y + h)), outline=COL["gt"], width=3)

    sw = traces.get("sliding_window")
    if sw:
        for i, b in enumerate(sw.crops):
            hit = sw.found and i == len(sw.crops) - 1 and not sw.params.get("parallel")
            d.rectangle(tuple(S(v) for v in b), outline=COL["tile_hit"] if hit else COL["tile"], width=3 if hit else 1)

    for name in ["full_frame", "downscale", "sliding_window", "eye_search"]:
        t = traces.get(name)
        if t and t.final_point:
            _marker(d, S(t.final_point[0]), S(t.final_point[1]), COL[name], SHORT[name], f_small)

    eye = traces.get("eye_search")
    if eye:
        crops = eye.crops
        for i, b in enumerate(crops):
            box = tuple(S(v) for v in b)
            confirmed = eye.found and i == len(crops) - 1
            if confirmed:
                d.rectangle(box, outline=COL["eye"], width=4)
            else:
                _dashed_rect(d, box, COL["eye"], width=3)
        pts = [(S(x), S(y)) for x, y in eye.fixations]
        if len(pts) > 1:
            d.line(pts, fill=COL["eye"], width=3)
        for i, (x, y) in enumerate(pts, 1):
            d.ellipse((x - 13, y - 13, x + 13, y + 13), fill=COL["eye"], outline=(0, 0, 0), width=2)
            t = str(i)
            d.text((x - d.textlength(t, font=f_med) / 2, y - 10), t, fill=(0, 0, 0), font=f_med)

    if thumb is not None:  # inset: what eye search saw first, pixelated (nearest) on purpose
        k = max(1, round(140 / thumb.width))
        inset = thumb.resize((thumb.width * k, thumb.height * k), Image.Resampling.NEAREST)
        cap = f"thumbnail {thumb.width}x{thumb.height}"
        x0, y0 = W - max(inset.width, round(d.textlength(cap, font=f_small))) - 12, 10
        d.rectangle((x0 - 3, y0 - 3, W - 8, y0 + inset.height + 20), fill=(0, 0, 0))
        img.paste(inset, (x0, y0))
        d.text((x0, y0 + inset.height + 2), cap, fill=COL["text"], font=f_small)

    head = 64
    lines = summary_lines or []
    foot = 10 + 20 * len(lines)
    out = Image.new("RGB", (W, H + head + foot), COL["bg"])
    out.paste(img, (0, head))
    o = ImageDraw.Draw(out)
    tag = "NEGATIVE (target absent)" if meta.get("is_negative") else f"{meta.get('num_instances', '?')} instance(s), {meta.get('size_bucket')}"
    o.text((10, 6), f"{meta.get('query_id', '')}  find: \"{meta.get('query', '')}\"  [{tag}]", fill=COL["text"], font=f_big)
    legend = [("ground truth", COL["gt"]), ("F full frame", COL["full_frame"]), ("D downscale", COL["downscale"]),
              ("S sliding window (tiles)", COL["sliding_window"]), ("E eye search 1->2->3", COL["eye_search"])]
    x = 10
    for text, c in legend:
        o.rectangle((x, 40, x + 12, 52), fill=c)
        o.text((x + 16, 37), text, fill=COL["text"], font=f_small)
        x += 26 + o.textlength(text, font=f_small)
    for i, line in enumerate(lines):
        o.text((10, H + head + 6 + 20 * i), line, fill=COL["bad"] if "WRONG" in line else COL["text"], font=f_small)
    return out


def save_png(img: Image.Image, path, colors: int = 224) -> None:
    """Small palette PNG that keeps the annotation colours exact: the photo gets `colors` palette
    entries, the remaining entries are reserved for COL (plain quantize turned blue markers grey)."""
    base = img.quantize(colors, method=Image.Quantize.MEDIANCUT)
    pal = base.getpalette()[: colors * 3]
    for c in COL.values():
        pal += list(c)
    pal += [0, 0, 0] * (256 - len(pal) // 3)
    pimg = Image.new("P", (1, 1))
    pimg.putpalette(pal)
    img.quantize(palette=pimg, dither=Image.Dither.NONE).save(path, optimize=True)
