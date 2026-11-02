"""Step 9 photo intake: phone photo -> upright, metadata-free, resized demo copy.

  python -m foveated_search.demo.intake            (every photo in demo.input_dir)
  python -m foveated_search.demo.intake IMG.HEIC   (one file)

For each original in test_pictures/ (JPG / PNG / HEIC):
1. apply the EXIF orientation (pillow-heif already applies HEIF rotation on open and resets the tag to 1, so
   exif_transpose is then a no-op: no double rotation; NOTES.md Step 9),
2. convert an embedded ICC profile to sRGB (phones shoot Display P3; the model and browsers assume sRGB),
3. rebuild the image from raw pixels, so no EXIF / GPS / XMP / ICC / maker notes can be carried over,
4. resize so the long side is demo.long_side (never upscaled),
5. save test_pictures/clean/<stem>.jpg + <stem>.json (sizes, orientation applied, what was stripped),
6. re-open the output and FAIL if any metadata survived (EXIF, GPS IFD, XMP, ICC, comments).
Only test_pictures/clean/ is committed; the originals are gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

from foveated_search.config import REPO_ROOT, Config, load_config

GPS_IFD = 0x8825
ORIENTATION = 0x0112


def demo_config(overlays: list[str] | None = None) -> Config:
    cfg = load_config(overlays=["configs/demo.yaml", *(overlays or [])])
    for k in ["demo.input_dir", "demo.clean_dir", "demo.long_side", "demo.jpeg_quality", "demo.extensions",
              "demo.to_srgb"]:
        cfg.get_key(k)
    return cfg


def register_heif() -> bool:
    try:
        import pillow_heif
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    return True


def demo_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Size with the long side = long_side, aspect kept (short side rounded), never upscaled."""
    if w <= 0 or h <= 0:
        raise ValueError(f"bad size {w}x{h}")
    if max(w, h) <= long_side:
        return w, h
    s = long_side / max(w, h)
    return (long_side, max(1, round(h * s))) if w >= h else (max(1, round(w * s)), long_side)


def describe_metadata(im: Image.Image) -> dict:
    """What metadata an opened image carries (presence only: never the GPS values or device strings)."""
    ex = im.getexif()
    gps = ex.get_ifd(GPS_IFD) if GPS_IFD in ex else {}
    orient = im.info.get("original_orientation", ex.get(ORIENTATION))
    return {
        "exif_tags": len(ex),
        "has_gps": bool(gps) or GPS_IFD in ex,
        "orientation": int(orient) if orient is not None else None,
        "has_xmp": bool(im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")),
        "has_icc": bool(im.info.get("icc_profile")),
        "has_exif_blob": bool(im.info.get("exif")),
    }


def metadata_problems(path: Path) -> list[str]:
    """Everything metadata-like found in a written file; [] means clean. Checks the decoded tags AND raw bytes."""
    probs = []
    with Image.open(path) as im:
        m = describe_metadata(im)
        if m["exif_tags"]:
            probs.append(f"{m['exif_tags']} EXIF tags")
        if m["has_gps"]:
            probs.append("GPS IFD")
        if m["has_xmp"]:
            probs.append("XMP")
        if m["has_icc"]:
            probs.append("ICC profile")
        if m["has_exif_blob"]:
            probs.append("EXIF blob")
        if im.info.get("comment"):
            probs.append("comment")
        text_keys = [k for k in im.info if k not in ("jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi",
                                                     "progressive", "progression", "adobe", "adobe_transform")]
        if text_keys:
            probs.append(f"info keys {text_keys}")
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xd8":
        extra = [f"APP{m - 0xE0}" if 0xE0 <= m <= 0xEF else "COM" for m in jpeg_header_markers(raw)
                 if (0xE1 <= m <= 0xEF) or m == 0xFE]
        if extra:
            probs.append(f"JPEG metadata segments {extra}")   # APP1 = EXIF/XMP, APP2 = ICC, COM = comment
    return probs


def jpeg_header_markers(raw: bytes) -> list[int]:
    """Marker bytes of every JPEG segment before the image data (start of scan)."""
    out, i = [], 2
    while i + 4 <= len(raw) and raw[i] == 0xFF:
        m = raw[i + 1]
        if m == 0xDA:   # start of scan: compressed data follows
            break
        out.append(m)
        i += 2 + int.from_bytes(raw[i + 2:i + 4], "big")
    return out


def to_srgb(im: Image.Image) -> tuple[Image.Image, str | None]:
    """Convert an embedded ICC profile to sRGB. Returns (image, source profile description or None)."""
    icc = im.info.get("icc_profile")
    if not icc:
        return im, None
    from PIL import ImageCms

    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    desc = ImageCms.getProfileDescription(src).strip()
    out = ImageCms.profileToProfile(im.convert("RGB"), src, ImageCms.createProfile("sRGB"), outputMode="RGB")
    return out, desc


def clean_image(im: Image.Image, long_side: int, srgb: bool = True) -> tuple[Image.Image, dict]:
    """Upright, sRGB, metadata-free, resized copy of an opened image + a record of what was done."""
    before = describe_metadata(im)
    orig_size = im.size
    up = ImageOps.exif_transpose(im)        # no-op if the orientation tag is 1 / absent (HEIC: already applied)
    if up is None:
        up = im
    profile = None
    if srgb:
        up, profile = to_srgb(up)
    if up.mode != "RGB":
        up = up.convert("RGB")
    bare = Image.frombytes("RGB", up.size, up.tobytes())   # pixels only: no info dict, no exif, no icc
    size = demo_size(*bare.size, long_side)
    if size != bare.size:
        bare = bare.resize(size, Image.Resampling.LANCZOS)
    rec = {
        "original_size": list(orig_size),            # as stored, before orientation
        "upright_size": list(up.size),
        "demo_size": list(bare.size),
        "orientation_tag": before["orientation"],
        "rotated": list(up.size) != list(orig_size) or (before["orientation"] or 1) != 1,
        "icc_converted_from": profile,
        "stripped": {k: before[k] for k in ("exif_tags", "has_gps", "has_xmp", "has_icc")},
    }
    return bare, rec


def save_clean(img: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No exif=/icc_profile= arguments, 4:4:4 chroma, baseline: deterministic, metadata-free JPEG.
    img.save(path, format="JPEG", quality=int(quality), subsampling=0, optimize=False, progressive=False)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def intake_one(src: Path, cfg: Config, processor_check: bool = True) -> dict:
    d = cfg["demo"]
    clean_dir = REPO_ROOT / d["clean_dir"]
    out = clean_dir / f"{src.stem}.jpg"
    with Image.open(src) as im:
        im.load()
        fmt = im.format
        img, rec = clean_image(im, int(d["long_side"]), bool(d["to_srgb"]))
    save_clean(img, out, int(d["jpeg_quality"]))
    probs = metadata_problems(out)
    if probs:
        out.unlink()
        raise RuntimeError(f"{out.name}: metadata survived stripping: {probs}")
    rec.update({"source_file": src.name, "source_format": fmt, "source_bytes": src.stat().st_size,
                "clean_file": out.relative_to(REPO_ROOT).as_posix(), "clean_bytes": out.stat().st_size,
                "clean_sha256": sha256_file(out), "metadata_check": "clean (no EXIF, GPS, XMP, ICC)",
                "long_side_setting": int(d["long_side"]), "jpeg_quality": int(d["jpeg_quality"])})
    if processor_check:
        rec["processor"] = processor_sizes(cfg, img)
    (clean_dir / f"{src.stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def processor_sizes(cfg: Config, img: Image.Image) -> dict:
    """What the real processor does to the demo copy at the demo server's pixel limits. Fails on hidden downscaling."""
    from foveated_search.serving.vision_tokens import VisionTokenCounter

    serve = load_config(overlays=["configs/serve_demo.yaml", "configs/serve_hires.yaml"], set_seed=False)
    c = VisionTokenCounter(serve.get_key("model.id"), int(serve.get_key("server.image_min_pixels")),
                           int(serve.get_key("server.image_max_pixels")))
    info = c.count(img)
    # smart_resize rounds each side to the 32-px grid (e.g. 1080 -> 1088); more than one grid step down = downscaled
    downscaled = info.resized_w < img.width - 32 or info.resized_h < img.height - 32
    if downscaled:
        raise RuntimeError(f"processor would downscale {img.width}x{img.height} -> {info.resized_w}x{info.resized_h}; "
                           "lower demo.long_side")
    return {"resized": [info.resized_w, info.resized_h], "vision_tokens": info.tokens,
            "max_pixels": int(serve.get_key("server.image_max_pixels"))}


def find_inputs(cfg: Config) -> list[Path]:
    d = cfg["demo"]
    root = REPO_ROOT / d["input_dir"]
    exts = {e.lower() for e in d["extensions"]}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts) if root.exists() else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="originals (default: every photo in demo.input_dir)")
    ap.add_argument("--no-processor-check", action="store_true")
    a = ap.parse_args(argv)
    cfg = demo_config()
    heif = register_heif()
    files = [Path(f) for f in a.files] or find_inputs(cfg)
    if not files:
        print(f"no photos in {cfg['demo']['input_dir']}/ ({', '.join(cfg['demo']['extensions'])})")
        return 1
    if any(f.suffix.lower() in (".heic", ".heif") for f in files) and not heif:
        print("pillow-heif is not installed in this venv: `make setup` (requirements/serve.lock)")
        return 1
    for f in files:
        r = intake_one(f, cfg, processor_check=not a.no_processor_check)
        s = r["stripped"]
        p = r.get("processor", {})
        print(f"{f.name}: {r['source_format']} {r['original_size'][0]}x{r['original_size'][1]} "
              f"(orientation {r['orientation_tag']}) -> {r['clean_file']} {r['demo_size'][0]}x{r['demo_size'][1]}, "
              f"{r['clean_bytes'] / 1e3:.0f} KB | stripped: {s['exif_tags']} EXIF tags, GPS {s['has_gps']}, "
              f"XMP {s['has_xmp']}, ICC {s['has_icc']} ({r['icc_converted_from'] or 'none'} -> sRGB) | "
              f"output check: {r['metadata_check']}"
              + (f" | processor {p['resized'][0]}x{p['resized'][1]}, {p['vision_tokens']} vision tokens" if p else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
