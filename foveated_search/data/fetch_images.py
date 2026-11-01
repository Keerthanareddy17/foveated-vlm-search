"""Fetch only the eval images the manifest uses, by URL (never the image zip).

Drive (persistent, small): paths.drive_eval_images_dir/<file_name>
Local (fast reads, wiped each session): paths.data_dir/<fetch.local_images_subdir>/<file_name>
Idempotent: an image is skipped if its copy opens with PIL and matches the annotated size.
New session: local copies are restored from Drive, not re-downloaded.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from foveated_search.config import REPO_ROOT, Config


def verify_image(path: Path, width: int, height: int) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:
            im.load()    # full decode (catches truncated files)
            return im.size == (width, height)
    except Exception:
        return False


def local_eval_dir(cfg: Config) -> Path:
    return Path(cfg.get_key("paths.data_dir")) / cfg.get_key("fetch.local_images_subdir")


def eval_image_path(cfg: Config, file_name: str) -> Path:
    """Fast local path if present, else the Drive copy."""
    p = local_eval_dir(cfg) / file_name
    return p if p.exists() else Path(cfg.get_key("paths.drive_eval_images_dir")) / file_name


def _download(url: str, dest: Path, width: int, height: int, f: dict) -> None:
    """Download to dest via a .part file, verify (PIL decode + size), retry with backoff."""
    last = None
    for attempt in range(int(f["retries"])):
        try:
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=float(f["timeout_s"])) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out)
            if not verify_image(tmp, width, height):
                raise ValueError(f"{dest.name}: does not open as a {width}x{height} image")
            tmp.replace(dest)
            return
        except Exception as e:  # network or verify failure -> retry with backoff
            last = e
            time.sleep(float(f["backoff_s"]) * 2**attempt)
    raise RuntimeError(f"failed to fetch {dest.name} after {f['retries']} attempts: {last}")


def fetch_to_dir(rows: list[dict], out_dir: Path, f: dict, workers: int | None = None) -> dict:
    """Local-only fetch (training images, storage policy): rows need file_name, coco_url, width, height.
    Idempotent: verified files are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq = {r["file_name"]: r for r in rows}

    def one(r: dict) -> str:
        dest = out_dir / r["file_name"]
        if dest.exists() and verify_image(dest, r["width"], r["height"]):
            return "skipped"
        _download(r["coco_url"], dest, r["width"], r["height"], f)
        return "downloaded"

    t0 = time.perf_counter()
    status: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=int(workers or f["workers"])) as ex:
        for i, fu in enumerate(as_completed([ex.submit(one, r) for r in uniq.values()]), 1):
            s = fu.result()
            status[s] = status.get(s, 0) + 1
            if i % 1000 == 0:
                print(f"[fetch] {i}/{len(uniq)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    dt = time.perf_counter() - t0
    size = sum((out_dir / n).stat().st_size for n in uniq)
    print(f"[fetch] {len(uniq)} images in {dt:.1f}s: {status}; all verified. {size / 1e6:.1f} MB at {out_dir}")
    return {"n": len(uniq), "status": status, "seconds": round(dt, 1), "mb": round(size / 1e6, 1)}


def _fetch_one(row: dict, drive_dir: Path, local_dir: Path, f: dict) -> str:
    name, w, h = row["file_name"], row["width"], row["height"]
    drive, local = drive_dir / name, local_dir / name
    if drive.exists() and verify_image(drive, w, h):
        if not (local.exists() and verify_image(local, w, h)):
            shutil.copyfile(drive, local)
            return "restored-local"
        return "skipped"
    _download(row["coco_url"], local, w, h, f)
    dtmp = drive.with_suffix(".part")
    shutil.copyfile(local, dtmp)
    dtmp.replace(drive)
    return "downloaded"


def fetch_eval_images(cfg: Config, rows: list[dict]) -> dict:
    f = cfg["fetch"]
    drive_dir = Path(cfg.get_key("paths.drive_eval_images_dir"))
    if not drive_dir.parent.parent.exists():
        raise SystemExit(f"Drive project folder missing ({drive_dir.parent.parent}): run the notebook setup cell")
    local_dir = local_eval_dir(cfg)
    drive_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    uniq = {r["file_name"]: r for r in rows}

    t0 = time.perf_counter()
    status: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=int(f["workers"])) as ex:
        futs = [ex.submit(_fetch_one, r, drive_dir, local_dir, f) for r in uniq.values()]
        for fu in as_completed(futs):
            s = fu.result()
            status[s] = status.get(s, 0) + 1
    dt = time.perf_counter() - t0

    # Final verification of every Drive copy + sha256 list (freezes the exact pixels used).
    lines, bad = [], []
    for name in sorted(uniq):
        r = uniq[name]
        p = drive_dir / name
        if not verify_image(p, r["width"], r["height"]):
            bad.append(name)
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {name}\n")
    if bad:
        raise RuntimeError(f"{len(bad)} eval images failed verification: {bad[:5]}")
    (REPO_ROOT / cfg.get_key("outputs.image_sha256")).write_text("".join(lines))
    size = sum((drive_dir / n).stat().st_size for n in uniq)
    print(f"[fetch] {len(uniq)} images in {dt:.1f}s: {status}; all verified (PIL decode + size). "
          f"Drive folder {size / 1e6:.1f} MB at {drive_dir}")
    return {"n": len(uniq), "status": status, "seconds": round(dt, 1), "drive_mb": round(size / 1e6, 1)}
