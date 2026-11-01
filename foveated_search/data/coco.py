"""COCO 2017 annotations: download (local disk), verify, extract, and a small index for querying.

Never downloads image zips. Images are fetched one by one by URL (see fetch_images.py).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from foveated_search.config import Config


def coco_dir(cfg: Config) -> Path:
    return Path(cfg.get_key("paths.data_dir")) / "coco"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_annotations(cfg: Config) -> dict[str, Path]:
    """Download + verify the annotations zip once, extract the configured JSONs. Idempotent."""
    c = cfg["coco"]
    root = coco_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / Path(c["annotations_url"]).name
    want_bytes, want_md5 = int(c["annotations_bytes"]), str(c["annotations_md5"])
    out = {Path(m).name: root / Path(m).name for m in c["extract"]}

    if all(p.exists() and p.stat().st_size > 0 for p in out.values()):
        print(f"[coco] annotations present in {root}, skip download")
        return out

    if not (zpath.exists() and zpath.stat().st_size == want_bytes):
        print(f"[coco] downloading {c['annotations_url']} ({want_bytes / 1e6:.0f} MB) -> {zpath}")
        tmp = zpath.with_suffix(".part")
        with urllib.request.urlopen(c["annotations_url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(zpath)
    size = zpath.stat().st_size
    if size != want_bytes:
        raise RuntimeError(f"annotations zip size {size} != expected {want_bytes}")
    md5 = _md5(zpath)
    if md5 != want_md5:
        raise RuntimeError(f"annotations zip md5 {md5} != expected {want_md5}")
    print(f"[coco] verified zip: {size} bytes, md5 {md5}")

    with zipfile.ZipFile(zpath) as z:
        for member in c["extract"]:
            dest = out[Path(member).name]
            with z.open(member) as src, open(dest.with_suffix(".part"), "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)  # zipfile checks the CRC on read
            dest.with_suffix(".part").rename(dest)
            print(f"[coco] extracted {member} ({dest.stat().st_size / 1e6:.0f} MB)")
    zpath.unlink()  # the two JSONs are all we need; the zip is re-downloadable
    return out


@dataclass
class CocoIndex:
    images: dict[int, dict]                       # image_id -> image record
    cats: dict[int, dict]                         # category_id -> {"id", "name", "supercategory"}
    anns_by_image: dict[int, list[dict]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CocoIndex":
        idx = cls(images={im["id"]: im for im in d["images"]}, cats={c["id"]: c for c in d["categories"]})
        for a in d["annotations"]:
            idx.anns_by_image.setdefault(a["image_id"], []).append(a)
        for anns in idx.anns_by_image.values():
            anns.sort(key=lambda a: a["id"])
        return idx

    @classmethod
    def load(cls, path: str | Path) -> "CocoIndex":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def supercat_members(self, supercat: str) -> set[int]:
        return {cid for cid, c in self.cats.items() if c["supercategory"] == supercat}
