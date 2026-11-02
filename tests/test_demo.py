"""Step 9 (GPU-free): photo intake (EXIF orientation, metadata stripping, resize math), trace -> replay conversion,
sent-image verification, page rendering, and the page's JS timeline logic (node, if installed)."""

import asyncio
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from foveated_search.config import REPO_ROOT
from foveated_search.demo import intake as I
from foveated_search.demo import replay as R
from foveated_search.search.base import run_strategy, search_config
from tests.mock_client import MockClient

RED = (255, 0, 0)


def phone_photo(w=400, h=300, orientation=6, gps=True, icc=False) -> tuple[Image.Image, bytes]:
    """Landscape pixels with a red top-left marker, plus EXIF bytes as a phone would write them."""
    im = Image.new("RGB", (w, h), (20, 90, 200))
    im.paste(RED, (0, 0, w // 8, h // 8))
    ex = Image.Exif()
    ex[I.ORIENTATION] = orientation
    ex[0x010F] = "PhoneMaker"
    ex[0x0110] = "PhoneModel X"
    if gps:
        ex.get_ifd(I.GPS_IFD).update({1: "N", 2: (37.0, 25.0, 19.2), 3: "W", 4: (122.0, 5.0, 6.0)})
    if icc:
        im.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return im, ex.tobytes()


def write_jpeg(path: Path, **kw) -> Path:
    im, exif = phone_photo(**kw)
    im.save(path, format="JPEG", quality=95, exif=exif, icc_profile=im.info.get("icc_profile"),
            comment=b"shot on a phone")
    return path


def red_corner(img: Image.Image) -> str:
    a = np.asarray(img).astype(int)
    h, w = a.shape[:2]
    pts = {"top-left": a[2, 2], "top-right": a[2, w - 3], "bottom-left": a[h - 3, 2], "bottom-right": a[h - 3, w - 3]}
    return min(pts, key=lambda k: np.abs(pts[k] - RED).sum())


# ---------------------------------------------------------------- resize math
@pytest.mark.parametrize("w,h,L,want", [
    (4032, 3024, 1920, (1920, 1440)),     # iPhone 4:3 landscape -> exactly server.image_max_pixels
    (3024, 4032, 1920, (1440, 1920)),     # portrait
    (4000, 2250, 1920, (1920, 1080)),     # 16:9
    (4000, 2251, 1920, (1920, 1080)),     # short side rounded, not floored to 1080.48 -> 1080
    (1920, 1080, 1920, (1920, 1080)),     # already at size
    (800, 600, 1920, (800, 600)),         # never upscaled
    (5000, 5000, 1920, (1920, 1920)),
])
def test_demo_size(w, h, L, want):
    assert I.demo_size(w, h, L) == want


def test_demo_size_rejects_empty():
    with pytest.raises(ValueError):
        I.demo_size(0, 10, 1920)


# ---------------------------------------------------------------- orientation + stripping
@pytest.mark.parametrize("orientation,size,corner", [
    (1, (400, 300), "top-left"), (3, (400, 300), "bottom-right"),
    (6, (300, 400), "top-right"), (8, (300, 400), "bottom-left"),
])
def test_exif_orientation_applied(tmp_path, orientation, size, corner):
    src = write_jpeg(tmp_path / "p.jpg", orientation=orientation)
    with Image.open(src) as im:
        out, rec = I.clean_image(im, 1920)
    assert out.size == size and red_corner(out) == corner
    assert rec["orientation_tag"] == orientation and rec["demo_size"] == list(size)


def test_metadata_stripped_and_verified(tmp_path):
    src = write_jpeg(tmp_path / "p.jpg", icc=True)
    probs = I.metadata_problems(src)                     # the checker sees everything on the original
    assert "GPS IFD" in probs and any("APP1" in p for p in probs) and "ICC profile" in probs
    with Image.open(src) as im:
        assert I.describe_metadata(im)["has_gps"]
        out, rec = I.clean_image(im, 1920)
    assert rec["stripped"]["has_gps"] and rec["stripped"]["has_icc"] and rec["stripped"]["exif_tags"] >= 3
    dst = tmp_path / "clean.jpg"
    I.save_clean(out, dst, 92)
    assert I.metadata_problems(dst) == []
    with Image.open(dst) as im:
        ex = im.getexif()
        assert len(ex) == 0 and I.GPS_IFD not in ex and not ex.get_ifd(I.GPS_IFD)
        assert "exif" not in im.info and "icc_profile" not in im.info and "comment" not in im.info
    raw = dst.read_bytes()
    assert b"Exif\x00\x00" not in raw and b"PhoneModel" not in raw and b"shot on a phone" not in raw
    assert I.jpeg_header_markers(raw)[0] == 0xE0 and all(m < 0xE1 or m > 0xEF for m in I.jpeg_header_markers(raw))


def test_png_metadata_stripped(tmp_path):
    from PIL import PngImagePlugin
    im, exif = phone_photo(orientation=1)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Location", "somewhere private")
    src = tmp_path / "p.png"
    im.save(src, pnginfo=meta, exif=exif)
    with Image.open(src) as o:
        out, _ = I.clean_image(o, 1920)
    dst = tmp_path / "c.jpg"
    I.save_clean(out, dst, 92)
    assert I.metadata_problems(dst) == [] and b"somewhere private" not in dst.read_bytes()


def test_heic_orientation_not_applied_twice(tmp_path):
    pytest.importorskip("pillow_heif")
    assert I.register_heif()
    im, exif = phone_photo(orientation=6)
    src = tmp_path / "p.heic"
    im.save(src, exif=exif, quality=95)
    with Image.open(src) as o:
        out, rec = I.clean_image(o, 1920)
    assert out.size == (300, 400) and red_corner(out) == "top-right"   # rotated once, like the JPEG
    assert rec["orientation_tag"] == 6 and rec["stripped"]["has_gps"]
    dst = tmp_path / "c.jpg"
    I.save_clean(out, dst, 92)
    assert I.metadata_problems(dst) == []


def test_intake_one_writes_clean_copy_and_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "REPO_ROOT", tmp_path)
    cfg = I.demo_config()
    cfg["demo"]["clean_dir"] = "clean"
    src = write_jpeg(tmp_path / "IMG_1.jpg", w=4032, h=3024)
    rec = I.intake_one(src, cfg, processor_check=False)
    out = tmp_path / "clean" / "IMG_1.jpg"
    side = json.loads((tmp_path / "clean" / "IMG_1.json").read_text())
    assert side == rec and rec["original_size"] == [4032, 3024] and rec["demo_size"] == [1440, 1920]
    assert rec["clean_file"] == "clean/IMG_1.jpg" and rec["stripped"]["has_gps"]
    assert "PhoneModel" not in json.dumps(side) and "37" not in json.dumps(side["stripped"])   # no values recorded
    with Image.open(out) as im:
        assert im.size == (1440, 1920) and len(im.getexif()) == 0


def test_gitignore_keeps_originals_out(tmp_path):
    """Originals in test_pictures/ are ignored; cleaned copies are not."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    def ignored(p):
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", p], cwd=REPO_ROOT).returncode == 0
    assert ignored("test_pictures/IMG_0001.HEIC") and ignored("test_pictures/IMG_0001.jpg")
    assert not ignored("test_pictures/clean/IMG_0001.jpg") and not ignored("test_pictures/clean/IMG_0001.json")


# ---------------------------------------------------------------- trace -> replay
def demo_cfg():
    return search_config(overlays=["configs/demo.yaml"])


def frame_img(w=1280, h=960, seed=1):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


ANSWERS = ['{"point_2d":[100,100]}', '{"y":false}',
           '{"point_2d":[110,110]}',                       # repeat: inside the crop already checked
           '{"point_2d":[800,700]}', '{"y":true,"point_2d":[500,500],"l":"a red cup"}']


def eye_trace(cfg, frame, answers=ANSWERS):
    return asyncio.run(run_strategy("eye_search", frame, "cup", MockClient(list(answers)), cfg)).to_dict()


def test_to_replay_eye_search_timeline():
    cfg, fr = demo_cfg(), frame_img()
    tr = eye_trace(cfg, fr)
    rp = R.to_replay(tr)
    ev = rp["events"]
    assert [e["role"] for e in ev] == ["point", "confirm", "point", "point", "confirm"]
    # call k starts when call k-1 ends; the timeline ends at time_to_answer
    assert ev[0]["t0"] == 0 and all(abs(a["t1"] - b["t0"]) < 1e-3 for a, b in zip(ev, ev[1:]))
    assert abs(rp["t_end"] - tr["time_to_answer_s"]) < 1e-3
    assert ev[0]["point"] == tr["fixations"][0] and not ev[0]["masked"] and not ev[0]["repeat"]
    assert ev[1]["yes"] is False and ev[1]["region"] == tr["crops"][0]
    assert ev[2]["repeat"] and ev[2]["masked"] and ev[2]["checked"] == [tr["crops"][0]]
    assert ev[3]["point"] == tr["fixations"][1] and not ev[3]["repeat"]
    assert ev[4]["yes"] and ev[4]["label"] == "a red cup"
    assert rp["found"] and rp["final_point"] == ev[4]["point"] == tr["final_point"]
    assert sum(e["pixels"] for e in ev) == tr["pixels_sent"] and sum(e["bytes"] for e in ev) == tr["png_bytes"]


def test_to_replay_absent_and_full_frame():
    cfg, fr = demo_cfg(), frame_img()
    rp = R.to_replay(eye_trace(cfg, fr, ['{"absent":true}']))
    assert not rp["found"] and len(rp["events"]) == 1 and rp["events"][0]["point"] is None
    ff = asyncio.run(run_strategy("full_frame", fr, "cup", MockClient(['{"point_2d":[250,500]}']), cfg)).to_dict()
    rp = R.to_replay(ff)
    assert rp["found"] and rp["events"][0]["point"] == [320.0, 480.0] == rp["final_point"]
    assert rp["events"][0]["sent"] == [1280, 960] and rp["events"][0]["pixels"] == 1280 * 960


def fake_run(cfg, fr, path: Path) -> dict:
    fr.save(path)
    traces = {"full_frame": asyncio.run(run_strategy("full_frame", fr, "cup", MockClient(['{"absent":true}']), cfg)).to_dict(),
              "eye_search": eye_trace(cfg, fr)}
    s = cfg["search"]
    import hashlib
    return {"run_id": "t__cup", "image": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_size": list(fr.size), "query": "cup", "model": "m", "precision": "FP8", "gpu": "L4", "concurrency": 1,
            "warmup_rounds": 1, "created": "now", "summary": {k: {"evidence": "e", "status": "x"} for k in traces},
            "settings": {"thumbnail_long_side": s["thumbnail_long_side"], "fixation_size": s["fixation_size"],
                         "max_fixations": s["max_fixations"], "look_again_mask": s["look_again_mask"],
                         "look_again_mask_rgb": s["look_again_mask_rgb"], "png_compress_level": 6},
            "traces": traces}


def test_sent_images_verified_and_tamper_detected(tmp_path):
    cfg, fr = demo_cfg(), frame_img()
    rec = fake_run(cfg, fr, tmp_path / "f.png")
    pngs = R.sent_images(rec, fr)
    assert len(pngs["eye_search"]) == 5 and len(pngs["full_frame"]) == 1
    assert pngs["eye_search"][0][1] != pngs["eye_search"][2][1]          # fixation 2 thumbnail carries the mask
    bad = fr.copy()
    bad.putpixel((5, 5), (0, 0, 0))
    with pytest.raises(R.ImageMismatch):
        R.sent_images(rec, bad)


def test_render_is_self_contained(tmp_path):
    cfg, fr = demo_cfg(), frame_img(640, 480)
    rec = fake_run(cfg, fr, tmp_path / "f.png")
    rec["query"] = "cup</script><script>alert(1)"
    payload = R.build_payload([rec], R.replay_config())
    html = R.render(payload)
    assert R.PLACEHOLDER not in html and "</script><script>alert" not in html
    data = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(data)["runs"][0]["query"].startswith("cup</script>")
    urls = set(re.findall(r"https?://[^\s\"')]+", html))
    assert urls <= {"http://www.w3.org/2000/svg"}                         # namespace string only, no requests
    assert payload["verified_images"] == 6 and set(payload["images"]) == {e["sha"] for e in
                                                                        payload["runs"][0]["panels"]["eye_search"]["events"]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_timeline_logic_in_node(tmp_path):
    js = re.search(r'<script id="logic">(.*?)</script>', R.TEMPLATE.read_text(), re.S).group(1)
    cfg, fr = demo_cfg(), frame_img()
    panel = R.to_replay(eye_trace(cfg, fr))
    (tmp_path / "logic.js").write_text(js)
    (tmp_path / "panel.json").write_text(json.dumps(panel))
    e = panel["events"]
    probe = f"""
const L = require('./logic.js'); const P = require('./panel.json');
const at = (t) => L.stateAt(P, t);
const out = {{ start: at(0), mid: at({(e[1]['t0'] + e[1]['t1']) / 2}), end: at({panel['t_end']}), after: at(99),
  thumbStart: L.thumbAt(P, 0), thumbLate: L.thumbAt(P, {e[3]['t0']}), bytes: L.fmtBytes(1234567) }};
console.log(JSON.stringify(out));"""
    (tmp_path / "probe.js").write_text(probe)
    r = json.loads(subprocess.run(["node", "probe.js"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout)
    assert r["start"]["sent"] == [0] and r["start"]["done"] == [] and r["start"]["current"] == 0
    assert r["start"]["pixels"] == e[0]["pixels"] and not r["start"]["finished"]
    assert r["mid"]["sent"] == [0, 1] and r["mid"]["done"] == [0] and r["mid"]["current"] == 1
    assert r["end"]["finished"] and r["end"]["done"] == [0, 1, 2, 3, 4]
    assert r["end"]["tokens"] == sum(x["vision_tokens"] for x in e) and r["end"]["bytes"] == sum(x["bytes"] for x in e)
    assert abs(r["after"]["timer"] - panel["t_end"]) < 1e-9          # timer stops at the measured time to answer
    assert r["thumbStart"] == 0 and r["thumbLate"] == 3 and r["bytes"] == "1.23 MB"


# ---------------------------------------------------------------- repeats and scoring (run_demo)
def test_median_index_picks_the_median_run():
    from foveated_search.demo.run_demo import median_index
    assert median_index([0.9, 0.5, 0.7, 0.6, 0.8]) == 2          # odd count: the median run itself
    assert median_index([0.5, 0.5, 0.5]) == 0                     # ties: first
    assert median_index([0.4, 0.6, 1.0, 0.5]) in (1, 3)           # even count: one of the two middle runs


def test_score_present_absent_and_unscored():
    from foveated_search.demo.run_demo import score
    box = [[100, 100, 200, 200]]
    assert score({"status": "found", "final_point": [150, 150]}, "present", box) is True
    assert score({"status": "found", "final_point": [250, 150]}, "present", box) is False   # wrong place
    assert score({"status": "absent", "final_point": None}, "present", box) is False
    assert score({"status": "absent", "final_point": None}, "absent", None) is True
    assert score({"status": "found", "final_point": [1, 1]}, "absent", None) is False      # false positive
    assert score({"status": "found", "final_point": [1, 1]}, None, None) is None
    with pytest.raises(ValueError):
        score({"status": "found", "final_point": [1, 1]}, "maybe", box)


def test_answer_note_only_for_wrong_answers_in_a_region():
    from foveated_search.demo.run_demo import answer_note, notes_for
    regions = [{"box": [0, 0, 10, 10], "queries": ["food"], "note": "packaged sweets"}]
    wrong = {"final_point": [5, 5], "correct": False}
    assert answer_note(wrong, "food", regions) == "packaged sweets"
    assert answer_note(wrong, "mouse", regions) is None                         # other query
    assert answer_note({"final_point": [50, 5], "correct": False}, "food", regions) is None   # outside the region
    assert answer_note({"final_point": [5, 5], "correct": True}, "food", regions) is None    # correct: no note
    assert answer_note({"final_point": None, "correct": False}, "food", regions) is None     # absent: no point
    spec = {"notes": {"test_pictures/clean/a.jpg": regions}}
    assert notes_for(spec, "test_pictures/clean/a.jpg") == regions == notes_for(spec, "/elsewhere/a.jpg")
    assert notes_for(spec, "b.jpg") == []
