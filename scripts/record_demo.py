#!/usr/bin/env python
"""Step 10: record the replay page to docs/assets/demo.gif and docs/assets/demo.mp4 (`make record-demo`).

Deterministic: instead of filming the page while it plays, every frame is the page's own state at an exact time
of the measured timeline (the replay's scrubber is set to t, then the page is screenshotted). Frames are taken at
FPS frames per second of real (measured) time, so the video plays at 1x real time; holds are frame durations.
Sequence (configs in SEQUENCE below): positive2 · food to the end, hold, then negative · apple to the end, hold.
Runs in .venv-record (Playwright + headless Chromium + Pillow); MP4 with ffmpeg, GIF with Pillow.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
VIEWPORT = {"width": 1280, "height": 1000}
SCALE = 1.5                     # device pixels per CSS pixel (sharp MP4; the GIF is scaled down)
SEQUENCE = [("positive2__food", 1.0 + 2.0), ("negative__apple", 3.0)]   # (run, hold at the end in s)
GIF_WIDTH = 960
GIF_FPS = 15


def capture(page_url: str, frames_dir: Path) -> list[tuple[Path, float]]:
    """Returns [(frame png, duration s)]."""
    out: list[tuple[Path, float]] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(page_url)
        pg.wait_for_timeout(300)
        def scrub(v):
            pg.evaluate(f"document.getElementById('scrub').value = {v}; "
                        "document.getElementById('scrub').dispatchEvent(new Event('input'))")
        # one clip height for every frame (the tallest layout in the sequence), so all frames have the same size
        height = 0
        for run, _ in SEQUENCE:
            pg.select_option("#run", run)
            pg.wait_for_timeout(100)
            scrub(1000)
            box = pg.locator("#panels").bounding_box()
            height = max(height, box["y"] + box["height"] + 8)
        clip = {"x": 0, "y": 0, "width": VIEWPORT["width"], "height": height}
        n = 0
        for run, hold in SEQUENCE:
            pg.select_option("#run", run)
            pg.wait_for_timeout(100)
            pg.click("#play")          # select() autoplays; pause it, then drive the scrubber
            scrub(1000)                # tMax = the slower panel's measured time: read the clock at the end
            t_max = float(pg.inner_text("#clock").split()[0])
            times = [i / FPS for i in range(int(t_max * FPS) + 1)]
            if times[-1] < t_max:
                times.append(t_max)    # always end on the exact final state (both answers shown)
            for k, t in enumerate(times):
                scrub(round(1000 * t / t_max))
                f = frames_dir / f"f{n:05d}.png"
                pg.screenshot(path=str(f), clip=clip)
                nxt = times[k + 1] if k + 1 < len(times) else t
                out.append((f, max(nxt - t, 1 / FPS) if k + 1 < len(times) else 1 / FPS))
                n += 1
            steps = len(times) - 1
            out[-1] = (out[-1][0], out[-1][1] + hold)
            print(f"  {run}: {steps + 1} frames over {t_max:.3f} s measured time, hold {hold:.1f} s")
        b.close()
        if errors:
            raise SystemExit(f"page errors while recording: {errors}")
    return out


def encode(frames: list[tuple[Path, float]], frames_dir: Path, mp4: Path, gif: Path) -> None:
    lst = frames_dir / "frames.txt"
    lines = []
    for f, d in frames:
        lines += [f"file '{f.name}'", f"duration {d:.5f}"]
    lines.append(f"file '{frames[-1][0].name}'")          # concat demuxer needs the last file repeated
    lst.write_text("\n".join(lines) + "\n")
    common = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    subprocess.run(common + ["-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow",
                             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    write_gif(frames, gif)


def write_gif(frames: list[tuple[Path, float]], gif: Path) -> None:
    """GIF with Pillow: GIF_FPS (every other 30 fps frame, durations merged), one global palette from four sampled
    frames, Floyd-Steinberg dither, holds as single long frames. (ffmpeg 6.1's paletteuse failed on this input with
    'Internal bug', and repeated hold frames made its GIF twice the size limit.)"""
    from PIL import Image

    step = FPS // GIF_FPS
    merged: list[tuple[Path, float]] = []
    for k, (f, d) in enumerate(frames):
        if merged and k % step and d < 2 / FPS:
            merged[-1] = (merged[-1][0], merged[-1][1] + d)
        else:
            merged.append((f, d))

    def load(f):
        im = Image.open(f).convert("RGB")
        return im.resize((GIF_WIDTH, round(im.height * GIF_WIDTH / im.width)), Image.Resampling.LANCZOS)
    imgs = [load(f) for f, _ in merged]
    w, h = imgs[0].size
    sample = Image.new("RGB", (w, 4 * h))
    for j, k in enumerate([0, len(imgs) // 3, 2 * len(imgs) // 3, len(imgs) - 1]):
        sample.paste(imgs[k], (0, j * h))
    pal = sample.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for im in imgs]
    q[0].save(gif, save_all=True, append_images=q[1:], duration=[round(1000 * d) for _, d in merged], loop=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--page", type=Path, default=ROOT / "results/demo/replay.html")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/assets")
    ap.add_argument("--work", type=Path, default=Path("/content/fvs_local/record_demo"))
    a = ap.parse_args()
    if a.work.exists():
        shutil.rmtree(a.work)
    a.work.mkdir(parents=True)
    print(f"recording {a.page.relative_to(ROOT)} at {VIEWPORT['width']} px wide x{SCALE}, {FPS} fps of measured time")
    frames = capture(a.page.resolve().as_uri(), a.work)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    mp4, gif = a.out_dir / "demo.mp4", a.out_dir / "demo.gif"
    encode(frames, a.work, mp4, gif)
    total = sum(d for _, d in frames)
    print(f"{mp4.relative_to(ROOT)} {mp4.stat().st_size / 1e6:.2f} MB | {gif.relative_to(ROOT)} "
          f"{gif.stat().st_size / 1e6:.2f} MB | {total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
