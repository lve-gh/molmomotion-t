"""Build the 2x2 comparison grid animation (real / tracking / DaS trajectory / DaS static control)
for the "other clips" (flamingo, car-turn), the same way part3_comparison() in make_expected_media.py
does it for bmx-trees. Writes docs/expected/part3_<clip>_comparison_2x2.{mp4,gif}.

Run from the repo root, in the DaS environment (needs imageio/cv2):
    venv_das/Scripts/python.exe scripts_local/make_other_clips_grid.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "expected"
OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
N_FRAMES = 49


def read_video(p):
    r = imageio.get_reader(p)
    fs = [np.asarray(f) for f in r]
    r.close()
    return fs


def reencode(src: Path, name: str):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(OUT / f"{name}.mp4")], check=True)


def to_gif(name: str, width: int, fps: int):
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(OUT / f"{name}.mp4"), "-vf", vf, "-loop", "0",
                    str(OUT / f"{name}.gif")], check=True)


def write_video(frames, name, fps):
    tmp = OUT / f"_{name}.mp4"
    imageio.mimsave(tmp, frames, fps=fps, codec="libx264", macro_block_size=2,
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "23"])
    reencode(tmp, name)
    tmp.unlink()


def build_grid(clip_dir: Path, out_name: str):
    real_full = read_video(clip_dir / "real_continuation.mp4")
    n_real = len(real_full)
    real = [real_full[round(k * (n_real - 1) / (N_FRAMES - 1))] for k in range(N_FRAMES)]
    panels = [("real continuation (DAVIS)", real),
              ("tracking video (control signal)", read_video(clip_dir / "tracking_video.mp4")),
              ("DaS + MolmoMotion trajectory", read_video(clip_dir / "generated_tracked.mp4")),
              ("DaS, static control (no motion)", read_video(clip_dir / "generated_static.mp4"))]
    frames = []
    for k in range(N_FRAMES):
        tiles = []
        for title, fs in panels:
            f = cv2.resize(fs[k], (480, 320)).copy()
            cv2.rectangle(f, (0, 0), (480, 22), (0, 0, 0), -1)
            cv2.putText(f, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(f)
        frames.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))
    write_video(frames, out_name, fps=8)
    to_gif(out_name, 720, 6)
    print(f"wrote {out_name}.mp4 / .gif ({(OUT / f'{out_name}.mp4').stat().st_size / 1024:.0f} KB / "
          f"{(OUT / f'{out_name}.gif').stat().st_size / 1024:.0f} KB)")


def main():
    build_grid(ROOT / "outputs" / "part3_flamingo", "part3_flamingo_comparison_2x2")
    build_grid(ROOT / "outputs" / "part3_carturn", "part3_carturn_comparison_2x2")


if __name__ == "__main__":
    main()
