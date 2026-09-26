"""Build the example images and videos shown in README.md ("Expected results") from the outputs of parts 1-3.

Run from the repo root in the MolmoMotion environment after the three parts have produced their outputs:
    python scripts_local/make_expected_media.py
Writes docs/expected/*.jpg and docs/expected/*.mp4 (kept small: JPEG q86 <= 1500 px wide, H.264 yuv420p).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "expected"
OUT.mkdir(parents=True, exist_ok=True)
P1, P2, P3 = (ROOT / "outputs" / f"part{i}" for i in (1, 2, 3))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def to_jpg(src: Path, name: str, max_w: int = 1500):
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(OUT / f"{name}.jpg", quality=86, optimize=True)


def reencode(src: Path, name: str):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(OUT / f"{name}.mp4")], check=True)


def to_gif(name: str, width: int, fps: int):
    """Animated preview (GitHub strips <video> from READMEs, GIFs render inline)."""
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


def read_video(p):
    r = imageio.get_reader(p)
    fs = [np.asarray(f) for f in r]
    r.close()
    return fs


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    return np.stack([K[0, 0] * xyz[..., 0] / z + K[0, 2], K[1, 1] * xyz[..., 1] / z + K[1, 2]], axis=-1)


def part1_animation():
    """Growing trails, predicted (magenta) vs real (green) 3D tracks, both projected into the t0 camera."""
    d = np.load(P1 / "pred_and_gt.npz")
    pred, gt, vis, p0, K = d["pred"], d["gt"], d["vis"], d["points_2d_at_t0"], d["intrinsics"]
    pp, gp = project(pred, K), project(gt, K)
    img = np.asarray(Image.open(P1 / "t0_frame.jpg").convert("RGB"))
    H, W = img.shape[:2]
    xs = np.concatenate([pp[..., 0].ravel(), gp[..., 0][vis].ravel(), p0[:, 0]])
    ys = np.concatenate([pp[..., 1].ravel(), gp[..., 1][vis].ravel(), p0[:, 1]])
    x0, x1 = min(0, xs.min() - 20), max(W, xs.max() + 20)
    y0, y1 = min(0, ys.min() - 20), max(H, ys.max() + 20)
    frames = []
    for k in range(pred.shape[1]):
        fig = plt.figure(figsize=((x1 - x0) / 100, (y1 - y0) / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(img, extent=[0, W, H, 0]); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        for p in range(pred.shape[0]):
            tr = np.concatenate([p0[p:p + 1], pp[p, :k + 1]])
            ax.plot(tr[:, 0], tr[:, 1], color="magenta", lw=2, label="predicted" if p == 0 else None)
            v = np.where(vis[p, :k + 1])[0]
            if len(v):
                g = np.concatenate([p0[p:p + 1], gp[p, v]])
                ax.plot(g[:, 0], g[:, 1], color="lime", lw=2, ls="--", label="real (PointMotionBench)" if p == 0 else None)
            ax.scatter(*pp[p, k], s=25, c="magenta", edgecolors="white")
        ax.scatter(p0[:, 0], p0[:, 1], s=30, c="yellow", edgecolors="black", zorder=5)
        ax.text(10, y0 + 22, f"t0 + {k + 1} frames", color="white", fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5, pad=3))
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w // 2 * 2, h // 2 * 2)) for f in frames]
    write_video(frames, "part1_prediction_vs_real", fps=8)


def part3_comparison():
    T0 = 2
    real_dir = ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees"
    real = []
    for k in range(49):
        j = T0 + 1 + int(round(k * 29 / 48))
        real.append(cv2.resize(cv2.cvtColor(cv2.imread(str(real_dir / f"{j:05d}.jpg")), cv2.COLOR_BGR2RGB), (720, 480)))
    panels = [("real continuation (DAVIS)", real),
              ("tracking video (control signal)", read_video(P3 / "tracking_video.mp4")),
              ("DaS + MolmoMotion trajectory", read_video(P3 / "generated_tracked_480x720_cfg1_10steps_offload.mp4")),
              ("DaS, static control (no motion)", read_video(P3 / "generated_static_480x720_cfg1_10steps_offload.mp4"))]
    frames = []
    for k in range(49):
        tiles = []
        for title, fs in panels:
            f = cv2.resize(fs[k], (480, 320)).copy()
            cv2.rectangle(f, (0, 0), (480, 22), (0, 0, 0), -1)
            cv2.putText(f, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(f)
        frames.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))
    write_video(frames, "part3_comparison_2x2", fps=8)


def main():
    for src, name in [(P1 / "fig1_input_points.png", "part1_input_points"),
                      (P1 / "fig2_pred_vs_gt.png", "part1_predicted_vs_real"),
                      (P1 / "fig4_future_frames_strip.png", "part1_real_future_frames"),
                      (P2 / "fig1_input_points.png", "part2_input_points"),
                      (P2 / "fig2_pred_vs_gt.png", "part2_predicted_vs_annotated"),
                      (P2 / "fig3_frame15_comparison.png", "part2_real_frame15"),
                      (P3 / "fig_part3_frames.png", "part3_das_frames")]:
        to_jpg(src, name)
    part1_animation()
    reencode(P3 / "generated_tracked_480x720_cfg1_10steps_offload.mp4", "part3_das_trajectory")
    reencode(P3 / "generated_static_480x720_cfg1_10steps_offload.mp4", "part3_das_static_control")
    reencode(P3 / "tracking_video.mp4", "part3_tracking_video")
    part3_comparison()
    for name, w, fps in [("part1_prediction_vs_real", 760, 8), ("part3_comparison_2x2", 720, 6),
                         ("part3_das_trajectory", 400, 6), ("part3_das_static_control", 400, 6),
                         ("part3_tracking_video", 400, 6)]:
        to_gif(name, w, fps)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:40s} {p.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
