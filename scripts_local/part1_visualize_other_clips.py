"""Part 1 visualizations for the flamingo / car-turn multi-example clips, mirroring
part1_visualize.py's Figure 1 (input query points) and Figure 2 (pred vs. GT trajectory)
for bmx-trees. Source data: outputs/multi/davis_<clip>/{pred,gt}.npz (written by the
multi-example run), t0 frame reused from outputs/part3_<clip>/t0_frame.jpg (same source
frame), meta from repos/molmo-motion/examples/data/davis_<clip>/meta.json.

Run from the repo root, in the MolmoMotion environment:
    venv_molmomotion/Scripts/python.exe scripts_local/part1_visualize_other_clips.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def build(clip: str, t0_image_dir: str):
    multi_dir = ROOT / "outputs" / "multi" / f"davis_{clip}"
    out_dir = ROOT / "outputs" / f"part1_{clip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_npz = np.load(multi_dir / "pred.npz")
    gt_npz = np.load(multi_dir / "gt.npz")
    pred, p0, K = pred_npz["pred"], pred_npz["points_2d_at_t0"], pred_npz["K"]
    gt, vis = gt_npz["gt"], gt_npz["vis"]
    meta = json.loads((ROOT / "repos" / "molmo-motion" / "examples" / "data" /
                        f"davis_{clip}" / "meta.json").read_text())
    P, F, _ = pred.shape

    pred_2d = project(pred, K)
    gt_proj = project(gt, K)

    t0_img = np.asarray(Image.open(ROOT / "outputs" / t0_image_dir / "t0_frame.jpg").convert("RGB"))
    H_img, W_img = t0_img.shape[:2]

    # ---- Figure 1: input points on t0 frame -----------------------------
    fig, ax = plt.subplots(figsize=(W_img / 100, H_img / 100), dpi=100)
    ax.imshow(t0_img)
    ax.scatter(p0[:, 0], p0[:, 1], s=90, c="yellow", edgecolors="black", linewidths=1.2, zorder=5)
    for i, (x, y) in enumerate(p0):
        ax.annotate(str(i + 1), (x, y), color="white", fontsize=9, ha="center", va="center",
                    fontweight="bold", zorder=6)
    ax.set_title(f"Input: t0 frame + {P} query points  (video={meta['video']}, obj={meta['obj']})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_input_points.png", dpi=100)
    plt.close(fig)

    # ---- Figure 2: predicted (magma) vs GT (green) 3D tracks in the t0-camera frame
    allx = np.concatenate([pred_2d[..., 0].ravel(), gt_proj[..., 0][vis].ravel(), p0[:, 0]])
    ally = np.concatenate([pred_2d[..., 1].ravel(), gt_proj[..., 1][vis].ravel(), p0[:, 1]])
    x0, x1 = min(0, allx.min() - 30), max(W_img, allx.max() + 30)
    y0, y1 = min(0, ally.min() - 30), max(H_img, ally.max() + 30)
    fig, ax = plt.subplots(figsize=((x1 - x0) / 100, (y1 - y0) / 100), dpi=100)
    ax.imshow(t0_img, extent=[0, W_img, H_img, 0])
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    cmap = matplotlib.colormaps["magma"]
    for p in range(P):
        pred_line = np.concatenate([p0[p:p + 1], pred_2d[p]], axis=0)
        ax.plot(pred_line[:, 0], pred_line[:, 1], color=cmap(0.6), linewidth=2.0,
                 label="predicted (MolmoMotion)" if p == 0 else None, zorder=4)
        if vis[p].any():
            gl = np.concatenate([p0[p:p + 1], gt_proj[p][vis[p]]], axis=0)
            ax.plot(gl[:, 0], gl[:, 1], color="lime", linewidth=2.0, linestyle="--",
                     label="ground truth (3D, PointMotionBench)" if p == 0 else None, zorder=3)
        ax.scatter([p0[p, 0]], [p0[p, 1]], s=50, c="yellow", edgecolors="black", zorder=5)
    ax.legend(loc="upper left")
    ax.set_title(f"Predicted vs. real continuation, projected into the t0 camera ({meta['video']})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_pred_vs_gt.png", dpi=100)
    plt.close(fig)
    print(f"wrote {out_dir}/fig1_input_points.png, fig2_pred_vs_gt.png")


def main():
    build("flamingo", "part3_flamingo")
    build("car_turn", "part3_carturn")


if __name__ == "__main__":
    main()
