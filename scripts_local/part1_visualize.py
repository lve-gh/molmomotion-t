"""Part 1 visualizations: static comparison figure + side-by-side GT-vs-pred
overlay video, built from outputs/part1/pred_and_gt.npz (written by
part1_infer_and_eval.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "part1"
DAVIS_FRAMES = ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees"


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def main():
    data = np.load(OUT_DIR / "pred_and_gt.npz")
    pred, gt, vis = data["pred"], data["gt"], data["vis"]
    gt_2d, p0, K = data["gt_2d"], data["points_2d_at_t0"], data["intrinsics"]
    meta = json.loads((ROOT / "repos" / "molmo-motion" / "examples" / "data" /
                        "davis_bmx_trees" / "meta.json").read_text())
    t0 = meta["t0_absolute"]
    P, F, _ = pred.shape

    pred_2d = project(pred, K)  # (P, F, 2)
    gt_proj = project(gt, K)    # 3D GT projected with the SAME t0 intrinsics -> same frame as pred

    t0_img = np.asarray(Image.open(OUT_DIR / "t0_frame.jpg").convert("RGB"))
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
    fig.savefig(OUT_DIR / "fig1_input_points.png", dpi=100)
    plt.close(fig)

    # ---- Figure 2: predicted (magma) vs GT (green), both = 3D tracks in the t0-camera
    # frame projected with the same intrinsics. (The 2D tracks shipped with the
    # benchmark are in each frame's own image coords; the camera pans with the
    # rider there, so they are NOT comparable to a t0-frame projection.)
    allx = np.concatenate([pred_2d[..., 0].ravel(), gt_proj[..., 0][vis].ravel(), p0[:, 0]])
    ally = np.concatenate([pred_2d[..., 1].ravel(), gt_proj[..., 1][vis].ravel(), p0[:, 1]])
    x0, x1 = min(0, allx.min() - 30), max(W_img, allx.max() + 30)
    y0, y1 = min(0, ally.min() - 30), max(H_img, ally.max() + 30)
    fig, ax = plt.subplots(figsize=((x1 - x0) / 100, (y1 - y0) / 100), dpi=100)
    ax.imshow(t0_img, extent=[0, W_img, H_img, 0])
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    cmap = matplotlib.colormaps["magma"]
    for p in range(P):
        pred_line = np.concatenate([p0[p:p+1], pred_2d[p]], axis=0)
        ax.plot(pred_line[:, 0], pred_line[:, 1], color=cmap(0.6), linewidth=2.0,
                 label="predicted (MolmoMotion)" if p == 0 else None, zorder=4)
        if vis[p].any():
            gl = np.concatenate([p0[p:p+1], gt_proj[p][vis[p]]], axis=0)
            ax.plot(gl[:, 0], gl[:, 1], color="lime", linewidth=2.0, linestyle="--",
                     label="ground truth (3D, PointMotionBench)" if p == 0 else None, zorder=3)
        ax.scatter([p0[p, 0]], [p0[p, 1]], s=50, c="yellow", edgecolors="black", zorder=5)
    ax.legend(loc="upper left")
    ax.set_title("Predicted vs. real continuation, projected into the t0 camera (canvas extended)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_pred_vs_gt.png", dpi=100)
    plt.close(fig)

    # ---- Figure 2b: metric 3D view (X right, Z depth) and (X, Y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p in range(P):
        axes[0].plot(pred[p, :, 0], pred[p, :, 2], color=cmap(0.6), alpha=0.8, label="pred" if p == 0 else None)
        axes[0].plot(gt[p][vis[p], 0], gt[p][vis[p], 2], "g--", alpha=0.8, label="GT" if p == 0 else None)
        axes[1].plot(pred[p, :, 0], pred[p, :, 1], color=cmap(0.6), alpha=0.8)
        axes[1].plot(gt[p][vis[p], 0], gt[p][vis[p], 1], "g--", alpha=0.8)
    axes[0].set_xlabel("X, m (right)"); axes[0].set_ylabel("Z, m (depth)"); axes[0].set_title("Top view (X-Z), t0 camera frame")
    axes[1].set_xlabel("X, m (right)"); axes[1].set_ylabel("Y, m (down)"); axes[1].invert_yaxis(); axes[1].set_title("Front view (X-Y)")
    axes[0].legend(); fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2b_3d_views.png", dpi=110)
    plt.close(fig)

    # ---- Figure 3: per-frame error curve ---------------------------------
    err = np.linalg.norm(pred - gt, axis=-1)  # (P, F)
    err_masked = np.where(vis, err, np.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in range(P):
        ax.plot(range(1, F + 1), err_masked[p], alpha=0.4, linewidth=1)
    mean_err = np.nanmean(err_masked, axis=0)
    ax.plot(range(1, F + 1), mean_err, color="black", linewidth=2.5, label="mean over points")
    ax.set_xlabel("future frame (t0 + k)")
    ax.set_ylabel("L2 error (m)")
    ax.set_title("Per-frame 3D displacement error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_error_curve.png", dpi=120)
    plt.close(fig)

    # ---- Figure 4: strip of real future frames with GT (green) / pred (magenta) points
    sample_ks = [0, 9, 19, 29]  # future indices (0-based) -> frames t0+1, t0+10, t0+20, t0+30
    fig, axes = plt.subplots(1, len(sample_ks), figsize=(4 * len(sample_ks), 4))
    for ax, k in zip(axes, sample_ks):
        frame_idx = t0 + 1 + k
        frame_path = DAVIS_FRAMES / f"{frame_idx:05d}.jpg"
        img = np.asarray(Image.open(frame_path).convert("RGB"))
        ax.imshow(img)
        v = vis[:, k]
        if v.any():
            ax.scatter(gt_2d[v, k, 0], gt_2d[v, k, 1], s=40, c="lime", edgecolors="black",
                        label="GT" if k == sample_ks[0] else None)
        ax.scatter(pred_2d[:, k, 0], pred_2d[:, k, 1], s=40, c="magenta", marker="x",
                    label="pred" if k == sample_ks[0] else None)
        ax.set_title(f"t0+{k+1} (#{frame_idx:05d})" + chr(10) + "GT: 2D track in this frame; pred: t0-camera proj.", fontsize=8)
        ax.axis("off")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_future_frames_strip.png", dpi=100)
    plt.close(fig)

    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
