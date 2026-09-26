"""Part 2 visualizations: input points, predicted-vs-annotated 2D path on
frame_0, and the predicted endpoint overlaid on the real frame_15."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sharerobot_example"
OUT_DIR = ROOT / "outputs" / "part2"


def main():
    data = np.load(OUT_DIR / "pred.npz")
    pred_2d, p0 = data["pred_2d"], data["points_2d_at_t0"]
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    gt_path = np.asarray(meta["gt_2d_path_frame0"], dtype=np.float32)  # (4,2)

    t0_img = np.asarray(Image.open(OUT_DIR / "t0_frame.jpg").convert("RGB"))
    frame15 = np.asarray(Image.open(DATA_DIR / "frame_future_real.jpg").convert("RGB"))
    H_img, W_img = t0_img.shape[:2]

    # Fig 1: 8 query points on t0
    fig, ax = plt.subplots(figsize=(W_img / 100, H_img / 100), dpi=100)
    ax.imshow(t0_img)
    ax.scatter(p0[1:, 0], p0[1:, 1], s=60, c="yellow", edgecolors="black", linewidths=1, zorder=4)
    ax.scatter(p0[0, 0], p0[0, 1], s=110, c="orange", edgecolors="black", linewidths=1.5,
                marker="*", zorder=5, label="anchor (=ShareRobot annotated point)")
    ax.legend(loc="lower right")
    ax.set_title(f"ShareRobot input: t0 frame + 8 query points on gripper\naction=\"{meta['action']}\"")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_input_points.png", dpi=100)
    plt.close(fig)

    # Fig 2: predicted anchor path (magma) vs ShareRobot annotated path (lime) on t0
    fig, ax = plt.subplots(figsize=(W_img / 100, H_img / 100), dpi=100)
    ax.imshow(t0_img)
    cmap = matplotlib.colormaps["magma"]
    anchor_path = np.concatenate([p0[0:1], pred_2d[0]], axis=0)
    ax.plot(anchor_path[:, 0], anchor_path[:, 1], color=cmap(0.6), linewidth=2.5, label="predicted (anchor point)")
    ax.plot(gt_path[:, 0], gt_path[:, 1], color="lime", linewidth=2.5, linestyle="--",
             marker="o", label="ShareRobot annotated path (sparse waypoints)")
    ax.scatter([p0[0, 0]], [p0[0, 1]], s=90, c="orange", edgecolors="black", zorder=5)
    ax.legend(loc="lower right")
    ax.set_title("Predicted trajectory vs. ShareRobot 2D annotation")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_pred_vs_gt.png", dpi=100)
    plt.close(fig)

    # Fig 3: real frame_15 with predicted final position + GT final waypoint overlaid
    fig, ax = plt.subplots(figsize=(W_img / 100, H_img / 100), dpi=100)
    ax.imshow(frame15)
    ax.scatter([pred_2d[0, -1, 0]], [pred_2d[0, -1, 1]], s=140, c="magenta", marker="X",
                edgecolors="black", linewidths=1.5, label="predicted final position (t0+30)")
    ax.scatter([gt_path[-1, 0]], [gt_path[-1, 1]], s=140, c="lime", marker="o",
                edgecolors="black", linewidths=1.5, label="ShareRobot last annotated waypoint")
    ax.legend(loc="lower right")
    ax.set_title("Real frame_15 (only future frame ShareRobot provides) vs. prediction")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_frame15_comparison.png", dpi=100)
    plt.close(fig)

    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
