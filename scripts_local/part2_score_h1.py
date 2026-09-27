"""Score a ShareRobot prediction produced by run_examples.py (e.g. the H1-F32 checkpoint, outputs/h1/sharerobot_example)
with the same 2D path metrics as part 2, and draw the projected paths of the 8 query points over the frame_15 image.

usage: python scripts_local/part2_score_h1.py [--pred outputs/h1/sharerobot_example/pred.npz] [--out-json results/part2_h1_metrics.json]
       [--fig outputs/h1/fig_h1_pred_vs_annotated.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2_infer_and_eval import compute_2d_path_metrics, project  # noqa: E402

DATA = ROOT / "data" / "sharerobot_example"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/h1/sharerobot_example/pred.npz")
    ap.add_argument("--out-json", default="results/part2_h1_metrics.json")
    ap.add_argument("--fig", default="outputs/h1/fig_h1_pred_vs_annotated.png")
    ap.add_argument("--label", default="H1-F32, real history = 1 frame")
    a = ap.parse_args()

    meta = json.loads((DATA / "meta.json").read_text())
    d = np.load(ROOT / a.pred)
    pred, K = d["pred"], d["K"]
    pred_2d = project(pred, K)
    m = compute_2d_path_metrics(pred_2d[0], meta["gt_2d_path_frame0"])
    m["future_horizon"] = int(pred.shape[1])
    m["label"] = a.label
    disp = np.linalg.norm(pred_2d[0, -1] - pred_2d[0, 0])
    m["predicted_anchor_displacement_px"] = float(disp)
    m["predicted_3d_path_length_m_point0"] = float(np.linalg.norm(np.diff(pred[0], axis=0), axis=-1).sum())
    (ROOT / a.out_json).write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k != "waypoint_to_path_px"}, indent=1))

    img = Image.open(DATA / "frame_future_real.jpg")
    gt = np.asarray(meta["gt_2d_path_frame0"])
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.imshow(img)
    ax.plot(gt[:, 0], gt[:, 1], "-o", c="lime", lw=2, label="annotated gripper path (ShareRobot)")
    ax.plot(gt[0, 0], gt[0, 1], "*", c="yellow", ms=16, mec="k")
    for p in range(pred_2d.shape[0]):
        ax.plot(pred_2d[p, :, 0], pred_2d[p, :, 1], "-", c="magenta", lw=1.5, alpha=0.9,
                label=f"predicted paths (8 points), {a.label}" if p == 0 else None)
        ax.plot(pred_2d[p, -1, 0], pred_2d[p, -1, 1], "x", c="magenta", ms=7)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("ShareRobot: predicted vs annotated 2D path (background: real frame_15)")
    ax.axis("off")
    (ROOT / a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / a.fig, dpi=130, bbox_inches="tight")
    print("figure ->", a.fig)


if __name__ == "__main__":
    main()
