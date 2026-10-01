"""Part 2e: a SECOND real, non-duplicated, multi-frame-history ShareRobot episode (bridge#episode_4801,
"lift the pot" -> "move the pot towards the blue cloth"), built with the SAME fixed (bug-free) methodology
as part2b_build_real_history.py (backproject_shared_z: all 8 query points share one depth per frame, no
per-point depth-map resampling). Purpose: the first episode (4263, "move the pan...") gave zero/wrong-direction
predictions across every input-construction variant tried; this checks whether that is specific to this one
scene, or a systematic failure across ShareRobot/bridge episodes in general (small-sample-size concern raised
by the user after seeing only one scene tested with H3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2b_build_real_history import (
    find_arm_xy, sample_query_points, estimate_depth, sample_depth_at, backproject_shared_z,
    IMG_W, IMG_H, FX, FY, CX, CY,
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "sharerobot_real_history3"
OUT_DIR = ROOT / "data" / "sharerobot_real_history3_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_FRAMES = [10, 16, 22]
FUTURE_REAL_FRAME = 29
ACTION = "lift the pot and move it towards the blue cloth"
ARM_ROI = (120, 0, 420, 280)  # same ROI convention as episode 4263


def main():
    arm_xy = {i: find_arm_xy(SRC_DIR / f"frame_{i}.png", roi=ARM_ROI) for i in HIST_FRAMES + [FUTURE_REAL_FRAME]}
    print("tracked arm 2D position per frame:", arm_xy)

    points_2d_t0 = sample_query_points(arm_xy[HIST_FRAMES[-1]], radius=35, seed=0)
    points_3d_history = np.zeros((3, 8, 3), dtype=np.float32)
    depth_map_t0 = estimate_depth(Image.open(SRC_DIR / f"frame_{HIST_FRAMES[-1]}.png").convert("RGB"))

    for fi, frame_idx in enumerate(HIST_FRAMES):
        if fi == len(HIST_FRAMES) - 1:
            points_2d = points_2d_t0
        else:
            shift = np.array(arm_xy[frame_idx]) - np.array(arm_xy[HIST_FRAMES[-1]])
            points_2d = points_2d_t0 + shift[None, :]
            points_2d[:, 0] = np.clip(points_2d[:, 0], 2, IMG_W - 2)
            points_2d[:, 1] = np.clip(points_2d[:, 1], 2, IMG_H - 2)
        anchor_z = sample_depth_at(points_2d[0, 0], points_2d[0, 1], depth_map_t0)
        points_3d_history[fi] = backproject_shared_z(points_2d, anchor_z)
        print(f"frame {frame_idx}: anchor 2D {points_2d[0]}, anchor depth {points_3d_history[fi, 0, 2]:.3f} m")

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
    torch.save(torch.from_numpy(points_2d_t0), OUT_DIR / "points_2d_at_t0.pt")
    torch.save(torch.from_numpy(points_3d_history), OUT_DIR / "points_3d_history.pt")
    torch.save(torch.from_numpy(K), OUT_DIR / "intrinsics_K.pt")
    for suffix, frame_idx in zip(("t-2", "t-1", "t+0"), HIST_FRAMES):
        Image.open(SRC_DIR / f"frame_{frame_idx}.png").convert("RGB").save(OUT_DIR / f"frame_{suffix}.jpg", quality=95)
    Image.open(SRC_DIR / f"frame_{FUTURE_REAL_FRAME}.png").convert("RGB").save(OUT_DIR / "frame_future_real.jpg", quality=95)

    # real future anchor (for direction-cosine eval), same shared-depth convention
    real_future_2d = np.array(arm_xy[FUTURE_REAL_FRAME])
    z_future = sample_depth_at(real_future_2d[0], real_future_2d[1], depth_map_t0)
    real_future_3d = np.array([(real_future_2d[0] - CX) / FX * z_future,
                                (real_future_2d[1] - CY) / FY * z_future, z_future])

    meta = {
        "id": "sharerobot_planning_bridge_episode_4801",
        "dataset": "BAAI/ShareRobot planning split, SECOND episode (to check sample-size / scene-specificity)",
        "action": ACTION,
        "image_size_wh": [IMG_W, IMG_H],
        "history_frame_indices_in_episode": HIST_FRAMES,
        "future_real_frame_index_in_episode": FUTURE_REAL_FRAME,
        "tracked_arm_xy_per_frame": {str(k): v for k, v in arm_xy.items()},
        "real_future_3d_anchor": real_future_3d.tolist(),
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "notes": "Uses the FIXED (bug-free) backproject_shared_z method from the start -- no companion-point "
                 "depth-resampling issue here. Estimated/assumed depth and intrinsics (not real GT like the "
                 "episode_4263 calibrated attempt).",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
