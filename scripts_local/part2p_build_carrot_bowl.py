"""Part 2p: a SEVENTH real H3 episode (bridge#episode_4441, "move towards the bowl" / "drop the carrot
into the bowl") -- the gripper already holds a carrot from frame 0 and carries it toward the one bowl in
the scene. Background has a toy toaster + 2 cans + a towel + a brush, but none of them are black (gripper's
own color), so the static/moving dark-pixel split cleanly isolates the gripper the same way as part2n/part2o.
History frames end just before the drop (contact begins ~frame 19); future frame is right at contact.
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
    sample_query_points, estimate_depth, sample_depth_at, backproject_shared_z,
    IMG_W, IMG_H, FX, FY, CX, CY,
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "sharerobot_real_history9"
OUT_DIR = ROOT / "data" / "sharerobot_real_history9_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_FRAMES = [6, 10, 14]
FUTURE_REAL_FRAME = 18
ACTION = "move towards the bowl"
ARM_ROI = (20, 0, 380, 260)
STATIC_SAMPLE_FRAMES = list(range(0, 21))


def find_arm_xy_excl_static(path, roi, static_dark):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = roi
    crop = im[y0:y1, x0:x1]
    dark = crop.sum(axis=-1) < 180
    sel = dark & ~static_dark
    ys, xs = np.nonzero(sel)
    assert len(xs) >= 20, f"too few moving-dark pixels in {path}"
    return float(xs.mean() + x0), float(ys.mean() + y0)


def main():
    x0, y0, x1, y1 = ARM_ROI
    darks = []
    for i in STATIC_SAMPLE_FRAMES:
        im = np.asarray(Image.open(SRC_DIR / f"frame_{i}.png").convert("RGB"))[y0:y1, x0:x1].astype(np.float32)
        darks.append(im.sum(axis=-1) < 180)
    static_dark = np.all(darks, axis=0)
    print("static-dark pixel count:", int(static_dark.sum()))

    arm_xy = {i: find_arm_xy_excl_static(SRC_DIR / f"frame_{i}.png", ARM_ROI, static_dark)
              for i in HIST_FRAMES + [FUTURE_REAL_FRAME]}
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

    real_future_2d = np.array(arm_xy[FUTURE_REAL_FRAME])
    z_future = sample_depth_at(real_future_2d[0], real_future_2d[1], depth_map_t0)
    real_future_3d = np.array([(real_future_2d[0] - CX) / FX * z_future,
                                (real_future_2d[1] - CY) / FY * z_future, z_future])

    meta = {
        "id": "sharerobot_planning_bridge_episode_4441_carrot_to_bowl",
        "dataset": "BAAI/ShareRobot planning split, SEVENTH episode: gripper already holds the carrot, "
                   "carries it toward the one bowl in the scene",
        "action": ACTION,
        "image_size_wh": [IMG_W, IMG_H],
        "history_frame_indices_in_episode": HIST_FRAMES,
        "future_real_frame_index_in_episode": FUTURE_REAL_FRAME,
        "tracked_arm_xy_per_frame": {str(k): v for k, v in arm_xy.items()},
        "real_future_3d_anchor": real_future_3d.tolist(),
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "notes": "Dataset's own annotation: 'move towards the bowl' -> 'drop the carrot into the bowl' -- the "
                 "carrot is grasped from frame 0, so this directly tests the 'already-holding-something, "
                 "nearest-bowl' hypothesis case with real video. Background has a toy toaster + 2 cans + towel "
                 "+ brush (not minimal) but none are black, so static/moving dark-pixel exclusion (same method "
                 "as part2n/part2o) cleanly isolates the gripper (static pixel count only 655 vs ~18000 "
                 "gripper pixels per frame).",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
