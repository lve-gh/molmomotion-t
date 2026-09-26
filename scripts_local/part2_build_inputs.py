"""Part 2: build MolmoMotion inputs for a ShareRobot clip.

ShareRobot's `trajectory` split (BAAI/ShareRobot) is NOT prepared for
MolmoMotion: for a chosen episode it ships exactly two RGB frames
(frame_0, frame_15 -- no continuous video, no depth, no camera intrinsics,
no per-frame timestamps) plus a handful of 2D pixel waypoints describing the
*end-effector's* path (not a free object's), annotated relative to one of
those two frames.

Episode chosen: bridge #episode_25423 (rtx_frames_success_13), 640x480,
gripper reaching across the scene -- clearly visible motion between the two
frames. Two annotations exist for it:
  id=2633 anchored on frame_0: instruction "reach for the spoon",
      trajectory = 4 waypoints (gripper's path starting at t0)
  id=2634 anchored on frame_15: instruction "move the spoon to the right",
      trajectory = 4 waypoints of the following sub-motion

We use id=2633 as the (t0, action, GT-path) triple: t0 = frame_0,
query anchor = trajectory[0] (335.64, 57.74) px -- verified by overlay to
sit exactly on the gripper -- and the remaining 3 waypoints as a coarse,
temporally-unaligned 2D ground truth for the future path. frame_15 is kept
as the one real future frame we have for qualitative comparison.

Decisions made explicit here (see report for full discussion):
  1. History: MolmoMotion-4B-H3-F30 needs 3 history frames with real
     apparent motion; ShareRobot gives us exactly one usable frame at t0.
     We replicate frame_0 three times as a documented simplification
     (zero apparent input velocity) rather than pulling a second 18GB
     H1-F32 checkpoint just for this -- flagged as a limitation.
  2. 3D history: no depth/calibration is shipped. We estimate metric depth
     with Depth-Anything-V2-Metric-Indoor-Small (indoor-trained monocular
     metric depth) and assume a pinhole camera with a RealSense-D435-like
     69.4 deg horizontal FOV (a common fixed tabletop camera in Bridge/
     WidowX-style setups) -- both are explicit, documented assumptions,
     not measured values.
  3. P=8 query points: the checkpoint hard-requires P=8 (config.num_points).
     ShareRobot only annotates one path per instruction, so we sample a
     small jittered cluster of 8 pixels around the annotated anchor point
     (i.e. treat the gripper as the tracked "object", covering its visible
     extent) -- point index 0 is exactly the annotated anchor and is the
     only one used for the quantitative 2D comparison; the rest illustrate
     spatial coherence of the prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / "data" / "sharerobot_candidates"
OUT_DIR = ROOT / "data" / "sharerobot_example"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 640, 480
ANCHOR_XY = (335.6390977443609, 57.744360902255636)
GT_PATH_FRAME0 = [
    [335.6390977443609, 57.744360902255636],
    [253.83458646616543, 117.89473684210525],
    [174.4360902255639, 196.09022556390977],
    [138.34586466165413, 259.8496240601504],
]
ACTION = "reach for the spoon"

# Assumed pinhole intrinsics (RealSense-D435-like 69.4deg horizontal FOV).
FOV_H_DEG = 69.4
FX = (IMG_W / 2) / np.tan(np.deg2rad(FOV_H_DEG) / 2)
FY = FX
CX, CY = IMG_W / 2, IMG_H / 2


def sample_query_points(anchor=ANCHOR_XY, n=8, radius=35, seed=0):
    rng = np.random.default_rng(seed)
    pts = [list(anchor)]
    while len(pts) < n:
        dx, dy = rng.uniform(-radius, radius, size=2)
        x, y = anchor[0] + dx, anchor[1] + dy
        x = float(np.clip(x, 2, IMG_W - 2))
        y = float(np.clip(y, 2, IMG_H - 2))
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)  # (8, 2), row 0 == anchor exactly


def estimate_depth(image: Image.Image):
    from transformers import pipeline
    depther = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    result = depther(image)
    # `result["depth"]` is an 8-bit visualization image; the real metric
    # values (meters) are in `predicted_depth`, at the model's native
    # (possibly downsampled) resolution.
    depth = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    if depth.shape != (IMG_H, IMG_W):
        depth_img = Image.fromarray(depth).resize((IMG_W, IMG_H), Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)
    return depth


def backproject(points_2d, depth_map):
    out = np.zeros((points_2d.shape[0], 3), dtype=np.float32)
    for i, (x, y) in enumerate(points_2d):
        xi, yi = int(round(x)), int(round(y))
        xi = np.clip(xi, 0, IMG_W - 1)
        yi = np.clip(yi, 0, IMG_H - 1)
        z = float(depth_map[yi, xi])
        if z <= 1e-3:
            # fall back to a small neighborhood median if the exact pixel is invalid
            y0, y1 = max(0, yi - 3), min(IMG_H, yi + 4)
            x0, x1 = max(0, xi - 3), min(IMG_W, xi + 4)
            patch = depth_map[y0:y1, x0:x1]
            z = float(np.median(patch[patch > 1e-3])) if (patch > 1e-3).any() else 0.6
        X = (x - CX) / FX * z
        Y = (y - CY) / FY * z
        out[i] = [X, Y, z]
    return out


def main():
    frame0 = Image.open(CAND_DIR / "spoon_frame0.png").convert("RGB")
    assert frame0.size == (IMG_W, IMG_H), frame0.size

    depth_map = estimate_depth(frame0)
    print("depth range (m):", depth_map.min(), depth_map.max(),
          "at anchor:", depth_map[int(ANCHOR_XY[1]), int(ANCHOR_XY[0])])

    points_2d = sample_query_points()
    points_3d_t0 = backproject(points_2d, depth_map)  # (8, 3)
    print("points_3d at t0:\n", points_3d_t0)

    # H=3 pseudo-history: replicate the single real frame/point-cloud 3x.
    points_3d_history = np.repeat(points_3d_t0[None, :, :], 3, axis=0)  # (3, 8, 3)

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)

    torch.save(torch.from_numpy(points_2d), OUT_DIR / "points_2d_at_t0.pt")
    torch.save(torch.from_numpy(points_3d_history), OUT_DIR / "points_3d_history.pt")
    torch.save(torch.from_numpy(K), OUT_DIR / "intrinsics_K.pt")
    for suffix in ("t-2", "t-1", "t+0"):
        frame0.save(OUT_DIR / f"frame_{suffix}.jpg", quality=95)
    frame15 = Image.open(CAND_DIR / "spoon_frame15.png").convert("RGB")
    frame15.save(OUT_DIR / "frame_future_real.jpg", quality=95)

    meta = {
        "id": "sharerobot_bridge_episode_25423",
        "action": ACTION,
        "image_size_wh": [IMG_W, IMG_H],
        "assumed_fov_h_deg": FOV_H_DEG,
        "assumed_intrinsics_K": K.tolist(),
        "gt_2d_path_frame0": GT_PATH_FRAME0,
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "notes": "3D history is a repeated single frame (no real motion cues); "
                 "depth + intrinsics are model-estimated / assumed, not measured.",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved ShareRobot MolmoMotion inputs to", OUT_DIR)


if __name__ == "__main__":
    main()
