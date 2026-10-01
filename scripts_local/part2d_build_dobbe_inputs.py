"""Part 2d: test the "camera framing" hypothesis for the Part-2 ShareRobot direction problem.

Across all 4 earlier attempts (part2_build_inputs.py, part2b/c), every one used a `bridge`-sourced ShareRobot
episode: a fixed tripod, third-person "lab bench" camera looking down at a WidowX arm. MolmoMotion's egocentric
examples (EgoDex: a chest-mounted camera filming a human hand) predict very accurately (our-vs-authors mean L2
1.2-1.9cm -- see README Part 1). This script builds the same kind of input, with the same (honestly limited,
estimated-depth) methodology as the original part2_build_inputs.py, but for a `dobbe`-sourced ShareRobot episode
instead: DobbE ("On Bringing Robots Home") records with a handheld, stick-mounted phone camera in real homes --
visually much closer to EgoDex's handheld/egocentric style than Bridge's fixed lab tripod. If direction comes out
right (or clearly better) here with an otherwise-identical pipeline, that isolates camera framing/domain as the
likely cause of the Part-2 failures, rather than "robot manipulation in general."

Episode: `55_dobbe#episode_3368` ("approach the cabinet door"), chosen as the largest-displacement (250.6px)
trajectory-split entry out of 759 dobbe episodes (same split structure as the original bridge example: exactly
one usable frame_0 + a 4-waypoint annotated 2D path, no continuous video, no depth/intrinsics -- same honest
limitations as before, intentionally, to keep this a controlled single-variable test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sharerobot_dobbe_example"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 256, 256
ANCHOR_XY = (113.56390977443608, 254.0751879699248)
GT_PATH_FRAME0 = [
    [113.56390977443608, 254.0751879699248],
    [93.83458646616543, 189.11278195488723],
    [84.21052631578948, 80.84210526315789],
    [83.7293233082707, 5.293233082706767],
]
ACTION = "approach the cabinet door"

FOV_H_DEG = 69.4  # same assumption as the bridge example -- intentionally unchanged, isolating camera FRAMING
FX = (IMG_W / 2) / np.tan(np.deg2rad(FOV_H_DEG) / 2)
FY = FX
CX, CY = IMG_W / 2, IMG_H / 2


def sample_query_points(anchor=ANCHOR_XY, n=8, radius=18, seed=0):
    # radius scaled down from the bridge example's 35px (640-wide image) to this 256-wide image, same proportion
    rng = np.random.default_rng(seed)
    pts = [list(anchor)]
    while len(pts) < n:
        dx, dy = rng.uniform(-radius, radius, size=2)
        x, y = anchor[0] + dx, anchor[1] + dy
        x = float(np.clip(x, 2, IMG_W - 2))
        y = float(np.clip(y, 2, IMG_H - 2))
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def estimate_depth(image: Image.Image):
    from transformers import pipeline
    depther = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    result = depther(image)
    depth = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    if depth.shape != (IMG_H, IMG_W):
        depth = np.array(Image.fromarray(depth).resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.float32)
    return depth


def backproject(points_2d, depth_map):
    out = np.zeros((points_2d.shape[0], 3), dtype=np.float32)
    for i, (x, y) in enumerate(points_2d):
        xi, yi = int(np.clip(round(x), 0, IMG_W - 1)), int(np.clip(round(y), 0, IMG_H - 1))
        z = float(depth_map[yi, xi])
        if z <= 1e-3:
            y0, y1 = max(0, yi - 3), min(IMG_H, yi + 4)
            x0, x1 = max(0, xi - 3), min(IMG_W, xi + 4)
            patch = depth_map[y0:y1, x0:x1]
            z = float(np.median(patch[patch > 1e-3])) if (patch > 1e-3).any() else 0.6
        out[i] = [(x - CX) / FX * z, (y - CY) / FY * z, z]
    return out


def main():
    frame0 = Image.open(OUT_DIR / "raw_frame0.png").convert("RGB")
    assert frame0.size == (IMG_W, IMG_H), frame0.size

    depth_map = estimate_depth(frame0)
    print("depth range (m):", depth_map.min(), depth_map.max(),
          "at anchor:", depth_map[int(ANCHOR_XY[1]), int(ANCHOR_XY[0])])

    points_2d = sample_query_points()
    points_3d_t0 = backproject(points_2d, depth_map)
    print("points_3d at t0:\n", points_3d_t0)

    points_3d_history = np.repeat(points_3d_t0[None, :, :], 3, axis=0)  # H1 only uses the last row anyway
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)

    torch.save(torch.from_numpy(points_2d), OUT_DIR / "points_2d_at_t0.pt")
    torch.save(torch.from_numpy(points_3d_history), OUT_DIR / "points_3d_history.pt")
    torch.save(torch.from_numpy(K), OUT_DIR / "intrinsics_K.pt")
    for suffix in ("t-2", "t-1", "t+0"):
        frame0.save(OUT_DIR / f"frame_{suffix}.jpg", quality=95)

    meta = {
        "id": "sharerobot_dobbe_episode_3368",
        "action": ACTION,
        "image_size_wh": [IMG_W, IMG_H],
        "assumed_fov_h_deg": FOV_H_DEG,
        "assumed_intrinsics_K": K.tolist(),
        "gt_2d_path_frame0": GT_PATH_FRAME0,
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "notes": "Same methodology as the original bridge example (part2_build_inputs.py): 1 real frame "
                 "(H1-F32), estimated depth, assumed intrinsics. The ONLY thing changed is the source domain: "
                 "a dobbe-sourced ShareRobot episode (handheld stick-mounted phone camera, real home) instead of "
                 "a bridge-sourced one (fixed lab tripod over a WidowX arm) -- testing whether MolmoMotion's "
                 "direction accuracy depends on camera framing/domain rather than 'robot manipulation' per se.",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved dobbe ShareRobot MolmoMotion inputs to", OUT_DIR)


if __name__ == "__main__":
    main()
