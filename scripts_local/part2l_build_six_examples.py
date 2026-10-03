"""Part 2l: build H1-F32 inputs for 6 maximally-different ShareRobot source datasets (different robot
platforms/cameras/tasks), to map out where MolmoMotion transfers well vs. poorly across ShareRobot's full
diversity -- not just the `bridge` source used throughout parts 2b-2k. Same single-real-frame + estimated-depth
methodology as part2_build_inputs.py / part2d_build_dobbe_inputs.py (H1-F32 needs only 1 real frame; ShareRobot's
`trajectory` split gives exactly that + a 4-waypoint annotated 2D path per episode, usable as 2D-only ground
truth for direction evaluation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "six_examples"
OUT_ROOT = ROOT / "data" / "six_examples_inputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FOV_H_DEG = 69.4  # same assumption used throughout -- isolating the SOURCE DATASET as the variable, not intrinsics


def sample_query_points(anchor, img_w, radius, n=8, seed=0):
    rng = np.random.default_rng(seed)
    pts = [list(anchor)]
    while len(pts) < n:
        dx, dy = rng.uniform(-radius, radius, size=2)
        x, y = anchor[0] + dx, anchor[1] + dy
        x = float(np.clip(x, 2, img_w - 2))
        y = float(np.clip(y, 2, img_w - 2))  # clip loosely; refined per-call below with real img_h
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def estimate_depth(image: Image.Image, img_w, img_h):
    from transformers import pipeline
    depther = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    result = depther(image)
    depth = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    if depth.shape != (img_h, img_w):
        depth = np.array(Image.fromarray(depth).resize((img_w, img_h), Image.BILINEAR), dtype=np.float32)
    return depth


def backproject(points_2d, depth_map, K, img_w, img_h):
    out = np.zeros((points_2d.shape[0], 3), dtype=np.float32)
    FX, CX, CY = K[0, 0], K[0, 2], K[1, 2]
    for i, (x, y) in enumerate(points_2d):
        xi, yi = int(np.clip(round(x), 0, img_w - 1)), int(np.clip(round(y), 0, img_h - 1))
        z = float(depth_map[yi, xi])
        if z <= 1e-3:
            y0, y1 = max(0, yi - 3), min(img_h, yi + 4)
            x0, x1 = max(0, xi - 3), min(img_w, xi + 4)
            patch = depth_map[y0:y1, x0:x1]
            z = float(np.median(patch[patch > 1e-3])) if (patch > 1e-3).any() else 0.6
        out[i] = [(x - CX) / FX * z, (y - CY) / FX * z, z]
    return out


def main():
    recs = json.loads((ROOT / "results" / "six_examples_records.json").read_text())
    for src, rec in recs.items():
        frame0_path = SRC_DIR / f"{src}_frame0.png"
        if not frame0_path.exists():
            print(f"[skip] {src}: {frame0_path} not downloaded")
            continue
        out_dir = OUT_ROOT / src
        out_dir.mkdir(parents=True, exist_ok=True)
        img_w, img_h = rec["meta_data"]["original_width"], rec["meta_data"]["original_height"]
        frame0 = Image.open(frame0_path).convert("RGB")
        if frame0.size != (img_w, img_h):
            img_w, img_h = frame0.size  # trust the actual downloaded image size

        FX = (img_w / 2) / np.tan(np.deg2rad(FOV_H_DEG) / 2)
        K = np.array([[FX, 0, img_w / 2], [0, FX, img_h / 2], [0, 0, 1]], dtype=np.float32)

        depth_map = estimate_depth(frame0, img_w, img_h)
        anchor = rec["trajectory"][0]
        radius = max(8, min(img_w, img_h) * 0.05)
        rng = np.random.default_rng(0)
        pts = [list(anchor)]
        while len(pts) < 8:
            dx, dy = rng.uniform(-radius, radius, size=2)
            x, y = anchor[0] + dx, anchor[1] + dy
            x, y = float(np.clip(x, 2, img_w - 2)), float(np.clip(y, 2, img_h - 2))
            pts.append([x, y])
        points_2d = np.array(pts, dtype=np.float32)

        points_3d_t0 = backproject(points_2d, depth_map, K, img_w, img_h)
        points_3d_history = np.repeat(points_3d_t0[None, :, :], 3, axis=0)

        torch.save(torch.from_numpy(points_2d), out_dir / "points_2d_at_t0.pt")
        torch.save(torch.from_numpy(points_3d_history), out_dir / "points_3d_history.pt")
        torch.save(torch.from_numpy(K), out_dir / "intrinsics_K.pt")
        for suffix in ("t-2", "t-1", "t+0"):
            frame0.save(out_dir / f"frame_{suffix}.jpg", quality=95)

        meta = {
            "id": f"sixex_{src}",
            "source_dataset": src,
            "action": rec["instruction"],
            "image_size_wh": [img_w, img_h],
            "assumed_fov_h_deg": FOV_H_DEG,
            "gt_2d_path_frame0": rec["trajectory"],
            "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            "notes": "H1-F32, 1 real frame, estimated depth + assumed intrinsics, same methodology as the "
                     "original bridge/dobbe H1 examples -- only the source dataset (robot platform/camera) differs.",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{src}: anchor depth {points_3d_t0[0,2]:.3f}m, saved to {out_dir}")


if __name__ == "__main__":
    main()
