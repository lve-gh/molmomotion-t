"""Part 2b: build MolmoMotion H3 inputs from a ShareRobot `planning` episode that has REAL consecutive frames
(unlike the `trajectory` split used in part2_build_inputs.py, which only ships 2 frames/episode).

Episode: rtx_frames_success_14/49_bridge#episode_4263 (BAAI/ShareRobot, `planning` split), action (from
`planning/train/future_prediction_task.json`): "move the pan towards the right side of the yellow knife". The
`planning` split's images are packed in a single ~510GB split tar.gz archive
(`planning/images/rt_frames_success.tar.gz.part.{aa..cq}`, 95 parts x 5.37GB); this episode happens to sit in the
first few MB of part .aa, so only that one part needs to be (partially) streamed -- see the report for how the
frames were extracted (`curl ... | gzip -dc | tar -x <specific members>`, stopped once found, no full download).

The object tracked is the robot ARM/gripper itself (the thing whose future position we ask the model to predict),
located automatically per frame as the centroid of near-black pixels in a fixed ROI (the arm is the only very dark
object in that region of the scene) -- this is coarse (lands near the wrist joint, not the fingertip exactly) but
consistent frame to frame, which is what matters for a real (non-zero, non-duplicated) velocity signal.

History: frames 15, 20, 25 of the episode (t-2, t-1, t0) -- picked because the arm's tracked x-position moves
consistently rightward across them (229 -> 295 -> 301 px), matching the episode's real motion (the arm carries the
pan to the right, ending near the blue holder). Frame 29 (4 frames past t0) is the only later real frame kept, for
a qualitative/directional check of the prediction (not a quantitative future GT -- no annotated path exists for
this episode, unlike the `trajectory`-split example).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "sharerobot_real_history"
OUT_DIR = ROOT / "data" / "sharerobot_real_history_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 640, 480
HIST_FRAMES = [15, 20, 25]  # t-2, t-1, t0
FUTURE_REAL_FRAME = 29
ACTION = "move the pan towards the right side of the yellow knife"
ARM_ROI = (120, 0, 420, 280)  # x0,y0,x1,y1: excludes the blue side panels and bottom-left mount hardware

FOV_H_DEG = 69.4  # same assumption as part2_build_inputs.py (no real intrinsics shipped)
FX = (IMG_W / 2) / np.tan(np.deg2rad(FOV_H_DEG) / 2)
FY = FX
CX, CY = IMG_W / 2, IMG_H / 2


def find_arm_xy(path: Path, roi=ARM_ROI) -> tuple[float, float]:
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = roi
    crop = im[y0:y1, x0:x1]
    dark = crop.sum(axis=-1) < 180  # the arm/gripper is the only near-black object in this ROI
    ys, xs = np.nonzero(dark)
    assert len(xs) >= 20, f"too few dark pixels in {path}"
    return float(xs.mean() + x0), float(ys.mean() + y0)


def sample_query_points(anchor, n=8, radius=35, seed=0):
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
    arm_xy = {i: find_arm_xy(SRC_DIR / f"frame_{i}.png") for i in HIST_FRAMES + [FUTURE_REAL_FRAME]}
    print("tracked arm 2D position per frame:", arm_xy)

    points_2d_t0 = sample_query_points(arm_xy[HIST_FRAMES[-1]])  # (8, 2), row 0 == anchor
    points_3d_history = np.zeros((3, 8, 3), dtype=np.float32)
    # Depth from an independent per-frame monocular estimate is NOT temporally consistent (a first pass gave
    # 1.11m -> 0.65m -> 0.30m across these 3 frames -- a 0.8m "motion" in a fraction of a second that isn't
    # physically real, just per-frame depth-model noise); it swamped the real, modest lateral (XY) motion and the
    # model reasonably predicted no further motion given a 3D history that didn't look physically plausible.
    # Fix: estimate depth ONCE at t0 and reuse that single depth map for all 3 history frames, so only the
    # anchor's real tracked XY position changes across frames (its Z is held at the t0 value) -- this keeps the
    # genuine lateral velocity signal while removing the spurious depth-jump noise.
    depth_map_t0 = estimate_depth(Image.open(SRC_DIR / f"frame_{HIST_FRAMES[-1]}.png").convert("RGB"))
    for fi, frame_idx in enumerate(HIST_FRAMES):
        if fi == len(HIST_FRAMES) - 1:
            points_2d = points_2d_t0  # t0: use all 8 sampled points
        else:
            # earlier frames: re-center the same 8-point cluster on that frame's tracked arm position
            # (rigid translation of the jittered cluster -- we don't re-detect each of the 7 jitter points
            # individually, only the anchor moves with the tracked arm).
            shift = np.array(arm_xy[frame_idx]) - np.array(arm_xy[HIST_FRAMES[-1]])
            points_2d = points_2d_t0 + shift[None, :]
            points_2d[:, 0] = np.clip(points_2d[:, 0], 2, IMG_W - 2)
            points_2d[:, 1] = np.clip(points_2d[:, 1], 2, IMG_H - 2)
        points_3d_history[fi] = backproject(points_2d, depth_map_t0)
        print(f"frame {frame_idx}: anchor 2D {points_2d[0]}, anchor depth {points_3d_history[fi, 0, 2]:.3f} m")

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
    torch.save(torch.from_numpy(points_2d_t0), OUT_DIR / "points_2d_at_t0.pt")
    torch.save(torch.from_numpy(points_3d_history), OUT_DIR / "points_3d_history.pt")
    torch.save(torch.from_numpy(K), OUT_DIR / "intrinsics_K.pt")
    for suffix, frame_idx in zip(("t-2", "t-1", "t+0"), HIST_FRAMES):
        Image.open(SRC_DIR / f"frame_{frame_idx}.png").convert("RGB").save(OUT_DIR / f"frame_{suffix}.jpg", quality=95)
    Image.open(SRC_DIR / f"frame_{FUTURE_REAL_FRAME}.png").convert("RGB").save(OUT_DIR / "frame_future_real.jpg", quality=95)

    meta = {
        "id": "sharerobot_planning_bridge_episode_4263",
        "dataset": "BAAI/ShareRobot planning split (real consecutive frames, unlike trajectory split)",
        "action": ACTION,
        "image_size_wh": [IMG_W, IMG_H],
        "assumed_fov_h_deg": FOV_H_DEG,
        "history_frame_indices_in_episode": HIST_FRAMES,
        "future_real_frame_index_in_episode": FUTURE_REAL_FRAME,
        "tracked_arm_xy_per_frame": {str(k): v for k, v in arm_xy.items()},
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "notes": "History is 3 REAL, distinct frames (not a replicated single frame): the tracked arm position "
                 "moves right across them (229 -> 295 -> 301 px), matching the episode's real rightward motion. "
                 "Depth + intrinsics are still model-estimated / assumed (same caveat as the main Part 2 example). "
                 "The query 'object' is the arm/gripper itself (auto-tracked via a dark-pixel centroid in a fixed "
                 "ROI, not hand-annotated), so the anchor point sits near the wrist joint rather than a specific "
                 "fingertip pixel.",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved real-history ShareRobot MolmoMotion inputs to", OUT_DIR)


if __name__ == "__main__":
    main()
