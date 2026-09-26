"""Part 1: run MolmoMotion on the authors' bundled DAVIS bmx-trees example,
compare the predicted 3D trajectory against the real PointMotionBench GT
track, compute ADE / FDE / PWT, and render visual comparisons.

Ground-truth alignment (verified by hand against the bundled
points_3d_history.pt / points_2d_at_t0.pt tensors):
  - data/pointmotionbench/davis/tracks/bmx-trees_3d.npz holds
    {"bike_rider": (86, 80, 3), "bmx_bike": (80, 80, 3)} float32 arrays,
    frame-index-aligned 1:1 with the 80 raw DAVIS JPEGImages/480p/bmx-trees
    frames (verified: history frames [0,1,2] from the npz match the bundled
    points_3d_history.pt bit-for-bit).
  - meta.json for the bundled example gives: obj="bike_rider",
    t0_absolute=2, history_frame_indices=[0,1,2],
    point_indices=[9,12,17,18,27,29,32,34] (indices into the 86 tracked
    points), n_points=8.
  - The model's out.future_3d is documented + code-verified (modeling.py) to
    be ABSOLUTE camera-frame-at-t0 XYZ (delta + anchor added back), so GT for
    metrics is the raw (nan_to_num'd) npz future slice -- no further anchor
    arithmetic needed on our side.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowmem_model import load_lowmem

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "molmo-motion"
EXAMPLE_DIR = REPO / "examples" / "data" / "davis_bmx_trees"
TRACKS_3D = ROOT / "data" / "pointmotionbench" / "davis" / "tracks" / "bmx-trees_3d.npz"
TRACKS_2D = ROOT / "data" / "pointmotionbench" / "davis" / "tracks" / "bmx-trees_2d.npz"
DAVIS_FRAMES = ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees"
CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H3-F30"
OUT_DIR = ROOT / "outputs" / "part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUTURE_HORIZON = 30
PWT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)


def load_meta():
    return json.loads((EXAMPLE_DIR / "meta.json").read_text())


def load_gt(meta):
    """Return (gt_future_xyz (P,F,3), gt_vis (P,F)) aligned to the model's
    future_3d convention: frame t0_absolute+1 .. t0_absolute+F."""
    with np.load(TRACKS_3D, allow_pickle=True) as f:
        d = f["points_3d"].item()
    pts = d[meta["obj"]].astype(np.float32)  # (N, T, 3)
    point_indices = meta["point_indices"]
    t0 = meta["t0_absolute"]
    fut_idx = list(range(t0 + 1, t0 + 1 + FUTURE_HORIZON))
    T = pts.shape[1]
    assert fut_idx[-1] < T, f"need {fut_idx[-1]+1} frames, npz only has {T}"
    sub = pts[point_indices][:, fut_idx, :]  # (P, F, 3)
    vis = np.isfinite(sub).all(axis=-1)      # (P, F) -- before nan_to_num
    gt = np.nan_to_num(sub, nan=0.0)
    return gt, vis, fut_idx


def load_gt_2d(meta, fut_idx):
    with np.load(TRACKS_2D, allow_pickle=True) as f:
        tracks_raw = f["tracks"]
    d = tracks_raw.item()
    tr = d[meta["obj"]]  # (T, N, 2)
    point_indices = meta["point_indices"]
    return tr[fut_idx][:, point_indices, :].transpose(1, 0, 2)  # (P, F, 2)


def compute_metrics(pred, gt, vis):
    """pred, gt: (P, F, 3) numpy. vis: (P, F) bool. Mirrors
    launch_scripts/eval_pointmotionbench.py::_compute_metrics."""
    err = np.linalg.norm(pred - gt, axis=-1)  # (P, F)
    err_vis = err[vis]
    ade = float(err_vis.mean()) if err_vis.size else float("nan")
    pwt_per_tau = {}
    for tau in PWT_THRESHOLDS:
        pwt_per_tau[tau] = float((err_vis <= tau).mean()) if err_vis.size else 0.0
    pwt = float(np.mean(list(pwt_per_tau.values()))) if pwt_per_tau else 0.0
    fde_list = []
    for p in range(gt.shape[0]):
        v = vis[p]
        if v.any():
            last_visible = int(np.where(v)[0].max())
            fde_list.append(float(err[p, last_visible]))
    fde = float(np.mean(fde_list)) if fde_list else float("nan")
    return {
        "ADE_m": ade, "FDE_m": fde, "PWT": pwt,
        "PWT_per_threshold": {str(k): v for k, v in pwt_per_tau.items()},
        "n_points_visible_frames": int(err_vis.size),
    }


def run_inference(meta):
    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    # low-memory loader (bf16 shards + GPU/CPU dispatch) -- see lowmem_model.py
    model = load_lowmem(str(CKPT))
    H = processor.config.history_size

    history_frames = [
        Image.open(EXAMPLE_DIR / f"frame_t{i:+d}.jpg").convert("RGB")
        for i in range(-(H - 1), 1)
    ]
    points_2d_at_t0 = torch.load(EXAMPLE_DIR / "points_2d_at_t0.pt")
    points_3d_history = torch.load(EXAMPLE_DIR / "points_3d_history.pt")[-H:]
    intrinsics = torch.load(EXAMPLE_DIR / "intrinsics_K.pt")
    action = (EXAMPLE_DIR / "caption.txt").read_text().strip()

    inputs = processor(
        history_frames=history_frames,
        points_2d_at_t0=points_2d_at_t0,
        points_3d_history=points_3d_history,
        action=action,
        future_horizon=FUTURE_HORIZON,
    )
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    import time
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.predict_trajectory(**inputs)
    dt = time.perf_counter() - t0

    return out, points_2d_at_t0, intrinsics, history_frames[-1], dt


def main():
    meta = load_meta()
    print("meta:", meta)

    gt, vis, fut_idx = load_gt(meta)
    gt_2d = load_gt_2d(meta, fut_idx)
    print("GT future_3d shape", gt.shape, "visible entries", vis.sum(), "/", vis.size)

    out, points_2d_at_t0, intrinsics, t0_image, infer_seconds = run_inference(meta)
    pred = out.future_3d.float().cpu().numpy()  # (P, F, 3)
    print("pred shape", pred.shape, "inference time (s):", infer_seconds)

    metrics = compute_metrics(pred, gt, vis)
    metrics["inference_seconds"] = infer_seconds
    metrics["gpu"] = torch.cuda.get_device_name(0)
    metrics["checkpoint"] = "allenai/MolmoMotion-4B-H3-F30 (local model.pt)"
    metrics["video"] = meta["video"]
    metrics["raw_model_text_head"] = out.future_text[:600]  # sanity check of what the model actually emitted
    metrics["object"] = meta["obj"]
    metrics["future_horizon"] = FUTURE_HORIZON
    print(json.dumps(metrics, indent=2))

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(OUT_DIR / "pred_and_gt.npz", pred=pred, gt=gt, vis=vis,
              gt_2d=gt_2d, points_2d_at_t0=points_2d_at_t0.numpy(),
              intrinsics=intrinsics.numpy())
    t0_image.save(OUT_DIR / "t0_frame.jpg")
    print("Saved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
