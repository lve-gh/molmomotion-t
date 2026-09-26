"""Part 2: run MolmoMotion on the ShareRobot inputs built by
part2_build_inputs.py, and score the prediction with 2D-only metrics
(explicitly NOT the same as part 1's 3D ADE/FDE/PWT -- see docstring below
and the report for what each number actually measures).
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
CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H3-F30"
DATA_DIR = ROOT / "data" / "sharerobot_example"
OUT_DIR = ROOT / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUTURE_HORIZON = 30
IMG_DIAG = float(np.hypot(640, 480))


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def run_inference(meta):
    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    # low-memory loader (bf16 shards + GPU/CPU dispatch) -- see lowmem_model.py
    model = load_lowmem(str(CKPT))
    H = processor.config.history_size
    assert H == 3, H

    history_frames = [
        Image.open(DATA_DIR / f"frame_t{i}.jpg").convert("RGB")
        for i in ("-2", "-1", "+0")
    ]
    points_2d_at_t0 = torch.load(DATA_DIR / "points_2d_at_t0.pt")
    points_3d_history = torch.load(DATA_DIR / "points_3d_history.pt")
    intrinsics = torch.load(DATA_DIR / "intrinsics_K.pt")

    inputs = processor(
        history_frames=history_frames,
        points_2d_at_t0=points_2d_at_t0,
        points_3d_history=points_3d_history,
        action=meta["action"],
        future_horizon=FUTURE_HORIZON,
    )
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    import time
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.predict_trajectory(**inputs)
    dt = time.perf_counter() - t0

    return out, points_2d_at_t0, intrinsics, history_frames[-1], dt


def compute_2d_path_metrics(pred_2d_anchor, gt_path):
    """pred_2d_anchor: (F, 2) projected predicted path of the anchor point.
    gt_path: (K, 2) ShareRobot-annotated waypoints INCLUDING the t0 anchor
    at index 0 (so gt_path[1:] are the future waypoints to score against).

    These are coarse, NOT time-aligned: ShareRobot gives no per-frame
    timestamps for its waypoints, so we cannot compute a per-frame ADE like
    in Part 1. What we *can* measure and report honestly:
      - waypoint_to_path_px: for each GT future waypoint, the min pixel
        distance to any point on the predicted F-step path (does the
        predicted path pass near where the arm actually went?).
      - endpoint_px: pixel distance between the last GT waypoint and the
        LAST predicted step (a coarse proxy for FDE; conflates whatever the
        true frame offset is with genuine prediction error).
      - direction_cosine: cosine similarity between the GT displacement
        vector (last-first waypoint) and the predicted displacement vector
        (last-first predicted step) -- did the model get the *direction* of
        motion right, independent of exact magnitude/timing.
    All pixel numbers are also reported normalized by the image diagonal
    (800px for 640x480) as a resolution-independent fraction.
    """
    gt_future = np.asarray(gt_path[1:], dtype=np.float32)  # (K-1, 2)
    dists = np.linalg.norm(pred_2d_anchor[None, :, :] - gt_future[:, None, :], axis=-1)  # (K-1, F)
    waypoint_to_path_px = dists.min(axis=1)  # (K-1,)

    endpoint_px = float(np.linalg.norm(pred_2d_anchor[-1] - gt_future[-1]))

    gt_vec = gt_future[-1] - np.asarray(gt_path[0], dtype=np.float32)
    pred_vec = pred_2d_anchor[-1] - pred_2d_anchor[0]
    denom = np.linalg.norm(gt_vec) * np.linalg.norm(pred_vec)
    # undefined (None) when the model predicts no motion at all (zero vector)
    cos_sim = float(np.dot(gt_vec, pred_vec) / denom) if denom > 1e-6 else None

    return {
        "waypoint_to_path_px": waypoint_to_path_px.tolist(),
        "waypoint_to_path_px_mean": float(waypoint_to_path_px.mean()),
        "waypoint_to_path_frac_of_diag_mean": float(waypoint_to_path_px.mean() / IMG_DIAG),
        "endpoint_px": endpoint_px,
        "endpoint_frac_of_diag": endpoint_px / IMG_DIAG,
        "direction_cosine_similarity": cos_sim,
    }


def main():
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    print("meta:", meta)

    out, points_2d_at_t0, intrinsics, t0_image, infer_seconds = run_inference(meta)
    pred = out.future_3d.float().cpu().numpy()  # (P, F, 3)
    K = intrinsics.numpy()
    pred_2d = project(pred, K)  # (P, F, 2)

    metrics = compute_2d_path_metrics(pred_2d[0], meta["gt_2d_path_frame0"])
    metrics["inference_seconds"] = infer_seconds
    metrics["gpu"] = torch.cuda.get_device_name(0)
    metrics["checkpoint"] = "allenai/MolmoMotion-4B-H3-F30 (local model.pt, H3 pseudo-history)"
    metrics["action"] = meta["action"]
    metrics["raw_model_text_head"] = out.future_text[:600]  # sanity check of what the model actually emitted
    metrics["future_horizon"] = FUTURE_HORIZON
    metrics["what_this_measures"] = (
        "Coarse 2D pixel-space path agreement between the predicted "
        "projected trajectory of the anchor query point and ShareRobot's "
        "4-waypoint annotated gripper path. NOT a time-aligned per-frame 3D "
        "error (no such GT exists for ShareRobot); see compute_2d_path_metrics "
        "docstring for exact definitions."
    )
    print(json.dumps(metrics, indent=2))

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(OUT_DIR / "pred.npz", pred=pred, pred_2d=pred_2d,
              points_2d_at_t0=points_2d_at_t0.numpy(), K=K)
    t0_image.save(OUT_DIR / "t0_frame.jpg")
    print("Saved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
