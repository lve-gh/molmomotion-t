"""Run MolmoMotion-4B-H3-F30 on the FIXED real-history ShareRobot input (part2b_build_real_history.py, after
removing the per-point depth-map-resampling bug -- see that script's docstring/notes) and score direction
against the real frame_29 continuation, backprojected with the same shared-anchor-depth method for consistency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowmem_model import load_lowmem  # noqa: E402
from part2b_build_real_history import (  # noqa: E402
    ACTION, FUTURE_REAL_FRAME, FX, FY, CX, CY, SRC_DIR, estimate_depth, sample_depth_at,
)

OUT_DIR = ROOT / "data" / "sharerobot_real_history_inputs"
RESULT_DIR = ROOT / "outputs" / "part2b_fixed"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from molmo_motion import MolmoMotionProcessor

    CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H3-F30"
    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    model = load_lowmem(str(CKPT))

    history_frames = [Image.open(OUT_DIR / f"frame_t{n}.jpg").convert("RGB") for n in ("-2", "-1", "+0")]
    points_2d_at_t0 = torch.load(OUT_DIR / "points_2d_at_t0.pt")
    points_3d_history = torch.load(OUT_DIR / "points_3d_history.pt")
    K = torch.load(OUT_DIR / "intrinsics_K.pt")

    inputs = processor(history_frames=history_frames, points_2d_at_t0=points_2d_at_t0,
                        points_3d_history=points_3d_history, action=ACTION, future_horizon=30)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    import time
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.predict_trajectory(**inputs)
    dt = time.perf_counter() - t0

    pred = out.future_3d.float().cpu().numpy()  # (P, F, 3)
    pred_anchor_path = pred[0]
    pred_vec = pred_anchor_path[-1] - pred_anchor_path[0]

    # Real frame_29 continuation: same tracked arm 2D position + the SAME shared-anchor-depth method (reuse the
    # t0 depth map sampled at its own location -- not the per-point bug, this is a single point anyway).
    meta = json.loads((OUT_DIR / "meta.json").read_text())
    arm_xy_29 = meta["tracked_arm_xy_per_frame"][str(FUTURE_REAL_FRAME)]
    depth_map_t0 = estimate_depth(Image.open(SRC_DIR / "frame_25.png").convert("RGB"))
    z29 = sample_depth_at(arm_xy_29[0], arm_xy_29[1], depth_map_t0)
    real_future_anchor = np.array([(arm_xy_29[0] - CX) / FX * z29, (arm_xy_29[1] - CY) / FY * z29, z29])
    real_t0_anchor = points_3d_history[-1, 0].numpy()
    real_future_vec = real_future_anchor - real_t0_anchor
    real_history_vec = points_3d_history[-1, 0].numpy() - points_3d_history[0, 0].numpy()

    cos_future = float(np.dot(pred_vec, real_future_vec) /
                        (np.linalg.norm(pred_vec) * np.linalg.norm(real_future_vec) + 1e-9))
    cos_history = float(np.dot(pred_vec, real_history_vec) /
                         (np.linalg.norm(pred_vec) * np.linalg.norm(real_history_vec) + 1e-9))

    result = {
        "id": "sharerobot_planning_bridge_episode_4263_FIXED_estimated_depth",
        "fix_applied": "all 8 query points now share one depth (the anchor's) per frame instead of each "
                        "re-sampling the depth map at its own shifted location -- see part2b_build_real_history.py",
        "real_history_vec": real_history_vec.tolist(),
        "real_future_vec": real_future_vec.tolist(),
        "predicted_vec": pred_vec.tolist(),
        "predicted_path_len_m": float(np.linalg.norm(np.diff(pred_anchor_path, axis=0), axis=-1).sum()),
        "direction_cosine_pred_vs_real_future": cos_future,
        "direction_cosine_pred_vs_real_history": cos_history,
        "inference_seconds": dt,
        "action": ACTION,
        "raw_model_text_head": out.future_text[:400],
    }
    print(json.dumps(result, indent=2))
    (RESULT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    np.savez(RESULT_DIR / "pred.npz", pred=pred, K=K.numpy())
    print("Saved to", RESULT_DIR / "result.json")


if __name__ == "__main__":
    main()
