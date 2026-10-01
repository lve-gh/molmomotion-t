"""Run MolmoMotion-4B-H3-F30 on the second real-history ShareRobot episode (part2e_build_second_episode.py)."""
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

DATA_DIR = ROOT / "data" / "sharerobot_real_history3_inputs"
RESULT_DIR = ROOT / "outputs" / "part2e"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from molmo_motion import MolmoMotionProcessor

    CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H3-F30"
    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    model = load_lowmem(str(CKPT))

    meta = json.loads((DATA_DIR / "meta.json").read_text())
    history_frames = [Image.open(DATA_DIR / f"frame_t{n}.jpg").convert("RGB") for n in ("-2", "-1", "+0")]
    points_2d_at_t0 = torch.load(DATA_DIR / "points_2d_at_t0.pt")
    points_3d_history = torch.load(DATA_DIR / "points_3d_history.pt")
    K = torch.load(DATA_DIR / "intrinsics_K.pt")

    inputs = processor(history_frames=history_frames, points_2d_at_t0=points_2d_at_t0,
                        points_3d_history=points_3d_history, action=meta["action"], future_horizon=30)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    import time
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.predict_trajectory(**inputs)
    dt = time.perf_counter() - t0

    pred = out.future_3d.float().cpu().numpy()
    pred_anchor_path = pred[0]
    pred_vec = pred_anchor_path[-1] - pred_anchor_path[0]

    real_future_3d = np.array(meta["real_future_3d_anchor"])
    real_t0_3d = points_3d_history[-1, 0].numpy()
    real_history_3d = points_3d_history[0, 0].numpy()
    real_future_vec = real_future_3d - real_t0_3d
    real_history_vec = real_t0_3d - real_history_3d

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    result = {
        "id": meta["id"],
        "action": meta["action"],
        "real_history_vec": real_history_vec.tolist(),
        "real_future_vec": real_future_vec.tolist(),
        "predicted_vec": pred_vec.tolist(),
        "predicted_path_len_m": float(np.linalg.norm(np.diff(pred_anchor_path, axis=0), axis=-1).sum()),
        "direction_cosine_pred_vs_real_future": cos(pred_vec, real_future_vec),
        "direction_cosine_pred_vs_real_history": cos(pred_vec, real_history_vec),
        "real_history_to_future_cosine_sanity": cos(real_history_vec, real_future_vec),
        "inference_seconds": dt,
        "raw_model_text_head": out.future_text[:400],
    }
    print(json.dumps(result, indent=2))
    (RESULT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    np.savez(RESULT_DIR / "pred.npz", pred=pred, K=K.numpy())
    print("Saved to", RESULT_DIR / "result.json")


if __name__ == "__main__":
    main()
