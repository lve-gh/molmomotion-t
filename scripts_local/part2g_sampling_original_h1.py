"""Part 2g: re-run the VERY FIRST ShareRobot example (episode_25423, "reach for the spoon", H1-F32, single
real frame -- the one whose README caption says "the model has no way to tell which direction the gripper is
already moving in, so it is essentially guessing", cosine -0.62) with the sampled-decoding fix from attempt 7
(results/part2_direction_investigation.json) instead of greedy decoding. Same checkpoint, same real input,
same 2D-path evaluation as the original (part2_score_h1.py) -- only the decoding strategy changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowmem_model import load_lowmem  # noqa: E402

DATA_DIR = ROOT / "data" / "sharerobot_example"
OUT_DIR = ROOT / "outputs" / "part2g"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPERATURE = 0.8
SEED = 42


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def main():
    torch.manual_seed(SEED)
    meta = json.loads((DATA_DIR / "meta.json").read_text())

    from molmo_motion import MolmoMotionProcessor
    from molmo_motion.nn.beam_search import MultinomialSampler

    CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H1-F32"
    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    H = processor.config.history_size
    assert H == 1, H
    model = load_lowmem(str(CKPT))

    history_frames = [Image.open(DATA_DIR / "frame_t+0.jpg").convert("RGB")]
    points_2d_at_t0 = torch.load(DATA_DIR / "points_2d_at_t0.pt")
    points_3d_history = torch.load(DATA_DIR / "points_3d_history.pt")[-1:]  # (1, 8, 3)
    K = torch.load(DATA_DIR / "intrinsics_K.pt").numpy()

    inputs = processor(history_frames=history_frames, points_2d_at_t0=points_2d_at_t0,
                        points_3d_history=points_3d_history, action=meta["action"], future_horizon=32)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    max_new_tokens = 160 * 32
    batch = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
    for k, v in inputs.items():
        if k in ("target_tokens", "loss_masks", "input_ids", "attention_mask"):
            continue
        batch[k] = v

    sampler = MultinomialSampler(temperature=TEMPERATURE)

    import time
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen_out = model._internal.generate(
            batch=batch, max_steps=max_new_tokens, is_distributed=False, sampler=sampler,
        )
    dt = time.perf_counter() - t0

    token_ids = gen_out.token_ids[:, 0].detach().cpu().numpy()[0]
    tokenizer = model._internal.config.llm.build_tokenizer()
    future_text = tokenizer.decode(token_ids[token_ids >= 0])

    from molmo_motion.modeling import parse_tracks_text, tracks_to_array
    parsed = parse_tracks_text(future_text)
    anchor = points_3d_history[-1, 0].numpy()
    if parsed is None:
        pred = np.zeros((8, 32, 3), dtype=np.float32) + anchor[None, None, :]
    else:
        delta, _vis = tracks_to_array(parsed, num_points=8, num_frames=32, start_timestamp=float(H))
        pred = np.asarray(delta, dtype=np.float32) + anchor[None, None, :]

    pred_2d = project(pred, K)

    # same 2D-path metric as the original H1 run (part2_infer_and_eval.compute_2d_path_metrics)
    gt_path = np.asarray(meta["gt_2d_path_frame0"], dtype=np.float32)  # (4, 2), index 0 = t0 anchor
    gt_future = gt_path[1:]
    pred_anchor_2d = pred_2d[0]  # (F, 2)
    dists = np.linalg.norm(pred_anchor_2d[None, :, :] - gt_future[:, None, :], axis=-1)
    waypoint_to_path_px = dists.min(axis=1)
    endpoint_px = float(np.linalg.norm(pred_anchor_2d[-1] - gt_future[-1]))
    gt_vec = gt_future[-1] - gt_path[0]
    pred_vec = pred_anchor_2d[-1] - pred_anchor_2d[0]
    denom = np.linalg.norm(gt_vec) * np.linalg.norm(pred_vec)
    cos_sim = float(np.dot(gt_vec, pred_vec) / denom) if denom > 1e-6 else None

    result = {
        "id": meta["id"], "action": meta["action"], "checkpoint": "H1-F32",
        "decoding": f"sampled, temperature={TEMPERATURE}, seed={SEED} (vs. original greedy)",
        "original_greedy_cosine": -0.62,
        "direction_cosine_similarity": cos_sim,
        "waypoint_to_path_px_mean": float(waypoint_to_path_px.mean()),
        "endpoint_px": endpoint_px,
        "predicted_anchor_displacement_px": float(np.linalg.norm(pred_anchor_2d[-1] - pred_anchor_2d[0])),
        "predicted_3d_path_length_m": float(np.linalg.norm(np.diff(pred[0], axis=0), axis=-1).sum()),
        "inference_seconds": dt,
        "future_text_sample": future_text[:600],
    }
    print(json.dumps(result, indent=2))
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    np.savez(OUT_DIR / "pred.npz", pred=pred, K=K)

    img = Image.open(DATA_DIR / "frame_future_real.jpg")
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.imshow(img)
    ax.plot(gt_path[:, 0], gt_path[:, 1], "-o", c="lime", lw=2, label="annotated gripper path (ShareRobot)")
    ax.plot(gt_path[0, 0], gt_path[0, 1], "*", c="yellow", ms=16, mec="k")
    for p in range(pred_2d.shape[0]):
        ax.plot(pred_2d[p, :, 0], pred_2d[p, :, 1], "-", c="magenta", lw=1.3, alpha=0.85,
                label=f"predicted (T={TEMPERATURE}, seed={SEED})" if p == 0 else None)
        ax.plot(pred_2d[p, -1, 0], pred_2d[p, -1, 1], "x", c="magenta", ms=8, mew=2)
        ax.plot(pred_2d[p, 0, 0], pred_2d[p, 0, 1], "o", c="magenta", ms=6, mec="white", mew=1)
    all_x = np.concatenate([pred_2d[:, :, 0].ravel(), gt_path[:, 0], [0, img.width]])
    all_y = np.concatenate([pred_2d[:, :, 1].ravel(), gt_path[:, 1], [0, img.height]])
    pad = 20
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.max() + pad, all_y.min() - pad)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"episode_25423 (H1-F32): sampled decoding T={TEMPERATURE} vs. original greedy (cosine -0.62)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig.png", dpi=130)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
