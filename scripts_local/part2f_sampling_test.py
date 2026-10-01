"""Part 2f: test whether the Part-2 "frozen/degenerate zero-motion" prediction is a greedy-decoding artifact,
fixable with a legitimate (non-training) decoding-strategy change -- not an input-construction issue.

predict_trajectory() (molmo_motion/modeling.py) always calls generate() with beam_size=1 and no sampler, i.e.
pure GREEDY decoding -- a well-known cause of repetitive/degenerate generation when a model is uncertain. The
underlying BeamSearch (molmo_motion/nn/beam_search.py) supports a `sampler` (MultinomialSampler etc., for
temperature-based stochastic decoding) and `constraints` (RepeatedNGramBlockingConstraint, which forbids
emitting an n-gram the model has already emitted) -- both fully supported, architecturally-provided generation
options, not a training/finetuning change and not touching the model's weights. This re-implements
predict_trajectory's logic directly (so we can pass sampler/constraints through, which the public wrapper
doesn't expose).

Works on any of the already-built real-history input directories (--data-dir), e.g.:
  data/sharerobot_real_history_inputs       (episode_4263, "move the pan...")
  data/sharerobot_real_history3_inputs      (episode_4801, "lift the pot...")

Saves the full predicted (P, F, 3) array (pred.npz) and a visualization (fig.png) overlaying real history
(green), real continuation (cyan) and the predicted path (magenta) on the real future frame, same style as
part2b/part2e's figures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from part2b_build_real_history import estimate_depth, sample_depth_at  # noqa: E402


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--ngram-block", type=int, default=0, help="0 disables; else RepeatedNGramBlockingConstraint size")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    DATA_DIR = ROOT / a.data_dir
    OUT_DIR = ROOT / a.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA_DIR / "meta.json").read_text())

    from molmo_motion import MolmoMotionProcessor
    from molmo_motion.nn.beam_search import MultinomialSampler, RepeatedNGramBlockingConstraint

    CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H3-F30"
    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    model = load_lowmem(str(CKPT))

    history_frames = [Image.open(DATA_DIR / f"frame_t{n}.jpg").convert("RGB") for n in ("-2", "-1", "+0")]
    points_2d_at_t0 = torch.load(DATA_DIR / "points_2d_at_t0.pt")
    points_3d_history = torch.load(DATA_DIR / "points_3d_history.pt")
    K = torch.load(DATA_DIR / "intrinsics_K.pt").numpy()

    inputs = processor(history_frames=history_frames, points_2d_at_t0=points_2d_at_t0,
                        points_3d_history=points_3d_history, action=meta["action"], future_horizon=30)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    # --- replicate predict_trajectory()'s logic, but with sampler/constraints passed through ---
    max_new_tokens = 160 * 30
    batch = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
    for k, v in inputs.items():
        if k in ("target_tokens", "loss_masks", "input_ids", "attention_mask"):
            continue
        batch[k] = v

    sampler = MultinomialSampler(temperature=a.temperature) if a.temperature != 0 else None
    constraints = [RepeatedNGramBlockingConstraint(ngram_size=a.ngram_block)] if a.ngram_block > 0 else None

    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen_out = model._internal.generate(
            batch=batch, max_steps=max_new_tokens, is_distributed=False,
            sampler=sampler, constraints=constraints,
        )
    dt = time.perf_counter() - t0

    token_ids = gen_out.token_ids[:, 0].detach().cpu().numpy()[0]
    tokenizer = model._internal.config.llm.build_tokenizer()
    future_text = tokenizer.decode(token_ids[token_ids >= 0])

    from molmo_motion.modeling import parse_tracks_text, tracks_to_array
    parsed = parse_tracks_text(future_text)
    anchor = points_3d_history[-1, 0].numpy()
    if parsed is None:
        pred = np.zeros((8, 30, 3), dtype=np.float32) + anchor[None, None, :]
    else:
        delta, _vis = tracks_to_array(parsed, num_points=8, num_frames=30, start_timestamp=3.0)
        pred = np.asarray(delta, dtype=np.float32) + anchor[None, None, :]

    pred_anchor_path = pred[0]
    pred_vec = pred_anchor_path[-1] - pred_anchor_path[0]
    path_len = float(np.linalg.norm(np.diff(pred_anchor_path, axis=0), axis=-1).sum())

    # real future anchor: from meta.json if present, else compute the same way part2b_infer_fixed.py did
    if "real_future_3d_anchor" in meta:
        real_future_3d = np.array(meta["real_future_3d_anchor"])
    else:
        future_frame_idx = meta["future_real_frame_index_in_episode"]
        arm_xy_future = meta["tracked_arm_xy_per_frame"][str(future_frame_idx)]
        t0_frame_idx = meta["history_frame_indices_in_episode"][-1]
        src_dir = ROOT / "data" / ("sharerobot_real_history" if "4263" in meta["id"] else "sharerobot_real_history3")
        depth_map_t0 = estimate_depth(Image.open(src_dir / f"frame_{t0_frame_idx}.png").convert("RGB"))
        z = sample_depth_at(arm_xy_future[0], arm_xy_future[1], depth_map_t0)
        real_future_3d = np.array([(arm_xy_future[0] - K[0, 2]) / K[0, 0] * z,
                                    (arm_xy_future[1] - K[1, 2]) / K[1, 1] * z, z])

    real_t0_3d = points_3d_history[-1, 0].numpy()
    real_history_3d = points_3d_history[0, 0].numpy()
    real_future_vec = real_future_3d - real_t0_3d
    real_history_vec = real_t0_3d - real_history_3d

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    result = {
        "id": meta["id"], "action": meta["action"],
        "temperature": a.temperature, "ngram_block": a.ngram_block, "seed": a.seed,
        "real_history_vec": real_history_vec.tolist(), "real_future_vec": real_future_vec.tolist(),
        "predicted_vec": pred_vec.tolist(), "predicted_path_len_m": path_len,
        "direction_cosine_pred_vs_real_future": cos(pred_vec, real_future_vec),
        "direction_cosine_pred_vs_real_history": cos(pred_vec, real_history_vec),
        "inference_seconds": dt,
        "future_text_sample": future_text[:800],
        "future_text_len": len(future_text),
    }
    print(json.dumps(result, indent=2))
    tag = f"temp{a.temperature}_ngram{a.ngram_block}_seed{a.seed}"
    (OUT_DIR / f"result_{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez(OUT_DIR / f"pred_{tag}.npz", pred=pred, K=K)

    # --- visualization ---
    pred_2d = project(pred, K)
    hist_2d = project(points_3d_history[:, 0].numpy(), K)
    future_2d = project(real_future_3d[None, :], K)[0]

    img = Image.open(DATA_DIR / "frame_future_real.jpg")
    fig, ax = plt.subplots(figsize=(7.5, 5.7))
    ax.imshow(img)
    ax.plot(hist_2d[:, 0], hist_2d[:, 1], "-o", c="lime", lw=2, label="real history (anchor)")
    ax.plot([hist_2d[-1, 0], future_2d[0]], [hist_2d[-1, 1], future_2d[1]], "-o", c="cyan", lw=2,
            label="real continuation")
    for p in range(pred_2d.shape[0]):
        ax.plot(pred_2d[p, :, 0], pred_2d[p, :, 1], "-", c="magenta", lw=1.3, alpha=0.85,
                label=f"predicted (T={a.temperature}, seed={a.seed})" if p == 0 else None)
        ax.plot(pred_2d[p, -1, 0], pred_2d[p, -1, 1], "x", c="magenta", ms=6)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{meta['id']}: sampled decoding (T={a.temperature}, ngram_block={a.ngram_block}, seed={a.seed})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_{tag}.png", dpi=130)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
