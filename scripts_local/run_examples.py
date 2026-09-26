"""Run MolmoMotion on several example directories in ONE process (model loaded once), resume-safe.

Each example directory holds frame_t{-2,-1,+0}.jpg, points_2d_at_t0.pt, points_3d_history.pt, intrinsics_K.pt and
either caption.txt or meta.json (with "action"/"caption"). The authors' bundled examples live in
repos/molmo-motion/examples/data/<name>; our ShareRobot example in data/sharerobot_example.

For every example it stores <out>/<name>/pred.npz (+ run.json) with the prediction, the raw generated text and timing.
When the authors' released prediction exists (examples/data/predictions_h3.jsonl, H3 model only) it also reports how our
output differs from theirs: mean L2 distance and the first position where the generated texts diverge. DAVIS clips are
scored against PointMotionBench tracks (same GT construction as part1_infer_and_eval.py).

usage:
  python scripts_local/run_examples.py --ckpt checkpoints/MolmoMotion-4B-H3-F30 --out outputs/multi \
      --examples davis_bmx_trees davis_car_turn davis_flamingo
  python scripts_local/run_examples.py --ckpt checkpoints/MolmoMotion-4B-H1-F32 --future 32 --out outputs/h1 \
      --example-dirs data/sharerobot_example
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowmem_model import load_lowmem  # noqa: E402

EX = ROOT / "repos" / "molmo-motion" / "examples" / "data"
TRACKS = ROOT / "data" / "pointmotionbench" / "davis" / "tracks"
TAUS = (0.01, 0.02, 0.05, 0.10, 0.20)


def metrics(pred, gt, vis):
    err = np.linalg.norm(pred - gt, axis=-1)
    e = err[vis]
    fde = [err[p, np.where(vis[p])[0].max()] for p in range(len(err)) if vis[p].any()]
    return {"ADE_m": float(e.mean()), "FDE_m": float(np.mean(fde)),
            "PWT": float(np.mean([(e <= t).mean() for t in TAUS])), "visible_pairs": int(vis.sum())}


def davis_gt(meta, future):
    d = np.load(TRACKS / f"{meta['video']}_3d.npz", allow_pickle=True)["points_3d"].item()
    pts = d[meta["obj"]].astype(np.float32)
    t0 = meta["t0_absolute"]
    sub = pts[meta["point_indices"]][:, t0 + 1: t0 + 1 + future, :]
    return np.nan_to_num(sub, nan=0.0), np.isfinite(sub).all(-1)


def load_inputs(d: Path, H: int):
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    names = ["-2", "-1", "+0"][-H:]
    frames = [Image.open(d / f"frame_t{n}.jpg").convert("RGB") for n in names]
    p2d = torch.load(d / "points_2d_at_t0.pt")
    p3d = torch.load(d / "points_3d_history.pt")[-H:]
    K = torch.load(d / "intrinsics_K.pt")
    action = (d / "caption.txt").read_text().strip() if (d / "caption.txt").exists() else meta.get("caption") or meta["action"]
    return meta, frames, p2d, p3d, K, action


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--future", type=int, default=30)
    ap.add_argument("--examples", nargs="*", default=[], help="names under repos/molmo-motion/examples/data")
    ap.add_argument("--example-dirs", nargs="*", default=[], help="arbitrary example directories")
    a = ap.parse_args()

    dirs = [EX / n for n in a.examples] + [Path(p) if Path(p).is_absolute() else ROOT / p for p in a.example_dirs]
    out_root = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    todo = [d for d in dirs if not (out_root / d.name / "run.json").exists()]
    print(f"{len(dirs)} examples, {len(todo)} to run: {[d.name for d in todo]}", flush=True)
    if not todo:
        return

    from molmo_motion import MolmoMotionProcessor
    ckpt = ROOT / a.ckpt if not Path(a.ckpt).is_absolute() else Path(a.ckpt)
    processor = MolmoMotionProcessor.from_pretrained(str(ckpt))
    H = processor.config.history_size
    model = load_lowmem(str(ckpt))
    print(f"model ready, history H={H}", flush=True)

    cached = {}
    jl = EX / "predictions_h3.jsonl"
    if jl.exists() and "H3" in str(ckpt):
        for l in open(jl, encoding="utf-8"):
            r = json.loads(l)
            cached[r["video"]] = r

    for d in todo:
        meta, frames, p2d, p3d, K, action = load_inputs(d, H)
        inputs = processor(history_frames=frames, points_2d_at_t0=p2d, points_3d_history=p3d,
                           action=action, future_horizon=a.future)
        inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.predict_trajectory(**inputs)
        dt = time.perf_counter() - t
        pred = out.future_3d.float().cpu().numpy()
        rec = {"example": d.name, "action": action, "inference_seconds": dt, "checkpoint": str(a.ckpt),
               "history": H, "future": a.future, "gpu": torch.cuda.get_device_name(0), "text_len": len(out.future_text)}
        vid = meta.get("video")
        if meta.get("dataset") == "davis_bench" and (TRACKS / f"{vid}_3d.npz").exists() and a.future == 30:
            gt, vis = davis_gt(meta, a.future)
            rec["metrics_ours"] = metrics(pred, gt, vis)
            (out_root / d.name).mkdir(parents=True, exist_ok=True)
            np.savez(out_root / d.name / "gt.npz", gt=gt, vis=vis)
        c = cached.get(vid)
        if c is not None:
            ap_ = np.array(c["pred_raw_combined"], dtype=np.float32)
            if ap_.shape == pred.shape:
                rec["our_vs_authors_mean_L2_m"] = float(np.linalg.norm(pred - ap_, axis=-1).mean())
                if "metrics_ours" in rec:
                    rec["metrics_authors"] = metrics(ap_, gt, vis)
            at = c["rollouts"][0]["pred_text"]
            rec["text_common_prefix_chars"] = common_prefix(out.future_text, at)
            rec["text_len_authors"] = len(at)
            rec["first_divergence_ours"] = out.future_text[max(0, rec["text_common_prefix_chars"] - 20):rec["text_common_prefix_chars"] + 30]
            rec["first_divergence_authors"] = at[max(0, rec["text_common_prefix_chars"] - 20):rec["text_common_prefix_chars"] + 30]
        (out_root / d.name).mkdir(parents=True, exist_ok=True)
        np.savez(out_root / d.name / "pred.npz", pred=pred, points_2d_at_t0=p2d.numpy(), K=K.numpy())
        (out_root / d.name / "future_text.txt").write_text(out.future_text, encoding="utf-8")
        (out_root / d.name / "run.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps({k: v for k, v in rec.items() if k not in ("first_divergence_ours", "first_divergence_authors")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
