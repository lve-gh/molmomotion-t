"""Score the authors' released predictions (examples/data/predictions_h3.jsonl) on the DAVIS clips and compare
them with our own predictions (same inputs, our bf16 + GPU/CPU-split pipeline).

The jsonl ships predictions only (the GT fields are empty), so the ground truth is rebuilt from the PointMotionBench
`davis/tracks/<clip>_3d.npz` exactly as in part1_infer_and_eval.py (frames t0+1 .. t0+30 of the given points).
Writes results/multi_example_metrics.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "repos" / "molmo-motion" / "examples" / "data"
TR = ROOT / "data" / "pointmotionbench" / "davis" / "tracks"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
TAUS = (0.01, 0.02, 0.05, 0.10, 0.20)
F = 30


def metrics(pred, gt, vis):
    err = np.linalg.norm(pred - gt, axis=-1)
    e = err[vis]
    fde = [err[p, np.where(vis[p])[0].max()] for p in range(len(err)) if vis[p].any()]
    return {"ADE_m": float(e.mean()), "FDE_m": float(np.mean(fde)),
            "PWT": float(np.mean([(e <= t).mean() for t in TAUS])), "visible_pairs": int(vis.sum())}


def gt_for(meta):
    d = np.load(TR / f"{meta['video']}_3d.npz", allow_pickle=True)["points_3d"].item()
    pts = d[meta["obj"]].astype(np.float32)
    t0 = meta["t0_absolute"]
    sub = pts[meta["point_indices"]][:, t0 + 1: t0 + 1 + F, :]
    return np.nan_to_num(sub, nan=0.0), np.isfinite(sub).all(-1)


def main():
    rows = {json.loads(l)["video"]: json.loads(l) for l in open(EX / "predictions_h3.jsonl", encoding="utf-8")}
    res = {}
    for ex in ("davis_bmx_trees", "davis_car_turn", "davis_flamingo"):
        meta = json.loads((EX / ex / "meta.json").read_text())
        gt, vis = gt_for(meta)
        auth = np.array(rows[meta["video"]]["pred_raw_combined"], dtype=np.float32)
        r = {"authors_prediction": metrics(auth, gt, vis)}
        ours = ROOT / "outputs" / "multi" / ex / "pred.npz"      # written by run_examples.py
        if not ours.exists() and ex == "davis_bmx_trees":
            ours = ROOT / "outputs" / "part1" / "pred_and_gt.npz"
        if ours.exists():
            o = np.load(ours)["pred"]
            r["our_prediction"] = metrics(o, gt, vis)
            r["our_vs_authors_mean_L2_m"] = float(np.linalg.norm(o - auth, axis=-1).mean())
        res[meta["video"]] = r
    (OUT / "multi_example_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    a = [v["authors_prediction"] for v in res.values()]
    print("\nmean over clips (authors' predictions):", {k: round(float(np.mean([x[k] for x in a])), 3) for k in ("ADE_m", "FDE_m", "PWT")})


if __name__ == "__main__":
    main()
