"""Extra trajectory metrics beyond ADE / FDE / PWT, for every clip in outputs/multi (ours) and, for DAVIS clips, for the
authors' released prediction (examples/data/predictions_h3.jsonl).

  * ADE / FDE / PWT (same as the official evaluation, visible pairs only)
  * normalized ADE / FDE: error divided by the ground-truth path length of the point (scale-free)
  * within-tolerance fraction: share of visible pairs whose error is below 10 % of the point's GT path length
  * Frechet distance and DTW (mean per point, in m): shape agreement that ignores the timing of the motion
  * direction cosine: cosine between predicted and real net displacement (start -> end) of each point, mean over points
  * speed ratio: predicted / real path length
  * common-motion-removed ADE: ADE after subtracting each trajectory's mean displacement offset (removes a shared shift,
    i.e. tests whether the shape/motion is right even when the position is off)

usage: python scripts_local/extra_metrics.py [--multi outputs/multi] [--out results/extra_metrics.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "repos" / "molmo-motion" / "examples" / "data"
TAUS = (0.01, 0.02, 0.05, 0.10, 0.20)


def frechet(p: np.ndarray, q: np.ndarray) -> float:
    n, m = len(p), len(q)
    d = np.linalg.norm(p[:, None] - q[None], axis=-1)
    ca = np.full((n, m), np.inf)
    ca[0, 0] = d[0, 0]
    for i in range(n):
        for j in range(m):
            if i == j == 0:
                continue
            best = min(ca[i - 1, j] if i else np.inf, ca[i, j - 1] if j else np.inf, ca[i - 1, j - 1] if i and j else np.inf)
            ca[i, j] = max(best, d[i, j])
    return float(ca[-1, -1])


def dtw(p: np.ndarray, q: np.ndarray) -> float:
    n, m = len(p), len(q)
    d = np.linalg.norm(p[:, None] - q[None], axis=-1)
    c = np.full((n + 1, m + 1), np.inf)
    c[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c[i, j] = d[i - 1, j - 1] + min(c[i - 1, j], c[i, j - 1], c[i - 1, j - 1])
    return float(c[n, m] / max(n, m))


def path_len(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(x, axis=0), axis=-1).sum()) if len(x) > 1 else 0.0


def score(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray) -> dict:
    err = np.linalg.norm(pred - gt, axis=-1)
    ade = float(err[vis].mean())
    fde, nade, nfde, inside, fr, dt, cos, spd, ade_cm = [], [], [], [], [], [], [], [], []
    for p in range(len(pred)):
        v = np.where(vis[p])[0]
        if len(v) < 2:
            continue
        g, q = gt[p, v], pred[p, v]
        L = max(path_len(g), 1e-6)
        fde.append(err[p, v[-1]])
        nade.append(err[p, v].mean() / L)
        nfde.append(err[p, v[-1]] / L)
        inside.append((err[p, v] <= 0.1 * L).mean())
        fr.append(frechet(q, g))
        dt.append(dtw(q, g))
        dg, dq = g[-1] - g[0], q[-1] - q[0]
        cos.append(float(dg @ dq / (np.linalg.norm(dg) * np.linalg.norm(dq) + 1e-9)))
        spd.append(path_len(q) / L)
        off = (q - g).mean(0)
        ade_cm.append(np.linalg.norm(q - off - g, axis=-1).mean())
    return {"ADE_m": ade, "FDE_m": float(np.mean(fde)),
            "PWT": float(np.mean([(err[vis] <= t).mean() for t in TAUS])),
            "normalized_ADE": float(np.mean(nade)), "normalized_FDE": float(np.mean(nfde)),
            "within_10pct_of_path_length": float(np.mean(inside)),
            "frechet_m": float(np.mean(fr)), "dtw_m": float(np.mean(dt)),
            "direction_cosine": float(np.mean(cos)), "speed_ratio": float(np.mean(spd)),
            "ADE_after_removing_mean_offset_m": float(np.mean(ade_cm)), "points_scored": len(fde)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", default="outputs/multi")
    ap.add_argument("--out", default="results/extra_metrics.json")
    a = ap.parse_args()
    cached = {}
    jl = EX / "predictions_h3.jsonl"
    if jl.exists():
        for l in open(jl, encoding="utf-8"):
            r = json.loads(l)
            cached[r["video"]] = r
    res = {}
    for d in sorted((ROOT / a.multi).glob("*")):
        if not (d / "gt.npz").exists() or not (d / "pred.npz").exists():
            continue
        g = np.load(d / "gt.npz")
        meta = json.loads((EX / d.name / "meta.json").read_text())
        row = {"ours": score(np.load(d / "pred.npz")["pred"], g["gt"], g["vis"])}
        c = cached.get(meta["video"])
        if c is not None:
            row["authors_released"] = score(np.array(c["pred_raw_combined"], dtype=np.float32), g["gt"], g["vis"])
        res[d.name] = row
        print(d.name, json.dumps({k: {m: round(v, 3) for m, v in s.items() if m in ("ADE_m", "FDE_m", "PWT", "frechet_m", "direction_cosine")}
                                  for k, s in row.items()}))
    (ROOT / a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
