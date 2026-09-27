"""Part 3, extended: compare every DaS variant in outputs/part3b (fixed tracking video, per-point motion, CFG / steps
ablations, full 50-step CFG-6 configuration, static control) and the original run in outputs/part3.

For each generated video: frame-0 fidelity, temporal consistency, sharpness, rider recognizability (template NCC),
motion adherence to the commanded curve, SSIM / PSNR / LPIPS against the real continuation (DAVIS frames mapped to the 49
generated frames as in part3_analyze.py). LPIPS needs `pip install lpips` (alexnet backbone) and is skipped without it.

usage (MolmoMotion environment, repo root): python scripts_local/part3b_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part3_analyze import DAVIS, H, T0, VIS_THR, W, gray, lap_var, read_video, track, video_stats  # noqa: E402

P3, P3B = ROOT / "outputs" / "part3", ROOT / "outputs" / "part3b"


def lpips_fn():
    try:
        import lpips
        import torch
        net = lpips.LPIPS(net="alex", verbose=False).eval()

        def f(a, b):
            ta = torch.from_numpy(a).permute(2, 0, 1)[None].float() / 127.5 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None].float() / 127.5 - 1
            with torch.no_grad():
                return float(net(ta, tb))
        return f
    except Exception as e:  # noqa: BLE001
        print("LPIPS unavailable:", e)
        return None


def main():
    t0_img = cv2.cvtColor(cv2.imread(str(P3 / "t0_480x720.png")), cv2.COLOR_BGR2RGB)
    d = np.load(ROOT / "outputs" / "part1" / "pred_and_gt.npz")
    p0 = d["points_2d_at_t0"].copy()
    p0[:, 0] *= W / 854.0
    p0[:, 1] *= H / 480.0
    m = 30
    x0, x1 = int(max(p0[:, 0].min() - m, 0)), int(min(p0[:, 0].max() + m, W))
    y0, y1 = int(max(p0[:, 1].min() - m, 0)), int(min(p0[:, 1].max() + m, H))
    tmpl = gray(t0_img)[y0:y1, x0:x1]
    real = []
    for k in range(49):
        j = T0 + 1 + int(round(k * 29 / 48))
        real.append(cv2.resize(cv2.cvtColor(cv2.imread(str(DAVIS / f"{j:05d}.jpg")), cv2.COLOR_BGR2RGB), (W, H),
                               interpolation=cv2.INTER_AREA))
    lp = lpips_fn()
    fixed_curve = np.load(P3B / "tracking_mean_fixed_motion_curve_px.npy")
    old_curve = np.load(P3 / "motion_curve_px.npy")

    runs = {"old_offset_tracking_10steps_cfg1": (P3 / "generated_tracked_480x720_cfg1_10steps_offload.mp4", old_curve),
            "old_static_10steps_cfg1": (P3 / "generated_static_480x720_cfg1_10steps_offload.mp4", None)}
    for p in sorted(P3B.glob("gen_*.mp4")):
        runs[p.stem] = (p, None if "static" in p.stem else fixed_curve)
    runs["real_continuation"] = (None, None)

    res = {}
    for name, (path, curve) in runs.items():
        if path is not None and not path.exists():
            continue
        fs = real if path is None else read_video(path)
        st = video_stats(fs, t0_img)
        tr = track(fs, tmpl)
        vis = tr[:, 0] >= VIS_THR
        st["rider_visible_frames(ncc>=0.5)"] = int(vis.sum())
        st["rider_ncc_mean"] = float(tr[:, 0].mean())
        if curve is not None and vis.sum() > 2:
            disp = tr[:, 1:] - tr[0, 1:]
            err = np.linalg.norm(disp[vis] - curve[vis], axis=1)
            fr = np.arange(len(fs))[vis]
            st["motion_rms_err_px_vs_commanded(visible frames)"] = float(np.sqrt((err ** 2).mean()))
            st["slope_tracked_dx_px_per_frame"] = float(np.polyfit(fr, disp[vis, 0], 1)[0])
            st["slope_commanded_dx_px_per_frame"] = float(np.polyfit(fr, curve[vis, 0], 1)[0])
            st["pearson_r_tracked_vs_commanded_dx"] = float(np.corrcoef(disp[vis, 0], curve[vis, 0])[0, 1])
            lost = np.where(~vis)[0]
            st["first_frame_rider_lost(ncc<0.5)"] = int(lost[0]) if len(lost) else None
        if path is not None:
            st["ssim_vs_real_mean"] = float(np.mean([ssim(gray(a), gray(b), data_range=255) for a, b in zip(fs, real)]))
            st["psnr_vs_real_mean"] = float(np.mean([psnr(b, a, data_range=255) for a, b in zip(fs, real)]))
            if lp is not None:
                st["lpips_vs_real_mean"] = float(np.mean([lp(a, b) for a, b in zip(fs, real)]))
        res[name] = st
        print(name, {k: round(v, 3) for k, v in st.items() if isinstance(v, float) and k in
                     ("ssim_vs_real_mean", "psnr_vs_real_mean", "lpips_vs_real_mean", "pearson_r_tracked_vs_commanded_dx",
                      "slope_tracked_dx_px_per_frame", "consec_ssim_mean")}, "visible", st["rider_visible_frames(ncc>=0.5)"], flush=True)
    (ROOT / "results" / "part3_ablations.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
