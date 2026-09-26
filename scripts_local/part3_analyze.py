"""Part 3 analysis: compare DaS output (trajectory-conditioned) with the static-control
baseline and with the real continuation (DAVIS bmx-trees), quantitatively.

Metrics (all no-reference or reference-vs-t0/real; definitions in the report):
  * frame-0 fidelity: PSNR/SSIM(generated[0], t0 image resized to 480x720)
  * temporal consistency: mean SSIM between consecutive frames; mean |frame diff|
  * sharpness: variance of Laplacian per frame (drop over time = blur)
  * object recognizability: normalized cross-correlation (TM_CCOEFF_NORMED) of the t0 rider
    patch against each frame (max over the frame); "visible" if >= 0.5
  * motion adherence: template-track position of the rider vs. the commanded pixel
    displacement curve (RMS error, only on frames where the rider is recognized)
  * vs real continuation: SSIM/PSNR against DAVIS frame t0+1+k*29/48 (resized 720x480)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "outputs" / "part3"
DAVIS = ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees"
T0 = 2
H, W = 480, 720
VIS_THR = 0.5


def read_video(p):
    r = imageio.get_reader(p)
    fs = [np.asarray(f) for f in r]
    r.close()
    return fs


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)


def lap_var(f):
    return float(cv2.Laplacian(gray(f), cv2.CV_64F).var())


def video_stats(fs, t0_img):
    s = [ssim(gray(fs[i]), gray(fs[i + 1]), data_range=255) for i in range(len(fs) - 1)]
    d = [float(np.abs(fs[i].astype(float) - fs[i + 1].astype(float)).mean()) for i in range(len(fs) - 1)]
    return {
        "frame0_psnr_vs_t0": float(psnr(t0_img, fs[0], data_range=255)),
        "frame0_ssim_vs_t0": float(ssim(gray(t0_img), gray(fs[0]), data_range=255)),
        "consec_ssim_mean": float(np.mean(s)),
        "consec_ssim_min": float(np.min(s)),
        "consec_absdiff_mean": float(np.mean(d)),
        "laplacian_var_first5": float(np.mean([lap_var(f) for f in fs[:5]])),
        "laplacian_var_last5": float(np.mean([lap_var(f) for f in fs[-5:]])),
    }


def track(fs, tmpl):
    out = []
    for f in fs:
        res = cv2.matchTemplate(gray(f), tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        out.append((float(mx), loc[0], loc[1]))
    return np.array(out)  # (T, 3): score, x, y


def main():
    tracked = read_video(P3 / "generated_tracked_480x720_cfg1_10steps_offload.mp4")
    static_p = P3 / "generated_static_480x720_cfg1_10steps_offload.mp4"
    static = read_video(static_p) if static_p.exists() else None
    t0_img = np.asarray(cv2.cvtColor(cv2.imread(str(P3 / "t0_480x720.png")), cv2.COLOR_BGR2RGB))
    curve = np.load(P3 / "motion_curve_px.npy")  # (49, 2) commanded displacement, px in 480x720

    d = np.load(ROOT / "outputs" / "part1" / "pred_and_gt.npz")
    p0 = d["points_2d_at_t0"].copy()
    p0[:, 0] *= W / 854.0
    p0[:, 1] *= H / 480.0
    m = 30
    x0, x1 = int(max(p0[:, 0].min() - m, 0)), int(min(p0[:, 0].max() + m, W))
    y0, y1 = int(max(p0[:, 1].min() - m, 0)), int(min(p0[:, 1].max() + m, H))
    tmpl = gray(t0_img)[y0:y1, x0:x1]

    # real continuation mapped to the 49 generated frames (30 predicted steps -> 49 frames)
    real = []
    for k in range(49):
        j = T0 + 1 + int(round(k * 29 / 48))
        im = cv2.cvtColor(cv2.imread(str(DAVIS / f"{j:05d}.jpg")), cv2.COLOR_BGR2RGB)
        real.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))

    res = {"template_bbox_xyxy": [x0, y0, x1, y1]}
    vids = {"tracked": tracked, "real_continuation": real}
    if static is not None:
        vids["static_baseline"] = static
    trk = {}
    for name, fs in vids.items():
        st = video_stats(fs, t0_img)
        tr = track(fs, tmpl)
        trk[name] = tr
        vis = tr[:, 0] >= VIS_THR
        st["rider_visible_frames(ncc>=0.5)"] = int(vis.sum())
        st["rider_ncc_frame0"] = float(tr[0, 0])
        st["rider_ncc_mean"] = float(tr[:, 0].mean())
        if name != "real_continuation":
            disp = tr[:, 1:] - tr[0, 1:]
            n = vis.sum()
            if n > 1:
                err = np.linalg.norm(disp[vis] - curve[vis], axis=1)
                st["motion_rms_err_px_vs_commanded(visible frames)"] = float(np.sqrt((err ** 2).mean()))
                st["mean_tracked_dx_px(visible)"] = float(disp[vis, 0].mean())
                st["mean_commanded_dx_px(visible frames)"] = float(curve[vis, 0].mean())
                fr = np.arange(len(fs))[vis]
                st["slope_tracked_dx_px_per_frame"] = float(np.polyfit(fr, disp[vis, 0], 1)[0])
                st["slope_commanded_dx_px_per_frame"] = float(np.polyfit(fr, curve[vis, 0], 1)[0])
                st["pearson_r_tracked_vs_commanded_dx"] = float(np.corrcoef(disp[vis, 0], curve[vis, 0])[0, 1])
                lost = np.where(~vis)[0]
                st["first_frame_rider_lost(ncc<0.5)"] = int(lost[0]) if len(lost) else None
                st["commanded_dx_at_frame0_px(offset baked into tracking video)"] = float(curve[0, 0])
        if name != "real_continuation":
            st["ssim_vs_real_mean"] = float(np.mean([ssim(gray(a), gray(b), data_range=255) for a, b in zip(fs, real)]))
            st["psnr_vs_real_mean"] = float(np.mean([psnr(b, a, data_range=255) for a, b in zip(fs, real)]))
        res[name] = st

    (P3 / "metrics_part3.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ---- figure: frames
    idx = [0, 12, 24, 36, 48]
    rows = [("real continuation (DAVIS)", real), ("tracking video (control)", read_video(P3 / "tracking_video.mp4")),
            ("DaS, MolmoMotion trajectory", tracked)]
    if static is not None:
        rows.append(("DaS, static control (no motion)", static))
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(3.2 * len(idx), 2.2 * len(rows)))
    for r, (title, fs) in enumerate(rows):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(fs[i]); ax.axis("off")
            if r == 0:
                ax.set_title(f"frame {i}", fontsize=9)
            if c == 0:
                ax.text(-0.02, 0.5, title, transform=ax.transAxes, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(P3 / "fig_part3_frames.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- figure: curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for name, tr in trk.items():
        ax[0].plot(tr[:, 0], label=name)
    ax[0].axhline(VIS_THR, color="k", ls=":"); ax[0].set_title("rider template NCC per frame"); ax[0].legend(fontsize=7)
    ax[1].plot(curve[:, 0], "k--", label="commanded dx")
    for name in ("tracked", "static_baseline"):
        if name in trk:
            tr = trk[name]; vis = tr[:, 0] >= VIS_THR
            xs = np.where(vis, tr[:, 1] - tr[0, 1], np.nan)
            ax[1].plot(xs, label=f"{name} tracked dx (visible only)")
    ax[1].set_title("horizontal displacement (px)"); ax[1].legend(fontsize=7)
    for name, fs in vids.items():
        ax[2].plot([lap_var(f) for f in fs], label=name)
    ax[2].set_title("sharpness (Laplacian variance)"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(P3 / "fig_part3_curves.png", dpi=110)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
