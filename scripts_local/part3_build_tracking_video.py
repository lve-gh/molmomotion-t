"""Part 3: turn a MolmoMotion prediction into a DaS "tracking video" so the
predicted trajectory can drive Diffusion-as-Shader's trajectory-conditioned
generation (`demo.py --tracking_path ...`).

Why this exists: DaS's own `--object_motion up/down/left/right` only supports
a constant-direction straight-line translation of a masked image region. To
actually use *MolmoMotion's* predicted (possibly curved, non-uniform) 3D
trajectory as the control signal, we replicate DaS's MoGe-based single-image
tracking path (`demo.py`'s `tracking_method == "moge"` branch /
`ObjectMotionGenerator.apply_motion`) but replace its fixed linear motion
with the actual per-frame 2D displacement of MolmoMotion's query points,
resampled from F=30 (MolmoMotion's horizon) to DaS's fixed 49-frame clip.

Pipeline:
  1. Run MoGe (`Ruicheng/moge-vitl`, bundled DaS submodule) on the t0 image
     -> dense per-pixel 3D point cloud + intrinsics + valid mask.
  2. Build an object mask = filled convex hull of the 8 MolmoMotion query
     points (dilated a bit), i.e. "the tracked object's silhouette", since
     ShareRobot/DAVIS don't ship a segmentation mask either.
  3. mean 2D displacement per future frame across the 8 points (projected
     MolmoMotion prediction minus their t0 anchor) -> resample 30 -> 49
     frames (hold last value if the DaS clip runs longer than MolmoMotion's
     2s horizon).
  4. Apply that per-frame vector as a rigid translation of the masked MoGe
     points (same math as `ObjectMotionGenerator.apply_motion`, just with a
     time-varying vector instead of `base_vec * t`).
  5. Render with `DiffusionAsShaderPipeline.visualize_tracking_moge` and
     save an mp4 usable with `demo.py --tracking_path`.

This script only prepares the tracking video; it must be run inside the
DaS venv (torch cu118 + diffusers + the DaS submodules), separately from
`demo.py`'s actual generation call (documented in the report / run
instructions), because of the very different, much heavier dependency set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DAS_REPO = ROOT / "repos" / "DiffusionAsShader"
sys.path.insert(0, str(DAS_REPO))
sys.path.insert(0, str(DAS_REPO / "submodules" / "MoGe"))

NUM_FRAMES = 49


def project(xyz, K):
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def convex_hull_mask(points_2d, shape_hw, dilate_px=25):
    import cv2
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = points_2d.round().astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1)
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def resample_time(curve, n_out):
    """curve: (F_in, 2) -> (n_out, 2), linear interp, hold last value if
    n_out spans past F_in's covered duration."""
    f_in = curve.shape[0]
    x_in = np.linspace(0, 1, f_in)
    x_out = np.linspace(0, 1, n_out)
    out = np.zeros((n_out, 2), dtype=np.float32)
    for d in range(2):
        out[:, d] = np.interp(x_out, x_in, curve[:, d])
    return out


def build_tracking_video(t0_image_path: Path, pred_npz_path: Path, out_mp4: Path,
                          device="cpu", height=480, width=720, static=False):
    """Same geometry as demo.py's MoGe single-image branch: MoGe point cloud
    (camera space, normalized intrinsics) -> move the masked points -> project
    with MoGe's intrinsics (CameraMotionGenerator.w2s_moge, identity poses)
    -> `visualize_tracking_moge`. The only change is the motion source: a
    time-varying pixel displacement from MolmoMotion instead of a fixed
    up/down/left/right translation. A pixel shift (du, dv) at depth z is
    applied as dx = du/W * z / fx_n, dy = dv/H * z / fy_n (fx_n, fy_n = MoGe
    normalized focal lengths), so masked points move by exactly (du, dv)
    pixels in the DaS 480x720 frame.
    """
    from models.pipelines import DiffusionAsShaderPipeline, CameraMotionGenerator
    from moge.model.v1 import MoGeModel

    data = np.load(pred_npz_path)
    pred, points_2d_at_t0, K = data["pred"], data["points_2d_at_t0"], data["intrinsics"]
    pred_2d = project(pred, K)  # (P, F, 2) in ORIGINAL image pixels (t0 camera)
    mean_disp = (pred_2d - points_2d_at_t0[:, None, :]).mean(axis=0)  # (F, 2)

    img = Image.open(t0_image_path).convert("RGB")
    H0, W0 = img.height, img.width
    img_r = transforms.Resize((height, width))(img)
    sx, sy = width / W0, height / H0
    img_t = transforms.ToTensor()(img_r)

    moge = MoGeModel.from_pretrained(str(Path(__file__).resolve().parents[1] / "checkpoints" / "moge-vitl" / "model.pt")).to(device)
    with torch.no_grad():
        res = moge.infer(img_t.to(device))
    pts = res["points"].float().cpu()            # (H, W, 3)
    mask_valid = res["mask"].cpu().numpy().astype(bool)
    intr = res["intrinsics"].float().cpu()       # normalized K
    Hh, Ww = pts.shape[:2]
    fxn, fyn = float(intr[0, 0]), float(intr[1, 1])
    del moge

    p0 = points_2d_at_t0.copy(); p0[:, 0] *= sx; p0[:, 1] *= sy
    obj_mask = convex_hull_mask(p0, (Hh, Ww)) & mask_valid
    disp = mean_disp.copy(); disp[:, 0] *= sx; disp[:, 1] *= sy
    curve = resample_time(disp, NUM_FRAMES)       # (49, 2) pixels
    if static:                                     # no-trajectory baseline: nothing moves
        curve = curve * 0.0

    tracks = pts.unsqueeze(0).repeat(NUM_FRAMES, 1, 1, 1).reshape(NUM_FRAMES, -1, 3).clone()
    sel = torch.from_numpy(obj_mask.reshape(-1))
    z = tracks[0, sel, 2]
    for t in range(NUM_FRAMES):
        du, dv = float(curve[t, 0]), float(curve[t, 1])
        tracks[t, sel, 0] += du / Ww * z / fxn
        tracks[t, sel, 1] += dv / Hh * z / fyn

    cam = CameraMotionGenerator(None, device="cpu")
    cam.set_intr(intr)
    poses = torch.eye(4).unsqueeze(0).repeat(NUM_FRAMES, 1, 1)
    uvd = cam.w2s_moge(tracks, poses).reshape(NUM_FRAMES, Hh, Ww, 3)

    das = DiffusionAsShaderPipeline.__new__(DiffusionAsShaderPipeline)
    das.output_dir = str(out_mp4.parent); das.fps = 8
    _, tracking_tensor = das.visualize_tracking_moge(uvd.numpy(), mask_valid, save_tracking=False)

    from moviepy.editor import ImageSequenceClip
    frames = (tracking_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ImageSequenceClip(list(frames), fps=8).write_videofile(str(out_mp4), codec="libx264", fps=8, logger=None)
    img_r.save(out_mp4.parent / f"t0_{height}x{width}.png")
    np.save(out_mp4.parent / f"motion_curve_px_{height}x{width}{'_static' if static else ''}.npy", curve)
    print("Wrote tracking video to", out_mp4, "| masked points:", int(obj_mask.sum()),
          "| max shift px:", float(np.abs(curve).max()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-image", required=True)
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--static", action="store_true", help="baseline: zero motion tracking video")
    args = ap.parse_args()
    build_tracking_video(Path(args.t0_image), Path(args.pred_npz), Path(args.out), device=args.device, height=args.height, width=args.width, static=args.static)
