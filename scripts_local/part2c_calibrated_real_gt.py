"""Part 2c: use REAL ground-truth 3D end-effector trajectory for the planning-split episode, instead of
estimated/invented depth, by (1) finding the original BridgeData V2 episode ShareRobot's planning split was
built from, (2) reading its real proprioceptive state (x,y,z end-effector pose, measured, not estimated), and
(3) estimating the camera->robot extrinsics ourselves (solvePnP) from pre-grasp frames, where our dark-pixel
2D arm tracker and the robot's end-effector position are known to refer to the same physical point.

Why this exists: the user asked "are you sure that data (depth / camera calibration) doesn't exist in the
500GB ShareRobot archive?" -- it doesn't, but it turns out ShareRobot is a CURATED SUBSET of BridgeData V2
(confirmed via trajectory.json's "original_dataset": "bridge" field and matching the arxiv:2310.08864
Open-X-Embodiment citation on ShareRobot's HF page), and BridgeData V2's episodes (mirrored in LeRobot/parquet
format at IPEC-COMMUNITY/bridge_orig_lerobot) DO ship the real, measured WidowX end-effector pose per frame.
The exact episode was found by text-matching ShareRobot's VLM-paraphrased instruction ("move the pan towards
the right side of the yellow knife") against bridge_orig_lerobot's meta/tasks.jsonl -- a single near-exact
match ("Move the pan to the right of the yellow knife", task_index 17429) -- then meta/episodes.jsonl gives
exactly one episode with that task string: episode_index 45932 (length 44, matching our known frame range).
Frame 0 of that episode's video is visually identical to our already-extracted ShareRobot frame_0.png,
confirming the match beyond doubt.

Camera calibration note: our dark-pixel-centroid 2D tracker (find_arm_xy in part2b) tracks a DIFFERENT
physical point before vs. after the gripper grasps the pan (the visual blob changes from "gripper fingers
alone" to "gripper + metal pan"), so a single rigid camera pose fit across all 12 locally-held frames has
~30px mean reprojection error. Restricting the PnP solve to the 8 PRE-GRASP frames only (0,1,2,3,4,5,8,10,
where the gripper is still open and the tracked blob is consistently just the gripper) gives a much tighter
fit (~12px mean reprojection error on a 640x480 image, ~2% of image width) -- this calibration is then used
to transform the REAL robot-frame 3D state at the history/future frames directly, without needing the 2D
tracker to be correct at those frames at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part2b_build_real_history import find_arm_xy, sample_query_points  # noqa: E402

SRC_DIR = ROOT / "data" / "sharerobot_real_history"
BRIDGE_DIR = ROOT / "outputs" / "bridge_match"
OUT_DIR = ROOT / "outputs" / "part2c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 640, 480
FOV_H_DEG = 69.4  # still assumed -- we solve extrinsics given this, not full intrinsics+extrinsics jointly
FX = (IMG_W / 2) / np.tan(np.deg2rad(FOV_H_DEG) / 2)
K = np.array([[FX, 0, IMG_W / 2], [0, FX, IMG_H / 2], [0, 0, 1]], dtype=np.float64)

HIST_FRAMES = [15, 20, 25]
FUTURE_REAL_FRAME = 29
PREGRASP_FRAMES = [0, 1, 2, 3, 4, 5, 8, 10]  # gripper open; 2D tracker and robot state are the same point here
ACTION = "move the pan towards the right side of the yellow knife"


def calibrate_extrinsics():
    df = pd.read_parquet(BRIDGE_DIR / "episode_045932.parquet")
    obj, img = [], []
    for f in PREGRASP_FRAMES:
        x, y = find_arm_xy(SRC_DIR / f"frame_{f}.png") if (SRC_DIR / f"frame_{f}.png").exists() else (None, None)
        # all PREGRASP_FRAMES are known to exist locally (0,1,2,3,4,5,8,10)
        xyz = np.array(df.iloc[f]["observation.state"][:3], dtype=np.float64)
        obj.append(xyz)
        img.append([x, y])
    obj, img = np.array(obj), np.array(img)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(4))
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    return R, t, df, {"n_calib_frames": len(PREGRASP_FRAMES), "mean_reproj_err_px": float(err.mean()),
                       "max_reproj_err_px": float(err.max())}


def to_cam(df, frame_idx, R, t):
    xyz_robot = np.array(df.iloc[frame_idx]["observation.state"][:3], dtype=np.float64)
    return R @ xyz_robot + t  # (3,) camera-frame point


def project(xyz_cam):
    u = K[0, 0] * xyz_cam[0] / xyz_cam[2] + K[0, 2]
    v = K[1, 1] * xyz_cam[1] / xyz_cam[2] + K[1, 2]
    return np.array([u, v])


def main():
    R, t, df, calib_info = calibrate_extrinsics()
    print("Calibration (pre-grasp PnP):", calib_info)

    anchor_cam = {f: to_cam(df, f, R, t) for f in HIST_FRAMES + [FUTURE_REAL_FRAME]}
    for f, p in anchor_cam.items():
        print(f"frame {f}: REAL calibrated camera-frame anchor xyz = {np.round(p, 4)}")

    # points_2d_at_t0: anchor = projection of the REAL calibrated t0 (frame 25) position; 7 jittered neighbors
    anchor_2d_t0 = project(anchor_cam[HIST_FRAMES[-1]])
    points_2d_t0 = sample_query_points(anchor_2d_t0, n=8, radius=35, seed=0)

    points_3d_history = np.zeros((3, 8, 3), dtype=np.float32)
    z_t0 = anchor_cam[HIST_FRAMES[-1]][2]
    for fi, f in enumerate(HIST_FRAMES):
        if fi == len(HIST_FRAMES) - 1:
            points_3d_history[fi, 0] = anchor_cam[f]
            for j in range(1, 8):
                x, y = points_2d_t0[j]
                z = z_t0
                points_3d_history[fi, j] = [(x - K[0, 2]) / K[0, 0] * z, (y - K[1, 2]) / K[1, 1] * z, z]
        else:
            # earlier frames: anchor uses the REAL calibrated position; neighbors keep the t0 jitter offset
            # in pixel space, backprojected at that frame's REAL anchor depth (consistent with part2b's
            # "reuse one depth, only xy moves" fix, except the depth now comes from real state, not a guess)
            points_3d_history[fi, 0] = anchor_cam[f]
            shift_2d = project(anchor_cam[f]) - anchor_2d_t0
            z = anchor_cam[f][2]
            for j in range(1, 8):
                x, y = points_2d_t0[j] + shift_2d
                points_3d_history[fi, j] = [(x - K[0, 2]) / K[0, 0] * z, (y - K[1, 2]) / K[1, 1] * z, z]

    real_future_vec = anchor_cam[FUTURE_REAL_FRAME] - anchor_cam[HIST_FRAMES[-1]]
    real_history_vec = anchor_cam[HIST_FRAMES[-1]] - anchor_cam[HIST_FRAMES[0]]
    hf_cos = float(np.dot(real_history_vec, real_future_vec) /
                    (np.linalg.norm(real_history_vec) * np.linalg.norm(real_future_vec)))
    print("real history->future displacement cosine (sanity, real GT only):", hf_cos)

    # ---- run MolmoMotion H3 inference on this calibrated-real input ----
    from molmo_motion import MolmoMotionProcessor
    from lowmem_model import load_lowmem

    CKPT = ROOT / "checkpoints" / "MolmoMotion-4B-H3-F30"
    processor = MolmoMotionProcessor.from_pretrained(str(CKPT))
    model = load_lowmem(str(CKPT))

    history_frames = [Image.open(SRC_DIR / f"frame_{f}.png").convert("RGB") for f in HIST_FRAMES]
    inputs = processor(
        history_frames=history_frames,
        points_2d_at_t0=torch.from_numpy(points_2d_t0),
        points_3d_history=torch.from_numpy(points_3d_history),
        action=ACTION,
        future_horizon=30,
    )
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    import time
    t0_perf = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.predict_trajectory(**inputs)
    dt = time.perf_counter() - t0_perf

    pred = out.future_3d.float().cpu().numpy()  # (P, F, 3)
    pred_anchor_path = pred[0]  # (F, 3)
    pred_vec = pred_anchor_path[-1] - pred_anchor_path[0]
    pred_cos_vs_real_future = float(np.dot(pred_vec, real_future_vec) /
                                     (np.linalg.norm(pred_vec) * np.linalg.norm(real_future_vec) + 1e-9))
    pred_cos_vs_real_history = float(np.dot(pred_vec, real_history_vec) /
                                      (np.linalg.norm(pred_vec) * np.linalg.norm(real_history_vec) + 1e-9))

    result = {
        "id": "sharerobot_planning_bridge_episode_4263_CALIBRATED_REAL_GT",
        "bridge_orig_lerobot_episode_index": 45932,
        "task_index": 17429,
        "calibration": calib_info,
        "real_history_frame_future_cosine_sanity": hf_cos,
        "real_future_vec_cam": real_future_vec.tolist(),
        "real_history_vec_cam": real_history_vec.tolist(),
        "predicted_vec_cam": pred_vec.tolist(),
        "predicted_path_len_m": float(np.linalg.norm(pred_anchor_path[-1] - pred_anchor_path[0])),
        "direction_cosine_pred_vs_REAL_future": pred_cos_vs_real_future,
        "direction_cosine_pred_vs_real_history": pred_cos_vs_real_history,
        "inference_seconds": dt,
        "action": ACTION,
        "raw_model_text_head": out.future_text[:400],
        "note": "Unlike part2b (estimated monocular depth), points_3d_history here uses the REAL measured "
                "WidowX end-effector 3D position (from the original BridgeData V2 episode, transformed into "
                "camera frame via our own solvePnP-estimated extrinsics), and the comparison target "
                "(real_future_vec_cam) is also REAL measured ground truth, not a self-estimated reconstruction.",
    }
    print(json.dumps(result, indent=2))
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print("Saved to", OUT_DIR / "result.json")


if __name__ == "__main__":
    main()
