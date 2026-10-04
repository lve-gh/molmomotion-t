"""Kaggle script kernel: DaS trajectory-conditioned generation in FULL bf16 (no NF4 quantization),
split across 2x T4 GPUs (32GB combined) via accelerate, to test whether quantization error is a
contributing factor to the seed-to-seed instability found locally with NF4 + block_offload on 8GB VRAM
(median r=-0.35 across 14 seeds, see molmomotion repo results/part3_scheduler_diagnosis.json).

Same config as the local sweep: 10 inference steps, CFG=1.0 (guidance_scale), DPM scheduler
(timestep_spacing="trailing"), 480x720, 49 frames, prompt "A BMX rider rides through the trees".
Tests several seeds, reuses the already-loaded model across seeds, computes the same
pearson_r_tracked_vs_commanded_dx metric used locally for direct comparison.
"""
import os
import sys
import gc
import json
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import subprocess
# Controlled rerun: Kaggle's default torch (2.11.0+cu128) differs from the LOCAL NF4 comparison's
# torch (2.5.1+cu118) by 6 minor versions -- an uncontrolled confound on top of the precision change
# we're actually testing (RNG/kernel numerics can and do change across torch releases). Pin torch to
# match local exactly, to isolate precision (fp16 vs bf16/NF4) as the only varying factor this run.
# torch==2.5.1 (matching local) has no torchvision wheel for Kaggle's Python 3.13 -- exact
# version-matching is infeasible on this image. Pivoting: use Kaggle's native torch (2.11.0+cu128,
# already proven working) and instead widen the seed sample to match local's n=14 exactly (same seed
# VALUES too), so the comparison is a statistically meaningful distribution-vs-distribution one
# (hit rate, median r) rather than a single-seed exact-reproduction chase, which is inherently fragile
# given this pipeline's already-documented extreme run-to-run sensitivity even with IDENTICAL config.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "diffusers==0.32.2", "transformers==4.49.0", "accelerate>=0.33",
                 "opencv-python-headless", "imageio[ffmpeg]", "scikit-image", "sentencepiece"],
                check=True)

import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image

import diffusers, transformers, accelerate
print(f"[versions] torch={torch.__version__} cuda={torch.version.cuda} diffusers={diffusers.__version__} "
      f"transformers={transformers.__version__} accelerate={accelerate.__version__}", flush=True)

def _find_assets_dir(root="/kaggle/input", max_depth=6):
    for dirpath, dirnames, filenames in os.walk(root):
        if "cogvideox_tracking.py" in filenames or "models.zip" in filenames or "tracking_video.mp4" in filenames:
            return dirpath
        if dirpath.count(os.sep) - root.count(os.sep) >= max_depth:
            dirnames[:] = []
    return None

for _dirpath, _dirnames, _filenames in os.walk("/kaggle/input"):
    print(f"[debug] {_dirpath}: dirs={_dirnames} files={_filenames}", flush=True)
ASSETS = _find_assets_dir() or "/kaggle/input/das-part3-bf16-assets"
print(f"[debug] using ASSETS={ASSETS}", flush=True)
if not os.path.isdir(f"{ASSETS}/models") and os.path.isfile(f"{ASSETS}/models.zip"):
    import zipfile
    workdir_assets = "/kaggle/working/assets"
    os.makedirs(workdir_assets, exist_ok=True)
    with zipfile.ZipFile(f"{ASSETS}/models.zip") as zf:
        zf.extractall(workdir_assets)
    print(f"[fixup] models.zip extracted, top-level: {os.listdir(workdir_assets)}", flush=True)
    if not os.path.isdir(f"{workdir_assets}/models") and os.path.isfile(f"{workdir_assets}/cogvideox_tracking.py"):
        # zip stored the .py at its root (no models/ prefix) -- recreate the models/ package dir ourselves
        os.makedirs(f"{workdir_assets}/models", exist_ok=True)
        import shutil
        shutil.move(f"{workdir_assets}/cogvideox_tracking.py", f"{workdir_assets}/models/cogvideox_tracking.py")
    for fn in ("tracking_video.mp4", "t0_480x720.png", "motion_curve_px.npy", "pred_and_gt.npz"):
        src, dst = f"{ASSETS}/{fn}", f"{workdir_assets}/{fn}"
        if not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)
    ASSETS = workdir_assets
    print(f"[fixup] extracted models.zip -> {ASSETS}, contents: {os.listdir(ASSETS)}", flush=True)
sys.path.insert(0, ASSETS)
print(f"[debug] ASSETS={ASSETS} contents={os.listdir(ASSETS)}", flush=True)
OUT = "/kaggle/working"

from models.cogvideox_tracking import (
    CogVideoXTransformer3DModelTracking, CogVideoXImageToVideoPipelineTracking,
)
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_video
from transformers import T5EncoderModel, T5Tokenizer

MODEL_PATH = "EXCAI/Diffusion-As-Shader"  # downloaded straight from HF, full precision (no NF4)
# T4 (Turing, sm75) rejects bf16 inputs in its memory-efficient SDPA kernel ("Expected query, key and
# value to all be of dtype: {Half, Float}. Got ... BFloat16" -- confirmed directly in an earlier run),
# forcing a fallback to the MATH backend which materializes the full O(seq^2) attention matrix (~56GB
# for this sequence length) and OOMs. fp16 IS natively supported by T4's efficient attention kernel.
DTYPE = torch.float16
PROMPT = "A BMX rider rides through the trees"
NEG_PROMPT = ("The video is not of a high quality, it has a low resolution. Watermark present in each frame. "
              "The background is solid. Strange body and strange trajectory. Distortion.")
SEEDS = [3, 4, 5, 6, 8, 9, 10, 11]  # the remaining 8 of the local 14-seed set (42,123,7,1,2,999
# already done in an earlier run, see outputs/part3_kaggle_bf16/results_bf16.json) -- completes the
# full n=14 same-seed-values comparison against results/part3_scheduler_diagnosis.json
NUM_STEPS = 10
GUIDANCE = 1.0
HEIGHT, WIDTH = 480, 720

t0_total = time.time()
results = {}

print("=== stage: load T5, encode prompts ===", flush=True)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=DTYPE).to("cuda:0")
text_encoder.eval()

from diffusers import CogVideoXImageToVideoPipeline
enc_pipe = CogVideoXImageToVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=None,
                                          transformer=None, scheduler=None)
with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = enc_pipe.encode_prompt(
        prompt=PROMPT, negative_prompt=NEG_PROMPT, do_classifier_free_guidance=True,
        num_videos_per_prompt=1, device="cuda:0", dtype=DTYPE)
prompt_embeds = prompt_embeds.to("cuda:0")
negative_prompt_embeds = negative_prompt_embeds.to("cuda:0")
del enc_pipe, text_encoder, tokenizer
gc.collect(); torch.cuda.empty_cache()
print(f"[mem] after freeing T5: cuda:0={torch.cuda.memory_allocated(0)/1e9:.2f}GB "
      f"cuda:1={torch.cuda.memory_allocated(1)/1e9:.2f}GB", flush=True)

print("=== stage: load VAE, encode tracking video + t0 image ===", flush=True)
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=DTYPE).to("cuda:0")
vae.eval()
vae.enable_slicing()
vae.enable_tiling()

import torchvision.transforms as transforms
tform = transforms.Compose([transforms.Resize((HEIGHT, WIDTH)), transforms.ToTensor()])
track_frames = load_video(f"{ASSETS}/tracking_video.mp4")
tracking_tensor = torch.stack([tform(f) for f in track_frames]).to("cuda:0", dtype=DTYPE)  # [T,C,H,W] in [0,1]
tracking_first_frame = tracking_tensor[0:1]  # pixel space, [1,C,H,W]

t0_img_pil = Image.open(f"{ASSETS}/t0_480x720.png").convert("RGB")
image_tensor = tform(t0_img_pil)  # [C,H,W]
# the reference _infer() converts the input image tensor back to a PIL Image before passing it to
# the pipeline call (only tracking_image/tracking_maps go in as raw tensors) -- match that exactly
image_for_pipe = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

with torch.no_grad():
    tm = tracking_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
    tracking_latent_dist = vae.encode(tm).latent_dist
    tracking_maps = tracking_latent_dist.sample() * vae.config.scaling_factor
    tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B,F,C,H,W]
del tracking_tensor, tm, tracking_latent_dist
gc.collect(); torch.cuda.empty_cache()
print(f"[mem] after VAE encode: cuda:0={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

print("=== stage: build transformer (bf16, no quantization), dispatch across cuda:0+cuda:1 ===", flush=True)
transformer = CogVideoXTransformer3DModelTracking.from_pretrained(MODEL_PATH, subfolder="transformer",
                                                                     torch_dtype=DTYPE)
transformer.eval()
from accelerate import dispatch_model
# NOT using infer_auto_device_map's greedy packing: it put ALL of combine_linears/transformer_blocks_copy
# on cuda:1 while splitting the first N main transformer_blocks across cuda:0+cuda:1 independently --
# since each layer i < num_tracking_blocks computes `hidden_states = hidden_states + combine_linears[i](
# transformer_blocks_copy[i](tracking_maps, ...))`, any main block on a DIFFERENT device than its paired
# copy/combine_linears causes "tensors on cuda:0 and cuda:1" errors accelerate's generic hooks don't
# resolve for this dual-branch architecture. Fix: explicitly co-locate transformer_blocks[i] with
# transformer_blocks_copy[i]/combine_linears[i] for every i that has a tracking branch.
n_tracking = transformer.num_tracking_blocks  # 18
n_layers = transformer.config.num_layers      # 42
split = n_tracking  # blocks [0, split) go with their tracking-branch counterpart on cuda:0;
                     # blocks [split, n_layers) have no tracking branch, go on cuda:1
dm = {"patch_embed": 0, "embedding_dropout": 0, "time_proj": 0, "time_embedding": 0}
for i in range(n_layers):
    dm[f"transformer_blocks.{i}"] = 0 if i < split else 1
for i in range(n_tracking):
    dm[f"transformer_blocks_copy.{i}"] = 0
    dm[f"combine_linears.{i}"] = 0
dm["initial_combine_linear"] = 0
dm["norm_final"] = 1
dm["norm_out"] = 1
dm["proj_out"] = 1
print("[device_map]", json.dumps({k: str(v) for k, v in dm.items()}), flush=True)
transformer = dispatch_model(transformer, device_map=dm)
print(f"[mem] after transformer dispatch: cuda:0={torch.cuda.memory_allocated(0)/1e9:.2f}GB "
      f"cuda:1={torch.cuda.memory_allocated(1)/1e9:.2f}GB", flush=True)

scheduler = CogVideoXDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
scheduler = CogVideoXDPMScheduler.from_config(scheduler.config, timestep_spacing="trailing")

pipe = CogVideoXImageToVideoPipelineTracking(
    vae=vae, text_encoder=None, tokenizer=None, transformer=transformer, scheduler=scheduler,
)
# CogVideoXImageToVideoPipelineTracking.__init__ unconditionally does `self.transformer =
# torch.compile(self.transformer)`. With the transformer sharded across 2 GPUs via accelerate hooks,
# dynamo tracing through those hooks causes a huge, broken memory allocation plan (56GB single alloc,
# reproduced identically across runs, unaffected by SDPA backend selection -- a classic dynamo graph
# break/recompile pathology). Un-wrap it: restore our own plain (uncompiled) dispatched transformer.
pipe.transformer = transformer
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


# ---- metrics (ported from scripts_local/part3b_analyze.py) ----
d = np.load(f"{ASSETS}/pred_and_gt.npz")
p0 = d["points_2d_at_t0"].copy()
p0[:, 0] *= WIDTH / 854.0
p0[:, 1] *= HEIGHT / 480.0
m = 30
x0b, x1b = int(max(p0[:, 0].min() - m, 0)), int(min(p0[:, 0].max() + m, WIDTH))
y0b, y1b = int(max(p0[:, 1].min() - m, 0)), int(min(p0[:, 1].max() + m, HEIGHT))
t0_img_np = np.asarray(t0_img_pil.resize((WIDTH, HEIGHT)))
tmpl = cv2.cvtColor(t0_img_np, cv2.COLOR_RGB2GRAY)[y0b:y1b, x0b:x1b]
curve = np.load(f"{ASSETS}/motion_curve_px.npy")  # (49,2) commanded dx,dy


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)


def track(fs, tmpl):
    out = []
    for f in fs:
        res = cv2.matchTemplate(gray(f), tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        out.append((float(mx), loc[0], loc[1]))
    return np.array(out)


def compute_metrics(frames_np):
    tr = track(frames_np, tmpl)
    vis = tr[:, 0] >= 0.5
    out = {"rider_visible_frames": int(vis.sum()), "rider_ncc_mean": float(tr[:, 0].mean())}
    if vis.sum() > 2:
        disp = tr[:, 1:] - tr[0, 1:]
        err = np.linalg.norm(disp[vis] - curve[vis], axis=1)
        out["motion_rms_err_px"] = float(np.sqrt((err ** 2).mean()))
        out["pearson_r_tracked_vs_commanded_dx"] = float(np.corrcoef(disp[vis, 0], curve[vis, 0])[0, 1])
        lost = np.where(~vis)[0]
        out["first_frame_lost"] = int(lost[0]) if len(lost) else None
    else:
        out["pearson_r_tracked_vs_commanded_dx"] = None
    return out


print("=== stage: generate per seed ===", flush=True)
# T4 (Turing, sm75) does not support the flash-attention SDPA backend (needs sm80+); PyTorch's default
# SDPA dispatch can silently fall back to the "math" backend, which materializes the full O(N^2) attention
# score matrix -- at 49 frames/480x720 that is tens of GB per layer and OOMs instantly. Force the
# memory-efficient (xformers-style) SDPA backend instead, which Turing does support and never
# materializes the full matrix.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def _sdpa_ctx():
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])  # no MATH fallback: force a hard error if
        # EFFICIENT_ATTENTION isn't actually usable for this shape, instead of silently falling back to
        # the memory-hungry MATH backend (which would explain why forcing [EFFICIENT, MATH] didn't help)
    print("[fixup] using torch.nn.attention.sdpa_kernel(EFFICIENT_ATTENTION only, no fallback)", flush=True)
except ImportError:
    import contextlib
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    def _sdpa_ctx():
        return contextlib.nullcontext()
    print("[fixup] using legacy torch.backends.cuda sdp flags (mem_efficient=True, flash=False)", flush=True)

for seed in SEEDS:
    t_seed = time.time()
    try:
        with torch.no_grad(), _sdpa_ctx():
            video = pipe(
                image=image_for_pipe,
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_videos_per_prompt=1,
                num_inference_steps=NUM_STEPS,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=GUIDANCE,
                generator=torch.Generator().manual_seed(seed),
                tracking_maps=tracking_maps,
                tracking_image=tracking_first_frame,
                height=HEIGHT,
                width=WIDTH,
            ).frames[0]
        out_path = f"{OUT}/gen_bf16_seed{seed}.mp4"
        export_to_video(video, out_path, fps=8)
        frames_np = [np.asarray(f) for f in video]
        met = compute_metrics(frames_np)
        met["seed"] = seed
        met["wall_seconds"] = time.time() - t_seed
        results[str(seed)] = met
        print(f"[seed {seed}] r={met.get('pearson_r_tracked_vs_commanded_dx')} "
              f"visible={met['rider_visible_frames']}/49 time={met['wall_seconds']:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        results[str(seed)] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"[seed {seed}] FAILED: {e}", flush=True)
    with open(f"{OUT}/results_bf16_batch2.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"=== DONE total={time.time()-t0_total:.0f}s ===", flush=True)
print("SUCCESS", flush=True)
