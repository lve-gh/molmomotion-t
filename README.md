# MolmoMotion test task: trajectory forecasting, ShareRobot transfer, DaS video generation

Code for three experiments (the write-up with figures is a separate report document):

1. Reproduce MolmoMotion's 3D point-trajectory forecast on an example from the authors' benchmark
   (DAVIS `bmx-trees`, PointMotionBench) and score it (ADE / FDE / PWT).
2. Try MolmoMotion on a ShareRobot episode (`bridge#episode_25423`), which has no 3D, no depth and no video,
   and evaluate with explicitly 2D-only measures.
3. Drive Diffusion-as-Shader (DaS) with the predicted trajectory and compare the generated clip with the real
   continuation and with a no-motion control.

Everything was run on Windows 10, RTX 3070 Ti (8 GB VRAM), 16 GB RAM. Several stock code paths do not fit or
crash on such a machine; the workarounds are part of the code (`scripts_local/`) and are described below.
Each part below lists how to run it and, directly under it, the expected results (one clip per part, single run,
single seed; the metrics are in `results/`, small deviations are normal across GPUs / dtypes, DaS is seeded with 42).
The images and videos are rebuilt by `python scripts_local/make_expected_media.py`.

## Setup

Third-party code is cloned next to the scripts (not vendored):

```
git clone https://github.com/allenai/molmo-motion repos/molmo-motion
git clone --recurse-submodules https://github.com/IGL-HKUST/DiffusionAsShader repos/DiffusionAsShader
```

* MolmoMotion environment: Python 3.11, `pip install -r requirements-molmomotion.txt`, then `pip install -e repos/molmo-motion[viz]`.
* DaS environment: Python 3.10, `pip install -r requirements-das.txt` (pins matter: DaS' own unpinned requirements
  resolve to mutually incompatible diffusers / transformers / torchao versions; `deepspeed` is not needed for inference).

Weights and data (plain `curl` was the only reliable way to download them here; check sha256 against the Hugging Face LFS hashes):

```
# MolmoMotion: native checkpoint (needed for predict_trajectory) + config.yaml / tokenizer files from the same repo
curl -L -o checkpoints/MolmoMotion-4B-H3-F30/model.pt https://huggingface.co/allenai/MolmoMotion-4B-H3-F30/resolve/main/model.pt
bash scripts_local/download_das.sh                                   # EXCAI/Diffusion-As-Shader, 25 GB
curl -L -o checkpoints/moge-vitl/model.pt https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt
```

Data: PointMotionBench `davis/tracks/bmx-trees_{2d,3d}.npz` and `davis/davis_captions.json`
(`allenai/PointMotionBench`), DAVIS-2017 trainval 480p frames of `bmx-trees`, ShareRobot
`trajectory/trajectory.json` and the two frames of `trajectory/images/rtx_frames_success_13/49_bridge#episode_25423/`
(`BAAI/ShareRobot`). Scripts expect them under `data/` (see the paths at the top of each script).

## Part 1: MolmoMotion on the authors' DAVIS example

```
python scripts_local/convert_ckpt_bf16.py checkpoints/MolmoMotion-4B-H3-F30/model.pt checkpoints/MolmoMotion-4B-H3-F30-bf16
python scripts_local/part1_infer_and_eval.py     # ~98 min on this GPU
python scripts_local/part1_visualize.py
```

Metrics follow `launch_scripts/eval_pointmotionbench.py`: ADE, FDE, PWT at 1/2/5/10/20 cm, in meters, camera frame at t0,
visible (point, frame) pairs only. `convert_ckpt_bf16.py` streams the fp32 checkpoint into bf16 shards without loading it
whole (`torch.load` crashed on this machine); `lowmem_model.py` builds the model in bf16 and splits it between GPU and CPU.

### Expected results (part 1)

ADE 0.381 m, FDE 0.727 m, PWT 8.6 % (`results/part1_metrics.json`). The run is deterministic (a second run reproduces the
prediction bit for bit).

**Comparison with the authors' released prediction** for the same clip (`examples/data/predictions_h3.jsonl`, score it with
`python scripts_local/eval_cached_predictions.py`): ADE 0.281 m, FDE 0.546 m, PWT 11.8 %, i.e. better than ours; the two
predictions differ by 11 cm on average. The generated texts have the same structure (1021 numbers each) but differ from the first
coordinate on (120 vs 121, 305 vs 307) and drift apart along the sequence (mean difference of the quantized values: 1.4 over the
first 40 numbers, 62 over the last 40), which points to numerical noise (GPU, kernels, bf16 accumulation) amplified by
autoregressive decoding rather than to a bug. Consequence: single-clip numbers depend on the hardware and should not be read as
model quality (`results/part1_vs_authors_prediction.json`, `results/multi_example_metrics.json`). Several clips are run in one
process with `scripts_local/run_examples.py`.

Input: frame t0 with the 8 query points on the rider (action: "A BMX rider rides through the trees").

![input frame and query points](docs/expected/part1_input_points.jpg)

Predicted (magenta) vs real (green, dashed) 3D tracks of the same points, both projected into the t0 camera (canvas
extended because the rider leaves the t0 field of view). Expected: same direction and speed, the prediction covers ~3.1 m
of the real ~3.6 m in 30 frames.

![predicted vs real trajectories](docs/expected/part1_predicted_vs_real.jpg)

The trails growing over the 30 predicted frames (click for the mp4):

[![prediction vs real trajectories](docs/expected/part1_prediction_vs_real.gif)](docs/expected/part1_prediction_vs_real.mp4)

Real future frames t0+1 / +10 / +20 / +30 (green: real 2D track in that frame, magenta x: prediction projected into the t0
camera). The camera follows the rider, so the prediction leaves the frame while the real rider stays in the centre; this is why
the metrics are computed in 3D, not on these projections.

![real future frames](docs/expected/part1_real_future_frames.jpg)

## Part 2: ShareRobot

```
python scripts_local/part2_build_inputs.py       # metric depth (Depth-Anything-V2 indoor), assumed 69.4 deg FOV, 8 query points, replicated history
python scripts_local/part2_infer_and_eval.py     # ~61 min
python scripts_local/part2_visualize.py
```

ShareRobot's `trajectory` split has two frames per episode and 2D end-effector waypoints only, so the 3D input is estimated
and the evaluation is coarse 2D path agreement (definitions in the docstrings). A history with real motion
(`frame_0, frame_0, frame_15`) has not been run yet.

### Expected results (part 2)

The model predicts zero motion for a replicated single-frame history; the 2D path metrics are then a "does not move"
baseline (`results/part2_metrics.json`).

Input frame ("reach for the spoon") with the 8 query points on the gripper (star = the annotated start point).

![ShareRobot input](docs/expected/part2_input_points.jpg)

Annotated 2D gripper path (green). Expected: the predicted trajectory collapses to the start point (the predicted path is
not visible), and the 2D metrics equal the baseline (mean 199 px, 25 % of the image diagonal).

![ShareRobot prediction vs annotation](docs/expected/part2_predicted_vs_annotated.jpg)

The only real future frame (frame_15): the gripper has moved towards the spoon.

![ShareRobot real frame_15](docs/expected/part2_real_frame15.jpg)

## Part 3: DaS driven by the MolmoMotion prediction

Run inside the DaS environment, from `repos/DiffusionAsShader`:

```
# 1) re-encode the DaS weights to bf16 pieces without mmap (safetensors reads crashed here), then rename the transformer pieces
#    to diffusion_pytorch_model-XXXXX-of-00014.bin and write diffusion_pytorch_model.bin.index.json
python ../../scripts_local/convert_safetensors_bf16.py ../../checkpoints/Diffusion-As-Shader/text_encoder  ../../checkpoints/Diffusion-As-Shader-bf16/text_encoder
python ../../scripts_local/convert_safetensors_bf16.py ../../checkpoints/Diffusion-As-Shader/transformer ../../checkpoints/Diffusion-As-Shader-bf16/transformer
# 2) tracking video from the predicted trajectory (add --static for the no-motion control)
python ../../scripts_local/part3_build_tracking_video.py --t0-image ../../outputs/part1/t0_frame.jpg \
   --pred-npz ../../outputs/part1/pred_and_gt.npz --out ../../outputs/part3/tracking_video.mp4 --device cuda
# 3) prompt encoding (T5 in its own process), then generation
python ../../scripts_local/part3_run_das_lowvram.py --stage encode --image ../../outputs/part3/t0_480x720.png \
   --tracking_video ../../outputs/part3/tracking_video.mp4 --prompt "A BMX rider rides through the trees"
python ../../scripts_local/part3_run_das_lowvram.py --stage generate --block_offload --height 480 --width 720 \
   --num_inference_steps 10 --guidance_scale 1.0 --image ../../outputs/part3/t0_480x720.png \
   --tracking_video ../../outputs/part3/tracking_video.mp4 --prompt "A BMX rider rides through the trees" \
   --output ../../outputs/part3/generated_tracked.mp4
python ../../scripts_local/part3_analyze.py      # metrics + figures (run from the repo root, in the MolmoMotion environment)
```

What it takes to fit 8 GB VRAM / 16 GB RAM (all in `part3_run_das_lowvram.py`):

* the DaS transformer (with the tracking branch) is 17.4 GB in bf16, more than the machine's RAM; it is loaded piece by
  piece as NF4 (bitsandbytes) straight onto the GPU (diffusers' quantized `from_pretrained` first merges all shards in RAM);
* DaS' `__init__` materialises 18 extra blocks in fp32 on the CPU (~16 GB): they are created on the meta device instead;
* T5 runs in a separate process, split between GPU and CPU; only the prompt embeddings are kept;
* `--block_offload` keeps the DiT blocks in host RAM and moves one block at a time to the GPU: a denoising step takes ~27 s;
  with all weights on the GPU (7.8 of 8 GB used) the first step did not finish in 17 minutes;
* `torch.compile` is unwrapped (no Triton on Windows); the resolution cannot be changed (diffusers 0.32 refuses it for this model).

Other files: `part3_diag_load.py`, `gpu_alloc_test.py`, `ram_selftest.py` are the small diagnostics used while debugging the above.

### Expected results (part 3)

480x720, 49 frames, 8 fps, NF4, 10 steps, CFG off, ~330 s and ~5.5 GB peak VRAM per clip. The rider follows the commanded
trajectory (slope 6.4 vs 9.5 px/frame, r = 0.85) and is gone by frame ~24; image quality degrades after ~12 frames
(`results/part3_metrics.json`).

Rows: real continuation, tracking video built from the prediction, DaS with the MolmoMotion trajectory, DaS with a static
(no-motion) tracking video; columns: frames 0 / 12 / 24 / 36 / 48. In the static control the rider stays in place while the
background dissolves.

![DaS frames](docs/expected/part3_das_frames.jpg)

The same four panels as an animation (click for the mp4):

[![real / control signal / DaS trajectory / DaS static control](docs/expected/part3_comparison_2x2.gif)](docs/expected/part3_comparison_2x2.mp4)

DaS with the MolmoMotion trajectory:

[![DaS with the MolmoMotion trajectory](docs/expected/part3_das_trajectory.gif)](docs/expected/part3_das_trajectory.mp4)

DaS with the no-motion control:

[![DaS with the no-motion control](docs/expected/part3_das_static_control.gif)](docs/expected/part3_das_static_control.mp4)

The control signal (tracking video) built from the prediction:

[![tracking video](docs/expected/part3_tracking_video.gif)](docs/expected/part3_tracking_video.mp4)

## Known limitations

* One clip per part, one seed; the full DaS configuration (50 steps, CFG 6.0) and ablations (NF4 / steps / CFG separately) were not run.
* The part 3 control is DaS with a static tracking video, not plain CogVideoX-I2V.
* The tracking video starts with the motion already offset by ~62 px (the first predicted step) instead of 0.
* Part 2 uses a replicated single frame as history, so the input has zero velocity.
