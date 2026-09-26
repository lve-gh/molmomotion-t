"""Low-memory loader for MolmoMotion (16GB RAM / 8GB VRAM machine).

The stock `MolmoMotion.from_pretrained(...); model._internal.to(bf16).cuda()`
needs (a) the 17.8GB fp32 state dict + a 19GB fp32 model in host RAM and
(b) ~9.7GB of bf16 weights in VRAM. Neither fits. This loader:
  1. builds the model directly in bf16 on CPU (default dtype trick, ~9.7GB),
  2. loads the fp32 checkpoint with mmap=True and copies tensor-by-tensor
     (load_state_dict casts fp32->bf16 in place, page cache is evictable),
  3. splits the model between GPU and CPU with accelerate's dispatch_model
     (weights that don't fit in VRAM are streamed layer-by-layer to GPU for
     each forward, i.e. slow but numerically identical to full-GPU bf16).
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from molmo_motion import MolmoMotion
from molmo_motion.public_config import MolmoMotionConfig
from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig
from molmo_motion.util import resource_path


def load_lowmem(ckpt_dir: str, gpu_gib: float = 6.0, cpu_gib: float = 10.0):
    ckpt_dir = str(ckpt_dir)
    cfg = Molmo2TrajectoryConfig.load(resource_path(ckpt_dir, "config.yaml"),
                                      key="model", validate_paths=False)
    public_cfg = MolmoMotionConfig(
        num_points=getattr(cfg, "num_points", 8),
        history_size=getattr(cfg, "history_size", 3),
        future_size=getattr(cfg, "num_future_frames", 8),
        max_sequence_length=cfg.llm.max_sequence_length,
    )

    # Build the model on the meta device (no host RAM), then place every tensor straight on its target device
    # (GPU or CPU) from the bf16 shards: host RAM peak = the CPU-resident part (~4 GB) + one shard, instead of the
    # whole ~9.7 GB bf16 model.
    import json
    from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
    from accelerate.utils import set_module_tensor_to_device

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights():
            model = MolmoMotion(public_cfg, internal_config=cfg)
    finally:
        torch.set_default_dtype(prev)
    internal = model._internal
    print("built bf16 model on the meta device", flush=True)

    no_split = sorted({type(m).__name__ for n, m in internal.named_modules() if "Block" in type(m).__name__})
    print("no_split classes:", no_split, flush=True)
    device_map = infer_auto_device_map(
        internal, max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"},
        no_split_module_classes=no_split, dtype=torch.bfloat16)
    n_gpu = sum(1 for v in device_map.values() if v == 0)
    print(f"device_map: {n_gpu}/{len(device_map)} top-level groups on GPU", flush=True)
    keys = sorted(device_map.keys(), key=len, reverse=True)

    def dev_for(name):
        return next(device_map[k] for k in keys if k == "" or name == k or name.startswith(k + "."))

    bf16_dir = Path(str(ckpt_dir) + "-bf16")
    index = json.loads((bf16_dir / "index.json").read_text())
    placed = 0
    for fn in sorted(set(index.values())):
        shard = torch.load(bf16_dir / fn, map_location="cpu", weights_only=True)
        for k, v in shard.items():
            dev = dev_for(k)
            set_module_tensor_to_device(internal, k, "cpu" if dev == "disk" else dev, value=v.to(torch.bfloat16)
                                        if v.is_floating_point() else v)
            placed += 1
        del shard
        gc.collect()
    left = [n for n, prm in internal.named_parameters() if prm.device.type == "meta"]
    all_keys = set(internal.state_dict().keys())
    print(f"placed {placed} tensors; params still on meta: {len(left)} {left[:3]}; "
          f"keys missing from checkpoint index: {len(all_keys - set(index.keys()))}", flush=True)
    assert not left, "some parameters were never loaded"
    internal.eval()
    model._internal = dispatch_model(internal, device_map=device_map)
    attach_progress(model)
    return model


def attach_progress(model, est_total_steps=2000, every=25):
    """Print generation progress: each decoder forward = one generated token
    (plus one prefill call). ETA is relative to `est_total_steps`, a rough
    guess (8 points x 30 frames x ~8 tokens); generation stops at EOS, so the
    real total can differ."""
    import time
    state = {"n": 0, "t0": None}

    def hook(module, inputs, output):
        if state["t0"] is None:
            state["t0"] = time.perf_counter()
        state["n"] += 1
        if state["n"] % every == 0:
            dt = time.perf_counter() - state["t0"]
            rate = state["n"] / dt
            eta = max(est_total_steps - state["n"], 0) / rate / 60
            print(f"[progress] step {state['n']} | {rate:.2f} tok/s | elapsed {dt/60:.1f} min | "
                  f"ETA ~{eta:.0f} min (vs ~{est_total_steps} tokens, approximate)", flush=True)

    model._internal.transformer.register_forward_hook(hook)
