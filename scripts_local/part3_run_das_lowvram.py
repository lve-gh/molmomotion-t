"""Part 3: attempt trajectory-conditioned generation with DaS on an 8GB GPU.

demo.py's default `_infer` path loads the T5 text encoder (~4.7B params),
the CogVideoX-5B tracking transformer (~5B params) and the VAE all in
bf16 and calls `pipe.to("cuda")` -- that alone is ~20GB of weights before
any activations, on an 8GB card. This script is the same generation call as
`DiffusionAsShaderPipeline._infer` (models/pipelines.py) but with the two
changes needed to even attempt it on 8GB:
  1. T5 text encoder loaded with `BitsAndBytesConfig(load_in_8bit=True)`
     (bitsandbytes is already a DaS dependency) -- ~4.7GB instead of ~9.4GB.
  2. `pipe.enable_sequential_cpu_offload()` instead of `pipe.to(device)` --
     keeps the ~5B-parameter transformer and the rest of the pipeline on
     CPU RAM, moving each submodule to GPU only for the instant it computes,
     trading speed for peak VRAM (the commented-out line in the original
     `models/pipelines.py:130`).

Records: wall-clock time, `torch.cuda.max_memory_allocated()`, and -- if it
OOMs -- exactly which stage/step that happened at, for the report.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

DAS_REPO = Path(__file__).resolve().parents[1] / "repos" / "DiffusionAsShader"
sys.path.insert(0, str(DAS_REPO))
sys.path.insert(0, str(DAS_REPO / "submodules" / "MoGe"))
sys.path.insert(0, str(DAS_REPO / "submodules" / "vggt"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torchvision.transforms as transforms
from diffusers.utils import load_image, load_video


def load_media_simple(path, max_frames=49, size=(480, 720)):
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".avi", ".mov"):
        frames = load_video(path)
    else:
        img = load_image(path)
        frames = [img] * max_frames
    if len(frames) < max_frames:
        import numpy as np
        idx = [int(i) for i in np.linspace(0, len(frames) - 1, max_frames)]
        frames = [frames[i] for i in idx]
    frames = frames[:max_frames]
    return torch.stack([transform(f) for f in frames])


EMB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "part3" / "prompt_embeds.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="t0 RGB image")
    ap.add_argument("--tracking_video", required=True, help="tracking mp4 from part3_build_tracking_video.py")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--checkpoint_path", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "Diffusion-As-Shader"))
    ap.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "outputs" / "part3" / "generated.mp4"))
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--block_offload", action="store_true",
                    help="keep NF4 transformer blocks in CPU RAM and stream one block at a time to the GPU (frees VRAM for activations)")
    ap.add_argument("--stage", choices=["encode", "generate"], required=True,
                    help="encode: T5 prompt encoding -> saved embeds; generate: NF4 transformer + VAE from saved embeds (separate processes keep peak memory low)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    stage = "init"
    try:
        from transformers import T5EncoderModel, T5Tokenizer, BitsAndBytesConfig
        from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
        from models.cogvideox_tracking import (
            CogVideoXTransformer3DModelTracking, CogVideoXImageToVideoPipelineTracking,
        )
        from diffusers.utils import export_to_video

        model_path = args.checkpoint_path
        dtype = torch.bfloat16

        stage = "load_vae"
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)

        import gc, json
        from transformers import AutoConfig
        from accelerate import dispatch_model, infer_auto_device_map
        bf16_root = Path(model_path).parent / "Diffusion-As-Shader-bf16"

        def load_pieces(module, sub):
            idx = json.loads((bf16_root / sub / "index.json").read_text())
            bad = []
            for fn in sorted(set(idx.values())):
                piece = torch.load(bf16_root / sub / fn, map_location="cpu", weights_only=True)
                r = module.load_state_dict(piece, strict=False)
                bad += list(r.unexpected_keys)
                del piece; gc.collect()
            missing = sorted(set(module.state_dict().keys()) - set(idx.keys()))
            print(f"[load {sub}] missing={len(missing)} unexpected={len(bad)}", flush=True)
            return missing, bad

        if args.stage == "encode":
            stage = "load_text_encoder_bf16"
            # bnb-8bit / safetensors from_pretrained crash on this machine (see report); T5 (~4.7B,
            # ~9.5GB bf16) is built from config, filled from bf16 pieces and split GPU/CPU with accelerate.
            tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
            t5_cfg = AutoConfig.from_pretrained(str(Path(model_path) / "text_encoder"))
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
            prev = torch.get_default_dtype(); torch.set_default_dtype(dtype)
            try:
                with init_empty_weights():
                    text_encoder = T5EncoderModel(t5_cfg)   # meta tensors: ~0 RAM
            finally:
                torch.set_default_dtype(prev)
            dm = infer_auto_device_map(text_encoder, max_memory={0: "5GiB", "cpu": "9GiB"},
                                       no_split_module_classes=["T5Block"], dtype=dtype)
            keys = sorted(dm.keys(), key=len, reverse=True)
            dev_for = lambda name: next(dm[k] for k in keys if k == "" or name == k or name.startswith(k + "."))
            idx = json.loads((bf16_root / "text_encoder" / "index.json").read_text())
            for fn in sorted(set(idx.values())):
                piece = torch.load(bf16_root / "text_encoder" / fn, map_location="cpu", weights_only=True)
                for k, v in piece.items():
                    dev = dev_for(k)
                    set_module_tensor_to_device(text_encoder, k, "cpu" if dev == "disk" else dev, value=v.to(dtype))
                del piece; gc.collect()
            sd_names = set(dict(text_encoder.named_parameters()).keys())
            for name, prm in text_encoder.named_parameters():
                if prm.device.type == "meta" and name == "encoder.embed_tokens.weight":
                    set_module_tensor_to_device(text_encoder, name, dev_for(name),
                                                value=text_encoder.shared.weight.detach().to("cpu"))
            left = [n for n, prm in text_encoder.named_parameters() if prm.device.type == "meta"]
            print(f"[load text_encoder] params still on meta: {len(left)} {left[:3]}", flush=True)
            text_encoder.eval()
            print("[mem] host RAM note: only CPU-resident T5 layers (~4.5GB) held in RAM", flush=True)
            text_encoder = dispatch_model(text_encoder, device_map=dm)
            print(f"[mem] T5 dispatched: {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM", flush=True)

            stage = "encode_prompts"
            neg = ("The video is not of a high quality, it has a low resolution. Watermark present in each frame. "
                   "The background is solid. Strange body and strange trajectory. Distortion.")
            from diffusers import CogVideoXImageToVideoPipeline
            enc_pipe = CogVideoXImageToVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=None,
                                                     transformer=None, scheduler=None)
            t_enc = time.time()
            with torch.no_grad():
                prompt_embeds, negative_prompt_embeds = enc_pipe.encode_prompt(
                    prompt=args.prompt, negative_prompt=neg, do_classifier_free_guidance=True,
                    num_videos_per_prompt=1, device="cuda:0", dtype=dtype)
            prompt_embeds, negative_prompt_embeds = prompt_embeds.cpu(), negative_prompt_embeds.cpu()
            print(f"[time] prompt encoding: {time.time()-t_enc:.0f}s", flush=True)
            del enc_pipe, text_encoder, dm
            gc.collect(); torch.cuda.empty_cache()
            print(f"[mem] after freeing T5: {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM; "
                  f"peak so far {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

            torch.save({"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds}, EMB_PATH)
            print(f"[stage encode] saved {EMB_PATH} | total time {time.time()-t_start:.0f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
            print("SUCCESS stage=encode", flush=True)
            return
        else:
            _e = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
            prompt_embeds, negative_prompt_embeds = _e["prompt_embeds"], _e["negative_prompt_embeds"]
            tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")

        stage = "load_transformer_nf4"
        # The DaS transformer (with the tracking branch) is ~17.4GB in bf16 -- more than the
        # machine's 16GB RAM, so CPU offload is impossible. Lighter config: 4-bit NF4
        # (bitsandbytes) loaded from the bf16 pieces -> ~4.7GB, fully resident on the GPU.
        from diffusers import BitsAndBytesConfig as DiffBnb
        # (1) DaS' __init__ materialises its 18 extra tracking blocks with `.to_empty(device="cpu")`
        #     in fp32 (~16GB of host RAM) BEFORE any weights load -> keep them on meta instead.
        # (2) diffusers' quantized from_pretrained first MERGES all shards into one ~17GB state dict in
        #     host RAM (machine has 16GB) and then fails with bogus CUDA "out of memory". So we build the
        #     model on meta, swap Linear -> bnb Linear4bit (NF4) and stream the bf16 pieces in one at a
        #     time, quantizing each weight straight onto the GPU.
        import torch.nn as nn, bitsandbytes as bnb
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from diffusers.quantizers.bitsandbytes.utils import replace_with_bnb_linear
        _orig_to_empty = nn.Module.to_empty
        nn.Module.to_empty = lambda self, *a, **k: self
        try:
            tcfg = CogVideoXTransformer3DModelTracking.load_config(str(bf16_root / "transformer"))
            prev = torch.get_default_dtype(); torch.set_default_dtype(dtype)
            try:
                with init_empty_weights():
                    transformer = CogVideoXTransformer3DModelTracking.from_config(tcfg)
            finally:
                torch.set_default_dtype(prev)
        finally:
            nn.Module.to_empty = _orig_to_empty
        qcfg = DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        skip = ["proj_out", "patch_embed", "time_embedding", "combine_linears", "initial_combine_linear", "norm_out"]
        transformer = replace_with_bnb_linear(transformer, modules_to_not_convert=skip, quantization_config=qcfg)
        idx = json.loads((bf16_root / "transformer" / "diffusion_pytorch_model.bin.index.json").read_text())["weight_map"]
        n_q = n_o = 0
        for fn in sorted(set(idx.values())):
            piece = torch.load(bf16_root / "transformer" / fn, map_location="cpu", weights_only=True)
            for k, w in piece.items():
                parent, _, leaf = k.rpartition(".")
                mod = transformer.get_submodule(parent) if parent else transformer
                if isinstance(mod, bnb.nn.Linear4bit) and leaf == "weight":
                    mod.weight = bnb.nn.Params4bit(w.to(dtype).contiguous(), requires_grad=False,
                                                   quant_type="nf4", compress_statistics=False).to("cuda")
                    n_q += 1
                else:
                    set_module_tensor_to_device(transformer, k, "cuda", value=w.to(dtype)); n_o += 1
            del piece; gc.collect()
            print(f"[load transformer] {fn}: quantized={n_q} other={n_o} | VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        for name, buf in transformer.named_buffers():
            if buf.device.type != "cuda":
                set_module_tensor_to_device(transformer, name, "cuda", value=buf.to("cpu") if buf.device.type != "meta" else buf)
        transformer.eval()
        n_meta = sum(1 for prm in transformer.parameters() if prm.device.type == "meta")
        print(f"[load transformer] params still on meta: {n_meta}", flush=True)
        print(f"[mem] transformer nf4 loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM", flush=True)

        if args.block_offload:
            # Weights of every DiT block live in host RAM; a pre-hook moves ONE block to the GPU for its
            # forward and a post-hook moves it back. NF4 blocks are ~90MB each, so the PCIe traffic is
            # ~5GB per denoising step (well under a second) while VRAM is freed for the 17.5k-token activations.
            def _to_gpu(m, a): m.to("cuda")
            def _to_cpu(m, a, o): m.to("cpu")
            blocks = list(transformer.transformer_blocks) + list(transformer.transformer_blocks_copy)
            for b in blocks:
                b.to("cpu")
                b.register_forward_pre_hook(_to_gpu)
                b.register_forward_hook(_to_cpu)
            import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
            print(f"[mem] block offload on: {len(blocks)} blocks in host RAM; VRAM now {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

        stage = "build_pipe"
        pipe = CogVideoXImageToVideoPipelineTracking(
            vae=vae, text_encoder=None, tokenizer=tokenizer,
            transformer=transformer, scheduler=scheduler)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        # DaS wraps the transformer in torch.compile (models/cogvideox_tracking.py:579); Triton/inductor
        # are unavailable on Windows -> unwrap to eager mode.
        if hasattr(pipe.transformer, "_orig_mod"):
            pipe.transformer = pipe.transformer._orig_mod
            print("[note] torch.compile wrapper removed (no Triton on Windows) -> eager mode", flush=True)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        stage = "move_vae"
        pipe.vae.to("cuda")  # small (0.4GB); transformer is already resident (nf4)

        stage = "load_media"
        image_tensor = load_media_simple(args.image, size=(args.height, args.width))[0]
        tracking_tensor = load_media_simple(args.tracking_video, size=(args.height, args.width))

        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        from PIL import Image
        image = Image.fromarray(image_np)
        height, width = image.height, image.width

        stage = "encode_tracking_maps"
        tracking_maps = tracking_tensor.float().unsqueeze(0).permute(0, 2, 1, 3, 4).to(dtype=dtype)
        tracking_first_frame = tracking_tensor[0:1]
        with torch.no_grad():
            tracking_latent_dist = pipe.vae.encode(tracking_maps.to(pipe.vae.device)).latent_dist
            tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)

        stage = "generate"
        video_generate = pipe(
            prompt=None, negative_prompt=None,
            prompt_embeds=prompt_embeds.to("cuda", dtype), negative_prompt_embeds=negative_prompt_embeds.to("cuda", dtype),
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(args.seed),
            tracking_maps=tracking_maps,
            tracking_image=tracking_first_frame,
            height=height,
            width=width,
        ).frames[0]

        stage = "export"
        export_to_video(video_generate, args.output, fps=8)

        dt = time.time() - t_start
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"SUCCESS stage={stage} time_s={dt:.1f} peak_vram_gb={peak_gb:.2f} output={args.output}")

    except torch.cuda.OutOfMemoryError as e:
        dt = time.time() - t_start
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"OOM at stage={stage} after time_s={dt:.1f} peak_vram_gb={peak_gb:.2f}")
        print(f"total_gpu_vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
        traceback.print_exc()
        raise SystemExit(1)
    except Exception:
        dt = time.time() - t_start
        print(f"FAILED at stage={stage} after time_s={dt:.1f}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
