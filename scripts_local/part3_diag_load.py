"""Diagnostic: call the base diffusers from_pretrained directly (DaS' override swallows the
first exception and falls back to a path that needs 17GB RAM) to get the real traceback."""
import os, sys, traceback
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "repos" / "DiffusionAsShader"))
import torch, torch.nn as nn
from diffusers import BitsAndBytesConfig as B
from models.cogvideox_tracking import CogVideoXTransformer3DModelTracking as T
print("free/total VRAM:", [x / 1e9 for x in torch.cuda.mem_get_info()], flush=True)
_o = nn.Module.to_empty
nn.Module.to_empty = lambda self, *a, **k: self
try:
    m = super(T, T).from_pretrained(str(R / "checkpoints/Diffusion-As-Shader-bf16/transformer"),
        torch_dtype=torch.bfloat16, use_safetensors=False,
        quantization_config=B(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    print("LOADED; VRAM GB:", torch.cuda.memory_allocated() / 1e9)
except Exception:
    traceback.print_exc()
    print("FAILED; VRAM allocated GB:", torch.cuda.memory_allocated() / 1e9, "free/total:", [x / 1e9 for x in torch.cuda.mem_get_info()])
finally:
    nn.Module.to_empty = _o
