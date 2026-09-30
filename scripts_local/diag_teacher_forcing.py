"""Teacher-forcing diagnostic: is the difference to the authors' released prediction numerical noise or a real input difference?

The generated text of the authors (examples/data/predictions_h3.jsonl) is appended to our prompt and the whole sequence is
run through the model in ONE forward pass. For each answer position we compare our argmax with the authors' token and look at
the logit margin (our best logit minus the logit of the authors' token). If mismatches only occur where that margin is tiny
(a few bf16 ulps of the logit), the two runs differ by numerical noise; a large margin at the first mismatch would point to a
real difference in the model input.

usage: python scripts_local/diag_teacher_forcing.py --ckpt checkpoints/MolmoMotion-4B-H3-F30 --example davis_bmx_trees \
           [--text-file outputs/multi/davis_bmx_trees/future_text.txt] [--n-answer-tokens 1200] --out outputs/diag/tf_x.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowmem_model import load_lowmem  # noqa: E402
from run_examples import EX, load_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--text-file", default=None, help="teacher text; default = the authors' released generated text")
    ap.add_argument("--frames", default="example", choices=["example", "davis_jpg", "mp4"],
                    help="frame source (bmx-trees only for the last two): the example JPEGs, the original DAVIS JPEGs, "
                         "or frames decoded from an mp4 re-encoded like reconstruct_davis.py (outputs/diag/mp4)")
    ap.add_argument("--n-answer-tokens", type=int, default=1200)
    ap.add_argument("--force-math-attn", action="store_true",
                    help="force PyTorch's exact, non-fused 'math' SDPA backend everywhere instead of the default "
                         "flash/efficient/cudnn kernel auto-selection -- tests whether the kernel choice itself is a "
                         "source of the noise that diverges from the authors' released prediction")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from molmo_motion import MolmoMotionProcessor
    ckpt = ROOT / a.ckpt
    processor = MolmoMotionProcessor.from_pretrained(str(ckpt))
    H = processor.config.history_size
    model = load_lowmem(str(ckpt))
    internal = model._internal
    llm_cfg = internal.config.llm
    tok = llm_cfg.build_tokenizer()

    d = EX / a.example
    meta, frames, p2d, p3d, K, action = load_inputs(d, H)
    if a.frames != "example":
        from PIL import Image
        if a.frames == "davis_jpg":
            frames = [Image.open(ROOT / "data/DAVIS/JPEGImages/480p" / meta["video"] / f"{i:05d}.jpg").convert("RGB")
                      for i in meta["history_frame_indices"]]
        else:
            frames = [Image.fromarray(np.load(ROOT / f"outputs/diag/mp4/bmx_frame{i}.npy")) for i in range(3)]
    inputs = processor(history_frames=frames, points_2d_at_t0=p2d, points_3d_history=p3d, action=action, future_horizon=30)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    if a.text_file:
        text = Path(a.text_file).read_text(encoding="utf-8")
        src = a.text_file
    else:
        rec = next(json.loads(l) for l in open(EX / "predictions_h3.jsonl", encoding="utf-8")
                   if json.loads(l)["video"] == meta["video"])
        text = rec["rollouts"][0]["pred_text"]
        src = "authors"
    ans = torch.tensor(tok.encode(text), dtype=torch.long)
    if ans.ndim > 1:
        ans = ans.flatten()
    ans = ans[: a.n_answer_tokens].cuda()
    ids = inputs["input_ids"]
    L = ids.shape[1]
    full = torch.cat([ids, ans[None]], dim=1)
    print(f"prompt tokens {L}, answer tokens used {len(ans)} (source: {src}), autocast=True", flush=True)

    attn = full != -1
    pos = torch.clamp(torch.cumsum(attn.to(torch.int32), dim=-1) - 1, min=0)
    image_args = dict(images=inputs.get("images"), image_masks=inputs.get("image_masks"),
                      low_res_token_pooling=inputs.get("low_res_token_pooling"), token_pooling=inputs.get("token_pooling"),
                      num_images=inputs.get("num_images"), multimodal_type=inputs.get("multimodal_type"),
                      num_image_starts=inputs.get("num_image_starts"))
    from contextlib import nullcontext
    if a.force_math_attn:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attn_ctx = sdpa_kernel([SDPBackend.MATH])
    else:
        attn_ctx = nullcontext()
    t = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16), attn_ctx:
            out = internal(full, attention_mask=attn, position_ids=pos, use_cache=False, **image_args)
    print(f"forward {time.perf_counter() - t:.0f}s, logits {tuple(out.logits.shape)} {out.logits.dtype}", flush=True)
    lg = out.logits[0, L - 1: L - 1 + len(ans)].float()
    top = lg.argmax(-1)
    tgt = ans.to(lg.device)
    match = (top == tgt)
    tl = lg.gather(1, tgt[:, None])[:, 0]
    margin = (lg.max(-1).values - tl)  # >= 0; 0 when our argmax == target
    mism = (~match).nonzero()[:, 0].cpu().numpy()
    res = {"example": a.example, "ckpt": a.ckpt, "source": src, "autocast": True, "frames": a.frames,
           "force_math_attn": a.force_math_attn, "prompt_tokens": L,
           "answer_tokens": int(len(ans)), "match_rate": float(match.float().mean()), "n_mismatch": int(len(mism)),
           "first_mismatch": int(mism[0]) if len(mism) else None,
           "mismatch_margins": [round(float(margin[i]), 4) for i in mism[:200]],
           "mismatch_positions": [int(i) for i in mism[:200]],
           "mismatch_context": [{"i": int(i), "authors": tok.decode([int(tgt[i])]), "ours": tok.decode([int(top[i])]),
                                 "margin": round(float(margin[i]), 4)} for i in mism[:20]]}
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("mismatch_margins", "mismatch_positions")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
