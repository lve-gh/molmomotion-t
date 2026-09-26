#!/usr/bin/env bash
# Download EXCAI/Diffusion-As-Shader with plain curl (resume-capable, explicit exit codes).
# (python/hf_transfer downloads hung on this machine; curl is reliable.)
set -u
DEST="$(cd "$(dirname "$0")/.." && pwd)/checkpoints/Diffusion-As-Shader"
BASE="https://huggingface.co/EXCAI/Diffusion-As-Shader/resolve/main"
mkdir -p "$DEST"
FILES="model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder/model.safetensors.index.json
tokenizer/added_tokens.json tokenizer/special_tokens_map.json tokenizer/spiece.model tokenizer/tokenizer_config.json
transformer/config.json transformer/diffusion_pytorch_model.safetensors.index.json vae/config.json
vae/diffusion_pytorch_model.safetensors spatracker/spaT_final.pth
text_encoder/model-00001-of-00002.safetensors text_encoder/model-00002-of-00002.safetensors
transformer/diffusion_pytorch_model-00001-of-00004.safetensors transformer/diffusion_pytorch_model-00002-of-00004.safetensors
transformer/diffusion_pytorch_model-00003-of-00004.safetensors transformer/diffusion_pytorch_model-00004-of-00004.safetensors"
for f in $FILES; do
  mkdir -p "$DEST/$(dirname "$f")"
  for attempt in 1 2 3 4 5 6; do
    curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors --connect-timeout 30 -sS "$BASE/$f" -o "$DEST/$f"
    rc=$?
    echo "$f attempt=$attempt rc=$rc size=$(stat -c%s "$DEST/$f" 2>/dev/null)"
    [ $rc -eq 0 ] && break
  done
done
echo ALL_DONE
