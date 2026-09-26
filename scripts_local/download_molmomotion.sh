#!/usr/bin/env bash
# usage: bash scripts_local/download_molmomotion.sh <hf-repo-name e.g. MolmoMotion-4B-H1-F32>
# Downloads the native checkpoint (model.pt) and the small config/tokenizer files with plain curl (resume-capable).
set -u
NAME="${1:?repo name}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/checkpoints/$NAME"
BASE="https://huggingface.co/allenai/$NAME/resolve/main"
mkdir -p "$DEST"
for f in config.json config.yaml generation_config.json preprocessor_config.json processor_config.json special_tokens_map.json \
         tokenizer.json tokenizer_config.json video_preprocessor_config.json added_tokens.json merges.txt vocab.json chat_template.jinja \
         model.safetensors.index.json configuration_molmo_motion.py modeling_molmo2.py processing_molmo_motion.py \
         image_processing_molmo_motion.py video_processing_molmo_motion.py; do
  curl -L -sS --retry 5 --retry-all-errors "$BASE/$f" -o "$DEST/$f"; echo "$f rc=$?"
done
for attempt in 1 2 3 4 5 6; do
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors --connect-timeout 30 -sS "$BASE/model.pt" -o "$DEST/model.pt"; rc=$?
  echo "model.pt attempt=$attempt rc=$rc size=$(stat -c%s "$DEST/model.pt")"; [ $rc -eq 0 ] && break
done
echo ALL_DONE
