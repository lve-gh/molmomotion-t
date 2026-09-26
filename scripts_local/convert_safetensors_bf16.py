"""Stream-convert safetensors shards to bf16 torch pieces without mmap.

On this machine `safetensors.load_file` / `from_pretrained` crash with a Windows
access violation while reading multi-GB shards (files verified intact by sha256).
We parse the safetensors header ourselves and read each tensor with plain
seek()+read(), cast fp32/fp16 -> bf16 and write ~1.2GB `.pt` pieces + index.json.

usage: python convert_safetensors_bf16.py <src_dir_with_*.safetensors> <dst_dir>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
DST.mkdir(parents=True, exist_ok=True)
PIECE = int(1.2e9)
NP = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "U8": np.uint8, "BOOL": np.bool_}


def read_shard(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            dt = meta["dtype"]
            if dt == "BF16":
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])
            else:
                t = torch.from_numpy(np.frombuffer(raw, dtype=NP[dt]).copy()).reshape(meta["shape"])
                if t.dtype in (torch.float32, torch.float16):
                    t = t.to(torch.bfloat16)
            yield name, t


def main():
    index, piece, size, k = {}, {}, 0, 0

    def flush():
        nonlocal piece, size, k
        if piece:
            fn = f"piece_{k:03d}.pt"
            torch.save(piece, DST / fn)
            for key in piece:
                index[key] = fn
            print("wrote", fn, len(piece), "tensors", flush=True)
            piece, size, k = {}, 0, k + 1

    for shard in sorted(SRC.glob("*.safetensors")):
        print("shard", shard.name, flush=True)
        for name, t in read_shard(shard):
            piece[name] = t
            size += t.numel() * t.element_size()
            if size >= PIECE:
                flush()
    flush()
    (DST / "index.json").write_text(json.dumps(index))
    print("done", len(index), "tensors")


if __name__ == "__main__":
    main()
