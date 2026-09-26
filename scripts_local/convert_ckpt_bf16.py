"""Stream-convert the 17.8GB fp32 `model.pt` (torch zip format) into bf16
shards without ever materialising the whole checkpoint in RAM.

Needed because on this machine (16GB RAM, Windows) both `torch.load(model.pt)`
and `torch.load(..., mmap=True)` crash with an access violation. We parse the
zip + pickle ourselves: a custom Unpickler returns lazy tensor descriptors,
then each storage is read from the zip entry, cast fp32->bf16 and written
out in ~1.5GB shards (bf16_shards/shard_XXX.pt) + an index json.
"""
import collections
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
DST.mkdir(parents=True, exist_ok=True)
SHARD_BYTES = int(1.5e9)

STORAGE_DTYPES = {
    "FloatStorage": torch.float32, "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16, "LongStorage": torch.int64,
    "IntStorage": torch.int32, "BoolStorage": torch.bool,
    "DoubleStorage": torch.float64, "ByteStorage": torch.uint8,
    "ShortStorage": torch.int16, "CharStorage": torch.int8,
}
NP_DTYPES = {torch.float32: np.float32, torch.float16: np.float16,
             torch.int64: np.int64, torch.int32: np.int32, torch.bool: np.bool_,
             torch.float64: np.float64, torch.uint8: np.uint8,
             torch.int16: np.int16, torch.int8: np.int8}


class LazyStorage:
    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


class LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild(storage, offset, size, stride, *a, **k):
    return LazyTensor(storage, offset, size, stride)


class Unp(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        if module == "torch" and name in STORAGE_DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        typ, storage_type, key, loc, numel = pid
        assert typ == "storage", typ
        name = storage_type if isinstance(storage_type, str) else storage_type.__name__
        return LazyStorage(key, STORAGE_DTYPES[name], numel)


def main():
    zf = zipfile.ZipFile(SRC)
    names = zf.namelist()
    pkl = [n for n in names if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    with zf.open(pkl) as f:
        sd = Unp(f).load()
    print("tensors:", len(sd), flush=True)

    cache_key, cache_arr = None, None
    shard, shard_bytes, shard_idx, index = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        fn = f"shard_{shard_idx:03d}.pt"
        torch.save(shard, DST / fn)
        for k in shard:
            index[k] = fn
        print("wrote", fn, len(shard), "tensors", flush=True)
        shard, shard_bytes, shard_idx = {}, 0, shard_idx + 1

    for i, (name, lt) in enumerate(sd.items()):
        if not isinstance(lt, LazyTensor):
            print("non-tensor entry", name, type(lt))
            continue
        st = lt.storage
        if cache_key != st.key:
            raw = zf.read(f"{prefix}data/{st.key}")
            cache_arr = np.frombuffer(raw, dtype=NP_DTYPES[st.dtype])
            cache_key = st.key
        n = int(np.prod(lt.size)) if lt.size else 1
        flat = cache_arr[lt.offset: lt.offset + (n if lt.size else 1)]
        t = torch.from_numpy(flat.copy()).view(lt.size) if lt.size else torch.from_numpy(flat.copy()).reshape(())
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)
        shard[name] = t
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= SHARD_BYTES:
            flush()
    flush()
    (DST / "index.json").write_text(json.dumps(index))
    print("done", len(index), "tensors ->", DST)


if __name__ == "__main__":
    main()
