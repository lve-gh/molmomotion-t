"""Quick user-space RAM self-test: fill 4GB with patterns, read back, count mismatches."""
import numpy as np, time
N = 512 * 1024 * 1024 // 8  # 512MB blocks of uint64
blocks = 8                   # 4GB total
pats = [0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555, None]
arrs = [np.empty(N, dtype=np.uint64) for _ in range(blocks)]
bad_total = 0
for pat in pats:
    t = time.time()
    for i, a in enumerate(arrs):
        if pat is None:
            a[:] = np.arange(N, dtype=np.uint64) ^ np.uint64(i * 0x9E3779B97F4A7C15 % (1 << 63))
        else:
            a[:] = np.uint64(pat)
    for i, a in enumerate(arrs):
        if pat is None:
            exp = np.arange(N, dtype=np.uint64) ^ np.uint64(i * 0x9E3779B97F4A7C15 % (1 << 63))
            bad = int((a != exp).sum())
        else:
            bad = int((a != np.uint64(pat)).sum())
        bad_total += bad
        if bad: print(f"pattern {pat} block {i}: {bad} mismatching words", flush=True)
    print(f"pattern {pat if pat is None else hex(pat)} done in {time.time()-t:.1f}s, cumulative mismatches={bad_total}", flush=True)
print("RESULT: mismatches =", bad_total)
