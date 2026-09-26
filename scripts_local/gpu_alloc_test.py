import sys, torch
use_bnb = sys.argv[1] == "bnb"
if use_bnb:
    import bitsandbytes as bnb
    print("bnb", bnb.__version__, flush=True)
xs = []
try:
    for i in range(80):            # up to ~5.6GB in 72MiB steps
        xs.append(torch.empty(72 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda"))
        if use_bnb and i == 0:
            p = bnb.nn.Params4bit(torch.randn(3072, 12288, dtype=torch.bfloat16), requires_grad=False, quant_type="nf4").to("cuda")
            print("Params4bit ok", flush=True)
    print("allocated", len(xs) * 72 / 1024, "GiB OK")
except Exception as e:
    print("FAILED after", len(xs) * 72 / 1024, "GiB:", str(e)[:200])
