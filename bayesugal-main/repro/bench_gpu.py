# -*- coding: utf-8 -*-
"""Benchmark real train-step cost of BayesSleepNet on this GPU.

Answers: which micro-batch size fits in 6 GB without host-memory spill, how long a
step takes, and whether the fp16 autocast path produces NaN.
"""
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import BayesianConvNet  # noqa: E402

T = 3000
SL = 20
TOTAL_MEM = torch.cuda.get_device_properties(0).total_memory


def run(bs, amp, device, steps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = BayesianConvNet(T, SL, 5, bayesian=True, mc_drop=True, device=device).to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), 12e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(bs, SL, 1, T, 1, device=device)
    y = torch.randn(bs).mul(1.4).add(2).clamp(0, 4.999).long().to(device)
    losses = []
    try:
        t0 = time.time()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                yp, kl = net(x)
                loss = F.cross_entropy(yp, y) + 2e-8 * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        peak = torch.cuda.max_memory_allocated()
        spill = peak > TOTAL_MEM
        print("bs=%-5d amp=%-5s  step=%6.3fs  peak=%5.2fGB  %s  loss=%s"
              % (bs, amp, dt, peak / 1024 ** 3, "SPILL->host" if spill else "fits     ",
                 ["%.4f" % v for v in losses]))
        return dt, peak, spill
    except torch.cuda.OutOfMemoryError:
        print("bs=%-5d amp=%-5s  OOM" % (bs, amp))
        return None, None, None
    finally:
        del net, opt, x, y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    print("GPU: %s  %.2f GB" % (torch.cuda.get_device_name(0), TOTAL_MEM / 1024 ** 3))
    for bs in (1024, 512, 256, 128, 64):
        for amp in (True, False):
            run(bs, amp, dev)
    print("\n== accumulation math (effective batch 1024) ==")
    for bs in (1024, 512, 256, 128):
        print("  micro=%4d accum=%2d  -> full-train 40 batches equiv" % (bs, 1024 // bs))
