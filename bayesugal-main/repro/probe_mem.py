"""Probe peak GPU memory / throughput of a NON-Bayesian Expert at various micro-batches.

Purpose: decide whether an ablation process can safely coexist on the same 6 GB
GTX 1660 SUPER as the already-running main job (~3.7 GB resident, 91% util).

Notes learned the hard way:
  * Expert.forward expects a 5-D tensor (B, SL, 1, T, 1).
  * On Windows/WDDM, torch.cuda.mem_get_info() can OVER-report free memory
    relative to nvidia-smi, so we report both.
"""
import argparse, os, subprocess, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsn import Expert  # noqa


def torch_free_gb():
    f, t = torch.cuda.mem_get_info()
    return f / 1e9, t / 1e9


def smi_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return o.stdout.strip()
    except Exception as e:                                   # pragma: no cover
        return "nvidia-smi failed: %r" % (e,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=20)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--bayesian", type=int, default=0)
    ap.add_argument("--infer-chunk", type=int, default=256)
    ap.add_argument("--mc", type=int, default=200)
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()

    dev = "cuda:0"
    fg, tg = torch_free_gb()
    print("[mem] before : torch_free=%.2f/%.2f GB   smi_used/total/util=%s" % (fg, tg, smi_mib()), flush=True)

    m = Expert(bayesian=bool(a.bayesian), mc_drop=True, prior_rho=-3.0, norm_mode="batch").to(dev)
    print("[model] params/expert = %d" % sum(p.numel() for p in m.parameters()), flush=True)

    B, SL, L = a.micro, a.seqlen, 3000

    if not a.skip_train:
        opt = torch.optim.AdamW(m.parameters(), lr=1.2e-3)
        crit = nn.CrossEntropyLoss()
        m.train()
        x = torch.randn(B, SL, 1, L, 1, device=dev)
        # Expert.forward pools (B*SL) segments then takes the LAST LSTM step, so the
        # logits are (B, 5) -- one prediction per *sequence*, not per 30-s epoch.
        y = torch.randint(0, 5, (B,), device=dev)
        for _ in range(2):                                    # warm-up / allocator settle
            opt.zero_grad(set_to_none=True)
            out, _kl = m(x)
            assert out.shape == (B, 5), out.shape
            crit(out, y).backward()
            opt.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            out, _kl = m(x)
            crit(out, y).backward()
            opt.step()
            torch.cuda.synchronize()
        dt = (time.time() - t0) / a.steps
        pk = torch.cuda.max_memory_allocated() / 1e9
        rsv = torch.cuda.max_memory_reserved() / 1e9
        print("[train] micro=%-4d peak_alloc=%.2f GB  peak_reserved=%.2f GB  %.4f s/step"
              % (B, pk, rsv, dt), flush=True)

    if not a.skip_infer:
        torch.cuda.reset_peak_memory_stats()
        m.eval()
        x = torch.randn(a.infer_chunk, SL, 1, L, 1, device=dev)
        t0 = time.time()
        with torch.no_grad():
            _ = m.influence(x, n_samples=a.mc)
            torch.cuda.synchronize()
        dt = time.time() - t0
        pk = torch.cuda.max_memory_allocated() / 1e9
        rsv = torch.cuda.max_memory_reserved() / 1e9
        print("[infer] chunk=%-4d mc=%-4d peak_alloc=%.2f GB peak_reserved=%.2f GB  %.2f s  (%.0f epochs/s)"
              % (a.infer_chunk, a.mc, pk, rsv, dt, a.infer_chunk * a.mc / dt), flush=True)

    fg2, tg2 = torch_free_gb()
    print("[mem] after  : torch_free=%.2f/%.2f GB   smi_used/total/util=%s" % (fg2, tg2, smi_mib()), flush=True)


if __name__ == "__main__":
    main()
