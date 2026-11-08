# -*- coding: utf-8 -*-
"""Smoke / feasibility probe for BayesSleepNet reproduction.

Checks (no dataset needed):
  1. torch.load defaults (weights_only) for pickled nn.Module lists
  2. model builds & parameter count matches the paper (5.3e5 total / 2.65e5 per expert)
  3. one train step at the paper batch size (1024) -> peak GPU memory + time
  4. MC inference (`influence`) cost at T=100
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


def mb(x):
    return x / 1024 ** 2


def probe_load():
    print("== 1) torch.load default ==")
    import inspect
    sig = inspect.signature(torch.load)
    print("   weights_only default:", sig.parameters["weights_only"].default)
    m = torch.nn.Linear(2, 2)
    p = os.path.join(os.environ.get("TEMP", "."), "_probemodel.pt")
    torch.save([m, m], p)
    try:
        torch.load(p)
        print("   load(list[nn.Module]) OK")
    except Exception as e:  # noqa: BLE001
        print("   load(list[nn.Module]) FAIL ->", type(e).__name__, str(e)[:160])
    return p


def probe_model(device):
    print("\n== 2) model build / params ==")
    net = BayesianConvNet(3000, 20, 5, bayesian=True, mc_drop=True, device=device).to(device)
    per = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("   params per expert : %d" % per)
    print("   params x2 experts : %d   (paper 5.3e5, per-expert 2.6e5)" % (per * 2))
    print("   LSTM bidirectional:", net.rnn_branch.lstm.bidirectional,
          "| hidden:", net.rnn_branch.hidden_size)
    return net


def probe_step(device, batch_size, seqlen=20, T=3000):
    print("\n== 3) train step batch_size=%d (seq=20, T=3000) ==" % batch_size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = BayesianConvNet(T, seqlen, 5, bayesian=True, mc_drop=True, device=device).to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), 12e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    # real shape produced by SeqEEGDataset + TransformEpoch: (BS, SL, 1, T, 1)
    x = torch.randn(batch_size, seqlen, 1, T, 1, device=device)
    y = torch.randint(0, 5, (batch_size,), device=device)
    try:
        t0 = time.time()
        with torch.amp.autocast("cuda"):
            yp, kl = net(x)
            loss = F.cross_entropy(yp, y.long()) + 2e-8 * kl
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        dt = time.time() - t0
        print("   OK  loss=%.4f  step=%.3fs  peak=%.2f GB"
              % (loss.item(), dt, mb(torch.cuda.max_memory_allocated()) / 1024))
        return True, dt
    except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
        print("   OOM:", str(e)[:120])
        return False, None
    finally:
        del net, opt, x, y
        torch.cuda.empty_cache()


def probe_influence(device, batch_size, T=3000, n_samples=100):
    print("\n== 4) influence (MC) batch_size=%d n_samples=%d ==" % (batch_size, n_samples))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = BayesianConvNet(T, 20, 5, bayesian=True, mc_drop=True, device=device).to(device)
    net.eval()
    x = torch.randn(batch_size, 20, 1, T, 1, device=device)
    try:
        t0 = time.time()
        with torch.no_grad():
            mu, std = net.influence(x, n_samples=n_samples)
        torch.cuda.synchronize()
        print("   OK  mu%s std%s  %.2fs  peak=%.2f GB  (mu sum=%s)"
              % (tuple(mu.shape), tuple(std.shape), time.time() - t0,
                 mb(torch.cuda.max_memory_allocated()) / 1024,
                 mu.sum(1)[:2].tolist()))
        return True
    except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
        print("   OOM:", str(e)[:120])
        return False
    finally:
        del net, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0),
          "mem %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    p = probe_load()
    probe_model(dev)
    for bs in (1024, 512, 256):
        ok, _ = probe_step(dev, bs)
        if ok:
            print("   -> largest workable batch size probed: %d" % bs)
            break
    probe_influence(dev, 256, n_samples=100)
    try:
        del p
        os.remove(p)
    except Exception:  # noqa: BLE001
        pass
