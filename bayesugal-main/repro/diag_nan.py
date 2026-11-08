# -*- coding: utf-8 -*-
"""Diagnose why the released fp16-autocast path yields NaN loss."""
import os
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import BayesianConvNet, BayesConv, kl_loss_calculate  # noqa: E402

dev = torch.device("cuda:0")
torch.manual_seed(0)

print("== A) isolate the KL term in fp16 vs fp32 ==")
bc = BayesConv(1, 64, 19, dev, 4, 9, bias=False, prior_w_sigma=-3).to(dev)
x = torch.randn(512, 1, 3000, device=dev)
for amp in (False, True):
    with torch.amp.autocast("cuda", enabled=amp):
        _, kl = bc(x, sample=True)
    print("   autocast=%-5s  kl dtype=%-12s kl=%s" % (amp, kl.dtype, float(kl)))

print("\n== B) full model, print ce / kl / total separately ==")
net = BayesianConvNet(3000, 20, 5, True, True, dev).to(dev)
net.train()
xb = torch.randn(128, 20, 1, 3000, 1, device=dev)
yb = torch.randint(0, 5, (128,), device=dev)
for amp in (False, True):
    with torch.amp.autocast("cuda", enabled=amp):
        yp, kl = net(xb)
        ce = F.cross_entropy(yp, yb.long())
    print("   autocast=%-5s  yp dtype=%-12s  ce=%-12s  kl=%-16s  total=%s"
          % (amp, yp.dtype, float(ce), float(kl), float(ce + 2e-8 * kl)))

print("\n== C) grad magnitudes / GradScaler behaviour (fp16) ==")
net2 = BayesianConvNet(3000, 20, 5, True, True, dev).to(dev)
net2.train()
opt = torch.optim.AdamW(net2.parameters(), 12e-4, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    yp, kl = net2(xb)
    loss = F.cross_entropy(yp, yb.long()) + 2e-8 * kl
print("   loss before backward      :", float(loss))
print("   scaler.get_scale()        :", scaler.get_scale())
scaler.scale(loss).backward()
gn = {}
for n, p in net2.named_parameters():
    if p.grad is not None:
        gn[n] = float(p.grad.abs().max())
print("   max|grad| per param (first 6):", dict(list(gn.items())[:6]))
print("   any inf/nan grad          :", any(not torch.isfinite(p.grad).all() for p in net2.parameters() if p.grad is not None))
opt.step()
scaler.update()
print("   scale after update        :", scaler.get_scale())

print("\n== D) does BN/batch-level z-score differ between 128 and 1024? ==")
net3 = BayesianConvNet(3000, 20, 5, False, False, dev).to(dev)
net3.eval()
for bs in (128, 1024):
    xs = torch.randn(bs, 20, 1, 3000, 1, device=dev)
    with torch.no_grad():
        v = xs.view(bs * 20, 1, 3000)
        m = v.mean(dim=(0, 2), keepdim=True)
        s = v.std(dim=(0, 2), keepdim=True)
    print("   bs=%-5d batch-mean=%.5f batch-std=%.5f  <-- normalisation statistics seen by conv1"
          % (bs, float(m), float(s)))
