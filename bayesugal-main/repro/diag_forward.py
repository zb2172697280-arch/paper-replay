# -*- coding: utf-8 -*-
"""Locate the first non-finite activation in the fp16-autocast forward pass."""
import os
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import BayesianConvNet  # noqa: E402

dev = torch.device("cuda:0")


def trace(amp, bs=128, benchmark=False):
    torch.backends.cudnn.benchmark = benchmark
    torch.manual_seed(0)
    net = BayesianConvNet(3000, 20, 5, True, True, dev).to(dev)
    net.train()
    x = torch.randn(bs, 20, 1, 3000, 1, device=dev)
    log = []

    def rec(tag, t):
        if isinstance(t, torch.Tensor):
            fin = bool(torch.isfinite(t).all())
            log.append((tag, str(t.dtype), float(t.abs().max()) if fin else float("nan"), fin))

    with torch.amp.autocast("cuda", enabled=amp):
        a, b, _, c, _ = x.size()
        v = x.view(a * b, 1, c)
        mean = torch.mean(v, dim=(0, 2), keepdim=True)
        std = torch.std(v, dim=(0, 2), keepdim=True)
        v = (v - mean) / std
        rec("input-norm", v)
        if net.bayesian:
            v, kl = net.conv1(v, sample=True)
        else:
            v, kl = net.conv1(v), torch.tensor(0.0, device=dev)
        rec("conv1", v)
        v = net.bn1(v); rec("bn1#1", v)
        v = F.relu(v); rec("relu1", v)
        v = net.conv2(v); rec("conv2", v)
        v = net.bn1(v); rec("bn1#2", v)
        v = F.relu(v); rec("relu2", v)
        v = net.pool1(v); rec("pool1", v)
        v = net.drop(v); rec("drop1", v)
        for i, cv in enumerate((net.conv3, net.conv4, net.conv5), 3):
            v = cv(v); rec("conv%d" % i, v)
            v = net.bn1(v); rec("bn1#%d" % i, v)
            v = F.relu(v); rec("relu%d" % i, v)
        v = net.pool2(v); rec("pool2", v)
        v = net.drop(v); rec("drop2", v)
        v = v.view(v.size(0), -1).view(-1, net.n_seqlen, 448)
        rec("reshape", v)
        v = net.rnn_branch(v); rec("lstm", v)
        v = net.drop(v); rec("drop3", v)
        v = net.linear(v); rec("linear", v)
        v = F.sigmoid(v); rec("sigmoid", v)
    print("  autocast=%-5s cudnn.benchmark=%-5s ->" % (amp, benchmark))
    for tag, d, mx, fin in log:
        print("      %-12s %-14s max=%10.4f  %s" % (tag, d, mx, "ok" if fin else "*** NON-FINITE ***"))
    del net, x
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    print("== fp32 ==")
    trace(False)
    print("== fp16 autocast ==")
    trace(True)
    print("== fp16 autocast + cudnn.benchmark=True ==")
    trace(True, benchmark=True)
