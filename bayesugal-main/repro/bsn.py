# -*- coding: utf-8 -*-
"""
BayesSleepNet — paper-faithful re-implementation.

Paper : "Efficient Sleep Staging with Bayesian Uncertainty-Guided Active Learning"
        IEEE Journal of Biomedical and Health Informatics, DOI 10.1109/JBHI.2026.3678608
Code  : https://github.com/yuty2009/bayesugal  (data pipeline / architecture / hyper-params reused)

Deliberate deviations from the released code, all in the direction of the PAPER
(see REPRO_AUDIT.md for the evidence table):

  1. output activation is Softmax (released code used F.sigmoid + renormalisation, and
     applied a second softmax on top of already-normalised probabilities during test)
  2. Bayesian entropy follows Eq.10  H = -sum_j mu^a * sigma^(1-a) * log(mu),  a = 0.5
     (released code used (sigma/mu)^1 in one routine, (sigma/mu)^0.7 in another, and the
      active-learning driver actually called the *plain entropy* routine)
  3. expert fusion uses the paper's uncertainty gating  w_i = softmax(-H_i)  (Eq.11/12)
     (released code used a mu/sigma-normalised sum)
  4. the two experts are trained on disjoint shards of the training set (paper Sec. III.B)
  5. MC sampling is T = 100 per expert = 200 total everywhere, incl. stage-1 selection
     (released code used only 50 total for stage 1)
  6. stage-1 selection sorts the WHOLE unlabelled set globally (released code sorted inside
     each shuffled mini-batch, so selection was batch-local)
  7. KL weight is the constant 2e-8 (released code halved it every mini-batch)
  8. one BatchNorm per conv layer, as in paper Table I (released code reused a single
     BatchNorm1d module across conv2..conv5)
  9. `optimizer.zero_grad()` is called (it is missing from the released tools.py)

Kept from the released code because the paper is ambiguous / the code is authoritative:
  * single-channel EEG, seqlen 20, 30 s @100 Hz (3000 samples)
  * Bayesian conv only on conv1, MC dropout on, unidirectional LSTM
    (single expert = 265 536 params -> x2 = 531 072 = the paper's 5.3e5, so the paper's
     "BiLSTM" in Table I is a typo)
  * KL is the SUM over parameters (paper Eq.4-6)
  * prior rho = -3  (i.e. prior sigma = softplus(-3) ~ 0.0486, NOT the paper's stated 1.0;
     see `--prior-rho`, it is exposed because the paper contradicts its own code here)
  * input z-scoring over the whole mini-batch, exactly as the released forward()
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ======================================================================================
# Bayesian layer  (Bayes-by-Backprop, Blundell et al.)
# ======================================================================================
def kl_gauss(p_mu, p_rho, q_mu, q_rho):
    """KL( N(p_mu, softplus(p_rho)^2) || N(q_mu, softplus(q_rho)^2) ), summed over elements."""
    p_sigma = 1e-6 + F.softplus(p_rho)
    q_sigma = 1e-6 + F.softplus(q_rho)
    return (torch.log(q_sigma / p_sigma) - 0.5
            + 0.5 * (p_sigma / q_sigma).pow(2)
            + 0.5 * ((p_mu - q_mu) / q_sigma).pow(2)).sum()


class BayesConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=False, prior_rho=-3.0):
        super().__init__()
        self.stride, self.padding, self.use_bias = stride, padding, bias
        self.w_mu = nn.Parameter(torch.Tensor(out_ch, in_ch, k).normal_(0.0, 0.1))
        self.w_rho = nn.Parameter(torch.Tensor(out_ch, in_ch, k).normal_(prior_rho, 0.1))
        self.register_buffer("prior_w_mu", torch.zeros(out_ch, in_ch, k))
        self.register_buffer("prior_w_rho", torch.full((out_ch, in_ch, k), float(prior_rho)))
        if bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_ch).normal_(0.0, 0.1))
            self.b_rho = nn.Parameter(torch.Tensor(out_ch).normal_(prior_rho, 0.1))
            self.register_buffer("prior_b_mu", torch.zeros(out_ch))
            self.register_buffer("prior_b_rho", torch.full((out_ch,), float(prior_rho)))
        else:
            self.register_parameter("b_mu", None)
            self.register_parameter("b_rho", None)

    def forward(self, x, sample=True):
        if not sample:
            return F.conv1d(x, self.w_mu, self.b_mu, self.stride, self.padding), x.new_zeros(())
        w = self.w_mu + (1e-6 + F.softplus(self.w_rho)) * torch.randn_like(self.w_mu)
        b = None
        kl = kl_gauss(self.w_mu, self.w_rho, self.prior_w_mu, self.prior_w_rho)
        if self.use_bias:
            b = self.b_mu + (1e-6 + F.softplus(self.b_rho)) * torch.randn_like(self.b_mu)
            kl = kl + kl_gauss(self.b_mu, self.b_rho, self.prior_b_mu, self.prior_b_rho)
        return F.conv1d(x, w, b, self.stride, self.padding), kl


# ======================================================================================
# One expert  (paper Table I: Conv1(19,64,s4) BN ReLU / Conv2(11,64,s4) BN ReLU /
#              Pool1(6) Drop / Conv3-5(7,64,s1) BN ReLU / Pool2(4) Drop /
#              reshape 448 / LSTM 64 / Drop / FC 64->5 / Softmax)
# ======================================================================================
class Expert(nn.Module):
    def __init__(self, n_timepoints=3000, n_seqlen=20, n_classes=5,
                 bayesian=True, mc_drop=True, prior_rho=-3.0, dropout=0.5, n_filters=64,
                 norm_mode="batch", share_bn=False, out_act="logits", infer_bn="running"):
        super().__init__()
        self.n_seqlen, self.n_classes = n_seqlen, n_classes
        self.bayesian, self.mc_drop = bayesian, mc_drop
        self.norm_mode = norm_mode          # 'batch' = released code, 'epoch' = per 30-s epoch
        self.share_bn = share_bn            # True = released-code fidelity (one bn1 for all 5)
        self.out_act = out_act              # 'sigmoid' = released-code fidelity (model.py:234)
        self.infer_bn = infer_bn            # 'batch' = released-code fidelity (no .eval() ever)

        self.conv1 = (BayesConv1d(1, n_filters, 19, 4, 9, bias=False, prior_rho=prior_rho)
                      if bayesian else nn.Conv1d(1, n_filters, 19, 4, 9, bias=False))
        self.conv2 = nn.Conv1d(n_filters, n_filters, 11, 4, 5, bias=False)
        self.conv3 = nn.Conv1d(n_filters, n_filters, 7, 1, "same", bias=False)
        self.conv4 = nn.Conv1d(n_filters, n_filters, 7, 1, "same", bias=False)
        self.conv5 = nn.Conv1d(n_filters, n_filters, 7, 1, "same", bias=False)
        if share_bn:
            # released code model.py:168 declares ONLY `self.bn1`; lines 191/195/206/210/214
            # all reuse it -> one BatchNorm1d shared by the five conv layers.
            self.bn1 = nn.BatchNorm1d(n_filters)
            self.bn2 = self.bn3 = self.bn4 = self.bn5 = self.bn1
        else:
            self.bn1, self.bn2, self.bn3, self.bn4, self.bn5 = (nn.BatchNorm1d(n_filters) for _ in range(5))
        self.pool1, self.pool2 = nn.MaxPool1d(6, 6), nn.MaxPool1d(4, 4)
        self.drop = nn.Dropout(dropout)
        flat = n_timepoints // 4 // 4 // 6 // 4 * n_filters           # 448 in the paper setting
        self.rnn = nn.LSTM(flat, n_filters, 1, batch_first=True, dropout=dropout, bidirectional=False)
        self.linear = nn.Linear(n_filters, n_classes, bias=False)

    # ---- forward ---------------------------------------------------------------------
    def forward(self, x, train=True, norm_stats=None):
        """x: (B, SL, 1, T, 1) -> logits (B, C), kl (scalar)"""
        a, b, _, c, _ = x.size()
        x = x.view(a * b, 1, c)
        if norm_stats is not None:
            x = (x - norm_stats[0]) / norm_stats[1]
        elif self.norm_mode == "epoch":            # z-score each 30 s segment independently
            x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True)
        else:                                      # released code: whole-mini-batch z-score
            x = (x - x.mean(dim=(0, 2), keepdim=True)) / x.std(dim=(0, 2), keepdim=True)

        if self.bayesian:
            x, kl = self.conv1(x, sample=True)
        else:
            x, kl = self.conv1(x), x.new_zeros(())
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self._drop(self.pool1(x), train)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self._drop(self.pool2(x), train)

        x = x.reshape(x.size(0), -1)
        x = x.reshape(-1, self.n_seqlen, x.size(-1))
        out, _ = self.rnn(x)
        x = self._drop(out[:, -1, :], train)
        x = self.linear(x)
        if self.out_act == "sigmoid":
            # released code model.py:234 feeds a sigmoid output straight into F.cross_entropy
            x = torch.sigmoid(x)
        return x, kl

    def _drop(self, x, train):
        return self.drop(x) if (train or self.mc_drop) else x

    # ---- MC inference ----------------------------------------------------------------
    @torch.no_grad()
    def influence(self, x, n_samples=100):
        """(mu, sigma) of the predictive distribution, size (B, C).  Paper Eq.8/9."""
        if self.infer_bn == "batch":
            # released-code fidelity: model.py / tools.py never call .train() or .eval(),
            # so the module stays in TRAINING mode and BatchNorm keeps using BATCH
            # statistics even at inference.  Dropout is still governed by the
            # forward(x, train=False) argument (see _drop), exactly as in model.py:199-231.
            self.train()
        else:
            self.eval()
        preds = x.new_zeros(n_samples, x.size(0), self.n_classes)
        for t in range(n_samples):
            y, _ = self.forward(x, train=False)
            if self.out_act == "sigmoid":
                # released code model.py:242 row-normalises the sigmoid output rather than
                # applying softmax to logits -- kept faithful when the switch is on.
                preds[t] = y / y.sum(dim=1, keepdim=True)
            else:
                preds[t] = F.softmax(y, dim=1)
        return preds.mean(0), preds.std(0, unbiased=False)


# ======================================================================================
# Uncertainty measures  (paper Sec. III.B-2)
# ======================================================================================
def bayes_entropy(mu, sigma, alpha=0.5):
    """Eq.10:  H = - sum_j  mu_j^a * sigma_j^(1-a) * log(mu_j)"""
    return -(mu.pow(alpha) * sigma.pow(1 - alpha) * torch.log(mu + EPS)).sum(dim=1)


def fuse(mu_list, sigma_list, alpha=0.5):
    """Eq.11/12:  w_i = softmax(-H_i) ;  yhat = sum_i w_i * mu_i ;  QU = sum_i H_i"""
    H = torch.stack([bayes_entropy(m, s, alpha) for m, s in zip(mu_list, sigma_list)], 0)   # (M,B)
    w = torch.softmax(-H, dim=0)                                                           # (M,B)
    mu_stack = torch.stack(mu_list, 0)                                                     # (M,B,C)
    yhat = (w.unsqueeze(-1) * mu_stack).sum(0)
    return yhat, H.sum(0), H


def experts_predict(experts, x, n_samples=100, alpha=0.5):
    mus, sigmas = [], []
    per = max(1, int(round(n_samples / len(experts))))
    for e in experts:
        mu, sg = e.influence(x, n_samples=per)
        mus.append(mu)
        sigmas.append(sg)
    return fuse(mus, sigmas, alpha)


# ======================================================================================
# Data
# ======================================================================================
def load_dataset(data_root, n_subjects=None):
    """Mirrors the released `load_dataset_preprocessed`: sorted(npz)[:n_subjects]."""
    files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
    if n_subjects is not None:
        files = files[:n_subjects]
    data, labels = [], []
    for f in files:
        with np.load(f) as d:
            data.append(np.ascontiguousarray(d["x"].astype(np.float32)[:, :, np.newaxis]))
            labels.append(d["y"].astype(np.int64))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return data, labels, names


def seq_index_matrix(n_epochs, seqlen):
    """Row i = epoch indices of the sequence ending at epoch i (front zero-padded, -1 = pad)."""
    idx = np.full((n_epochs, seqlen), -1, dtype=np.int64)
    for i in range(n_epochs):
        end = i + 1
        if end < seqlen:
            idx[i, seqlen - end:] = np.arange(end)
        else:
            idx[i] = np.arange(end - seqlen, end)
    return idx


class _SeqGather:
    """Gathers sequence batches: x[b] = epochs[seq_idx[b]] with the pad slots zeroed."""

    seqlen = 20
    seq_idx = None
    epochs = None
    labels = None

    def gather(self, ids):
        """ids: array-like (B,) -> (B, seqlen, 1, 3000, 1)"""
        si = self.seq_idx[torch.as_tensor(np.asarray(ids), dtype=torch.long)]
        ok = (si >= 0).unsqueeze(-1)                 # (B, SL, 1)
        g = self.epochs[si.clamp(min=0)]             # (B, SL, 3000, 1)
        return (g * ok.unsqueeze(-1)).unsqueeze(2)   # (B, SL, 1, 3000, 1)

    def target(self, ids):
        si = self.seq_idx[torch.as_tensor(np.asarray(ids), dtype=torch.long)]
        return self.labels[si[:, -1]]


class Bank(_SeqGather):
    """All sequences of one recording."""

    def __init__(self, epochs, labels, seqlen):
        self.epochs = torch.from_numpy(epochs) if isinstance(epochs, np.ndarray) else epochs.clone()
        self.labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels.clone()
        self.seqlen = seqlen
        self.seq_idx = torch.from_numpy(seq_index_matrix(len(self.labels), seqlen))

    def __len__(self):
        return len(self.labels)


class BankSet(_SeqGather):
    """Concatenated view over several Bank objects, addressed by global sample ids."""

    def __init__(self, banks, seqlen):
        self.seqlen = seqlen
        self.epochs = torch.cat([b.epochs for b in banks], 0)
        self.labels = torch.cat([b.labels for b in banks], 0)
        seq_idx, offset = [], 0
        for b in banks:
            seq_idx.append(b.seq_idx + offset)
            offset += len(b)
        self.seq_idx = torch.cat(seq_idx, 0)
        self.n_epochs = offset

    def __len__(self):
        return self.seq_idx.size(0)


# ======================================================================================
# Metrics  (paper Eq.14-16)
# ======================================================================================
def metrics_from_preds(y_true, y_pred, n_classes=5):
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_f1 = f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
    return dict(
        acc=float(accuracy_score(y_true, y_pred) * 100),
        mf1=float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)),
                           zero_division=0) * 100),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        per_f1=[float(v) * 100 for v in per_f1],
        n=int(len(y_true)),
    )


def as_minutes(s):
    return "%dm %.1fs" % (int(s // 60), s % 60)
