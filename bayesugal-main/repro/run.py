# -*- coding: utf-8 -*-
"""
Full SleepEDF-20 reproduction driver (paper-faithful).

    python repro/run.py --mode all --device cuda:0

Modes
    cv     : 10-fold cross-validation  -> paper Table III row "BayesSleepNet"
    ugal   : two-stage uncertainty-guided active learning on the held-out subjects
             -> paper Table III row "BayesSleepNet (UGAL)"  (eval on the remaining 80%)
    all    : cv then ugal, fold by fold (default)

Resumable: every finished (fold, stage) writes results/<stage>_fold<k>.json and is skipped
on a later run.  Run log: results/run.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsn import (Bank, BankSet, Expert, experts_predict,
                 load_dataset, metrics_from_preds, as_minutes)  # noqa: E402

DEV = torch.device("cpu")


# --------------------------------------------------------------------------------------
@torch.no_grad()
def mc_qu(experts, bankset, ids, n_samples, alpha, chunk, device):
    """Chunked MC inference over `ids` -> (quantified uncertainty, fused probs)."""
    qus, probs = [], []
    for s in range(0, len(ids), chunk):
        part = ids[s:s + chunk]
        x = bankset.gather(part).to(device)
        yhat, qu, _ = experts_predict(experts, x, n_samples=n_samples, alpha=alpha)
        qus.append(qu.cpu())
        probs.append(yhat.cpu())
    return torch.cat(qus), torch.cat(probs)


def adjust_lr(opt, epoch, epochs, lr):
    warm = epochs // 10
    cur = lr * (epoch + 1) / warm if epoch < warm else \
        lr * 0.5 * (1.0 + np.cos(np.pi * ((epoch + 1) - warm) / (epochs - warm)))
    for g in opt.param_groups:
        g["lr"] = cur
    return cur


_NVSMI = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                      "System32", "nvidia-smi.exe")


def _smi_free_gb():
    """Free GPU memory in GiB, straight from the driver.  None if unavailable."""
    try:
        r = subprocess.run([_NVSMI, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=25)
        txt = (r.stdout or "").strip().splitlines()[0]
        return int(float(txt)) / 1024.0
    except Exception:
        return None


def free_gb():
    """Conservative estimate of free GPU memory, in GiB.

    torch.cuda.mem_get_info() is NOT trustworthy on this box.  Under WDDM it was
    observed reporting 5.38 GB free while nvidia-smi showed 2.4 GB, and 0.00 GB free
    while nvidia-smi showed 0.46 GB.  Because the failure mode we care about is
    *over*-reporting (which would let a second job OOM the main run), we take the
    minimum of the two estimates.
    """
    vals = []
    s = _smi_free_gb()
    if s is not None:
        vals.append(s)
    try:
        vals.append(torch.cuda.mem_get_info(torch.device(torch.cuda.current_device()))[0] / 1024 ** 3)
    except Exception:
        pass
    return min(vals) if vals else float("inf")


def mem_guard(cfg, tag=""):
    """Never let a concurrent job push the GPU into OOM: block until `free >= mem_guard_gb`."""
    if cfg.mem_guard_gb <= 0:
        return
    waited = 0
    while True:
        free = free_gb()
        if free >= cfg.mem_guard_gb:
            if waited:
                print("      %s mem-guard released after %ds (free %.2f GiB)" % (tag, waited, free), flush=True)
            return
        if waited == 0:
            print("      %s mem-guard: free %.2f GiB < %.2f GiB, waiting..." % (tag, free, cfg.mem_guard_gb),
                  flush=True)
        time.sleep(15)
        waited += 15


def make_experts(cfg, device):
    return [Expert(3000, cfg.seqlen, 5, bayesian=cfg.bayesian, mc_drop=cfg.mc_drop,
                   prior_rho=cfg.prior_rho, norm_mode=cfg.norm,
                   share_bn=cfg.share_bn, out_act=cfg.out_act,
                   infer_bn=cfg.infer_bn).to(device)
            for _ in range(cfg.models)]


def train_experts(experts, bankset, shards, cfg, device, lr, epochs, tag="", schedule=True):
    """Train each expert on its own shard; effective batch = micro_batch * accum,
    z-score statistics aggregated over the whole effective batch (matches the released
    forward() at the paper's batch size 1024)."""
    opts = [torch.optim.AdamW(e.parameters(), lr, betas=(0.9, 0.95)) for e in experts]
    micro, accum = cfg.micro_batch, max(1, cfg.batch_size // cfg.micro_batch)
    rng = np.random.RandomState(cfg.seed)
    t0 = time.time()
    for ep in range(epochs):
        mem_guard(cfg, tag)
        cur_lr = adjust_lr(opts[0], ep, epochs, lr) if schedule else lr
        if not schedule:
            for o in opts:
                for g in o.param_groups:
                    g["lr"] = lr
        stats = []
        for ei, ex in enumerate(experts):
            shard = shards[ei]
            perm = rng.permutation(len(shard))
            tot_ce = tot_kl = nb = 0
            ce_acc = torch.zeros((), device=device)
            kl_acc = torch.zeros((), device=device)
            ex.train()
            for st in range(0, len(perm), cfg.batch_size):
                sel = shard[perm[st:st + cfg.batch_size]]
                x = bankset.gather(sel).to(device)
                y = bankset.target(sel).to(device)
                flat = x.view(-1, 1, x.size(3))
                ns = (flat.mean(dim=(0, 2), keepdim=True), flat.std(dim=(0, 2), keepdim=True))
                nsub = max(1, (len(sel) + micro - 1) // micro)
                opts[ei].zero_grad(set_to_none=True)
                for m in range(0, len(sel), micro):
                    xb = x[m:m + micro]
                    yb = y[m:m + micro]
                    logits, kl = ex(xb, train=True, norm_stats=ns)
                    ce = torch.nn.functional.cross_entropy(logits, yb)
                    loss = (ce + cfg.kl_weight * kl) / nsub
                    loss.backward()
                    ce_acc += ce.detach()
                    kl_acc += kl.detach()
                    nb += 1
                    del logits, kl, loss, ce
                if cfg.clip:
                    torch.nn.utils.clip_grad_norm_(ex.parameters(), 5.0)
                opts[ei].step()
                tot_ce += float(ce_acc)
                tot_kl += float(kl_acc)
                ce_acc.zero_()
                kl_acc.zero_()
                del x, y, flat, ns
            stats.append((tot_ce / max(nb, 1), tot_kl / max(nb, 1)))
        if ep % 10 == 0 or ep == epochs - 1:
            print("      %s ep %3d/%d lr=%.2e ce=%.4f kl=%.1f  %s"
                  % (tag, ep + 1, epochs, cur_lr, stats[0][0], stats[0][1],
                     as_minutes(time.time() - t0)), flush=True)
    return experts


# --------------------------------------------------------------------------------------
def eval_cv(experts, bankset, ids, cfg, device):
    x_all, y_all = [], []
    for s in range(0, len(ids), cfg.infer_chunk):
        part = ids[s:s + cfg.infer_chunk]
        x = bankset.gather(part).to(device)
        yhat, qu, _ = experts_predict(experts, x, n_samples=cfg.mc_samples, alpha=cfg.alpha)
        x_all.append(yhat.argmax(1).cpu().numpy())
        y_all.append(bankset.target(part).numpy())
        del x, yhat, qu
    return metrics_from_preds(np.concatenate(y_all), np.concatenate(x_all))


def systematic_picks(order, k, budget):
    """Paper Sec. IV.E.2: take every k-th sample; if the list ends early, continue from
    the beginning until the budget is met."""
    picks = []
    for off in range(k):
        picks += list(order[off::k])
        if len(picks) >= budget:
            break
    return np.asarray(picks[:budget], dtype=np.int64)


def run_ugal(experts, bankset, ids, cfg, device, tag=""):
    """Stage 1 -> fine-tune -> Stage 2 -> evaluate on the remaining 80%."""
    n = len(ids)
    b1 = int(n * cfg.stage1_ratio)
    b2_target = int(n * cfg.stage2_ratio)

    # ---- stage 1: systematic sampling across the uncertainty spectrum -------------
    qu, _ = mc_qu(experts, bankset, ids, cfg.mc_samples, cfg.alpha, cfg.infer_chunk, device)
    order = np.argsort(qu.numpy(), kind="stable")
    s1_local = systematic_picks(order, cfg.k, b1)
    s1 = np.asarray(ids)[s1_local]
    print("      %s stage1: %d/%d selected (k=%d)" % (tag, len(s1), n, cfg.k), flush=True)

    # ---- fine-tune both experts on S1 --------------------------------------------
    ft_epochs = cfg.ft_epochs if cfg.ft_epochs > 0 else cfg.epochs
    # paper: fine-tuning uses a FIXED learning rate of 3.6e-4 (no warm-up / cosine)
    train_experts(experts, bankset, [s1] * len(experts), cfg, device,
                  lr=cfg.lr * cfg.ft_lr_mul, epochs=ft_epochs, tag=tag + " ft", schedule=False)

    # ---- stage 2: recompute QU on U \ S1, annotate the top 10%, keep the rest -----
    mask = np.ones(n, dtype=bool)
    mask[s1_local] = False
    rest = np.asarray(ids)[mask]
    qu2, _ = mc_qu(experts, bankset, rest, cfg.mc_samples, cfg.alpha, cfg.infer_chunk, device)
    o2 = np.argsort(qu2.numpy(), kind="stable")
    n_keep = int(len(rest) * (1.0 - 2 * cfg.stage2_ratio) / (1.0 - cfg.stage1_ratio)) \
        if cfg.eval_ratio is None else int(len(rest) * cfg.eval_ratio)
    keep = rest[o2[:n_keep]]
    print("      %s stage2: %d annotated, eval on %d (%.1f%% of the subject)"
          % (tag, len(rest) - n_keep, len(keep), 100.0 * len(keep) / n), flush=True)

    res = eval_cv(experts, bankset, keep, cfg, device)
    res.update(n_subject=n, n_stage1=len(s1), n_stage2=len(rest) - n_keep, n_eval=len(keep))
    return res


# --------------------------------------------------------------------------------------
def folds_of(n_subjects, folds, seed):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.arange(n_subjects))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"D:\dataset\sleep-edf-20\processed")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--mode", default="all", choices=["cv", "ugal", "all"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-subjects", type=int, default=39)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ft-epochs", type=int, default=0, help="0 = same as --epochs")
    ap.add_argument("--lr", type=float, default=12e-4)
    ap.add_argument("--ft-lr-mul", type=float, default=0.3)
    ap.add_argument("--kl-weight", type=float, default=2e-8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--infer-chunk", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=20)
    ap.add_argument("--models", type=int, default=2)
    ap.add_argument("--mc-samples", type=int, default=200, help="TOTAL, split evenly over experts")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--stage1-ratio", type=float, default=0.1)
    ap.add_argument("--stage2-ratio", type=float, default=0.1)
    ap.add_argument("--eval-ratio", type=float, default=None,
                    help="default: 0.8/0.9 of the remaining set (paper: evaluate on 80%% of all)")
    ap.add_argument("--prior-rho", type=float, default=-3.0)
    ap.add_argument("--norm", default="batch", choices=["batch", "epoch"])
    ap.add_argument("--share-bn", action="store_true",
                    help="released-code fidelity: ONE BatchNorm1d shared by all 5 conv layers "
                         "(model.py:168 declares only bn1; 191/195/206/210/214 all reuse it)")
    ap.add_argument("--out-act", default="logits", choices=["logits", "sigmoid"],
                    help="released-code fidelity: 'sigmoid' applies torch.sigmoid to the "
                         "classifier output (model.py:234) and row-normalises in influence() "
                         "instead of softmax (model.py:242)")
    ap.add_argument("--infer-bn", default="running", choices=["running", "batch"],
                    help="'batch' = released-code fidelity: the original never calls "
                         ".train()/.eval(), so BatchNorm keeps using BATCH statistics at "
                         "inference too ('running' = the default, our stricter behaviour)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bayesian", type=int, default=1)
    ap.add_argument("--mc-drop", type=int, default=1)
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--force", type=int, default=0)
    ap.add_argument("--mem-guard-gb", type=float, default=0.0,
                    help="block until this much GPU memory is free (protects a concurrent job)")
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help="hard cap this process at this fraction of total GPU memory "
                         "(0 = uncapped); makes the OOM hit US, never the concurrent job")
    ap.add_argument("--only-folds", default="", help="e.g. 0,1,2  (default: all)")
    ap.add_argument("--tag", default="")
    cfg = ap.parse_args()
    cfg.bayesian, cfg.mc_drop, cfg.clip = bool(cfg.bayesian), bool(cfg.mc_drop), bool(cfg.clip)

    global DEV
    DEV = torch.device(cfg.device)
    if cfg.mem_frac > 0 and DEV.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.mem_frac, DEV)
    os.makedirs(cfg.out, exist_ok=True)
    logf = open(os.path.join(cfg.out, "run.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    log("=" * 100)
    log("RUN %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), vars(cfg)))
    if DEV.type == "cuda":
        log("GPU %s | free %.2f GiB of %.2f GiB"
            % (torch.cuda.get_device_name(DEV), free_gb(),
               torch.cuda.get_device_properties(DEV).total_memory / 1024 ** 3))

    data, labels, names = load_dataset(cfg.data_root, cfg.n_subjects)
    log("loaded %d recordings, %d epochs total" % (len(data), sum(len(y) for y in labels)))

    banks = [Bank(d, l, cfg.seqlen) for d, l in zip(data, labels)]
    splits = folds_of(len(data), cfg.folds, cfg.seed)
    sfx = ("_" + cfg.tag) if cfg.tag else ""

    order = list(range(cfg.folds))
    if cfg.only_folds:
        order = [int(x) for x in cfg.only_folds.split(",") if x.strip() != ""]
    for fold in order:
        tr, te = splits[fold]
        f_cv = os.path.join(cfg.out, "cv_fold%d%s.json" % (fold, sfx))
        f_ug = os.path.join(cfg.out, "ugal_fold%d%s.json" % (fold, sfx))
        need_cv = cfg.mode in ("cv", "all") and (cfg.force or not os.path.exists(f_cv))
        need_ug = cfg.mode in ("ugal", "all") and (cfg.force or not os.path.exists(f_ug))
        if not (need_cv or need_ug):
            log("fold %d already done, skip" % fold)
            continue
        log("-" * 100)
        log("FOLD %d  train=%d recordings  test=%d recordings (%s)"
            % (fold, len(tr), len(te), ", ".join(names[i] for i in te)))
        t_fold = time.time()
        try:
            mem_guard(cfg, "fold%d" % fold)
            train_set = BankSet([banks[i] for i in tr], cfg.seqlen)
            ids_all = np.arange(len(train_set))
            rng = np.random.RandomState(cfg.seed + fold)
            perm = rng.permutation(len(ids_all))
            shards = [ids_all[perm[:len(perm) // cfg.models]],
                      ids_all[perm[len(perm) // cfg.models:]]] if cfg.models == 2 else \
                np.array_split(ids_all[perm], cfg.models)

            experts = make_experts(cfg, DEV)
            n_par = sum(p.numel() for p in experts[0].parameters() if p.requires_grad)
            log("      params/expert=%d  x%d = %d   (paper 5.3e5)" % (n_par, cfg.models, n_par * cfg.models))
            experts = train_experts(experts, train_set, shards, cfg, DEV,
                                    lr=cfg.lr, epochs=cfg.epochs, tag="fold%d" % fold)

            # MC inference is the memory high-water mark of the whole run, so re-check
            # the guard before entering it.
            mem_guard(cfg, "fold%d-eval" % fold)

            if need_cv:
                preds, gts = [], []
                for ti in te:
                    b = banks[ti]
                    ids = np.arange(len(b))
                    for s in range(0, len(ids), cfg.infer_chunk):
                        part = ids[s:s + cfg.infer_chunk]
                        x = b.gather(part).to(DEV)
                        yhat, _, _ = experts_predict(experts, x, n_samples=cfg.mc_samples,
                                                     alpha=cfg.alpha)
                        preds.append(yhat.argmax(1).cpu().numpy())
                        gts.append(b.labels[part].numpy())
                        del x, yhat
                m = metrics_from_preds(np.concatenate(gts), np.concatenate(preds))
                json.dump(m, open(f_cv, "w"), indent=1)
                log("      CV  acc=%.2f mf1=%.2f k=%.4f  per-class F1=%s"
                    % (m["acc"], m["mf1"], m["kappa"], ["%.1f" % v for v in m["per_f1"]]))
                del preds, gts

            if need_ug:
                ug = []
                for ti in te:
                    b = banks[ti]
                    ex = make_experts(cfg, DEV)
                    for a, c in zip(ex, experts):
                        a.load_state_dict(c.state_dict())
                    r = run_ugal(ex, b, np.arange(len(b)), cfg, DEV, tag="%s" % names[ti])
                    r["subject"] = names[ti]
                    ug.append(r)
                    log("      %-12s acc=%.2f mf1=%.2f k=%.4f" % (names[ti], r["acc"], r["mf1"], r["kappa"]))
                    del ex
                json.dump(ug, open(f_ug, "w"), indent=1)

            log("      fold %d done in %s" % (fold, as_minutes(time.time() - t_fold)))
            del experts, train_set
            torch.cuda.empty_cache()
        except Exception:
            log("!! fold %d FAILED\n%s" % (fold, traceback.format_exc()))
            torch.cuda.empty_cache()
            continue

    # ---- summary -----------------------------------------------------------------
    summary = {}
    for stage, pat in (("cv", "cv_fold%d%s.json"), ("ugal", "ugal_fold%d%s.json")):
        rows = []
        for f in range(cfg.folds):
            p = os.path.join(cfg.out, pat % (f, sfx))
            if os.path.exists(p):
                rows.append((f, json.load(open(p))))
        if not rows:
            continue
        if stage == "cv":
            arr = lambda k: np.array([r[k] for _, r in rows])               # noqa: E731
            summary[stage] = dict(
                folds=len(rows),
                acc=float(arr("acc").mean()), mf1=float(arr("mf1").mean()),
                kappa=float(arr("kappa").mean()),
                per_f1=[float(x) for x in np.mean([r["per_f1"] for _, r in rows], 0)],
            )
        else:
            allr = [u for _, lst in rows for u in lst]
            n = sum(u["n_eval"] for u in allr)
            w = lambda k: float(sum(u[k] * u["n_eval"] for u in allr) / n)   # noqa: E731
            summary[stage] = dict(subjects=len(allr), n_eval=n,
                                 acc=w("acc"), mf1=w("mf1"), kappa=w("kappa"),
                                 per_f1=[float(sum(u["per_f1"][j] * u["n_eval"] for u in allr) / n)
                                         for j in range(5)])
    json.dump(summary, open(os.path.join(cfg.out, "summary%s.json" % sfx), "w"), indent=1)
    log("=" * 100)
    log("SUMMARY %s" % json.dumps(summary, indent=1))
    log("paper ref: CV 86.75 / 81.12 / 0.819   UGAL 94.58 / 89.24 / 0.925   (SleepEDF-20)")


if __name__ == "__main__":
    main()
