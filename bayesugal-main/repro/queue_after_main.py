# -*- coding: utf-8 -*-
"""
Queue the full-fidelity "w/o ensemble" ablation behind the running 10-fold main job.

WHY A QUEUE INSTEAD OF RUNNING IN PARALLEL
------------------------------------------
A measured single-fold trial (2026-09-20 21:04-21:26) showed that running a second
job concurrently is a net LOSS on this 6 GB GTX 1660 SUPER:

    main job alone       : 46.0 s/epoch
    main job + trial     : 106  s/epoch   (2.3x slower)
    trial alone-equiv    : 112  s/epoch
    aggregate throughput : 0.0183 epoch/s  vs  0.0217 epoch/s solo   -> 16% net LOSS

The GPU is already saturated by one job, so sharing it only adds overhead.  Queuing
is strictly better: the main job keeps its 30 h ETA, and the ablation then runs at
full fidelity (micro=128 / infer-chunk=256, the same settings as the main job)
instead of the reduced settings a concurrent run would have forced.

WHY NO torch IMPORT
-------------------
Importing torch and touching CUDA would create a CUDA context (~400 MB) that this
watcher would then hold for ~30 h, stealing headroom from the main job.  This script
uses only the standard library plus nvidia-smi, so it holds ZERO GPU memory.

USAGE
    python repro/queue_after_main.py            # start the watcher (blocks)
    python repro/queue_after_main.py --dry-run  # just report what it sees

To cancel: kill the watcher process before the main job ends.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NVSMI = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
PY = r"D:\Program Files\anaconda\envs\pytorch\python.exe"

MAIN_LOG = os.path.join(ROOT, "repro", "results", "run.log")
MAIN_SUMMARY = os.path.join(ROOT, "repro", "results", "summary.json")
ABLATE_OUT = os.path.join(ROOT, "repro", "results_ablate_noensemble")
ABLATE_STDOUT = os.path.join(ROOT, "repro", "ablate_noensemble_stdout.log")
WATCH_LOG = os.path.join(ROOT, "repro", "queue_watcher.log")

# Full fidelity: identical to the main job, only the ensemble is removed.
ABLATE_ARGS = [
    "-u", os.path.join("repro", "run.py"),
    "--mode", "cv",
    "--models", "1",              # paper Table IV: w/o ensemble {bayesian:T, mc_drop:T, num:1}
    "--bayesian", "1",
    "--mc-drop", "1",
    "--micro-batch", "128",       # same as the main job now that we run alone
    "--batch-size", "1024",
    "--infer-chunk", "256",       # same as the main job
    "--epochs", "200",
    "--n-subjects", "39",
    "--data-root", os.path.join("D:/", "dataset", "sleep-edf-20", "processed"),
    "--out", os.path.join("repro", "results_ablate_noensemble"),
    "--tag", "noens",
]

IDLE_FREE_GIB = 4.5      # main job gone -> the card is essentially empty
STALE_LOG_S = 900        # main run.log untouched for this long = stopped producing
POLL_S = 60
# NOTE: run.py only writes FOLD-level events into run.log (the per-epoch progress goes
# to stdout), so `stale` is true for most of a fold and cannot be used on its own.  The
# real signal is free memory.  Between folds the main job calls torch.cuda.empty_cache()
# and free memory can briefly jump above IDLE_FREE_GIB, so the idle state must persist
# well past any inter-fold gap (that gap is ~1-2 min: del + empty_cache + reload 39 npz).
# 8 consecutive 60 s polls gives a wide margin against a false trigger.
CONFIRM_POLLS = 8


def free_gib():
    try:
        r = subprocess.run([NVSMI, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=25)
        return int(float((r.stdout or "").strip().splitlines()[0])) / 1024.0
    except Exception:
        return None


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def launch(dry_run):
    os.makedirs(ABLATE_OUT, exist_ok=True)
    cmd = [PY] + ABLATE_ARGS
    log("LAUNCH %s" % " ".join('"%s"' % c if " " in c else c for c in cmd))
    if dry_run:
        log("(dry-run: not actually launching)")
        return None
    f = open(ABLATE_STDOUT, "a", encoding="utf-8")
    f.write("\n" + "=" * 100 + "\n[v/0] queued ablation launched "
            + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    log("started pid=%d" % p.pid)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    log("watcher up | idle<=%.1f GiB | stale_log>%ds | poll=%ds | confirm=%d"
        % (IDLE_FREE_GIB, STALE_LOG_S, POLL_S, CONFIRM_POLLS))
    idle_streak = 0
    while True:
        fr = free_gib()
        done = os.path.exists(MAIN_SUMMARY)
        try:
            stale = (time.time() - os.path.getmtime(MAIN_LOG)) > STALE_LOG_S
        except OSError:
            stale = False

        if done:
            log("main summary.json found -> main job finished cleanly")
            launch(a.dry_run)
            return
        if fr is not None and fr >= IDLE_FREE_GIB and stale:
            idle_streak += 1
            if idle_streak >= CONFIRM_POLLS:
                log("main run.log stale >%ds and free=%.2f GiB for %d polls -> assuming main "
                    "job stopped (possibly crashed); launching anyway" % (STALE_LOG_S, fr, idle_streak))
                launch(a.dry_run)
                return
        else:
            idle_streak = 0

        if a.dry_run:
            log("dry-run: free=%.2f GiB stale=%s done=%s streak=%d" % (fr or -1, stale, done, idle_streak))
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
