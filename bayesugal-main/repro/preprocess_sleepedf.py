# -*- coding: utf-8 -*-
"""
Preprocess Sleep-EDF Expanded (sleep-cassette) into the *.npz format the released
pipeline expects:  x (n_epochs, 3000, 1) float32 / y (n_epochs,) int32 / fs.

Faithful re-implementation of `EDFreader/sleepreader.py::load_eegdata_sleepedfx`
(kept 1:1: Fpz-Cz channel, 30 s epochs, 30-min wake trimming, MOVE/UNK removed),
but parameterised and with per-stage counts printed so the result can be checked
against paper Table II.

    python repro/preprocess_sleepedf.py --subset 20 --src D:/EDF/sleep-edf-database-expanded-1.0.0/sleep-cassette --out D:/dataset/sleep-edf-20
    python repro/preprocess_sleepedf.py --subset 78 --src ... --out D:/dataset/sleep-edf-78
"""
import argparse
import collections
import os
import re
import sys
import time

import numpy as np
import pyedflib

# ---- exactly as in pre/sleepstage.py --------------------------------------
W, N1, N2, N3, REM, MOVE, UNK = 0, 1, 2, 3, 4, 5, 6
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,   # AASM: merge N3+N4
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5,
}
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
PAPER_TABLE_II = {                     # subject num, epochs per stage
    20: dict(subj=20, W=10197, N1=2804, N2=17799, N3=5703, REM=7717, total=44220),
    78: dict(subj=78, W=69824, N1=21522, N2=69132, N3=13039, REM=25835, total=199352),
}


def subject_of(fname):
    m = re.match(r"(SC\d{3})", fname)
    return m.group(1) if m else fname


def select_pairs(src, n_subjects):
    """PSG/Hypnogram pairs of the first `n_subjects` SC subjects, alphabetically sorted.

    Mirrors `load_dataset_preprocessed` -> sorted(npzfiles)[:n_subjects], so the
    subset ordering (and therefore the KFold splits) is reproducible.
    """
    fs = sorted(os.listdir(src))
    psg = [f for f in fs if f.endswith("-PSG.edf")]
    hyp = [f for f in fs if "Hypnogram" in f]
    assert len(psg) == len(hyp)
    subjects = []
    for f in psg:
        s = subject_of(f)
        if s not in subjects:
            subjects.append(s)
    keep = set(subjects[:n_subjects])
    return sorted([(p, h) for p, h in zip(psg, hyp) if subject_of(p) in keep],
                  key=lambda x: x[0]), sorted(keep)


def load_pair(psg_path, hyp_path, select_ch="EEG Fpz-Cz"):
    psg_f = pyedflib.EdfReader(psg_path)
    ann_f = pyedflib.EdfReader(hyp_path)
    assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = psg_f.getStartdatetime()
    file_duration = psg_f.getFileDuration()
    epoch_duration = psg_f.datarecord_duration
    if psg_f.datarecord_duration == 60:      # SC4362F0 / SC4362FC fix, as upstream
        epoch_duration = epoch_duration / 2

    ch_names = psg_f.getSignalLabels()
    idx = -1
    for s in range(psg_f.signals_in_file):
        if ch_names[s] == select_ch:
            idx = s
            break
    if idx == -1:
        raise Exception("Channel not found: %s in %s" % (select_ch, psg_path))
    fs = psg_f.getSampleFrequency(idx)
    n_epoch_samples = int(epoch_duration * fs)
    signals = psg_f.readSignal(idx).reshape(-1, n_epoch_samples)

    n_epochs = psg_f.datarecords_in_file
    if psg_f.datarecord_duration == 60:
        n_epochs = n_epochs * 2
    assert len(signals) == n_epochs, "signal %s != %s" % (signals.shape, n_epochs)

    labels, total_duration = [], 0
    onsets, durations, stages = ann_f.readAnnotations()
    for a in range(len(stages)):
        onset_sec, duration_sec = int(onsets[a]), int(durations[a])
        ann_str = "".join(stages[a])
        assert onset_sec == total_duration, "non-contiguous annotation in %s" % hyp_path
        label = ann2label[ann_str]
        if duration_sec % epoch_duration != 0:
            raise Exception("bad duration %s %s" % (duration_sec, epoch_duration))
        labels.append(np.ones(int(duration_sec / epoch_duration), dtype=int) * label)
        total_duration += duration_sec
    labels = np.hstack(labels)[:len(signals)]
    psg_f._close()
    ann_f._close()

    x = signals.astype(np.float32)
    y = labels.astype(np.int32)

    # keep only sleep period: 30-min margins around the first/last non-W epoch
    w_edge_mins = 30
    nw = np.where(y != W)[0]
    start_idx = nw[0] - (w_edge_mins * 2)
    end_idx = nw[-1] + (w_edge_mins * 2)
    if start_idx < 0:
        start_idx = 0
    if end_idx >= len(y):
        end_idx = len(y) - 1
    sel = np.arange(start_idx, end_idx + 1)
    x, y = x[sel], y[sel]

    # drop movement / unknown
    bad = np.where((y == MOVE) | (y == UNK))[0]
    if len(bad):
        keep = np.setdiff1d(np.arange(len(x)), bad)
        x, y = x[keep], y[keep]

    return dict(x=x, y=y, fs=fs, ch_label=select_ch, start_datetime=start_datetime,
                file_duration=file_duration, epoch_duration=epoch_duration,
                n_all_epochs=n_epochs, n_epochs=len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, required=True, choices=[20, 78])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proc = os.path.join(args.out, "processed")
    os.makedirs(proc, exist_ok=True)
    pairs, subjects = select_pairs(args.src, args.subset)
    print("subset=%d  subjects=%d (%s..%s)  recordings=%d"
          % (args.subset, len(subjects), subjects[0], subjects[-1], len(pairs)))

    t0, counter = time.time(), collections.Counter()
    for i, (p, h) in enumerate(pairs, 1):
        name = p[:-8]
        dst = os.path.join(proc, name + ".npz")
        if os.path.exists(dst):
            d = np.load(dst)
            y = d["y"]
        else:
            d = load_pair(os.path.join(args.src, p), os.path.join(args.src, h))
            np.savez(dst, **d)
            y = d["y"]
        counter.update(y.tolist())
        if i % 5 == 0 or i == len(pairs):
            print("  [%3d/%3d] %-12s n_epochs=%-6d  %.0fs"
                  % (i, len(pairs), name, len(y), time.time() - t0))
            sys.stdout.flush()

    print("\n-- epoch counts by stage (all %d recordings) --" % len(pairs))
    ref = PAPER_TABLE_II[args.subset]
    ok = True
    print("%-8s %10s %10s %8s" % ("stage", "reproduced", "paper", "delta"))
    tot = sum(counter[k] for k in range(5))
    for k, nm in enumerate(CLASS_NAMES):
        d = counter[k] - ref[nm]
        ok &= (d == 0)
        print("%-8s %10d %10d %8s" % (nm, counter[k], ref[nm], "%+d" % d))
    print("%-8s %10d %10d %8s" % ("TOTAL", tot, ref["total"], "%+d" % (tot - ref["total"])))
    print("\n[%s] class counts %s paper Table II"
          % ("MATCH" if ok and tot == ref["total"] else "DIFF", "==" if ok and tot == ref["total"] else "!="))


if __name__ == "__main__":
    main()
