# -*- coding: utf-8 -*-
"""
Download Sleep-EDF Expanded (physionet sleep-edfx v1.0.0) sleep-cassette subset.

Usage:
    python repro/download_sleepedf.py --out D:/dataset/sleep-edf-20 --n-subjects 20
    python repro/download_sleepedf.py --out D:/dataset/sleep-edf-78 --n-subjects 78

The paper's SleepEDF-20  = SC subjects SC400..SC419  -> 39 recordings (SC413 lost 1 night)
The paper's SleepEDF-78  = the whole SC subset       -> 153 recordings (78 subjects)
Recordings are selected in the SAME alphabetical order the original loader uses
(`load_dataset_preprocessed` -> sorted(npzfiles)[:n_subjects]), so the resulting
`processed/*.npz` numbering is reproducible.
"""
import argparse
import os
import re
import sys
import time
import urllib.request

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
UA = {"User-Agent": "Mozilla/5.0 (research reproduction)"}


def list_dir(url):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode(
        "utf-8", "replace")
    rows = re.findall(r'<a href="([^"]+?\.edf)">[^<]*</a>\s+\S+\s+\S+\s+(\d+)', html)
    return [(n, int(s)) for n, s in rows]


def subject_id(fname):
    m = re.match(r"(SC\d{3})", fname)
    return m.group(1) if m else fname


def pick(files, n_subjects):
    """Recordings of the first `n_subjects` distinct SC subjects, alphabetically sorted."""
    psg = sorted([f for f in files if f[0].endswith("-PSG.edf")], key=lambda x: x[0])
    subjects = []
    for n, _ in psg:
        sid = subject_id(n)
        if sid not in subjects:
            subjects.append(sid)
    keep = set(subjects[:n_subjects])
    sel = [f for f in files if subject_id(f[0]) in keep]
    return sorted(sel, key=lambda x: x[0]), sorted(keep)


def fetch(url, dst, size):
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip"
    tmp = dst + ".part"
    for attempt in range(3):
        try:
            done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers=UA)
            if done:
                req.add_header("Range", "bytes=%d-" % done)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(tmp) == size:
                os.replace(tmp, dst)
                return "ok"
            done = os.path.getsize(tmp)
        except Exception as e:  # noqa: BLE001
            print("   retry %d: %s" % (attempt + 1, e))
            time.sleep(2)
            if os.path.exists(tmp) and os.path.getsize(tmp) >= size:
                os.replace(tmp, dst)
                return "ok"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-subjects", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = list_dir(BASE)
    sel, subjects = pick(files, args.n_subjects)
    total = sum(s for _, s in sel)
    print("subjects %d (%s .. %s), recordings/files %d, total %.2f GB"
          % (len(subjects), subjects[0], subjects[-1], len(sel), total / 1e9))

    t0 = time.time()
    for i, (name, size) in enumerate(sel, 1):
        dst = os.path.join(args.out, name)
        st = fetch(BASE + name, dst, size)
        print("[%3d/%3d] %-28s %8.2f MB  %s  (%dm%02ds)"
              % (i, len(sel), name, size / 1e6, st,
                 int((time.time() - t0) // 60), int((time.time() - t0) % 60)))
        sys.stdout.flush()
    print("done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
