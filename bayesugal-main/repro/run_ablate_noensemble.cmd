@echo off
REM =====================================================================================
REM  MANUAL FALLBACK -- full-fidelity "w/o ensemble" ablation (paper Table IV row).
REM
REM  Normally you never run this by hand: `repro/queue_after_main.py` starts exactly this
REM  command automatically once the main 10-fold job has finished.  Use this file only if
REM  the watcher died (check repro/queue_watcher.log).
REM
REM  WHY IT RUNS ALONE, NOT IN PARALLEL  (measured 2026-09-20, see REPRO_AUDIT.md section 十)
REM  A single-fold trial showed sharing the 6 GB GTX 1660 SUPER with the main job is a net
REM  LOSS:
REM      main alone       46.0 s/epoch
REM      main + trial    106   s/epoch   (2.3x slower)
REM      trial alone     112   s/epoch   (2.4x slower)
REM      aggregate       0.0183 epoch/s  vs  0.0217 epoch/s solo   ->  16% net LOSS
REM  So this runs AFTER the main job, and can therefore use the main job's exact settings
REM  instead of the reduced ones a concurrent run would have forced:
REM      --micro-batch 128   (micro=64 -> 1.35 GB reserved, micro=32 -> 0.70 GB, but those
REM                           cut the BatchNorm batch from 2560 to 1280/640 segments)
REM      --infer-chunk 256   (chunk=256 peaks at 3.02 GB -- fine when running alone)
REM
REM  Estimated wall clock: 10 folds x 200 epochs x ~46 s + CV eval  ~=  27 h.
REM  Resumable: finished folds write cv_fold<k>_noens.json and are skipped on re-run.
REM  NOTE: run it through a MANAGED background channel. A detached Popen dies as soon as
REM  the launching tool call returns in this sandbox.
REM =====================================================================================
setlocal
cd /d "%~dp0.."

"D:\Program Files\anaconda\envs\pytorch\python.exe" -u repro\run.py ^
  --mode cv ^
  --models 1 ^
  --bayesian 1 ^
  --mc-drop 1 ^
  --micro-batch 128 ^
  --batch-size 1024 ^
  --infer-chunk 256 ^
  --epochs 200 ^
  --n-subjects 39 ^
  --data-root "D:\dataset\sleep-edf-20\processed" ^
  --out repro\results_ablate_noensemble ^
  --tag noens
endlocal
