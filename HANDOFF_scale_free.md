# HANDOFF — TomoCPT v2 (state on 28 Oct 2026)

Read this first in any new session. Then read `CLAUDE.md`, `.claude/agents/00_ORCHESTRATOR.md`
and `.claude/state/gate_status.json` (all on branch `v2-dev`). Nothing has been pushed.

## 1. Where things are

| Tag | Commit | Content |
|---|---|---|
| `v2-gate-0-base` | ce6f7b0 | released v0.7.0 + `fix/v1-patches` + the plan |
| `v2-gate-1-passed` | 7cb354e | legacy goldens, checkpoints, metrics, config backfill fix |
| `v2-gate-2-passed` | 333c20a | pipeline v2: band rule, levels, multi-class points |
| `v2-gate-3-passed` | 05641bf | multi-class heads, FiLM scale conditioning, v2 loss |
| `v2-gate-4-passed` | fef5c48 | v2 predict (per-class files), calibration, benchmarks, README |
| `v2-gate-3b-passed` | d190287 | cross-class negative weighting (`train.particles.cross_class_neg_weight`) |
| `v2-gate-3c-passed` | b946416 | CLI config snapshot: checkpoint loads no longer leak into each other |
| `v2-gate-3d-passed` | (tag) | softmax class head with implicit background (`train.particles.class_head`) |
| `v2-gate-4b-passed` | (tag) | k-fold training, out-of-fold calibration, fold ensembles (`experiments/run_cv.sh`) |
| `v2-gate-5-passed` | (this commit) | SSL v2: wedge/scale-aware views, masked recon, global + dense contrastive, relative scale |

Fast suite: 587 passed, 27 skipped (~4.5 min). Gate 5 passed; full CZII SSL benchmarks
(`benchmarks/ssl_probe.py`, `ssl_finetune.py`) await a human decision. Next: softmax class head
(own gate, to plan) or Gate 6 (filaments). v2-dev and v2-* tags are pushed to origin.

Environment: `source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh && conda activate tomocpt`
(RTX A6000). The base worktree `../tomocpt-base` (detached at `v2-gate-0-base`) is used by the
golden generators; leave it.

## 2. How to work (user preferences)
- Usage limits are tight: limit subagents. Use a Test Author agent for frozen tests; small
  implementations can be done directly. Small sweeps, no reruns, stop at gate boundaries.
- Keep context lean (short tool outputs).
- Commits and local tags are authorised by the orchestrator protocol; never push or touch `main`.

## 3. Human decisions since 26 Oct (details in `gate_status.json` notes)
- Gate 3: C4 Hessian tolerance < 15 %; C13 frozen at seed 0 (known flake, k=0).
- Gate 4: one output file per class in every format (no merged STAR); `imodmodel==0.1.0`
  (0.0.13 could not write); v1 inference unpad axis-order defect fixed (v1 output changes only
  for Z-padded tomograms).
- Gate 3b added (cross-class negative weighting), recommended k = 0.25. Softmax class head deferred.
- CZII diameters: public values `1:130,2:180,3:260,4:300,5:120,6:270` Å.

## 4. Open items
- Checkpoint-load config leak: fixed in Gate 3c (CLI snapshot).
- k-fold CV on CZII: run `experiments/run_cv.sh` (not run yet).
- Softmax class head: done in Gate 3d (`train.particles.class_head: softmax`); not yet run on CZII.
- SSL benchmark: relaunch `experiments/run_ssl_benchmarks.sh` (softmax pickers, 60 epochs, SSL v2 90).
- Full benchmarks (`benchmarks/czii_multiclass.py`, `synthetic_multisize.py`) not run, only `--tiny`.

## 5. CZII v2 results (k = 0)
Config `data/czii/v2_config.yaml`, model `data/czii/weights/tomocpt_v2/tomocpt/checkpoints/weights.ckpt`
(best epoch 17 of 34), predictions on the validation tomograms TS_5_4, TS_69_2 in `data/czii/pred_v2/`.

| Class | P | R | F4 |
|---|---|---|---|
| beta-amylase | 0.04 | 0.60 | 0.34 |
| beta-galactosidase | 0.17 | 0.52 | 0.47 |
| thyroglobulin | 0.27 | 0.76 | 0.69 |
| ribosome | 0.69 | 0.72 | 0.72 |
| apo-ferritin | 0.64 | 0.63 | 0.63 |
| VLP | 1.00 | 0.85 | 0.86 |

Cross-class confusion after suppression is small; the dominant error is background false picks
(beta-amylase 275, thyroglobulin 133). k = 0.25 run: mean F4 0.60, mean AP 0.46 (same as k = 0);
no cross-class double picks but small classes hedge. Use k = 0 (at most 0.1). About 95 % of false
picks are background densities with particle-like contrast. Details and diagnostics:
`experiments/czii_v2_results.md`, `experiments/diag_czii_v2.py`.
