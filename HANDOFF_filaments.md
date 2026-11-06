# HANDOFF — Filament mode (TODO 3c), written 6 Nov 2026

Replaces `HANDOFF_gate6_benchmark.md` (31 Oct, stale: Gate 6 has since passed and the CZII recipe
changed). Read first: `CLAUDE.md`, `.claude/agents/06_AGENT_FILAMENTS.md`, `gate_06_report.json`
(T6.5 smoke run), `TODO_v2.md` (sections 3b, 3c). Branch `v2-dev` at tag `v2-gate-4g-passed`
(`8d62afc`), plus Gate 4h (relative threshold, tag `v2-gate-4h-passed`); fast suite 861 passed / 27 skipped. Environment:
`source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh && conda activate tomocpt`
(one RTX A6000). The user is usage-limited: orchestrator implements directly, no agent fan-out, small
sweeps, one commit per finished TODO item, stop at gate boundaries.

## 1. What exists for filaments (Gate 6, PASSED 1 Nov 2026)
- Inputs: `prepData.filament_masks` (`"<class>:<mask dir>"`, mask → centreline skeleton) or
  `prepData.filament_coords` (STAR/IMOD polylines, spline-fitted; `rlnOrigin*` subtracted, RELION 5
  centred coordinates rejected). Chunks store segments in `filaments/<chunk>.npy` (`[M, 8]`).
- Targets: tube channel per filament class, drawn by `render_targets`; filament classes are `kind`
  `filament`, after the particles in the class table.
- Predict: filament channels are traced (`predict/filaments.trace_class_maps`) into polylines,
  resampled every `infer.filament_spacing_ang` (40 Å), dropped below `filament_min_length_ang`
  (200 Å), smoothed `filament_smoothing_ang` (10 Å). Outputs: STAR (RELION / Warp) and IMOD models.
- Metric: `benchmarks/filaments.centerline_scores` (length-weighted precision / recall / F1 at a
  tolerance in voxels); `_summary` also gives particle F-beta.
- Tested on synthetic phantoms only (slow C8/C9: mt F1 1.00, actin 0.92 with the softmax head) plus
  one end-to-end smoke run on real data (1 Nov: tiny model, 4 runs per dataset, 1 epoch; no scores).

## 2. Data on disk (`data/filaments/`)
| Set | Kind | Voxel | Shape (z, y, x) | Runs | Masks |
|---|---|---|---|---|---|
| 10490 | simulated (CryoETSim), ground truth | 6.8 Å | 133×600×600 | 300 | microtubule 299, actin 299, membrane 300 |
| 10477 | experimental | 13.481 Å | 512×948×982 | 12 | microtubule 12, membrane 12 |
| 10513 | experimental, labels not ground truth | 20.579 Å | 250×1024×1024 | 20 | microtubule 20, membrane 20 |
DS-10490 particle points (141 STARs) are in `data/filaments/10490_points/` (prep snippet, diameters
still `???`). **DS-10490 is reserved as the "final boss" particle benchmark (TODO 6)**, except the
filament runs below (human decision 6 Nov 2026). It is the only set with both microtubules and actin.

## 3. Changes since Gate 6 and what they mean for filaments
| Change (gate) | Filament impact |
|---|---|
| Softmax head, k = 0 (3d; Gate 6 ruling) | Use it: filament channels share the softmax with particles and background. |
| Discriminative losses (7): focal value term, dense contrastive, negative masks | Focal is particle-only (filaments keep Huber, reconciled ruling). Contrastive anchors filaments on ridge voxels one diameter apart. Membrane negative masks exist for all three sets (`masks/membrane/`). Gate 7 was "failed as tested, re-judge after 3b"; focal off by default. |
| Built-in CV + ensembles (4c: `train.cv_all_folds`, OOF picks, `cv_summary.md`) | **Particle-only.** Calibration finds no points for filament classes (threshold `None`, falls back; the smoke run wasted 11 min on it), OOF reports have no centreline metric. Ensembles average maps, so tracing an ensemble should work but is untested. |
| Exact coordinates (4d) | `filament_coords` already reads floats; masks are voxel-aligned. Nothing to do. |
| Early stopping fields, schedule-free (4e) | General. |
| EMA / plateau SWA (4f) | General: legacy SWA never reached a checkpoint under early stopping in any run. |
| Core weight, offset head (4f) | Particle channels only by design; leave off for filament runs. |
| LR plateau matched to early stopping (4g, opt-in) + relative threshold (4h) | General: with `early_stop_rel_delta` both callbacks test `val_loss_smooth < best·(1−δ)`, independent of the loss scale. δ = 0.007 reproduces CZII's absolute 1e-4 at its floor (derived from one curve). |
| Annotation offset (3b.1b, open) | CZII labels sit (+0.4, +0.45, +1.4) voxels from image centres; random flips blur any such offset. Masks are drawn on the tomogram grid, so mask-derived centrelines should have none; STAR filament coordinates might. Human ruling: any fix must be general. |

CZII lessons that carry over: fold-0 (2-tomogram) scores cannot rank recipes (run-to-run noise in
`val_loss` ≈ 4e-4, β-gal F4 ±0.05); schedule variants (constant LR, LR drops, EMA, schedule-free) all
reach the same loss floor. Do not spend GPU on schedule sweeps for filaments.

## 4. Recipe for filament runs (the current v2 recipe)
```yaml
train:
  data_pipeline_version: 2
  chunk_sampling: repeat_factor
  augmentation: {label_interpolation: linear, rotation_mode: flips, flip_p: 0.5}
  particles: {class_head: softmax, cross_class_neg_weight: 0.0}
  early_stop_rel_delta: 0.007         # Gate 4h, relative; replaces early_stop_min_delta
  early_stop_patience: 10
  lr_plateau_match_early_stop: true
  lr_plateau_patience: 5
  weight_averaging: ema
prepData: {multiclass: true, label_border_policy: clip}
```
`benchmarks/filaments.py::_train_arm` (via `benchmarks/common._train_section`) does **not** set most
of this yet (it sets pipeline 2, linear labels, class head, conditioning on): update it first.

## 5. Next steps (TODO 3c), in order
1. **Decisions (human, 6 Nov 2026)**:
   a. Data: DS-10490 (simulated; microtubule + actin masks). Fixed runs `sample_1` … `sample_7` (both
      masks non-empty, checked): 5 train (`sample_1`–`sample_5`), 2 validation (`sample_6`, `sample_7`).
      These 7 runs are excluded from the final-boss test split (TODO 6). DS-10477 (microtubules only,
      experimental) is an optional out-of-distribution check of the trained model, not training data.
   b. Size: small model and few runs for quick checks and smoke tests; validation runs use 7 tomograms
      (5 / 2, as CZII). Model size for those is the default (as the CZII runs), unless the user says otherwise.
   c. Arms: filaments only first (3c.1); joint with particles later (needs DS-10490 diameters).
   d. Thresholds: relative, done (Gate 4h, δ = 0.007).
   Scale check for the smoke run: microtubule ≈ 250 Å is 37 voxels at 6.8 Å (above the band), actin is
   ≈ 10; the ratio 3.6 > 32/10, so `group_lengths` should give two levels. Confirm in `chunk_layout.json`.
2. **Calibration for filament classes** (small gate, `predict/calibration.py`, `cv_report.py`):
   skip filament rows in point calibration (saves the wasted time), and optionally calibrate the
   tracing threshold on validation masks with `centerline_scores` (F1 or F4). Without it the trace
   threshold is the fixed fallback.
3. **Update the driver** (`benchmarks/filaments.py`): recipe of section 4; mixed voxel sizes need one
   staged raw dir per dataset (`prepare_picking_star_multiclass` reads one pixel size per directory;
   the smoke run worked with filaments only, recheck for the joint arm); membrane negative masks for
   3c.2 (`prepData.negative_masks`).
4. Smoke run (`--tiny --max_runs 4 --epochs 1`) to recheck the pipeline on the current code, then
   the agreed run in the background with a watcher (`status` file with `start`/`exit`, as the CZII
   pilots in `benchmarks/out/czii_3b_*`).
5. Report: centreline precision / recall / F1 (and F4) on DS-10477 held-out and DS-10513, plus views of
   traced filaments (3c.3: missed / false / broken traces). Then 3c.2 (Gate 7 losses + membrane
   negatives) only if 3c.1 works.

## 6. Known traps
- Frozen tests: disputes go to `.claude/reports/gate_0N_test_dispute.md`; never edit a frozen file.
- Unpickled checkpoint configs lack new fields: use `safe_getattr` before `BaseModel.__init__`.
- `val_loss_smooth` is seeded by the sanity-check validation (starts high; settles in a few epochs).
- DS-10513 labels are not ground truth: report its scores as agreement, not accuracy.
- The pilot worktree `../tomocpt-3b-pilot` (at `v2-gate-4g-passed`) is free for long runs, so the
  main checkout can keep changing during a run.
