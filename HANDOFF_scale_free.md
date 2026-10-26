# HANDOFF — TomoCPT v2 (state on 26 Oct 2026)

Read this first in any new session. Then read `CLAUDE.md` and `.claude/agents/00_ORCHESTRATOR.md`
(both on branch `v2-dev`).

## 1. Where things are

| Branch | Content | Status |
|---|---|---|
| `origin/main` | v0.7.0 (Fourier augmentation, portal download, chunk-collision fix, contrast inversion p = 0.5) | production; v2 work never goes here |
| `v2-dev` | v0.7.0 + `fix/v1-patches` + the plan (`CLAUDE.md`, `.claude/agents/*`, `.claude/state/gate_status.json`, this file) | v2 integration branch; tests pass (109 passed, 26 skipped without the membrain-seg reference) |

All v2 branches start with `v2-` (`v2-dev`, `v2-feat/…`, `v2-fix/…`, `v2-plan/…`), and so do
tags (`v2-gate-0-base`, `v2-gate-N-passed`). See CLAUDE.md Section 8, rule 0.

The development machine has an RTX A6000 and the `tomocpt` conda env (Python 3.12,
torch 2.5.1+cu124, MONAI 1.4.0), so slow tests run there.

## 2. Goal (decided 26 Oct 2026)

One model and one `tomocpt predict` run pick several particle species of different sizes,
plus microtubules, from tomograms of different voxel sizes:
1. **Gate 1 — lock:** legacy goldens, legacy picking and SSL checkpoints, metrics.
2. **Gate 2 — data v2:** band rule (10–32 voxels) instead of rescaling every particle to 20
   voxels; all species of a tomogram in one multi-class target; levels when the species
   span more than the band; points stored per chunk.
3. **Gate 3 — model:** one output channel per class; FiLM conditioning on the working voxel
   size (a class trained at several voxel sizes needs to know its physical scale).
4. **Gate 4 — inference and benchmarks:** each class read at its level; per-class outputs;
   CZII and synthetic benchmarks against one v1 model per species.
5. **Gate 5 — SSL v2:** wedge-aware, scale-aware pretraining with dense embeddings; judged by
   the picker's gain with few labelled tomograms.
6. **Gate 6 — filaments:** microtubules from portal masks (DS-10490, DS-10477; DS-10513 as a
   held-out test with non-ground-truth labels) as centreline channels traced to polylines.

Dropped: mixture-of-experts; membrane, organelle and void segmentation; picking a species by
its diameter alone (FiLM on a requested length); PU / EMA-teacher training; distillation
into v2 models; the SSL wedge-restoration and even/odd tasks.

## 3. Fixed before the plan (`fix/v1-patches`, on `v2-dev`, not on `origin/main`)

- Fine-tuning trains the `out` layer (×1.0); it was in no optimizer group.
- SSL works again: it could not be built (since `5a3abc3`), not be loaded from checkpoints,
  its heads were never optimised, and it never logged `val_loss_smooth`. The recons panel
  shows the reconstruction.
- Resuming a fine-tuning run keeps its differential LR (it used to fail to load the optimizer).
- IMOD `prepare_data`: NameError fixed, and labels were drawn with X and Z swapped.
- `desired_particle_pixel_size` help text says diameter.

## 4. Decisions already taken

- Contrast inversion stays fixed at p = 0.5 (`constants.CONTRAST_INVERSION_PROB`), not configurable.
- `config.yaml` uses `infer.predictions_coord_format: relion_31`.
- Fine-tuning LR for `out`: ×1.0 (fine-tuning usually targets a new species).
- Pipeline v2 never upsamples: `apix_work >= apix_native`; a class below the band at native
  size is used there with `d_vox < band_min` (Gate 2).
- Multi-class training uses repeat-factor chunk sampling (`train.chunk_sampling`, Gate 2).
- Per-class thresholds are calibrated on the validation tomograms after v2 training and
  stored in the checkpoint; `predict` also writes a merged `heatmap.mrc` and a `classmap.mrc`
  (class numbers) (Gate 4).
- Not adopted: a global class catalogue across datasets, and a per-tomogram presence test
  that drops classes with few picks.

## 4b. Open research question (not planned)
The model picks only the classes of its class table. A cell contains an open-ended set of
objects; detecting objects outside the trained classes needs new ideas and probably new
architectures. More reading is needed before this becomes a gate.

## 5. Next steps, in order

1. Tag `v2-gate-0-base` on `v2-dev` and start Gate 1 with the orchestrator
   (`.claude/agents/00_ORCHESTRATOR.md`).
2. Optional, any time: finish the baseline arm of the Fourier-augmentation comparison
   (`experiments/NOTEBOOK.md`: the baseline run was stopped at epoch ~20) and score both
   runs with picking F-beta once `tomocpt/metrics.py` exists (Gate 1).

## 6. Data

- CZII 10440 (6 species, 10.012 Å/px):
  `python scripts/download_portal_data.py --dataset-id 10440 --out data/czii --voxel-spacing 10.012 --processing filtered --max-runs 7`
- Microtubules: DS-10490 (6.8 Å, simulated, also actin and protein points), DS-10477
  (13.48 Å), DS-10513 (20.58 Å, labels not ground truth). Mask download arrives in Gate 6.
