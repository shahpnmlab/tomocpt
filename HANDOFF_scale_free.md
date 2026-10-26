# HANDOFF — TomoCPT v2 (state on 26 Oct 2026)

Read this first in any new session. Then read `CLAUDE.md` and `.claude/agents/00_ORCHESTRATOR.md`
(both on branch `plan/scale-free`).

## 1. Where things are

| Branch | Content | Status |
|---|---|---|
| `main` | v0.7.0 (Fourier augmentation, portal download, chunk-collision fix, contrast inversion p = 0.5) + `fix/v1-patches` | merged; tests pass (109 passed, 26 skipped without the membrain-seg reference) |
| `plan/scale-free` | `main` + `CLAUDE.md`, `.claude/agents/*` (orchestrator, test author, Gates 1–6), `.claude/state/gate_status.json`, this file | plan only, no code |

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

## 3. Fixed on `main` before the plan (`fix/v1-patches`)

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

## 5. Next steps, in order

1. Merge `plan/scale-free` → `main`.
2. Tag `gate-0-base` on that `main` and start Gate 1 with the orchestrator
   (`.claude/agents/00_ORCHESTRATOR.md`).
3. Optional, any time: finish the baseline arm of the Fourier-augmentation comparison
   (`experiments/NOTEBOOK.md`: the baseline run was stopped at epoch ~20) and score both
   runs with picking F-beta once `benchmarks/metrics.py` exists (Gate 1).

## 6. Data

- CZII 10440 (6 species, 10.012 Å/px):
  `python scripts/download_portal_data.py --dataset-id 10440 --out data/czii --voxel-spacing 10.012 --processing filtered --max-runs 7`
- Microtubules: DS-10490 (6.8 Å, simulated, also actin and protein points), DS-10477
  (13.48 Å), DS-10513 (20.58 Å, labels not ground truth). Mask download arrives in Gate 6.
