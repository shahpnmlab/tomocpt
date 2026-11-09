# HANDOFF — Gate 8 spec (performance), written 2 Nov 2026

**Task of the receiving agent:** scan the codebase end to end and write the Gate 8 spec
`.claude/agents/08_AGENT_PERFORMANCE.md`. Do **not** implement anything, do not change `tomocpt/`,
tests or goldens. The spec is reviewed by the human before any Gate 8 work starts.

Read first: `CLAUDE.md`, `.claude/agents/00_ORCHESTRATOR.md`, `.claude/agents/TA_TEST_AUTHOR.md`,
`TODO_v2.md`, `.claude/state/gate_status.json`, all gate reports in `.claude/reports/`, and one or
two existing specs as the template (`07_AGENT_DISCRIMINATIVE.md`, `04b_AGENT_CV_ENSEMBLE.md`:
sections Why / Design / Config / Allowed write paths / Public names / Pass criteria / Acceptance for
humans). Branch `v2-dev`. Environment:
`source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh && conda activate tomocpt`
(RTX A6000, 48 GB). User is usage-limited: work directly, no fan-out of agents, stop when the spec
is written.

## 1. Preconditions and timing (human decision, 2 Nov 2026)

**Update 9 Nov 2026 (human decision): Gate 8 starts now.** Gate 7 and filament mode are closed
(3c.2: membrane negatives + contrastive 0.1, actin F1 0.810, microtubule 0.9995). Schedule-free (3b.3)
was tried and lost (0.771 vs 0.792): RAdam + EMA stays the default optimiser, profile with it.
TODO items 4 (SSL re-evaluation), 5 (~130-class smoke test) and 6 (DS-10490) are **deferred until
after Gate 8**, not dropped. Goal in the human's words: squeeze as much as possible out of a
**single-GPU box** (RTX A6000, 48 GB), measured here first.

**Multi-GPU (human decision 9 Nov 2026): also a target, but no multi-GPU box is available yet.**
Design every change for DDP too (no unused parameters, sampler and crop seeds per rank and epoch,
no rank-0-only work that stalls the others: calibration, snapshots, image logging; gradient
bucketing / `static_graph` where it applies; multi-GPU predict through the existing Dask workers).
Check correctness here with 2 processes on the gloo backend (NCCL refuses two ranks on one GPU)
and compare against the single-process result. Mark every multi-GPU speed claim as unmeasured,
and list the measurements to run once a box is available.

**Measuring speed (human decision 9 Nov 2026): no full training run per change**, or Gate 8 never
ends. Per change:
- speed: a fixed micro-benchmark, a few hundred steps after warm-up on fixed data and seed
  (steps/s, data wait, peak memory), the same script for before and after;
- equivalence: outputs, loss and gradients on a few fixed batches against the old code, with
  stated tolerances, plus a short run (a few epochs, or a few hundred steps) whose loss curve must
  overlay the baseline's.
Full training runs only for changes that alter the training distribution (on-the-fly sampling,
library upgrades), and then one acceptance run per group of such changes at the end of a
sub-step, not one per change. Changes that are exact (checkpointing off, batched Hessian) need no
full run. Consider folding the memory / step-time part of item
5.1 (C ≈ 134) into the 8a profile, since the per-class loops scale with C.

Measured on filament runs (DS-10490, 20 training tomograms, one level at 6.8 A/px, batch 16,
1059 steps per epoch, 18.4 GB GPU): 19 min per epoch with the 3c.1b recipe, 20 min with the Gate 7
contrastive loss (+5 %); early stop near epoch 50, 16.8-17.1 h per run. Predict 23.7 s per
600x600x~120 tomogram for both classes; tracing 0.4-1.9 s per class (CPU); post-fit calibration
traces every threshold 0.1-0.9 per filament class. `training/snapshots.py` (dashboard snapshots)
adds a predict pass of a few chunks per epoch when `snapshot_chunks > 0`.

The text below is the original plan of 2 Nov.

- **Gate 8 work starts only after all training and predict code has been validated**: Gate 7
  closed (CZII CV arms, filament benchmark with membranes), and the remaining validation items of
  `TODO_v2.md` (SSL re-evaluation, ~130-class scaling smoke test, DS-10490, schedule-free optimiser
  `RAdamScheduleFree`, item 3b.3) done or waived by the human. Profile the training step with the
  optimiser that will be the default. Writing the spec may happen earlier; it changes no code.
- **Never touch the working tree while a CV / benchmark run is in progress**: every fold starts a
  fresh `tomocpt train` from the repo, so a code change silently enters later folds. Gate 8
  implementation goes on `v2-feat/perf` in its own worktree.
- Check for running jobs before profiling (`nvidia-smi`, `pgrep -f "tomocpt|run_cv"`); profiling
  needs the GPU to itself.

## 2. Human decisions that the spec must encode (2 Nov 2026)

1. **Goldens describe the current (slow) implementation.** Every golden value in `tests/golden/`
   is true for today's setup. When a performance change alters outputs (even within float
   tolerance), the affected goldens are **recomputed** from the new code, after the change has
   passed its equivalence tests against the old code. The spec must say, per step, which goldens
   are expected to change and how they are regenerated (script, source commit, tolerance), and
   that each regeneration is recorded as a ruling in `gate_status.json`. Frozen tests whose
   assertions encode a slow-path detail (not behaviour) go through a test dispute, not silent edits.
2. **Holistic scope, training first.** Cover the whole process: `prepare_data`, chunking /
   sampling, dataloading and augmentation, the training step, post-fit calibration, CV / ensemble,
   `predict` (sliding window, peaks, filament tracing), and benchmarks. Training is the first
   target.
3. **Library upgrades are allowed** (torch, MONAI, Kornia, TorchIO, Lightning, ...) when they bring
   relevant optimisations; each upgrade is its own step with full suite + regenerated goldens.
   Invariant 8 still applies: list each proposed pin change in the spec for human approval.
4. **CuPy / cuCIM are allowed** as long as installation stays painless for end users: an optional
   extra (e.g. `tomocpt[gpu]`) with versions matched to the torch wheel's CUDA, automatic fallback
   to numpy/scipy when absent, identical results within a stated tolerance. Prefer native torch or
   hand-written torch code where it is reasonable; use CuPy/cuCIM where torch has no equivalent
   (EDT, connected components, skeletonisation, ...).
5. **On-the-fly sampling instead of chunk files** (the human's proposal): preprocess each tomogram
   once per level, precompute crop-centre indices weighted by an importance map, crop during
   training with a seeded generator (fixed seed for tests). Opt-in first; becomes the default only
   after a CZII CV comparison against the grid-chunk baseline.

## 3. Findings and ideas to verify (orchestrator, 2 Nov 2026; not measured unless stated)

- **Measured:** CZII training is GPU-bound: 100 % utilisation, 18.5 of 48 GB, batch 16,
  about 9 min per epoch (402–476 steps), about 4.3 h per fold with early stopping near epoch 28,
  about 21.5 h per 5-fold CV. Baseline fold chunks are 8.1 GB per fold
  (`benchmarks/out/czii_bench_20261029_2237/cv/fold0/chunks`).
- **Free GPU memory → turn off activation checkpointing** (`SwinUNETR.use_checkpoint`, and the
  per-class Hessian checkpointing in `training/multiclass_loss.py`) rather than raising the batch:
  same maths and optimisation, no recompute in backward. Raising the batch size changes the
  optimisation and breaks comparability with existing baselines (human was told; not chosen).
- Other training-step candidates: batched Hessian term (one call for all classes instead of the
  per-class loop), `torch.compile`, fused/SDPA attention (check what MONAI and torch versions
  provide before claiming it), bf16 coverage of the loss.
- **Chunk redundancy:** chunk 64 at stride 32 stores each voxel about 8 times, and every CV fold
  re-chunks the same tomograms (about 40x in total for 5 folds).
- **On-the-fly design sketch (to refine in the spec):** one memory-mappable float16 array per
  tomogram and level, plus points/segments/negative masks; a sampling map over crop centres
  (near labelled objects with per-class repeat factors, inside negative masks, and a background
  floor that replaces the `ALPHA` empty-chunk dropping); per-epoch draws seeded by (seed, epoch,
  worker); GPU-resident tomograms when they fit (CZII ≈ 1 GB in float16), memmap + pinned memory
  otherwise; targets rendered on the GPU from geometry; rotation by sampling a rotated crop from
  the full tomogram (`affine_grid` / `grid_sample`, no empty borders); a fixed seeded validation
  set so `val_loss_smooth` stays stable for early stopping and LR scheduling. Pipeline v1 and grid
  chunks stay available (invariant 1).
- **scipy / skimage use** (quick survey): mostly in `prepare_data` and inference —
  `labels/filaments.py` (EDT, `label`, `binary_fill_holes`, `skeletonize`, splines),
  `predict/filaments.py` (`map_coordinates`), `predict/helpers.py` and `metrics.py` (`cKDTree`),
  `dataManager/fourier_augmentations.py` / `filter_utils.py` (`gaussian_filter1d`, `interp1d`).
  The training loop's CPU cost is mostly TorchIO augmentation, `.nii.gz` decoding and numpy target
  rendering — profile before porting anything.
- Known waste: post-fit calibration runs even for filament-only models (≈ 5 min per validation
  tomogram); OOF picks per fold in the ensemble step (~8 min per CV).
- CV as a routine (moved to TODO_v2.md item 3a, human decision 4 Nov 2026; Gate 8 only profiles it): a built-in `tomocpt train` CV mode (today `experiments/run_cv.sh` loops folds,
  `tomocpt.predict.ensemble` pools OOF picks). Becomes worthwhile once a fold is fast.

## 4. Steps for the receiving agent

1. **Scan** every stage listed in 2.2: entry points, hot loops, CPU/GPU placement, data formats,
   per-call numpy/scipy/skimage use, host↔device copies, files written and read. Record file and
   function names (`path:line`).
2. **Profiling plan** (and, only if the GPU is free and the human agrees, a short measurement run):
   one CZII fold for a few hundred steps (forward, backward, Hessian, data wait), one `prepare_data`
   + chunking, one CZII `predict`, one calibration. Use `torch.profiler` / simple timers; write
   numbers into the spec's "Why" section and mark estimates as estimates.
3. **Write `.claude/agents/08_AGENT_PERFORMANCE.md`** as a sequence of sub-steps (proposed order,
   adjust from the scan): 8a profile; 8b training step without behaviour change (checkpointing,
   batched Hessian, compile); 8c on-the-fly sampler + GPU data path + GPU augmentation (opt-in);
   8d built-in CV mode; 8e prepare_data / predict on GPU (+ optional `[gpu]` extra); 8f library
   upgrades. For each sub-step: design, config fields with defaults (all defaults reproduce the
   current behaviour until the human flips them), allowed write paths, public names, pass criteria
   (equivalence vs the old code with explicit tolerances; speed targets as measured ratios on the
   development machine; which goldens are regenerated and how), and human acceptance (CZII CV
   within noise of the current baseline for anything that changes the training distribution).
4. Propose the `CLAUDE.md` additions (Section 6 config rows, Section 7 gate-map row) **inside the
   spec** as a block for the orchestrator; do not edit `CLAUDE.md` yourself.
5. Add a `TODO_v2.md` item pointing to the spec, commit both on `v2-dev` (never `main`), stop and
   report to the human with the open questions (dependency pins, CuPy/cuCIM packaging, speed
   targets, which sub-steps to run first).

## 5. Known traps

- Bit-identical lock tests (Gates 1–7, e.g. `tests/golden/gate7_lock.json`, CPU logits SHA) will
  break under compile / upgrades / kernel changes by design: plan their regeneration (decision 2.1),
  do not weaken them silently.
- CPU backward is not bit-reproducible run to run (Gate 7 compares gradients by absolute sum,
  rtol 1e-5); GPU kernels even less so — equivalence criteria need tolerances, and seeds.
- `ddp_find_unused_parameters_false`: any new module must give every parameter a gradient each step.
- Unpickled checkpoint configs lack new fields: read them with `safe_getattr` (CLAUDE.md 1.9).
- Contrast inversion is always on (human decision); keep it in any GPU augmentation path.
- Targets must stay label maps for TorchIO (invariant 10); a GPU augmentation path outside TorchIO
  must keep the same semantics (intensity transforms never touch targets; linear interpolation of
  soft targets in v2).
