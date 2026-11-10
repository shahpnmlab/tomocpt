# 08_AGENT_PERFORMANCE.md — Gate 8: performance (single GPU first, DDP-ready)

Written 9 Nov 2026 from `HANDOFF_gate8.md` (human decisions 2 and 9 Nov 2026). Open questions
answered by the human on 9 Nov 2026 (Section 12); the spec below includes the answers.

> **INVIOLABLE (human, 9 Nov 2026): all Gate 8 work, including amendments to this spec, lives on
> branch `v2-feat/perf` (worktree `../tomocpt-perf`). Nothing is committed to or merged into
> `v2-dev` until every sub-step works to spec within its stated margins.** Gate 8 runs before TODO items 4 (SSL re-evaluation), 5 (~130-class smoke test) and
6 (DS-10490), which are deferred, not dropped. Optimiser for every profile: RAdam + EMA
(`train.weight_averaging=ema`); schedule-free lost in 3b.3 (0.771 vs 0.792).

Goal (human): get as much as possible out of a **single-GPU box** (RTX A6000, 48 GB), measured
here first. Multi-GPU is also a target, but no such box is available yet: every change is
designed for DDP, its correctness is checked here with 2 processes on gloo, and every multi-GPU
speed claim is marked **unmeasured** (list in Section 9).

---

## 1. Why

### 1.1 Measured on production runs (before this spec)
| Run | Numbers |
|---|---|
| CZII CV (Gate 7 arms) | GPU-bound, 100 % utilisation, 18.5 of 48 GB, batch 16, ~9 min per epoch (402–476 steps), ~4.3 h per fold (early stop near epoch 28), ~21.5 h per 5-fold CV; fold chunks 8.1 GB each |
| Filaments (DS-10490, 3c.1b / 3c.2) | 20 train tomograms, one level 6.8 Å/px, batch 16, 1059 steps per epoch, 18.4 GB; 19 min per epoch (20 min with contrastive 0.1, +5 %); early stop near epoch 50; 16.8–17.1 h per run |
| Predict | 23.7 s per 600×600×~120 tomogram (both filament classes); tracing 0.4–1.9 s per class (CPU) |
| Calibration | ≈ 5 min per validation tomogram (also for filament-only models); filament calibration traces 9 thresholds 0.1–0.9 per class |

### 1.2 Measured for this spec (9 Nov 2026, synthetic, scratch scripts, nothing in the repo changed)
Training step on random batches, `MySwinUNETR` fs=36 v2 with dropout as the defaults, bf16 autocast,
RAdam, clip 1.0, `multiclass_picking_loss` (Huber). Mean of 30 steps after 10 warm-up steps, CUDA
events. "loop+ckpt" = today's per-class Hessian loop under gradient checkpointing.

| C | B | Swin ckpt | Hessian | ms/step | fwd | loss | bwd | peak GB |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | on (today) | loop+ckpt | 977 | 285 | 5 | 681 | 12.9 |
| 1 | 16 | **off** | loop+ckpt | 813 (**1.20×**) | 287 | 5 | 516 | 25.9 |
| 2 | 16 | on (today) | loop+ckpt | 1003 | 289 | 9 | 699 | 12.9 |
| 2 | 16 | off | loop / batched | 830 / 825 | 290 | 8 / 7 | 525 / 522 | 25.9 / 26.0 |
| 6 | 16 | on (today) | loop+ckpt | 1052 | 291 | 24 | 731 | 13.1 |
| 6 | 16 | off | loop+ckpt / loop / batched | 884 / 866 / 850 | 291 | 24 / 24 / 21 | 562 / 544 / 532 | 26.1 / 26.8 / 28.4 |
| 134 | 4 | on (today) | loop+ckpt | 1423 | 78 | 163 | 1176 | 5.6 |
| 134 | 4 | off | loop+ckpt | 1379 | 78 | 163 | 1132 | 10.3 |
| 134 | 4 | off | **batched** | 418 (**3.4×**) | 78 | 117 | 217 | 26.8 |

Forward time per top-level module (C=2, B=16, no checkpointing): `swinViT` 184 ms (65 %),
`decoder1` (64³) 56 ms, `encoder1` (64³) 28 ms, everything else < 8 ms each. CUDA kernel profile
(3 steps): conv wgrad/fprop/dgrad ≈ 25 %; layout copies (`direct_copy`, `nchwToNhwc`/`nhwcToNchw`,
device-to-device memcpy) ≈ 17 %; attention softmax forward + backward (fp32 under autocast) ≈ 8.6 %;
dropout kernels ≈ 4 %; the optimiser step 6 ms.

`torch.compile` of the whole `MySwinUNETR` (torch 2.5.1, default mode, training step with backward,
B=16): the first compiled step **had not finished after 15 min** (killed; GPU idle meanwhile, so the
time is graph capture / Inductor codegen). Whole-model compile is impractical as is; 8b.5 tries
regional compilation (each `SwinTransformerBlock` / `UnetrBasicBlock` compiled separately, so one
compiled region is reused across blocks of equal shape) and measures compile time first.

Data loader stages, one CPU thread (one DataLoader worker), 64³ chunk, 12 particles:
| Stage | C=1 | C=6 | C=134 |
|---|---|---|---|
| read volume `.nii.gz` (today) / `.npy` float16 | 12.4 / 0.6 ms | | |
| `render_targets` (points → dense) | 1.6 ms | 2.1 ms | 2.9 ms |
| default augmentation (`RandomAffine` p 0.8 + elastic/blur), target nearest | 41 ms | 103 ms | **2231 ms** |
| same, target linear (v2 recipe) | 34 ms | 80 ms | **1316 ms** |

### 1.3 Conclusions (drive the order of work)
1. At C ≤ 6 the training step is **≈ 98 % network** (forward + backward). The loss is 0.5–2.5 %.
   The batched Hessian gains nothing there; the network does.
2. **Swin checkpointing off is a free 1.20×** at unchanged maths (same RNG; MONAI checkpoints with
   `use_reentrant=False`, which restores the RNG state in the recompute), at 2× activation memory
   (12.9 → 25.9 GB at B 16, fits 48 GB). It is already a config field
   (`train.network.SwinUNETR.use_checkpoint`, default True). The per-class Hessian checkpointing
   (`train.particles.loss_checkpoint`) costs ~2 % at C=6.
3. The remaining network cost is in the Swin encoder (attention with an explicit fp32 softmax,
   window partition / roll copies) and in the two 64³ conv blocks. Candidates: fused attention
   (SDPA), `torch.compile`, channels-last 3-D convolutions — each measured in 8b before it is kept.
4. At C ≈ 134 two things break: the per-class Hessian loop (3.4× slower than batched, at B=4) and
   the CPU augmentation, which resamples every target channel (1.3–2.2 s per sample, so 16 samples
   on 8 workers ≈ 3–4.5 s per batch); a `[16, 134, 64³]` float32 target is also 2.25 GB per batch
   on the host→device link. Targets must be rendered **after** the spatial transform, from geometry,
   on the GPU (8c) — not resampled.
5. At C ≤ 6 with 8 workers the CPU path keeps up today (~0.1 s per sample → ~0.2 s per batch of 16
   against a ~1 s step); it will **not** keep up once the step is 1.5–2× faster, at higher C, or with
   the Fourier / intensity augmentations on. So the GPU data path (8c) matters after 8b.
6. Chunk files: each voxel stored ~8× (64 at stride 32), `.nii.gz` decode 12 ms per chunk, 8.1 GB per
   CZII fold. On-the-fly crops from one array per tomogram and level remove chunking, the disk use and
   the decode.
7. Predict / calibration (not yet profiled, 8a): predict reads patches with TorchIO `GridSampler` +
   `DataLoader(batch_size=infer.predictions_batch_size=2, num_workers=0)` from a CPU tensor and copies
   each batch to the GPU (`predict/multiclass.py:_predict_level`); peak NMS
   (`predict/helpers.py:peak_local_max_torch`) runs a Python loop over candidates with a GPU→CPU sync
   per candidate (`if suppress_grid[...]`), and calibration collects picks at the **lowest** grid
   threshold, where candidates are most numerous. Likely the ≈ 5 min per validation tomogram.

---

## 2. Scan: where the time goes (file:line at `81d43f1`)

| Stage | Entry / hot code | Placement and notes |
|---|---|---|
| prepare_data | `labels/run.py`, `labels/helpers.py`, `labels/filaments.py:675 _region_polylines` (ndimage.label, skimage `skeletonize`), `:719 _fill_lumen` (2× EDT, `binary_fill_holes`), `:213 fit_filament_spline` (scipy splprep) | CPU numpy/scipy; filament masks only |
| chunking v2 | `dataManager/chunking.py:588 process_single_tomo_task_v2`; Dask `_run_tasks:776` (one GPU worker per GPU) | load MRC → normalise (CPU) → `scaling.resample_volume` (GPU when available) → `symmetrize_and_pad` → TorchIO `GridSampler` → per chunk `.nii.gz` write in an 8-thread pool (`_save_chunk:122`), points `.npy`, packed negative-mask `.npy`; drop of empty chunks seeded per (seed, dir, level, scale) |
| chunk pool / folds | `chunking.py:976 chunk_pool_v2`, `:994 link_fold_chunks`; `training/cv.py` | Gate 4c: chunks once, symlinked per fold |
| dataset | `dataManager/dataloading.py:236 _getitem_v2` | per sample: `tio.ScalarImage(path)` (gzip decode), points `.npy`, filament segments `.npy`, negative mask; optional zoom via `grid_sample` (CPU); `render_targets` (`labels/helpers.py:571`, numpy loop per particle, `[C, 64³]` float32), `render_tubes` (`labels/filaments.py:136`, allocates `[C, 64³]` `umin`); TorchIO transforms on `[C, 64³]` |
| augmentation | `dataloading.py:36 build_training_transforms` | TorchIO `RandomAffine(otsu pad)` / `OneOf(elastic, blur)`, `RandomFourierDegradation` (numpy/scipy, off), `RandomContrastInversion` (always), intensity transforms (off) — CPU workers |
| loader | `dataloading.py:532 _get_dataloader`, `:541 train_dataloader` | `num_workers = n_cpus_for_dataloading` (8), **no `pin_memory`**, default prefetch, `persistent_workers` only when `use_gpus`; repeat-factor `WeightedRandomSampler` seeded once (Lightning wraps it under DDP) |
| step | `networks/pickingModel.py:361 training_step` | `MySwinUNETR.forward` / `forward_features` (`networks/swinunetr.py:46`); sigmoid/softmax in fp32 (`multiclass_loss.py:33`); `multiclass_picking_loss` (`multiclass_loss.py:107`): weights, Huber, **per-class Hessian loop under checkpoint** (`:173`), `bool(unavailable.any())` host sync (`:191`) |
| step extras | `pickingModel.py:209 contrastive_loss`, `training/discriminative.py`, `training/offsets.py` | anchors / hard negatives (sorting, `nonzero`) |
| logging | `pickingModel.py:452 _log_panels` | rank 0 only; per step until every class has a panel: `counts.argmax()` + `int(...)` host syncs, `.cpu()`; `_log_per_class_val_loss:427`: C extra loss calls + `int(sum)` sync per val batch |
| EMA | `training/averaging.py:107` | per-tensor `mul_`/`add_` loop over `state_dict()` every step (~200 tensors) |
| trainer | `training/helpers.py:520 _setup_trainer` | `ddp_find_unused_parameters_false`, `bf16-mixed`, clip 1.0, `TQDMProgressBar(refresh_rate=10)` |
| snapshots | `training/snapshots.py:91` | rank 0 only, at each validation end: predict of K chunks (others wait at the next all-reduce) |
| post-fit calibration | `training/run.py:112` → `predict/calibration.py:229 collect_validation_picks` | rank 0 only, after fit; predicts every validation tomogram (particle path even for filament-only models), then traces 9 thresholds per filament class (`calibration.py:96`) sequentially on the CPU |
| predict | `predict/run.py` (Dask, one worker per GPU) → `predict/multiclass.py:245 predict_class_maps` → `:182 _predict_level` | MRC load + normalise (CPU); resample on GPU; **pad then `.cpu()`** for TorchIO; batch 2; Hann accumulation per patch in a Python loop; TTA flips run sequentially; resample back per class on GPU then `.cpu()` |
| peaks | `predict/helpers.py:192 peak_local_max_torch`, `:394 _refine_peaks_subpixel_gpu`, `:1013 cross_class_suppression` (cKDTree) | max-pool on GPU, then greedy NMS loop in Python with a host sync per candidate |
| tracing | `labels/filaments.py:756 trace_filaments`, `predict/filaments.py` (`map_coordinates`) | CPU |
| CV / ensemble | `training/cv.py` (one subprocess per fold, sequential), `predict/ensemble.py`, `predict/cv_report.py` | OOF picks saved per fold (Gate 4c D3) |
| metrics | `metrics.py` (cKDTree) | CPU, cheap |

---

## 3. Ground rules for every sub-step

### 3.1 Where and when
- **Inviolable:** all work on branch `v2-feat/perf` in its own worktree `../tomocpt-perf` (created
  9 Nov 2026 from `v2-dev` b8042e7). No commit to `v2-dev` and no merge into it until **every**
  sub-step passes its criteria within the stated margins (human decision 9 Nov 2026); then one merge,
  with the human's go-ahead. Sub-steps commit and tag on `v2-feat/perf` only (`v2-gate-8x-passed`
  tags point at `v2-feat/perf` commits). Acceptance and benchmark runs start from the perf worktree.
  Fixes needed on `v2-dev` meanwhile are merged *into* `v2-feat/perf`, never the other way. Never
  commit to `main`.
- Never change the `v2-dev` working tree while a CV / benchmark run uses it (folds start fresh
  `tomocpt train` processes from the repo).
- Before any measurement: `nvidia-smi` and `pgrep -af "tomocpt|run_cv"` must show the GPU free.

### 3.2 Change classes
| Class | Meaning | Proof | Full training run |
|---|---|---|---|
| **E0 exact** | same arithmetic | CPU fp32 bit-identical outputs, loss, gradients on fixed batches; GPU within the run-to-run noise of the old code | none |
| **E1 equivalent** | same maths, different kernels / summation order | fixed batches: logits, loss, gradients vs old code within the tolerance of the sub-step (fp32 CPU and bf16 GPU); short run (≥ 300 steps, fixed data and seed) whose loss curve overlays the baseline | none |
| **E2 distribution** | changes what the network sees (sampling, augmentation implementation, library upgrade) | E1 checks where they apply, plus distribution checks (Section 6); short runs overlay within noise | **one acceptance run per group of E2 changes, at the end of the sub-step** (Section 8) |

Tolerances, unless a sub-step says otherwise: CPU fp32 E1 — logits `atol 1e-5`, loss `rtol 1e-6`,
per-parameter gradients `rtol 1e-4` on the absolute sum and `atol 1e-6` elementwise; GPU bf16 E1 —
loss `rtol 2e-3`, gradient cosine similarity ≥ 0.999 per parameter tensor with norm > 1e-6. The
loss-curve overlay: same fixed batches and seeds, mean absolute difference of the 50-step moving
average of the training loss over steps 100–300 ≤ 3 × that between two runs of the old code (bf16
GPU is not deterministic, so the old code's own spread sets the noise).

### 3.3 Speed measurements (no full training run per change; human decision 9 Nov 2026)
`benchmarks/perf/step.py` (8a) is the single micro-benchmark for every change, run before and
after with the same arguments: fixed seed, fixed data (synthetic batches, or a fixed list of real
CZII chunks / crops), 50 warm-up steps, 300 timed steps; reports steps/s, forward / loss / backward
/ optimiser ms (CUDA events), **data wait** (time `training_step` waits for the next batch),
peak allocated and reserved memory, and the git commit. A claim is a ratio of two runs on the
development machine (RTX A6000), each repeated twice; run-to-run spread is reported with it.

### 3.4 Goldens (human decision 2 Nov 2026)
Goldens in `tests/golden/` describe the current implementation. When a change alters a golden
value (even within float tolerance):
1. The change first passes its E0/E1 tests against the **old** code (values from a worktree of the
   commit before the change, never from the new code).
2. Then the affected goldens are regenerated with their own script
   (`tests/golden/make_goldens_gate{1,2,3,3b,4,6,7}.py`, `tests/gate5/make_golden_g5_c1.py`) at the
   new commit; `benchmarks/perf/compare_goldens.py` (8a) prints the max abs / rel difference of
   every array and scalar between old and new files, which must be inside the sub-step's tolerance.
3. The regeneration is recorded as a ruling in `.claude/state/gate_status.json → gates.8.notes`:
   file, script, source commit, old/new max difference, tolerance, reason; the frozen test hashes are
   updated in the same commit. The generator scripts write the source commit into the golden file.
4. A frozen test whose assertion encodes a slow-path detail (not behaviour) is changed only through
   `.claude/reports/gate_08_test_dispute.md` and a human ruling, never silently.

Per sub-step, the goldens expected to change are listed under "Goldens".

### 3.5 Defaults
Every new config field defaults to today's behaviour (invariant 1). An E0 change that only
removes work (a host sync, a copy) needs no field. Flipping a default is a separate, explicit
human decision, recorded in `gate_status.json`, with the goldens handled as in 3.4.

### 3.6 DDP rules (every sub-step)
- Every parameter gets a gradient every step (invariant 5; `ddp_find_unused_parameters_false`).
  A new module is created only when its feature is on.
- Random draws (crop centres, augmentation, dropout aside) are seeded by `(seed, epoch, global rank,
  worker)`; no two ranks draw the same stream; the union over ranks is deterministic for a seed.
- No rank-0-only work inside the training loop that other ranks wait for at the next collective
  (snapshots, image panels, per-class validation logging): shard it over ranks, make it
  asynchronous, or measure and accept its cost explicitly.
- Collectives only where Lightning puts them (gradient all-reduce, `sync_dist` metrics at epoch end);
  nothing new calls `all_gather` per step. `static_graph=True` / bucket size only where 8b shows the
  graph is static (it is not with the contrastive anchors' variable sizes — check).
- **Gloo check here:** 2 processes, `accelerator="cpu"`, `strategy="ddp"` (gloo), tiny model, no
  dropout, every `(b, c)` available: after one step, the all-reduced gradients equal the
  single-process gradients on the concatenated batch (fp32, `rtol 1e-5`); two epochs of a v2 fit
  finish and save one checkpoint. Test file `tests/gate8/test_g8_ddp_gloo.py` (slow marker not
  needed if < 60 s). NCCL refuses two ranks on one GPU, so GPU DDP is unmeasured.

---

## 4. Sub-steps (proposed order: 8a → 8b → 8e.1 → 8c → 8e.2 → 8f; 8d is profile-only)

### 8a — Profile harness and baseline (no behaviour change)
**Design.**
- `benchmarks/perf/step.py`: Section 3.3 micro-benchmark. Inputs: `--config <yaml>` (the recipe),
  `--data synthetic|chunks:<dir>|otf:<dir>`, `--classes C`, `--batch B`, `--steps`, `--warmup`,
  `--seed`, `--profile` (adds a `torch.profiler` trace and the per-module forward table of 1.2),
  `--out <json>`. Builds the model and loss exactly as `tomocpt train` does (`_create_model`,
  `BasePickingModel.training_step`), so recipe flags are measured, not re-implemented.
- `benchmarks/perf/loader.py`: data path alone (samples/s per worker count, per-stage ms as 1.2,
  for chunk dirs and later for on-the-fly), no GPU step.
- `benchmarks/perf/pipeline.py`: wall time of `prepare_data`, chunking (CZII, 7 tomograms), one
  `tomocpt predict` on a CZII test tomogram and on a DS-10490 filament tomogram (per stage: load,
  resample, patch loop, accumulate, resample back, peaks, refine, tracing, write), and one post-fit
  calibration (`calibrate_checkpoint`) per particle and filament model; `--profile` adds
  `cProfile` top-30.
- `benchmarks/perf/compare_goldens.py`: Section 3.4 step 2.
- **TODO 5.1 folded in (memory / step time part only):** `step.py --classes 134` with the
  Hessian loop and batched (8b.2), sigmoid and softmax heads, B = 4, 8, 16; peak memory; plus the
  loader at C = 134. The accuracy part of 5.1 stays in TODO 5.
- Baseline report `benchmarks/out/perf_baseline_<date>/` (JSON + `summary.md`): every table of
  Section 1.2 re-run through the harness on real CZII chunks (fold 0 of the Gate 7 pool) and on
  the DS-10490 filament chunks, plus pipeline timings. These are the "before" numbers of every later
  sub-step.

**Config.** None. **Goldens.** None.

**Pass criteria.**
- **G8a-C1** `step.py` on synthetic data (C=2, B=2, tiny model, CPU, 5 steps) writes a JSON with
  `steps_per_s`, `ms_forward`, `ms_loss`, `ms_backward`, `ms_optimizer`, `ms_data_wait`,
  `peak_mem_bytes`, `commit`, `config_hash` and `losses` (one float per timed step); two runs give the
  same loss sequence (CPU, fixed seed); the CLI (`python -m benchmarks.perf.step ... --out x.json`)
  writes the same keys.
- **G8a-C2** `compare_goldens.compare(old, new)` (`.json` or `.npz`) returns `{"max_abs", "max_rel",
  "keys": {name: {"max_abs", "max_rel"}}, "missing", "added", "mismatched"}` (JSON flattened to
  `a/b/0` paths; `mismatched` = non-numeric values or shapes that differ). A golden against itself:
  all zero, empty lists; a perturbed copy: the exact max difference at the perturbed key.
- **G8a-C3** (orchestrator check, not pytest) Repository unchanged except `benchmarks/perf/**`,
  `tests/gate8/**`, the spec, reports and state.
- Baseline report written (human-reviewed, not a test); on the A6000, `step.py` run-to-run spread
  of `steps_per_s` ≤ 3 %.

### 8b — Training step without behaviour change
Each item is its own commit with its own micro-benchmark before/after and equivalence test.

**8b.1 Swin checkpointing off (E0, config only).** `train.network.SwinUNETR.use_checkpoint=False`
already exists. Work: equivalence test, measurement on real data, memory head-room check at B=16
with contrastive and offset heads on, and a recipe entry. Measured 1.20× (1.2). *Default flip
(proposal): `False` for new runs* — maths identical, checkpoints unaffected (the field is in the
pickled config; `predict` is under `inference_mode`, where checkpointing does nothing useful) — human
decision. Also `train.particles.loss_checkpoint=False` (E0, ~2 % at C=6; at C=134 memory decides).

**8b.2 Batched Hessian in class blocks (E1).** `multiclass_picking_loss(..., hessian_block=1)`:
the Hessian term runs on blocks of `hessian_block` channels at once (`kornia.filters.
spatial_gradient3d` is depthwise per channel), `0` = all channels in one call, `1` = today's loop.
The target Hessian is computed under `torch.no_grad()` (it needs no graph). Checkpointing, if on,
wraps each block. Peak memory grows linearly with the block (at C=134, B=4: 26.8 GB for one block);
`benchmarks/perf/step.py` picks the largest block that keeps peak ≤ 40 GB at the recipe batch.
Also remove the per-step host sync `bool(unavailable.any())` (`multiclass_loss.py:191`): always add
`0 * y_pred[:, unavailable].sum()` (E0: adding an exact zero; the sum over an empty selection is 0).
Measured 3.4× at C=134, B=4; ≤ 2 % at C ≤ 6.

**8b.3 Host syncs, copies, loader flags (E0).**
- `train.pin_memory: bool = False`, `train.prefetch_factor: Optional[int] = None` (torch default) →
  `DataLoader(pin_memory=..., prefetch_factor=...)`; batches moved with `non_blocking=True`.
- `_log_panels` (`pickingModel.py:452`): look for a panel only every `train.panel_every_n_steps`
  (default 1 = today), or only on validation batches; no per-step `.cpu()` / `int()` on rank 0
  otherwise. `_log_per_class_val_loss`: one batched call returning `[C]` instead of C calls + C syncs.
- EMA update with `torch._foreach_mul_` / `torch._foreach_add_` over a cached list of tensors
  (`averaging.py:107`), bit-identical per element.
- Each kept only if `step.py` shows a gain; otherwise dropped with the measurement in the report.

**8b.4 Fused window attention (E1, opt-in).** `train.network.attention: eager | sdpa = eager`.
`tomocpt/networks/attention.py: SDPAWindowAttention` replaces MONAI's `WindowAttention` in a built
model (same parameter and buffer names, so state dicts and checkpoints are interchangeable) and
computes `F.scaled_dot_product_attention(q, k, v, attn_mask=relative_bias (+ shift mask),
dropout_p=attn_drop if training, scale=head_dim**-0.5)`. **Verify before claiming:** with fs=36 the
head dimension is 12 at every stage (36/3, 72/6, 144/12, 288/24); check which SDPA backend torch 2.5.1
selects for head_dim 12 with an additive float mask and dropout in bf16 (memory-efficient or math);
if it falls back to the math kernel the change is dropped. Dropout masks are drawn differently from
MONAI's `nn.Dropout` (E1 only with `attn_drop_rate=0`; with dropout on, the overlay run decides).
Applies to `MySwinUNETR` and `ScaleConditionedSwinUNETR`, picking and SSL.

**8b.5 `torch.compile` (E1, opt-in).** `train.network.compile: none | regional = none`; `regional`
compiles the repeated sub-modules (`SwinTransformerBlock`, `UnetrBasicBlock`/`UnetrUpBlock`) in place
with `nn.Module.compile()` (no `_orig_mod.` prefix in state-dict keys; checkpoints stay loadable
without compile). Whole-model compile did not finish a first step in 15 min (1.2), so it is not an
option unless an 8f upgrade changes that. Questions for 8b: compile time per run (and per fold process), recompiles when the batch shape changes (last
batch, validation batch size), interaction with Swin checkpointing (compile with checkpointing off),
with DDP (`DDPStrategy(static_graph=...)`, the optimizer hook), with the EMA swap (weights copied in
place: no recompile), and `predict` (compile there only if a tomogram has enough patches to repay
the compile time). Kept if steps/s, compile time included over a 1000-step epoch, improves
(numbers reported, no fixed bar).

**8b.6 Channels-last 3-D convolutions (E1, opt-in, only if 8a shows the layout copies).**
`train.network.channels_last: bool = False`: `model.to(memory_format=torch.channels_last_3d)` and
inputs converted once; the cudnn `nchwToNhwc` / `nhwcToNchw` kernels (≈ 5 % of GPU time) should vanish.

**Config.** `train.particles.hessian_block: int = 1`, `train.pin_memory: bool = False`,
`train.prefetch_factor: Optional[int] = None`, `train.panel_every_n_steps: int = 1`,
`train.network.attention: str = "eager"`, `train.network.compile: str = "none"`,
`train.network.channels_last: bool = False`. Existing: `train.network.SwinUNETR.use_checkpoint`,
`train.particles.loss_checkpoint`.

**Goldens.** None at defaults. If the human flips `hessian_block` (to 0 or a block > 1):
`tests/golden/gate3b_c1_loss.npz`, `gate6_c1.npz` (loss / gradient arrays only), `gate7_lock.json`
(`loss_hex`, `grads_abs_sum`) may change in the last bits → regenerate (3.4). If `attention=sdpa`
or `compile` ever become defaults for v2 models: `gate6_c1.npz` forward arrays, `gate7_lock.json`
`logits_sha256` → regenerate; v1 models (`gate1_forward.npz`, `gate4_v1_infer.*`) keep eager
attention unless the human rules otherwise. `use_checkpoint` / `loss_checkpoint` change no golden
(every CPU golden already uses `use_checkpoint=False`).

**Pass criteria.**
- **G8b-C1** Defaults: every earlier test passes unchanged; no golden changes.
- **G8b-C2** `use_checkpoint=False` vs `True` (tiny model, CPU fp32, dropout on, fixed seed): logits,
  loss and every gradient bit-identical over 3 steps.
- **G8b-C3** `hessian_block ∈ {0, 2, 3}` vs `1` for C = 1, 3, 7 (sigmoid and softmax, mixed
  availability, `class_weights`, cross-class weight, focal value term, core weight): loss and
  gradients within the CPU E1 tolerance; an unavailable class still gets a (zero) gradient path.
- **G8b-C4** `SDPAWindowAttention`: same `state_dict` keys and shapes as MONAI's module; with
  dropout 0 the output equals the eager module (CPU fp32 `atol 1e-5`; GPU bf16 `atol 2e-2`, cosine
  ≥ 0.9999) for shifted and unshifted windows; a checkpoint saved with `eager` loads with `sdpa`
  and vice versa (strict).
- **G8b-C5** `compile=regional`: state-dict keys unchanged; the checkpoint loads in a process without
  compile and predicts within GPU E1 tolerance of the eager model.
- **G8b-C6** One v2 training step with every 8b option on: every trainable parameter gets a
  gradient (invariant 5), with contrastive and offset heads on as well.
- **G8b-C7** Gloo DDP check of 3.6 with every 8b option that runs on CPU.
- **G8b-C8** (slow, GPU) 300-step overlay of the loss curve (3.2) for the recipe with the kept 8b
  options vs the 8a baseline, on fixed real CZII chunks.
- Speed (report-only, human decision 9 Nov 2026; no pass/fail bar): steps/s ratio and peak memory
  per item on the A6000. An opt-in item that measures no gain is dropped, with its numbers in the report.

### 8c — On-the-fly sampling, GPU data path, GPU augmentation (opt-in, E2)
Human proposal (2 Nov 2026). Two parts, 8c.1 changes only *where crops come from*, 8c.2 *how they
are augmented and rendered*; **one** acceptance run for both, at the end of 8c (human decision
9 Nov 2026, Section 8). Each part still passes its own E0/E1 and overlay checks first.

**8c.1 Tomogram store and crop sampler (CPU augmentation unchanged).**
- **Store** (`tomocpt/dataManager/tomostore.py`, written by `_prepare_data_if_needed` when
  `train.sampling=on_the_fly`): per tomogram and level/scale (same `plan_levels`, resample, padding
  and coordinate map as chunking; same `<dir>__L{li}s{k}` names, `assign_chunk_dir_names`) one
  `volume.npy` in **float16** (human decision 9 Nov 2026; `train.otf.volume_dtype`, float32 kept
  as an option for the exactness tests), and targets in **float32**: `points.npy` `[N, 5]`,
  `filaments.npy` `[M, 8]` (geometry, float32, so sub-voxel centres and the Gaussian profiles are not
  rounded), any dense target (`target_storage=dense`) float32, and the rendered target always float32
  (CPU and GPU); `negative_mask.npy` (packed bits),
  `chunk_meta.json` (unchanged `ChunkMeta`). Manifest `chunk_layout.json` gains `"storage":
  "tomogram"` (absent = chunks; invariant 4: training refuses a storage that does not match
  `train.sampling`). The Gate 4c pool / fold split reuses the store (folds = lists of tomogram dirs,
  no symlinks per chunk). Written once per dataset, not per fold.
- **Centre distribution.** Candidate crop corners on the grid of chunking (stride
  `CHUNK_STRIDE`) with a uniform sub-stride jitter in `[0, stride)` per axis, so the expected
  coverage equals the grid. Each candidate gets the weight `w = w_obj + w_neg + w_bg`:
  `w_obj` = chunk repeat factor of its classes (`repeat_factor_weights`, when
  `chunk_sampling=repeat_factor`; else 1 if it contains an available object), `w_neg` =
  `train.otf.negative_mask_weight` if it contains negative-mask voxels (default 0 = off), `w_bg` =
  the keep probability `1 - drop_p` of an empty candidate (the `ALPHA` rule of chunking, as an
  expected value instead of a one-off coin flip). With default settings the expected per-epoch mix
  of empty / object crops equals the grid baseline's.
- **Epoch.** `train.otf.samples_per_epoch: Optional[int] = None` → the number of chunks the grid would
  keep (expected), so steps per epoch, plateau / early-stop patience and EMA decay keep their meaning.
  Draws per epoch: `torch.multinomial(weights, n, replacement=True, generator=g)` with
  `g` seeded by `_stable_seed(seed, epoch, global_rank)`; each rank draws `n / world_size`; the
  jitter by the same generator. Fixed `train.seed` → identical crops (tests).
- **Validation is a fixed seeded random set** (human decision 9 Nov 2026): drawn once from the
  validation tomograms with the training centre distribution (same candidate weights, jitter on), by a
  generator seeded with `train.otf.val_seed` only (not the epoch, not the rank), `val_samples` crops
  (`None` = the expected grid count of the validation tomograms); the same crops every epoch and every
  run with that seed, so `val_loss_smooth`, early stopping and the LR plateau see a stable set. No
  augmentation. Under DDP the fixed list is split over ranks by index. Its values are **not**
  comparable with the chunk baseline's `val_loss`: the comparison is the CZII CV (Section 8).
- **Reading.** CPU path: `np.load(mmap_mode="r")`, crop, then the unchanged `_getitem_v2` code from
  `render_targets` on (TorchIO transforms in workers). GPU path (8c.2): tomograms resident on the
  GPU when `train.otf.device=cuda` and they fit (`auto`: if the store is ≤ 25 % of free GPU memory;
  CZII ≈ 7 × 0.15 GB float16 at 10 Å/px), else pinned host memory.
- Exactness check: with `volume_dtype=float32`, jitter off and every candidate kept, the crops equal
  the chunk files; with float16 they differ by the float16 rounding only (values lie in [-3, 3] after
  the robust normalisation, so `|Δ| ≤ 3 · 2⁻¹¹ ≈ 1.5e-3`).

**8c.2 GPU augmentation and target rendering.** `train.otf.augment_device: cpu | cuda = cpu`.
- Runs in `LightningDataModule.on_after_batch_transfer` on the GPU (no model code changes): the
  loader yields only crop descriptors (tomogram index, corner, per-sample seed).
- **Spatial:** one affine per sample — rotation (`affine_degrees`, `affine_p`), zoom
  (`scale.zoom_aug_max`), flips (`rotation_mode=flips`) — composed with an optional smooth elastic
  displacement field (coarse control grid, `elastic_weight`, TorchIO's defaults: 7 control points,
  max displacement 7.5 voxels, cubic upsampling), applied by one `F.grid_sample` that reads the
  rotated crop **from the full tomogram** (no `otsu` padding at the borders; a crop near a tomogram
  border is padded with the tomogram's own robust median, documented as a distribution change).
- **Targets:** rendered on the GPU **after** the spatial transform: points (and segment ends) are
  mapped with the inverse affine, then drawn with a torch port of `render_targets` /
  `render_tubes` (`tomocpt/labels/render_torch.py`, same Section 1.3 profile, clip policy); with an
  elastic field, the target is rendered in the source frame of the crop and warped with the same
  grid, `linear` (v2) or `nearest` (`label_interpolation=nearest`) — the semantics of a
  `tio.LabelMap` (invariant 10's intent: intensity transforms never touch targets).
- **Intensity:** blur (separable Gaussian, `blur_std`), contrast inversion at
  `CONTRAST_INVERSION_PROB` (**always on**), and the optional intensity transforms; the Fourier
  degradation (`use_mw_aug`, `use_fourier_aug`) after every spatial transform, ported to `torch.fft`
  when it is turned on in a recipe — until then `augment_device=cuda` with those options on raises
  `ValueError` naming the CPU path.
- Negative masks warped with the same grid (nearest), contrastive anchors unchanged (they read the
  dense target).
- SSL v1/v2 and pipeline v1 keep the CPU path (`ValueError` if `augment_device=cuda` is combined
  with them).

**Config** (`defaultConfigs/otf_config.py: OnTheFlyConfig`, field `train.otf`; plus `train.sampling`).
```python
train.sampling: str = "chunks"                 # "chunks" | "on_the_fly"
@dataclass
class OnTheFlyConfig:
    volume_dtype: str = "float16"              # "float16" | "float32" (targets/geometry always float32)
    device: str = "auto"                       # where tomograms live: "auto" | "cuda" | "cpu"
    augment_device: str = "cpu"                # "cpu" (TorchIO in workers) | "cuda" (8c.2)
    samples_per_epoch: Optional[int] = None    # None = expected grid count
    jitter: bool = True                        # sub-stride jitter of crop corners
    negative_mask_weight: float = 0.0          # extra weight of crops with negative-mask voxels
    seed: int = 42
    val_seed: int = 1234                       # fixed seeded validation set
    val_samples: Optional[int] = None          # None = expected grid count of the validation tomograms
```

**Goldens.** None changes (all new paths are opt-in). New goldens for the on-the-fly path, made by
its own generator script at the 8c commit (fixed seed): crop corners of epoch 0/1, one rendered
batch (sums and SHA), store manifest.

**Pass criteria.**
- **G8c-C1** Defaults (`sampling=chunks`): every earlier test passes unchanged.
- **G8c-C2** Store: for every tomogram, level and scale the store holds what chunking would chunk
  (volume after resample + padding equal to the chunker's padded array: float16 store within
  `atol 1.5e-3` and equal to `chunker_array.astype(float16)`, `atol 0` with `volume_dtype=float32`;
  points / segments float32 and identical, negative mask identical); written once for K folds.
- **G8c-C3** With `jitter=False` and every candidate kept, the crops and targets of the CPU path equal
  the chunk files (volume `atol 0` with `volume_dtype=float32`, `atol 1.5e-3` with float16; target
  float32, `atol 1e-6` in both cases).
- **G8c-C4** Sampler: same seed → same crops; different epoch or rank → different streams with no
  shared draws; the per-class frequency of object crops over 10⁴ draws matches the repeat-factor
  weights within 3 σ; the fraction of empty crops matches the grid's expected keep rate within 3 σ.
- **G8c-C5** Validation set: identical crops (corners and contents) across epochs, across two runs
  with the same `val_seed`, and under 2-rank gloo DDP (union of the ranks' crops = the 1-process
  list); a different `val_seed` gives a different set; drawn only from validation tomograms; no
  augmentation applied.
- **G8c-C6** GPU spatial transform: for 100 random affines, rendered targets after the transform
  have their maxima at the mapped point positions (≤ 0.5 voxel) and equal the CPU `render_targets`
  of the mapped points (`atol 1e-5`); identity transform → the crop equals the CPU crop.
- **G8c-C7** GPU intensity: contrast inversion fires at p 0.5 ± 3 σ over 10⁴ samples; targets
  unchanged by every intensity transform; elastic displacement statistics (mean |d|, max |d|) of the
  GPU field match TorchIO's within 10 % over 10³ draws.
- **G8c-C8** Gloo DDP check (3.6): two ranks draw disjoint streams, a 2-epoch fit finishes.
- **G8c-C9** (slow, GPU) Data wait < 5 % of the step time with `augment_device=cuda` at C = 6 and
  C = 134 (B=4); 300-step loss-curve overlay of 8c.1 vs the chunk baseline (same recipe) within the
  3.2 rule (the distribution check before the acceptance run).
- **G8c-C10** Contrastive anchors and negative-mask negatives work on GPU batches (one step, both on).

### 8d — CV / ensemble (profile only; built-in CV is Gate 4c, TODO 3a)
- 8a's pipeline profile covers one `cv_all_folds` run of a tiny recipe: chunk pool, fold processes,
  calibration, ensemble, report. Known waste to verify and remove if measured: post-fit calibration
  predicting particle maps for filament-only models (`training/run.py:108`; skip the particle path
  when the class table has no particle class); OOF picks predicted twice (Gate 4c saves
  `val_picks.npz`, check `build_ensemble` never re-predicts when they exist).
- Multi-GPU (design only, unmeasured): `train.cv_parallel_folds: int = 1` — run up to N fold
  processes at once, one GPU each (`CUDA_VISIBLE_DEVICES` per subprocess, `training/cv.py:178`),
  instead of DDP inside a fold. For a model of this size this is expected to scale better than DDP
  (no all-reduce); to be measured (Section 9).
- **Pass criteria:** **G8d-C1** filament-only model: calibration does not run the particle path and
  writes the same filament calibration as before (E0); **G8d-C2** `cv_parallel_folds=2` on CPU
  (tiny, K=3) gives the same fold checkpoints' validation picks as sequential (same seeds).
- Writes: `training/run.py` (calibration call only), `predict/calibration.py`, `training/cv.py`,
  `defaultConfigs/train_config.py` (`cv_parallel_folds`).

### 8e — Predict, calibration and prepare_data
**8e.1 Exact predict speed-ups (E0/E1, first after 8b: cheap and independent of training).**
- **Accumulator device** (done first, 9 Nov 2026): `infer.accumulate_device=auto` counts the caching
  allocator's reserved-but-free memory (`memory_reserved - memory_allocated`) as free. The 8a "falls
  back to CPU, 2.3x" finding was a cProfile artefact and is withdrawn (`BASELINE_20261109.md`); the
  change is kept as a robustness fix (no speed-up measured). Tests `tests/gate8/test_g8e1_accumulator.py`.
- `peak_local_max_torch` (`predict/helpers.py:192`): move the greedy suppression loop to the CPU on
  the (small) candidate list — copy candidates and values once, run the same loop on numpy, then the
  same plateau-centroid rule — no per-candidate host sync. **E0: identical picks**, same order.
- GPU patch grid (`infer.patch_sampler: torchio | gpu = torchio`): patches sliced from the padded
  volume on the GPU (`torch.Tensor.unfold`-style indexing in batches), the same patch locations as
  TorchIO's `GridSampler` (its location rule reproduced and tested), Hann accumulation with the same
  window, flips of TTA batched. E1 (summation order of the overlap). Also allow
  `infer.predictions_batch_size` larger by default in the recipe (today 2; a human decision; batch
  size changes only cudnn kernel choice).
- Calibration: collect particle picks once at the lowest threshold (unchanged) but reuse the class
  maps for the filament tracing (one predict per validation tomogram for both kinds); filament
  threshold tracing across thresholds and classes in a process pool
  (`infer.filament_calibration_workers: int = 1`, E0, identical traces).

**8e.2 GPU array backend for prepare_data and tracing (optional extra, after 8c).**
- `tomocpt/gpu_backend.py`: `has_cupy()`, `edt(mask)`, `label(mask, structure)`, `find_objects`,
  `binary_fill_holes`, `map_coordinates(order=1)`; CuPy / cuCIM when installed and the field is
  `auto`, else numpy/scipy; results identical (integer outputs) or within `atol 1e-5` (EDT float).
  Torch-native first where it is reasonable (e.g. trilinear `map_coordinates` = `grid_sample`).
- Skeletonisation: check whether cuCIM provides a 3-D `skeletonize` (cuCIM's `skimage.morphology`
  may only have 2-D thinning) before porting; if not, keep skimage (it is per connected component
  and already cropped).
- Fields: `prepData.array_backend: str = "numpy"`, `infer.array_backend: str = "numpy"`
  (`numpy | auto`).
- Packaging: optional extra `tomocpt[gpu]` with `cupy-cuda12x` and `cucim-cu12` matched to the CUDA
  12.4 wheel of torch 2.5.1 (exact pins in Section 10, need approval); without the extra everything
  runs as today.

**Goldens.** 8e.1 `peak_local_max_torch` E0: none. `patch_sampler=gpu` opt-in: none at defaults; if
it becomes the default, `tests/golden/gate4_v1_infer.npz` maps and `gate6_c1.npz` `pred:*` arrays
→ regenerate (3.4) after the E1 check (maps `atol 1e-5`, picks identical). 8e.2: none at defaults.

**Pass criteria.**
- **G8e-C1** Defaults: every earlier test passes; `peak_local_max_torch` returns identical peaks to
  the version at the 8a commit on random maps with plateaus, ties, borders and thousands of
  candidates (golden from the 8a commit).
- **G8e-C2** `patch_sampler=gpu`: patch locations equal TorchIO's for odd shapes, even/odd overlap
  and volumes smaller than a patch; class maps within `atol 1e-5` (CPU fp32) of the TorchIO path;
  identical picks on the Gate 4 round-trip tomograms; TTA maps equal.
- **G8e-C3** Calibration: filament thresholds and scores identical with `filament_calibration_workers`
  1 and 4; particle calibration unchanged.
- **G8e-C4** `gpu_backend` with and without CuPy (skip the CuPy half when not installed): `label`,
  `find_objects`, `binary_fill_holes` identical, `edt` `atol 1e-5`; `mask_to_centerlines` on the Gate 6
  phantoms gives identical polylines.
- Speed (report-only): CZII predict per tomogram and calibration per validation tomogram before/after.

### 8f — Library upgrades (each its own step, E2)
Candidates, each needing human approval of the pin (invariant 8) and a re-run of the full suite with
regenerated goldens (3.4). Before proposing a pin, 8a measures what the upgrade brings on the step
benchmark in a throw-away environment; no upgrade is proposed for its own sake.
| Package | Now | Why look | Check first |
|---|---|---|---|
| torch | 2.5.1+cu124 | newer SDPA backends (small head dims), `torch.compile` maturity, `nn.Module.compile` | Lightning / MONAI / TorchIO compatibility; CUDA wheel vs driver |
| MONAI | 1.4.0 | SwinUNETR changes in later releases (whether an SDPA / fused-attention option exists — unverified) | checkpoint key compatibility of `SwinUNETR` (invariant 2) |
| pytorch-lightning | 2.4.0 | DDP / compile integration fixes | callbacks: SWA, EarlyStopping, our `WeightAveraging` |
| TorchIO | 0.20.2 | only if the CPU path stays the default | `GridSampler` location rule (8e.2 tests) |
| Kornia | 0.7.4 | `spatial_gradient3d` speed | bit-level Hessian values (Gate 3/7 goldens) |
Goldens expected to change with any of these: potentially all bit-exact locks —
`gate1_forward.npz`, `gate1_goldens.json`, `gate2_v1_chunks.json`, `gate3_goldens.json`,
`gate3b_c1_loss.npz`, `gate4_v1_infer.{json,npz}`, `gate6_c1.{json,npz}`, `gate7_lock.json`, the
Gate 5 golden. Each upgrade: run the old goldens first and record which fail and by how much
(`compare_goldens.py`), regenerate, rule (3.4). Acceptance run as Section 8 (one per group of upgrades).

---

## 5. Allowed write paths (Gate 8 only; per sub-step the subset it needs)
- `benchmarks/perf/**` (new), `benchmarks/out/perf_*` (outputs, not committed if large)
- `tomocpt/training/multiclass_loss.py` (8b.2), `tomocpt/networks/pickingModel.py` (8b.3 logging,
  8b.5 compile hook), `tomocpt/networks/attention.py` (new, 8b.4), `tomocpt/networks/swinunetr.py`
  (attention swap / channels-last only), `tomocpt/networks/baseModel.py` (`build_model` hook only),
  `tomocpt/training/averaging.py` (8b.3 EMA), `tomocpt/training/helpers.py` (`_setup_trainer` DDP
  options, `_prepare_data_if_needed` store), `tomocpt/training/run.py` (`Data` / `OnTheFlyData`
  arguments, calibration call), `tomocpt/training/cv.py` (8d), `tomocpt/training/snapshots.py`
  (rank-0 stall only)
- `tomocpt/dataManager/dataloading.py` (loader flags), `tomocpt/dataManager/tomostore.py`,
  `tomocpt/dataManager/otf.py`, `tomocpt/dataManager/gpu_augment.py` (new, 8c),
  `tomocpt/dataManager/chunking.py` (store writer reuse only), `tomocpt/labels/render_torch.py` (new)
- `tomocpt/predict/helpers.py` (`peak_local_max_torch`), `tomocpt/predict/multiclass.py`
  (`_predict_level`), `tomocpt/predict/calibration.py`, `tomocpt/labels/filaments.py` (backend calls
  only), `tomocpt/gpu_backend.py` (new)
- `tomocpt/defaultConfigs/{train_config,network_config,particles_config,infer_config,prepdata_config}.py`
  (own fields only), `tomocpt/defaultConfigs/otf_config.py` (new)
- `pyproject.toml` (only the approved pins / the `gpu` extra), `README.md` (performance section)
- `tests/golden/**` (regeneration per 3.4 only), `.claude/reports/gate_08*_report.json`

## 6. Public names (TA imports only these)
- `tomocpt.training.multiclass_loss.multiclass_picking_loss(..., hessian_block: int = 1)`
- `tomocpt.networks.attention.{SDPAWindowAttention, swap_window_attention(model, impl) -> int}`
- `tomocpt.dataManager.tomostore.{build_tomogram_store, read_tomogram_store, TomogramStore}`
- `tomocpt.dataManager.otf.{crop_candidates, candidate_weights, draw_crops, OnTheFlyDataset, OnTheFlyData}`
- `tomocpt.dataManager.gpu_augment.{GpuAugment, random_affine, elastic_field}`
- `tomocpt.labels.render_torch.{render_targets_torch, render_tubes_torch}`
- `tomocpt.predict.helpers.peak_local_max_torch` (signature unchanged),
  `tomocpt.predict.multiclass.grid_locations(shape, patch, overlap) -> LongTensor [N, 6]`
- `tomocpt.gpu_backend.{has_cupy, edt, label, find_objects, binary_fill_holes, map_coordinates}`
- `tomocpt.defaultConfigs.otf_config.OnTheFlyConfig`
- `benchmarks.perf.step.run_step_benchmark(config, data, classes, batch, steps, warmup, seed, **options) -> dict`
  (`config`: a `MainConfig`, a YAML path or `None`; options: `overrides`, `device`, `logger`),
  `benchmarks.perf.compare_goldens.compare(old_path, new_path) -> dict`

## 7. Test plan notes for TA
- `tests/gate8/` with `__init__.py` and `conftest.py` as earlier gates; references for E0/E1 come
  from a worktree of the commit **before** each change (generator scripts in `tests/golden/
  make_goldens_gate8*.py`), never from the new code.
- CPU tests fp32 with tiny models; GPU checks `slow`; the overlay runs and speed numbers are
  benchmarks (human-reviewed JSON), not pytest.
- Each sub-step's criteria above map 1:1 to tests named `test_G8x_Cn_*`.

## 8. Acceptance for humans
- **8b, 8e.1** (E0/E1): no full training run. Report: micro-benchmark ratios, overlay plots,
  goldens changed (if any) with rulings.
- **8c** (E2): **one** CZII 5-fold CV for 8c.1 + 8c.2 together (human decision 9 Nov 2026; built-in CV
  mode, RAdam + EMA recipe, same seeds as the current baseline, run from the perf worktree), against
  the current grid-chunk baseline. Accept as default only if Kaggle-weighted OOF F4 (exact
  coordinates) is within noise of the baseline: |ΔF4| ≤ 0.01 (Gate 7 arms differed by ±0.01 on a
  single seed) and no class with weight > 0 drops by more than 0.03 (bar approved by the human,
9 Nov 2026); plus wall time per fold. A
  filament run (DS-10490 3c.2 recipe) with actin F1 within 0.02 of 0.810 before `on_the_fly` becomes
  the filament default.
- **8f**: the same CZII CV once per group of upgrades.
- Default flips: **E0 changes may become defaults inside Gate 8** (human decision 9 Nov 2026: Swin
  checkpointing off, `loss_checkpoint` off, pinned memory, prefetch), each recorded in
  `gate_status.json` with its goldens handled as in 3.4 — on `v2-feat/perf` only (3.1). E1/E2 flips
  (`hessian_block`, `attention`, `compile`, `patch_sampler`, `sampling`) stay separate human
  decisions after the evidence.

## 9. Multi-GPU: design checks here, measurements later (all unmeasured)
Checked here: 3.6 gloo tests per sub-step; seeds per rank; no unused parameters; no new rank-0
stall. To run on a multi-GPU box (record in `benchmarks/out/perf_multigpu_<date>/`):
1. Steps/s and scaling efficiency at 1, 2, 4 (8) GPUs, batch 16 per GPU, recipe with the kept 8b
   options; with and without `static_graph`, `gradient_as_bucket_view`, bucket size 25 / 100 MB.
2. Same with `sampling=on_the_fly`, `augment_device=cuda` (each rank holds its tomograms; memory per rank).
3. Data wait per rank; time other ranks spend waiting for rank 0 in `snapshots` / panels / calibration.
4. `cv_parallel_folds = n_gpus` vs DDP folds run one after another (wall time per 5-fold CV).
5. `tomocpt predict` on N test tomograms with 1 vs N Dask GPU workers; calibration sharded over GPUs.
6. Compile + DDP: compile time per rank, recompiles.

## 10. Proposed dependency changes (each needs human approval; invariant 8)
- `tomocpt[gpu]` extra (8e.2): `cupy-cuda12x` (13.x) and `cucim-cu12` (a release built for CUDA 12) —
  exact versions to be chosen in 8e.2 after an install test in a clean env against torch 2.5.1+cu124.
- Upgrades of 8f: none proposed yet; 8a measures candidates first and proposes exact pins with the
  numbers.

## 11. Proposed CLAUDE.md additions (for the orchestrator, after human approval)
Section 6 rows:
```
| `train.particles.hessian_block` | int = `1` (classes per Hessian call; 0 = all) | 8 |
| `train.pin_memory`, `train.prefetch_factor`, `train.panel_every_n_steps` | bool = `False`, Optional[int] = `None`, int = `1` | 8 |
| `train.network.attention`, `train.network.compile`, `train.network.channels_last` | `eager`, `none`, `False` | 8 |
| `train.sampling` | `chunks` \| `on_the_fly` = `chunks` | 8 |
| `train.otf` | `OnTheFlyConfig` (`defaultConfigs/otf_config.py`) | 8 |
| `train.cv_parallel_folds` | int = `1` | 8 |
| `infer.patch_sampler`, `infer.filament_calibration_workers` | `torchio`, int = `1` | 8 |
| `prepData.array_backend`, `infer.array_backend` | `numpy` \| `auto` = `numpy` | 8 |
```
Section 7 row:
```
| 8 | `08_AGENT_PERFORMANCE.md` | Performance: profile harness, exact training-step speed-ups, on-the-fly crops + GPU augmentation (opt-in), predict / calibration / prepare_data speed, library upgrades; DDP-ready (human decisions 2 and 9 Nov 2026) | see the spec's "Allowed write paths" (per sub-step) |
```
Section 1.6 note (after 8b, if the default is flipped): `use_checkpoint` default and the measured
cost. Invariant 8 list updated with any approved pin.

## 12. Human answers (9 Nov 2026)
1. **Order:** 8a → 8b → 8e.1 → 8c → 8e.2 → 8f, 8d alongside 8a — approved.
2. **Default flips of E0 changes inside Gate 8:** allowed (Section 8), on `v2-feat/perf` only.
3. **8c acceptance:** one CZII CV for 8c.1 + 8c.2 together; bar |ΔF4| ≤ 0.01 and per-class ≤ 0.03 approved.
4. **On-the-fly validation:** a fixed seeded random set (8c.1, G8c-C5).
5. **Precision:** volumes stored as float16; labels / targets float32 (the Gaussian profiles).
6. **CuPy / cuCIM:** optional `tomocpt[gpu]` extra approved in principle; exact pins still need approval (Section 10).
7. **Speed targets:** report-only; no pass/fail bar on speed.
8. **TODO 5.1:** memory / step-time part folded into 8a; the accuracy part stays in TODO 5.
9. **INVIOLABLE:** all changes on a separate branch (`v2-feat/perf`); no commits to `v2-dev` unless
   everything works to spec within the margins (Section 3.1).

### 8g - learning rate and EMA decay follow the effective batch size (human request 10 Nov 2026)
`train.optimizer.lr_scaling: none | sqrt | linear = none` (opt-in while tested; meant to become
`sqrt` by default in production, a human decision after the GPU check), and
`train.optimizer.lr_reference_batch: int = 16` (the batch the recipe's lr was tuned at). Not CLI flags.
- `k = batch_size x devices / lr_reference_batch` (effective batch; DDP counts every GPU).
- `sqrt`: `lr * sqrt(k)` (Krizhevsky 2014; Malladi et al. 2022, the rule for Adam-family optimisers
  such as RAdam). `linear`: `lr * k` (Goyal et al. 2017; derived for SGD, with warm-up).
- EMA (`weight_averaging=ema`): `ema_decay ** k` (Busbridge et al. 2023), the same averaging window
  in samples. SWA's `swa_lrs = 0.1 * lr` follows the scaled lr.
- Epochs unchanged (Goyal keeps epochs; matching step counts would cost k times the compute).
  Early stopping and patience are in epochs and need no change.
- `k = 1` reproduces the unscaled run bit for bit (invariant 1 at the default batch 16).
- Not ported from the illustrative script: batch size from a target lr (no use case), epoch
  scaling. **G8g-C1..C4** in `tests/gate8/test_g8g_lr_scaling.py`. GPU check (later, the GPU is
  busy with the 8c CVs): batch 16 vs 32 and 64 with `sqrt`, loss against samples seen.
