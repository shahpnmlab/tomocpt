# 07_AGENT_MULTI_STRUCTURE.md — Gate 7: one pass for particles, filaments, membranes, organelles and void

## Mission
Add segmentation heads so that one network, in one inference run, gives:
- centroid heatmaps for globular particles (the Gate 3 size-conditioned head), and
- probability maps for **membrane**, **filament**, **organelle** (lumen) and **void**
  (vacuum, ice outside the cell, outside the lamella).

Train it from **partial labels**: most tomograms are annotated for only some classes.
Keep every earlier behaviour unchanged when no segmentation class is enabled.
This gate is a dense (non-MoE) model. Gate 8 tests MoE on top of it.

## Why labels, and why this order
- Router maps do not become membrane or filament maps without supervision. Each class
  needs labelled examples. The portal has segmentation masks for many datasets, and
  MemBrain-seg can give membrane pseudo-labels.
- MoE helps when different tasks compete for capacity. That competition exists only once
  these heads exist, so MoE is tested after this gate (Gate 8).

## Scale: two resolutions, one network
At 10 Å/px a 64³ chunk covers 64 nm. A membrane bilayer (about 4 nm thick) and an actin
filament (about 7 nm) fit well. A mitochondrion (hundreds of nm) or a void region does
not. So the classes have a working voxel size each:
- **fine** (default 10 Å/px, clipped to the band rule for particles): particles,
  membranes, filaments;
- **context** (default 40 Å/px, `train.tasks.context_apix`): organelles, void.
The same weights serve both. The network gets `apix_work` as a second conditioning input
(`ConditionEmbedding(d_vox, apix_work)`), which is meaningful here because pipeline v2
lets `apix_work` vary. Inference runs the tomogram at both voxel sizes and merges the
maps at the native size. "One pass" means one model and one command.

## Allowed write paths
- `tomocpt/networks/heads.py` (new), `tomocpt/networks/conditioning.py`
- `tomocpt/networks/swinunetr.py` (new class `MultiStructureSwinUNETR` only)
- `tomocpt/networks/pickingModel.py` (multi-structure training step)
- `tomocpt/training/seg_losses.py` (new)
- `tomocpt/labels/segmentation.py` (new), `tomocpt/labels/run.py` (segmentation inputs)
- `tomocpt/dataManager/**` (segmentation targets in chunks and batches)
- `tomocpt/predict/**` (two-resolution inference, structure maps)
- `tomocpt/defaultConfigs/{tasks_config,prepdata_config,infer_config}.py`
- `scripts/download_portal_data.py` (segmentation masks)
- `benchmarks/multi_structure.py` (new)
- `.claude/reports/gate_07_report.json`

## Design

### Classes and output
`STRUCTURE_CLASSES = ("membrane", "filament", "organelle", "void")`, fixed order.
`train.tasks.classes` selects a subset (default `()`, which means picking only).

`MultiStructureSwinUNETR(SizeConditionedSwinUNETR)`:
- `forward(x, d_vox=None, apix=None) -> (logits, hidden4)`, `logits` has shape
  `[B, P + K, D, H, W]`. Channels `0..P-1` are the particle heatmaps from Gate 5
  (`P = 1` for a single-class picker, which is exactly the Gate 3 output).
  Channels `P..P+K-1` follow `train.tasks.classes`.
- Segmentation heads: one `UnetOutBlock(fs → K)` on `decoder1` output, plus an optional
  shared `1×1×1` conv. Independent sigmoid per class (multi-label: a voxel can be both
  "membrane" and "organelle boundary").
- Conditioning: `ConditionEmbedding(d_vox, apix)` = Gate 3 size embedding (or a learned
  "no particle" vector when `d_vox` is missing) + `apix` embedding
  (`t = log(apix/10)`, same Fourier features, zero-initialised output layer).
  With `apix=None`, the `apix` branch contributes exactly 0.
- `predict_step` returns `sigmoid(logits)` for all channels.

### Partial labels
Each training sample carries `seg_target [K, D, H, W]` (float, 0..1) and
`seg_available [K]` (bool). `seg_available[k] = True` means class `k` is annotated
**exhaustively** in that region. For unavailable classes the loss is 0 and no gradient
flows into that channel from that sample.

Loss per available class: `BCE + soft Dice`; filaments add **clDice** (centre-line Dice);
membranes use a 1-voxel distance-tolerant Dice (labels from different tools disagree by
about one voxel). Per-class weights in `train.tasks.class_weights`.
Total: `L = L_pick (Gate 3, only where a particle target exists) + Σ_k w_k L_k`.
If a batch has no available sample for class `k`, add `0 · logits[:, P+k].sum()` so the
head still gets a (zero) gradient (DDP rule).

### Labels
- `prepData` gets an optional list of segmentation inputs per dataset
  (`segmentation_dirs`, `segmentation_classes`): MRC masks matched to tomograms by name
  (the same matching as `labels.helpers.match_data_to_tomograms`).
- `labels/segmentation.py`:
  - `resample_mask(mask, apix_from, apix_to, shape_to) -> float mask` (linear, then
    clipped to [0, 1]).
  - `render_filament_traces(points_xyz, diameter_ang, apix, shape) -> mask` for filaments
    annotated as points along their axis (tubes between consecutive points).
  - `availability_from_manifest(...)`: a class is available for a tomogram only if the
    user or the portal manifest marks it exhaustive. Default for portal masks: available.
- Chunking stores `seg_target` per class at the class's working voxel size and the
  availability vector in `chunk_meta.json`.
- `scripts/download_portal_data.py --masks membrane,microtubule,...` also downloads
  `SegmentationMask` annotations (MRC) and maps portal object names to the four classes
  with `--class-map name=class,...`.

### Inference
`infer.export_structures: bool = False`. When on, `predict` writes
`<stem>_<class>.mrc` (float16, native shape, native voxel size) for each class, and keeps
the coordinate output for particles as in Gate 4. Fine classes run at the fine voxel
size; context classes at `context_apix`; both are resized to native size.

## Public names
`tomocpt.networks.swinunetr.MultiStructureSwinUNETR`,
`tomocpt.networks.conditioning.ConditionEmbedding`,
`tomocpt.networks.heads.STRUCTURE_CLASSES`,
`tomocpt.training.seg_losses.{partial_label_loss, soft_dice, cl_dice}`,
`tomocpt.labels.segmentation.{resample_mask, render_filament_traces}`,
`tomocpt.defaultConfigs.tasks_config.TasksConfig`.

## Pass criteria
- **G7-C1** With `train.tasks.classes = ()`, the model, loss and outputs equal Gate 5 (and
  Gate 3 for one particle class)
  (state-dict keys and logits, atol 1e-6).
- **G7-C2** The particle channels of `MultiStructureSwinUNETR` loaded from a Gate 3 or Gate 5 checkpoint equal
  the Gate 3 logits (atol 1e-6), for `apix=None` and any `apix` (zero-init).
- **G7-C3** Partial labels: with `seg_available[k] = False` for every sample, the gradient
  of head `k` is exactly 0 and not `None`; changing that class's target changes nothing.
- **G7-C4** `cl_dice` is 1 for identical tubes and drops below 0.5 when a tube is broken
  in three places, while plain Dice stays above 0.9 (shows why clDice is used).
- **G7-C5** `resample_mask` round trip keeps a sheet's position within 0.5 native voxel;
  `render_filament_traces` gives a tube of the right diameter (± 1 voxel).
- **G7-C6** Two-resolution inference: on a phantom with a context-scale blob, the merged
  map at native size aligns with the truth (centroid error < 1 native voxel).
- **G7-C7** Portal masks: on a fake portal client (as in the script's tests), `--masks`
  writes masks with the tomogram's shape and voxel size and records availability in the
  manifest.
- **G7-C8** DDP rule: one step with a batch that has only particle labels gives every
  parameter a gradient (not `None`).
- **G7-C9** (slow) Synthetic multi-structure phantom (spheres of two sizes, bilayer
  sheets, tubes, large organelle-like vesicles, a void slab; wedge ±60°; noise), with
  partial labels (each tomogram labels two random classes). Thresholds on held-out data:
  membrane Dice ≥ 0.6, filament clDice ≥ 0.6, organelle Dice ≥ 0.7, void Dice ≥ 0.8.
- **G7-C10** (slow) Interference check: picking F1 of the multi-structure model is at most
  0.03 below the Gate 5 (or Gate 3) picking-only model on the same particles, for every particle class.

## Benchmark (human review)
`benchmarks/multi_structure.py` runs on portal datasets that have membrane and filament
masks and particle picks. Report per-class Dice / clDice, picking F-beta, time and
memory, and the same numbers for single-task models trained on each class alone.
