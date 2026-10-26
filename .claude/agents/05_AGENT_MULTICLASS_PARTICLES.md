# 05_AGENT_MULTICLASS_PARTICLES.md — Gate 5: several particle classes in one model

## Mission
Pick K particle species in one model and one `predict` run. The model outputs **one
centroid heatmap channel per class** (each a normal 0–1 Gaussian target). All species of a
tomogram are labelled together, so a particle of one species is never a negative for
another. Keep everything identical when only one class is used.

## Why one channel per class (and not one channel with class-scaled peaks)
Coding the class in the peak height (class 1 peaks at 1, class 2 at 2, …) was considered
and rejected:
- peak height already means confidence; a low, unsure class-2 peak looks like a confident
  class-1 peak, so thresholds stop working;
- a regression network that cannot choose between class 1 and class 3 outputs about 2,
  which is a confident wrong class;
- it puts the species in an order that does not exist;
- where Gaussian tails of one class fall to the peak height of another, false peaks appear;
- the loss grows with the class number, so high classes dominate.
One sigmoid channel per class keeps the current target, loss and peak finding per class,
and partial labels are a per-class mask.

## Dependencies
- Gate 2: pipeline v2, points stored per chunk, targets drawn in the dataloader. All
  species of a tomogram must share one working voxel size (the band rule with several
  lengths). Multi-class needs pipeline v2; in v1 each species is rescaled differently.
- Gate 3: size-conditioned model (the optional generic channel uses it).
- Gate 4: inference planning and the benchmark harness.

## Allowed write paths
- `tomocpt/labels/**` (multi-class `prepare_data`)
- `tomocpt/dataManager/**` (class column in points, multi-channel target drawing,
  `class_available`)
- `tomocpt/networks/{swinunetr,pickingModel,inference,distillationModel}.py`
- `tomocpt/training/**`
- `tomocpt/predict/**` (per-class peaks, cross-class suppression, output files,
  accumulator precision)
- `tomocpt/defaultConfigs/{prepdata_config,particles_config,infer_config}.py`
- `benchmarks/multiclass_czii.py` (new), `README.md`
- `.claude/reports/gate_05_report.json`

## Design

### Data
- `prepData.multiclass = True` turns on one dataset that holds all species:
  - `coordinate_files`: STAR files with `rlnClassNumber` (for example the
    `coords/all_objects.star` written by `scripts/download_portal_data.py`).
  - `class_diameters = "1:120,2:130,3:180,4:300,5:260,6:270"` (class number : diameter Å).
  - `class_names = "1:apo-ferritin,2:beta-amylase,…"` (optional; used in outputs).
  - `class_annotated`: per dataset, which classes are annotated **exhaustively**
    (default: every class that appears in the STAR file). A class that is not annotated
    gets `class_available = False` for all chunks of that dataset.
- Class indices: the sorted class numbers map to channels `0..K-1`. Store the mapping in
  the checkpoint config and in `chunk_format.json`.
- The coordinates CSV (Gate 2) already has `class_index` and `diameter_ang` per point.
- Working voxel size: the band rule with all class diameters of the dataset. For CZII at
  10.012 Å/px (diameters about 120–300 Å, so 12–30 voxels) no resampling is needed.

### Batch
| Key | Shape | dtype |
|---|---|---|
| `target_data` | `[B, K, 64, 64, 64]` | float32, channel k = class k Gaussians |
| `class_available` | `[B, K]` | bool |
| `class_d_vox` | `[B, K]` | float32, class diameter in working voxels |
| `input_data`, `apix_work`, `dataset_id` | as Gate 2 | |

With K = 1 and no `multiclass`, the batch is exactly the Gate 2 batch (plus a constant
`class_available = True`).

### Model
- `NetworkConfig.build_model` sets `out_channels = K` (+1 with the generic channel).
  Nothing else in the network changes.
- **Warm start** from a single-class checkpoint (`load_single_class_into_multiclass`):
  copy the old `out` weights into every class channel, so each class starts as the old
  picker. Report this in the log.
- **Optional generic channel** (`train.particles.generic_channel = True`, default False):
  channel K is size-conditioned (Gate 3). Each sample picks one annotated class at random,
  uses its diameter as `d_vox`, and its channel as the target. This keeps
  "pick any particle of diameter L" for species without a class.

### Loss
For each class `k` where `class_available[b, k]`:
- the Gate 2/3 v2 loss on channel `k`, with inverse-frequency weights computed per sample
  **and per class**, and the Hessian scaled by `(class_d_vox[b, k] / 20)²`;
- unavailable `(b, k)` pairs contribute 0;
- the total is the mean over available `(b, k)` pairs (not divided by K), so rare classes
  are not diluted by unavailable ones;
- `train.particles.class_weights` (optional) multiplies each class.
- **Memory:** compute the Hessian term one class at a time inside
  `torch.utils.checkpoint` (`train.particles.loss_checkpoint = True`, default). The
  vectorised form stores about 6K full-size tensors per side. Optional
  `train.particles.hessian_box = True` computes the Hessian term only inside a box of
  1.5 × radius around each particle (outside, both ŷ and y are near 0).
- DDP rule: if a class is unavailable in the whole batch, add `0 · logits[:, k].sum()`.

### Inference
- One accumulator per class. `infer.accumulate_dtype = "float16"` and
  `infer.accumulate_device = "cpu"` are available for large tomograms (CZII 630×630×184:
  292 MB per class in float32). Defaults keep the current behaviour (float32).
- Peaks per class with that class's radius and distance threshold.
- **Cross-class suppression** (`infer.cross_class_nms = True`): when peaks of different
  classes are closer than `0.5 × min(r_a, r_b)`, keep the higher score. Log how many peaks
  were removed per class pair. This is the confusion signal.
- Outputs:
  - RELION 3.1 / 5.0: one STAR file with `rlnClassNumber` (the original class numbers) and
    `rlnAutopickFigureOfMerit` (the peak score);
  - Warp and IMOD: one file per class, `<name>_<class_name>.<ext>`;
  - with `save_prediction_confidence_map`, one map per class, `<stem>_<class_name>.mrc`.
- `infer.classes` (optional) restricts the output to some classes.

## Public names
`tomocpt.labels.multiclass.{parse_class_diameters, class_index_map}`,
`tomocpt.labels.helpers.render_targets` (from Gate 2, with `n_classes = K`),
`tomocpt.networks.swinunetr.load_single_class_into_multiclass`,
`tomocpt.training.multiclass_loss.multiclass_picking_loss`,
`tomocpt.predict.helpers.cross_class_suppression`,
`tomocpt.defaultConfigs.particles_config.ParticlesConfig`.

## Pass criteria
- **G5-C1** K = 1 lock: with `multiclass = False`, batches, loss and predictions equal
  Gate 4 (atol 1e-6).
- **G5-C2** `render_targets(..., n_classes=K)`: each particle appears only in its class channel,
  with that class's radius (± 1 voxel); overlapping particles of different classes keep
  both peaks in their own channels.
- **G5-C3** Partial labels: with `class_available[:, k] = False`, the gradient of output
  channel k is exactly 0 (not `None`), and changing that channel's target changes nothing.
- **G5-C4** The per-class checkpointed loss equals the vectorised loss (rtol 1e-5) on a
  K = 4 batch. (Slow, GPU, report only: peak memory of both forms.)
- **G5-C5** Per-class weights: a class with 10× fewer positive voxels gets 10× larger
  positive weights than a common class in the same sample (rtol 1e-3).
- **G5-C6** Warm start: after `load_single_class_into_multiclass`, every class channel's
  logits equal the single-class logits (atol 1e-6).
- **G5-C7** `cross_class_suppression`: two peaks of different classes 1 voxel apart → the
  higher one stays; peaks 3 radii apart → both stay; same-class peaks are not touched.
- **G5-C8** STAR round trip: the RELION output has `rlnClassNumber` with the original
  numbers and reads back with `tomocpt.labels.star.Star`.
- **G5-C9** `float16` CPU accumulation gives maps within 1e-3 of `float32` and the same
  peaks.
- **G5-C10** Config: `class_diameters` parsing (spaces, missing class, duplicate class →
  error that names the problem).
- **G5-C11** (slow) Synthetic: 4 classes, two of them the same size but different shape
  (solid vs hollow spheres). Held-out F1 ≥ 0.8 for every class, and fewer than 15 % of
  true particles of one same-size class are picked as the other.
- **G5-C12** (slow) Generic channel on: picking a held-out fifth size by `d_vox` alone
  reaches F1 ≥ 0.7.

## Benchmark (human review)
`benchmarks/multiclass_czii.py`: CZII 10440, split by tomogram. Arms: (a) one v1 model per
species (today's practice); (b) one multi-class model; (c) (b) warm-started from the public
TomoCPT model. Report per species F-beta (β = 4), precision, recall, the cross-class
suppression counts (confusion), wall time and memory. Expected: the largest gains for the
same-size pairs (apo-ferritin / β-amylase; thyroglobulin / virus-like particle / ribosome).
