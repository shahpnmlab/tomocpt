# 07_AGENT_DISCRIMINATIVE.md — Gate 7: discriminative training (contrastive + focal)

Added 30 Oct 2026 by human decision. **Implement after Gate 6 (filaments)**: look-alike
confusion also hits filaments (thin filaments such as actin and vimentin against membranes),
so this gate covers particle and filament classes from the start.

## Why
CZII 5-fold CV (`experiments/czii_v2_results.md`, 30 Oct 2026): centring is accurate (median
error 12–25 Å, F4 nearly unchanged from 0.5 R to 1.0 R match radius), but recall is low for the
small classes (0.54–0.56) and the ribosome (0.63), and about 95 % of false picks are confident
picks on real, unlabelled densities. In real data many species share a size, so size and
contrast cannot separate classes; the network must learn identity. The current loss is a
voxel-wise weighted regression: nothing singles out a missed instance or a confident false pick.
Target regime: data-poor, non-denoised tomograms.

## Design (both options off by default; defaults reproduce Gate 6 exactly)
### (a) Supervised dense contrastive loss (`train.discriminative.contrastive_weight > 0`)
- **Features.** A 1×1×1 projection head (reuse `tomocpt.networks.ssl.heads.DenseProjectionHead`,
  `embed_dim`, L2-normalised) on the `decoder2` output (32³, FiLM-conditioned when scale
  conditioning is on). `ScaleConditionedSwinUNETR` / `MySwinUNETR` gain
  `forward_features(x, apix=None) -> (logits, hidden4, decoder2)`; `forward` is unchanged
  (invariant 7). An SSL v2 checkpoint's dense head may initialise this head (T7.5).
- **Anchors from the augmented dense target** (no dataloader change, coordinates stay consistent
  with augmentation): particle anchors are voxels where target channel `c` is a local maximum
  ≥ `anchor_min_target` (centres); filament anchors are ridge voxels of a filament channel
  ≥ `anchor_min_target`, subsampled along the ridge at ≥ one diameter spacing. Only available
  `(b, c)` pairs give anchors; at most `anchors_per_class` per class per batch.
- **Hard negatives.** Voxels where every target channel is 0 after dilating the summed target by
  `exclusion_radius` × the class radius, ranked by the model's own highest class probability
  (detached); the top `hard_negatives` per batch with probability in
  `[hard_negative_min_score, hard_negative_max_score]`. The upper cap excludes near-certain picks,
  which are often unlabelled real particles. Optional explicit negatives from
  `prepData.negative_masks` (e.g. membrane segmentations from the CryoET Data Portal), batch key
  `negative_mask [B, 1, D, H, W]`, sampled up to `mask_negatives` per batch.
- **Memory bank.** Per class a FIFO queue of `bank_size` detached embeddings (and one for
  negatives), a registered buffer, not a parameter; updated after the loss; under DDP each rank
  keeps its own queue (no gather needed for correctness).
- **Loss.** SupCon (Khosla et al. 2020) over batch anchors against batch + bank: positives are
  same-class embeddings (particle or filament class), negatives are other classes and all
  negative embeddings; temperature `temperature`. Background embeddings are never anchors.
  Total loss = picking loss + `contrastive_weight` × SupCon. Finite under bf16; a batch without
  anchors contributes `0 × head(...).sum()` (DDP: the head always gets a gradient).

### (b) Focal value term (`train.discriminative.value_loss = "focal"`)
- Replaces the weighted Huber value term of Section 5.4 with the CenterNet penalty-reduced focal
  loss (Zhou et al. 2019) on the per-class probabilities, `alpha = focal_alpha` (2),
  `beta = focal_beta` (4), normalised by the number of instances (particle centres; filament
  ridge length in units of one diameter) per available `(b, c)`. The Hessian term is unchanged.
  Works with both class heads (sigmoid, softmax).

## Config — `defaultConfigs/discriminative_config.py`, `TrainConfig.discriminative`
```python
@dataclass
class DiscriminativeConfig:
    contrastive_weight: float = 0.0        # 0 = off, no head created
    embed_dim: int = 64
    temperature: float = 0.1
    anchors_per_class: int = 32
    anchor_min_target: float = 0.99
    bank_size: int = 1024
    hard_negatives: int = 64
    hard_negative_min_score: float = 0.3
    hard_negative_max_score: float = 0.9
    exclusion_radius: float = 1.0          # x class radius around labelled objects
    mask_negatives: int = 64
    init_from_ssl: bool = True             # copy an SSL v2 dense head when fine-tuning from one
    value_loss: str = "huber"              # "huber" | "focal"
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
```
`prepData.negative_masks: Optional[str] = None` (`"<name>:<mask dir>"`, masks of structures that
are never a class, matched to tomograms by stem as `prepData.filament_masks`).

## Allowed write paths
- `tomocpt/training/discriminative.py` (new: anchors, hard negatives, memory bank, SupCon, focal)
- `tomocpt/networks/pickingModel.py` (head creation, loss hook)
- `tomocpt/networks/swinunetr.py` (`forward_features` only)
- `tomocpt/training/multiclass_loss.py` (selecting the value term only)
- `tomocpt/training/helpers.py` (`_create_model`: SSL dense-head transfer only)
- `tomocpt/defaultConfigs/discriminative_config.py` (new), `train_config.py` (the field only),
  `prepdata_config.py` (`negative_masks` only)
- `tomocpt/dataManager/**`, `tomocpt/labels/**` (negative masks into chunks and the batch only)
- `benchmarks/discriminative.py`, `experiments/run_benchmarks.sh` (a `discriminative` step)
- `README.md`, `.claude/reports/gate_07_report.json`

## Public names
`tomocpt.training.discriminative.{sample_anchors, sample_hard_negatives, ContrastiveMemoryBank,
supcon_loss, focal_heatmap_loss}`, `tomocpt.defaultConfigs.discriminative_config.DiscriminativeConfig`.

## Pass criteria
- **G7-C1** Defaults: every earlier test passes unchanged; no head module exists.
- **G7-C2** `sample_anchors`: on rendered targets (spheres and tubes), particle anchors are the
  centres (± 0.5 voxel, before and after a flip augmentation), filament anchors lie on the
  centreline (≤ 0.5 voxel) at ≥ one diameter spacing; unavailable `(b, c)` give none; the per-class
  cap holds.
- **G7-C3** `sample_hard_negatives`: never within `exclusion_radius` × radius of a labelled object;
  scores inside `[min, max]`; ranked by score; mask negatives come only from the mask.
- **G7-C4** `ContrastiveMemoryBank`: FIFO per class, detached (no grad), stored in `state_dict`,
  size capped, not a parameter.
- **G7-C5** `supcon_loss`: lower for class-separated embeddings than for shuffled labels; equals a
  float64 reference on a toy case (rtol 1e-5); finite under bf16; batch without anchors gives 0
  with a gradient path.
- **G7-C6** `focal_heatmap_loss` equals the CenterNet formula on a toy case (rtol 1e-6); with
  `value_loss = "huber"` the loss equals Gate 6 exactly.
- **G7-C7** One training step with both options on (sigmoid and softmax, particle and filament
  classes): every trainable parameter gets a gradient; with both off, outputs and loss are
  bit-identical to Gate 6.
- **G7-C8** SSL v2 transfer: with `init_from_ssl`, the picker's head equals the SSL dense head.
- **G7-C9** (slow) Synthetic look-alikes at low SNR, tiny model: two particle species of the same
  size and similar contrast (solid vs hollow), a thin tube class (70 Å) and unlabelled sheets
  (membrane-like planes, with negative masks). With contrastive on, versus off (same seeds 0, 1):
  mean F1 of the look-alike pair improves by ≥ 0.05 and tube false picks on sheets fall by ≥ 30 %.

## Acceptance for humans (not a CI gate)
CZII 5-fold CV (`experiments/run_cv.sh`) for (a), (b) and (a)+(b) against the softmax baseline
(Kaggle-weighted OOF F4 0.631), plus a filament dataset with membranes (Gate 6 benchmark data).
Recommend an option on by default only if it gains ≥ 0.02 OOF F4 on CZII and does not lower
filament centreline F1.
