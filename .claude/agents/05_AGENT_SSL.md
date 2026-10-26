# 05_AGENT_SSL.md — Gate 5: wedge-aware, scale-aware self-supervised pretraining (SSL v2)

## Mission
Pretrain the Gate 3 network without labels so that the multi-class picker starts from
features that already separate structures of different size and shape, are robust to
wedge, CTF and dose differences between datasets, and carry dense (per-voxel) embeddings.
Replace the SSL v1 pretext tasks that the missing wedge breaks. Keep SSL v1 identical and
SSL → picking transfer working.

## Background: what is wrong with SSL v1
1. **The 4-way rotation task is ill-posed.** `rot_rand` rotates by k·90° about Z (plane
   (Y, X)). With the tilt axis along Y, the missing wedge is a double wedge in the `kz–kx`
   plane, constant along `ky`. 90° and 270° move it to the `kz–ky` plane; 0° and 180°
   leave it in place. Cellular content has no preferred in-plane direction, so the only
   reliable cue separates {0°, 180°} from {90°, 270°}: best cross-entropy ≈ ln 2 ≈ 0.69,
   accuracy ≈ 50 %. The head learns wedge orientation, and half the training views have a
   wedge orientation that never occurs at inference.
2. **Conflicting goals on one feature.** Rotation needs rotation-sensitive features; the
   contrastive loss pairs views with independent rotations and asks the same pooled 2³
   feature to be rotation-invariant.
3. **Inpainting is too easy and mostly copies.** Masked blocks hold noise in [0, 1] (mean
   0.5) while the data have median 0 and range ±3, so masks are obvious; the L1 covers the
   whole volume, so most of it rewards copying visible voxels.
4. **SSL chunks use the fixed-particle rescale**, so SSL needs a particle size and sees
   one scale only.
5. **Only a pooled 2³ embedding exists.** A 64³ chunk holds many particles; one vector per
   chunk cannot describe them.
6. The embedding L1 term uses `sum`, so its weight grows with batch size and width.

## SSL v2 design (`train.ssl.version = 2`)
- **Network.** The Gate 3 `ScaleConditionedSwinUNETR` with `out_channels = 1`
  (reconstruction). The decoder receives `apix_view`, so the FiLM weights pretrain too.
  The encoder is unconditioned.
- **Data.** Resample each tomogram once to
  `apix_ref = clip(apix_native, apix_ref_min, apix_ref_max)` (no particle size needed) with
  `dataManager.scaling.resample_volume(..., fourier)`. Cut source chunks of 128³, stride 64.
- **Views.** Per sample, a centre and two scale factors `s_a, s_b`, log-uniform in
  `[1, 2^scale_range_log2]`. A view takes a cube of `64·s` source voxels around the centre
  and Fourier-downsamples it to 64³, so `apix_view = apix_ref · s`. All views pass the same
  final low-pass (0.9 of Nyquist), so `s = 1` is not special. Isotropic scaling keeps the
  wedge angles.
- **Noise equalisation.** Estimate each view's noise std from the Fourier shell 0.8–0.9
  Nyquist; add white noise to the quieter view until both match; then robust-normalise.
  This removes the "noise level tells the scale" shortcut.
- **Wedge-preserving spatial augmentation** (per view): random flips of each axis, applied
  to the view and its reconstruction target. No other rotations.
- **Degradation `D`** (per view, independent parameters; input only). `RandomFourierDegradation`
  with missing wedge (`keep_angle` in `[30°, 60°]`, oriented for `(z, y, x)`) and amplitude
  randomisation (`degradation_p`, default 0.8); then intensity `a·x + b`,
  `a ∈ [0.8, 1.2]`, `b ∈ [-0.1, 0.1]`, Gaussian noise `σ ∈ [0, 0.1]`. The reconstruction
  target is the same view **without** `D` (restore a clean view from a degraded one); the
  contrastive views get different `D`, so embeddings ignore CTF, dose and wedge
  differences; random amplitude spectra hide the radial power-spectrum shape that would
  otherwise reveal the scale.
- **Polarity.** Contrast inversion with `ssl.inversion_p` (default 0.5), drawn per view and
  applied to the view **and** its target together.
- **Order per view:** crop and Fourier-resample → noise equalisation (both views) → flips
  (view and target) → polarity (view and target) → `D` (view only) → masking (view only)
  → robust normalisation of the input (the target uses the same affine normalisation).
- **Tasks.**
  1. *Masked reconstruction*: blocks of 8–16 voxels, mask ratio `U(0.15, 0.4)`, fill value 0.
     Loss `L1(masked) + recon_visible_weight · L1(visible)` (v2 default 0.1).
  2. *Global contrastive*: NT-Xent between views A and B (temperature 0.1) through a 2-layer
     projection head on pooled `hidden[4]`; L1 regulariser with `mean`.
  3. *Dense contrastive*: a 1×1×1 projection head (`dense_dim`, default 64) on the
     `decoder2` output (32³). Sample `n_anchors` (default 256) source-space points inside the
     region both views cover; map each to view coordinates (scale, centre offset and flips);
     read embeddings with `grid_sample`; InfoNCE where the positive is the same anchor in the
     other view and the negatives are all other anchors in the batch (temperature 0.1).
     This gives per-location embeddings, so each particle in a chunk has its own vector.
  4. *Relative scale*: predict `log2(s_a / s_b)` from
     `p = concat(avgpool(hidden[3]), avgpool(hidden[4]))` per view; head
     `r = (g([p_a, p_b]) - g([p_b, p_a])) / 2` (antisymmetric by construction), `g` a 2-layer
     MLP, Smooth-L1 loss. It reads encoder features only, which are unconditioned.
- **Rotation head**: not used in v2. `version = 2` with `use_rotation = True` raises
  `ValueError` citing point 1.

## Allowed write paths
- `tomocpt/networks/selfSupervisedModel.py`
- `tomocpt/networks/ssl/**` (new: `views.py`, `heads.py`, `losses.py`)
- `tomocpt/dataManager/ssl_views.py` (new: SSL v2 chunking and the view dataset)
- `tomocpt/defaultConfigs/ssl_config.py` (new)
- `tomocpt/training/helpers.py`: the self-supervised branches of `_create_model` and
  `_prepare_data_if_needed` only
- `benchmarks/ssl_probe.py`, `benchmarks/ssl_finetune.py` (new)
- `.claude/reports/gate_05_report.json`

## Config (this gate adds) — `defaultConfigs/ssl_config.py`, `TrainConfig.ssl: SSLConfig`
```python
@dataclass
class SSLConfig:
    version: int = 1
    use_rotation: Optional[bool] = None  # None: True for v1, False for v2
    contrast_weight: float = 1.0
    dense_weight: float = 1.0
    dense_dim: int = 64
    n_anchors: int = 256
    recon_weight: float = 10.0
    recon_visible_weight: Optional[float] = None   # None: 1.0 (v1, whole volume), 0.1 (v2)
    scale_weight: float = 1.0
    scale_range_log2: float = 1.0
    apix_ref_min: float = 8.0
    apix_ref_max: float = 16.0
    source_chunk: int = 128
    use_mw_aug: bool = True
    use_fourier_aug: bool = True
    degradation_p: float = 0.8
    keep_angle_min: float = 30.0
    keep_angle_max: float = 60.0
    inversion_p: float = 0.5
```
`None` defaults resolve per version, so `ssl.version=2` alone gives a valid v2 run.

## Tasks
- **T5.1** SSL v2 data: `make_ssl_source_chunks(...)` and `ScalePairDataset` in
  `dataManager/ssl_views.py`. The manifest records `"kind": "ssl"` and `apix_ref` per tomogram.
- **T5.2** `tomocpt.networks.ssl.views`: `make_scale_pair(source, centre, s_a, s_b, apix_ref, rng)
  -> (view_a, view_b, target_log2, coord_maps)`, `equalise_noise(view_a, view_b)`,
  `wedge_preserving_augment(view, target, rng) -> (view, target, flips)`,
  `degrade(view, ssl_cfg, rng) -> view` (reuses `RandomFourierDegradation._degrade_patch`,
  no copy of its numerics), `corresponding_points(n, coord_maps, flips_a, flips_b, rng)
  -> (pts_a, pts_b)`, and `make_training_pair(...) -> dict(input_a, target_a, mask_a,
  input_b, target_b, mask_b, target_log2, pts_a, pts_b, apix_a, apix_b)`.
- **T5.3** `tomocpt.networks.ssl.losses`: `masked_l1(pred, target, mask, visible_weight)`,
  `nt_xent(z_a, z_b, temperature, l1_weight)`, `dense_info_nce(f_a, f_b, pts_a, pts_b, temperature)`.
- **T5.4** `tomocpt.networks.ssl.heads`: `ProjectionHead`, `DenseProjectionHead`, `RelativeScaleHead`.
- **T5.5** `SelfSupervisedModel` v2 path: two views per sample, the four tasks, logging of
  each loss, scale MAE, and the `recons_*` panel. v1 path unchanged.
- **T5.6** Transfer: SSL v2 checkpoint → Gate 3 picking model loads backbone, decoder and
  `cond`; SSL heads are dropped explicitly; `out` follows the Gate 3 class-table rules.
- **T5.7** Benchmarks (human review):
  - `benchmarks/ssl_probe.py`: dense embeddings at the true CZII centres (no labels used in
    pretraining), k-NN accuracy per species with a split by tomogram; compare SSL v2,
    SSL v1, and a randomly initialised network.
  - `benchmarks/ssl_finetune.py`: Gate 3 picker from SSL v2 vs from scratch vs from SSL v1,
    with 1, 2 and all training tomograms; per-class F-beta, 3 seeds.

## Public names
`tomocpt.networks.ssl.views.{make_scale_pair, equalise_noise, wedge_preserving_augment, degrade, corresponding_points, make_training_pair}`,
`tomocpt.networks.ssl.losses.{masked_l1, nt_xent, dense_info_nce}`,
`tomocpt.networks.ssl.heads.{ProjectionHead, DenseProjectionHead, RelativeScaleHead}`,
`tomocpt.dataManager.ssl_views.{make_ssl_source_chunks, ScalePairDataset}`,
`tomocpt.defaultConfigs.ssl_config.SSLConfig`.

## Pass criteria
- **G5-C1** v1 lock: seeded `SelfSupervisedModel._step` on the tiny model equals the
  `v2-gate-0-base` golden loss and its three components (rtol 1e-5).
- **G5-C2** Cause check: on a wedged noise volume, rot90 about Z raises
  `wedge_power_fraction` (canonical orientation) above 0.2; rot180 about Z and every axis
  flip keep it below 0.02.
- **G5-C3** `wedge_preserving_augment` and `make_scale_pair` keep `wedge_power_fraction`
  < 0.02 (50 draws); `degrade` with only the wedge on also keeps it < 0.02.
- **G5-C4** `masked_l1`: with equal residuals, the gradient at a visible voxel is
  `visible_weight ×` the gradient at a masked voxel; masked input voxels are exactly 0.
- **G5-C5** `make_scale_pair`: target equals `log2(s_a/s_b)`; on a one-blob volume the blob
  centroid sits at the view centre in both views (± 0.5 voxel).
- **G5-C6** `corresponding_points`: on a volume of distinct point markers, every anchor pair
  lands on the same marker in both views (± 0.5 view voxel), with flips and different scales.
- **G5-C7** `dense_info_nce`: identical feature maps with perfect correspondences give a lower
  loss than shuffled correspondences; the loss is finite under bf16 autocast.
- **G5-C8** `RelativeScaleHead(a, b) == -RelativeScaleHead(b, a)` (atol 1e-6).
- **G5-C9** `equalise_noise` on pure noise: shell noise std of the two views within 10 %.
- **G5-C10** `version = 2` with `use_rotation = True` → `ValueError`; `version = 2` alone
  resolves `use_rotation = False` and `recon_visible_weight = 0.1`.
- **G5-C11** Clean target: in `make_training_pair`, `target_a` equals the view after flips
  and polarity only (bitwise, fixed seed), even with `degradation_p = 1`; `input_a` differs
  from `target_a` outside the mask. With `inversion_p = 1`, `target_a` is the inverted view.
- **G5-C12** `degrade` keeps the view's mean and standard deviation (rtol 0.02).
- **G5-C13** One v2 training step: every parameter of enabled modules gets a gradient;
  disabled optional modules do not exist; `val_loss_smooth` is logged.
- **G5-C14** Transfer: SSL v2 checkpoint → Gate 3 picking model (C = 1 and C = 4) through
  `_create_model` loads; only SSL heads are dropped and `out` follows Gate 3 rules.
- **G5-C15** (slow) Scale learning: structured phantoms (spheres of several sizes, tubes,
  wedge, noise); held-out MAE of the `log2` ratio < 0.15.
- **G5-C16** (slow) Shortcut control (`D` on): the same pipeline on pure-noise volumes gives
  absolute Pearson correlation between predicted and true ratio < 0.2.
- **G5-C17** (slow) Dense embeddings separate species: on held-out phantoms with four
  species (two of the same size, solid vs hollow), k-NN accuracy of dense embeddings at the
  true centres ≥ 0.8, and ≥ 0.2 above a randomly initialised network.

## Acceptance for humans (not a CI gate)
Recommend SSL v2 as the default initialisation for the Gate 3 picker only if, on CZII,
`ssl_finetune.py` shows a macro F-beta gain ≥ 0.03 with 1 or 2 training tomograms and no
loss (≥ −0.01) with all tomograms.

## Notes for the next gate
Report SSL v2 throughput (views per second) and memory compared with v1.
