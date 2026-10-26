# 06_AGENT_SSL_WEDGE.md — Gate 6: wedge-aware self-supervised pretraining (SSL v2)

## Mission
Replace the pretext tasks that the missing wedge breaks. Add a task that teaches the
backbone about scale, which the scale-free picker needs. Keep SSL v1 identical and
keep SSL → picking transfer working.

## Background: what is wrong with SSL v1
1. **The 4-way rotation task is ill-posed.** `rot_rand` rotates by k·90° about Z
   (plane (Y, X)). With the tilt axis along Y, the missing wedge is a double wedge
   in the `kz–kx` plane, constant along `ky`.
   - 90° and 270° move the wedge to the `kz–ky` plane. 0° and 180° leave it in place.
   - Cellular content has no preferred in-plane direction.
   - So the only reliable cue separates {0°, 180°} from {90°, 270°}. The best
     cross-entropy is about ln 2 ≈ 0.69 (chance is ln 4 ≈ 1.39), and accuracy is about
     50 %. The head learns wedge orientation, which has no value for picking.
   - Half the training views have a wedge orientation that never occurs at inference.
   - Check in old TensorBoard logs: `loss_rot` that stalls near 0.69 confirms this.
2. **Conflicting goals on one feature.** The rotation head needs rotation-sensitive
   features. The contrastive loss pairs views with independent rotations, so it asks
   the same pooled 2³ feature to be rotation-invariant.
3. **Inpainting is too easy and mostly copies.** Masked blocks are filled with noise
   scaled to [0, 1] (mean 0.5), while the data have median 0 and range ±3, so masks are
   obvious. The L1 loss covers the whole volume, so most of it rewards copying visible
   voxels.
4. **The "recons" TensorBoard panel shows the input** (`grid4` uses `x`). The
   reconstruction quality was never visible.
5. **SSL chunks use the fixed-particle rescale**, so SSL needs a particle size and
   sees one scale only.
6. The embedding L1 term uses `sum`, so its weight grows with batch size and width.

## SSL v2 design (`train.ssl.version = 2`)
- **Data.** Resample each tomogram once to `apix_ref = clip(apix_native, apix_ref_min, apix_ref_max)`
  (no particle size needed). Cut source chunks of 128³ with stride 64.
- **Views.** For each sample, pick a centre and two scale factors `s_a, s_b`,
  log-uniform in `[1, 2^scale_range_log2]`. A view takes a cube of `64·s` source voxels
  around the centre and Fourier-downsamples it to 64³ (`resample_volume(..., fourier)`).
  So `apix_view = apix_ref · s`. All views pass the same final low-pass (0.9 of
  Nyquist), so `s = 1` is not special. Isotropic scaling keeps the wedge angles.
- **Noise equalisation.** Estimate the noise std of each view from the Fourier shell
  0.8–0.9 Nyquist. Add white noise to the quieter view until both match. Then robust-
  normalise each view. This removes the "noise level tells the scale" shortcut.
- **Wedge-preserving spatial augmentation** (per view): random flips of each axis. No
  rotations except those that flips produce (180° about an axis). The same flips are
  applied to the view and to its reconstruction target.
- **Degradation augmentation `D`** (per view, independent parameters; on by default in v2).
  This is where the Fourier augmentations go. `D` changes only the **network input**;
  the reconstruction target is the same view **without** `D`.
  1. `RandomFourierDegradation` with `missing_wedge_aug` and `amplitude_aug` on
     (`ssl.use_mw_aug`, `ssl.use_fourier_aug`, `ssl.degradation_p`, default 0.8).
     Missing wedge: `keep_angle` in `[30°, 60°]` (models tilt ranges narrower than the
     data's), oriented for `(z, y, x)`. Amplitude: models different defocus, CTF handling
     and dose filtering. Local blend as in the transform.
  2. Intensity `a·x + b` with `a ∈ [0.8, 1.2]`, `b ∈ [-0.1, 0.1]`; Gaussian noise
     `σ ∈ [0, 0.1]`.
  Why this helps each task:
  - *Reconstruction* becomes "restore the clean view from a degraded one", which is
    IsoNet-style de-wedging and denoising. The network learns to infer structure in
    wedge-affected regions, which is what picking needs.
  - *Contrastive*: views A and B get different degradations, so the embedding learns to
    ignore CTF, dose and wedge differences between datasets. This is the main benefit
    for transfer to new datasets.
  - *Relative scale*: random amplitude spectra hide the radial power-spectrum shape,
    which is the main shortcut for guessing scale. (G6-C13 checks this with `D` on.)
  Do not apply `D` to the target. A degraded target teaches the network to reproduce the
  artefacts.
- **Polarity.** Contrast inversion with `ssl.inversion_p` (default 0.5), drawn per view
  and applied to the view **and** its target together, so reconstruction stays possible
  and the two contrastive views can have opposite polarity.
- **Order per view:** crop and Fourier-resample → noise equalisation (both views) →
  flips (view and target) → polarity (view and target) → `D` (view only) → masking
  (view only) → robust normalisation of the input (the target uses the same affine
  normalisation, so its values are comparable).
- **Tasks.**
  1. *Masked reconstruction*: blocks of 8–16 voxels, mask ratio `U(0.15, 0.4)`, fill
     value 0. Loss `L1(masked) + recon_visible_weight · L1(visible)` (v2 default 0.1).
     Target: the unmasked view, or the other half-tomogram if `even_odd_suffixes` is set
     (Noise2Noise target; random swap of even and odd).
  2. *Contrastive*: NT-Xent between views A and B of the same centre (temperature 0.1)
     through a 2-layer projection head on pooled `hidden[4]`. L1 regulariser uses `mean`.
  3. *Relative scale*: predict `log2(s_a / s_b)`. Features
     `p = concat(avgpool(hidden[3]), avgpool(hidden[4]))` per view. Head
     `r = (g([p_a, p_b]) - g([p_b, p_a])) / 2`, with `g` a 2-layer MLP, so the output is
     antisymmetric by construction. Smooth-L1 loss.
     The contrastive projection absorbs scale invariance; the scale head reads the
     un-projected features.
  4. *Wedge task* (optional, `wedge_task = True`, off by default; experimental). The
     degradation augmentation only adds extra wedge in the data's own orientation. This
     task trains filling of the real wedge direction:
     rotate a view by a random 3D rotation `R`, apply the canonical missing-wedge mask
     (`dataManager.fourier_augmentations.oriented_wedge_mask`, `array_order="zyx"`)
     in Fourier space, and train the decoder to restore `R·x` on the Fourier
     coefficients that `R·x` measured but the input lacks. This follows the idea of
     DeepDeWedge (Wiedemann & Heckel, Nat. Commun. 2024).
- **Rotation head**: not used in v2. `version = 2` with `use_rotation = True` raises
  `ValueError` that cites point 1 above.

## Allowed write paths
- `tomocpt/networks/selfSupervisedModel.py`
- `tomocpt/networks/ssl/**` (new: `views.py`, `heads.py`, `losses.py`, `wedge.py`)
- `tomocpt/dataManager/ssl_views.py` (new; SSL v2 chunking and the view dataset)
- `tomocpt/defaultConfigs/ssl_config.py` (add `apix_ref_min=8.0`, `apix_ref_max=16.0`,
  `source_chunk=128`; set v2 defaults inside the v2 code path only)
- Path exception: `tomocpt/training/helpers.py`, the self-supervised branch of
  `_create_model` and of `_prepare_data_if_needed` only.
- `.claude/reports/gate_06_report.json`

## Tasks
- **T6.1** Fix the TensorBoard bug for v1 too (log the reconstruction). Change nothing
  else in v1.
- **T6.2** SSL v2 data: `make_ssl_source_chunks(...)` and `ScalePairDataset` in
  `dataManager/ssl_views.py`. Write `chunk_format.json` with `"kind": "ssl"`.
- **T6.3** `tomocpt.networks.ssl.views`: `make_scale_pair(source, centre, s_a, s_b,
  apix_ref, rng) -> (view_a, view_b, target_log2)`, `equalise_noise(view_a, view_b)`,
  `wedge_preserving_augment(view, target, rng) -> (view, target)` (flips and polarity,
  applied to both), `degrade(view, ssl_cfg, rng) -> view` (the augmentation `D`; reuses
  `RandomFourierDegradation._degrade_patch`, no copy of its numerics).
  `make_training_pair(...) -> dict(input_a, target_a, mask_a, input_b, target_b, mask_b, target_log2)`
  runs the full order above.
- **T6.4** `tomocpt.networks.ssl.losses`: `masked_l1(pred, target, mask, visible_weight)`,
  `nt_xent(z_a, z_b, temperature, l1_weight)`.
- **T6.5** `tomocpt.networks.ssl.heads`: `ProjectionHead`, `RelativeScaleHead`.
- **T6.6** `SelfSupervisedModel` v2 path: two views per sample, the three tasks, the
  optional wedge task, logging of each loss and of scale MAE.
- **T6.7** Transfer: SSL v2 checkpoint → picking (unconditioned or conditioned) loads
  backbone and decoder; SSL heads are dropped explicitly; conditioned models use
  `load_legacy_state_dict` rules for `cond.*`.

## Public names
`tomocpt.networks.ssl.views.{make_scale_pair, equalise_noise, wedge_preserving_augment}`,
`tomocpt.networks.ssl.losses.{masked_l1, nt_xent}`,
`tomocpt.networks.ssl.heads.{ProjectionHead, RelativeScaleHead}`,
`tomocpt.dataManager.ssl_views.{make_ssl_source_chunks, ScalePairDataset}`.

## Pass criteria
- **G6-C1** v1 lock: seeded `SelfSupervisedModel._step` on the tiny model equals the
  `gate-0-base` golden loss and its three components (rtol 1e-5).
- **G6-C2** With a mock logger, the `recons_*` image equals the reconstruction slice.
- **G6-C3** Cause check: on a wedged noise volume, rot90 about Z raises
  `wedge_power_fraction` (canonical orientation) above 0.2; rot180 about Z and every
  axis flip keep it below 0.02.
- **G6-C4** `wedge_preserving_augment` and `make_scale_pair` keep
  `wedge_power_fraction` < 0.02 (50 draws). `degrade` with only the wedge on also keeps
  it < 0.02 (its wedge is oriented like the data's).
- **G6-C5** `masked_l1`: with equal residuals, the gradient at a visible voxel is
  `visible_weight ×` the gradient at a masked voxel; masked input voxels are exactly 0.
- **G6-C6** `make_scale_pair`: target equals `log2(s_a/s_b)`; on a one-blob volume the
  blob centroid sits at the view centre in both views (± 0.5 voxel).
- **G6-C7** `RelativeScaleHead(a, b) == -RelativeScaleHead(b, a)` (atol 1e-6).
- **G6-C8** `equalise_noise` on pure noise: shell noise std of the two views within 10 %.
- **G6-C9** v2 with `use_rotation=True` → `ValueError`.
- **G6-C10** One v2 training step: every parameter of enabled modules gets a gradient;
  disabled optional modules do not exist.
- **G6-C11** Transfer: SSL v2 checkpoint → picking model (unconditioned and
  conditioned) loads; only SSL heads are dropped and only `cond.*` is missing.
- **G6-C12** (slow) Scale learning: structured phantoms (blobs, sheets, tubes, wedge,
  noise); held-out MAE of `log2` ratio < 0.15.
- **G6-C15** Target is clean: in `make_training_pair`, `target_a` equals the view after
  flips and polarity only (bitwise, for a fixed seed), even with `degradation_p = 1`;
  `input_a` differs from `target_a` outside the mask.
- **G6-C16** Polarity is consistent: with `inversion_p = 1`, `target_a` is the inverted
  view and the masked-region L1 between a perfect predictor and the target is 0.
- **G6-C17** `degrade` keeps the view's mean and standard deviation (rtol 0.02).
- **G6-C13** (slow) Shortcut control (run with `D` on): same pipeline on pure-noise volumes; the absolute
  Pearson correlation between predicted and true ratio < 0.2.
- **G6-C18** (slow, report only) Degradation ablation: SSL v2 with `D` on vs off; then
  picking fine-tune on a synthetic set whose test tomograms use a different amplitude
  spectrum and tilt range from training. Report F1 for both.
- **G6-C14** (slow, report only) Picking fine-tune from SSL v2 vs from scratch on the
  Gate 3 G3-C12 task: report steps to reach F1 0.7. Not a pass condition.

## Notes for the next gate
Report the SSL v2 throughput (views per second) and memory compared with v1.
