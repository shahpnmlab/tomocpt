# Gate 5 tests (SSL v2: wedge-aware, scale-aware self-supervised pretraining)

Written by the Test Author from `.claude/agents/05_AGENT_SSL.md` before the implementer started.
Fast tier: `pytest tests/gate5 -m "not slow"` (CPU). Slow tier: `pytest tests/gate5 -m slow` (CUDA).
New public names are imported inside the test functions, so the v1 lock runs on code without them.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), `ssl_v2_fit` (one-step CPU fits of the v2 model, built once per session) |
| `g5_helpers.py` | Helpers (no tests): array conversion, wedged noise, crop wedge metric, shell noise std, blob centroid, flip detection, SSL configs, SSL source chunks, one-step v2 fit, encoder / `decoder2` feature readers |
| `g5_common.py` | Shared by the golden generator and the C1 test (no `tomocpt` import at module level) |
| `make_golden_g5_c1.py` | C1 golden generator (run in the `v2-gate-0-base` worktree) |
| `g5_c1_golden.json` | C1 golden |
| `test_g5_v1_lock.py` | C1 |
| `test_g5_wedge.py` | C2, C3, C9, C12 |
| `test_g5_views.py` | C5, C6, C11, C4 (input side), T5.2 key/shape contract of `make_training_pair` |
| `test_g5_losses_heads.py` | C4 (`masked_l1`), C7, C8 |
| `test_g5_config.py` | C10 |
| `test_g5_model.py` | C13, C14 |
| `test_g5_learning.py` | C15, C16, C17 (slow) |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G5-C1 v1 lock | `test_G5_C1_ssl_v1_step_matches_base_golden` |
| G5-C2 Cause check | `test_G5_C2_rot90_about_z_moves_the_wedge_rot180_and_flips_do_not`, `test_G5_C2_view_metric_separates_rotated_from_correct_crops` (self-check of the crop metric) |
| G5-C3 Wedge kept | `test_G5_C3_wedge_preserving_augment_keeps_the_wedge`, `test_G5_C3_make_scale_pair_keeps_the_wedge`, `test_G5_C3_degrade_with_only_the_wedge_keeps_the_wedge` |
| G5-C4 Masked L1 / zero fill | `test_G5_C4_masked_l1_gradient_ratio[0.1, 0.37, 1.0]`, `test_G5_C4_masked_input_voxels_are_exactly_zero` |
| G5-C5 Scale pair | `test_G5_C5_target_and_blob_at_the_view_centre[3 scale pairs]` |
| G5-C6 Correspondences | `test_G5_C6_corresponding_points_land_on_the_same_marker[2 scale pairs]` |
| G5-C7 Dense InfoNCE | `test_G5_C7_dense_info_nce_prefers_true_correspondences`, `test_G5_C7_dense_info_nce_finite_under_bf16_autocast` |
| G5-C8 Antisymmetric head | `test_G5_C8_relative_scale_head_is_antisymmetric` |
| G5-C9 Noise equalisation | `test_G5_C9_equalise_noise_on_white_noise_of_different_levels`, `test_G5_C9_equalise_noise_on_a_scale_pair_of_pure_noise` |
| G5-C10 Config | `test_G5_C10_defaults_reproduce_v1`, `test_G5_C10_version_2_alone_resolves`, `test_G5_C10_version_2_with_rotation_raises`, `test_G5_C10_version_set_after_construction_builds_a_v2_model` |
| G5-C11 Clean target | `test_G5_C11_target_is_the_clean_view`, `test_G5_C11_inversion_applies_to_view_and_target` (+ `test_G5_T52_make_training_pair_keys_and_shapes`) |
| G5-C12 Degrade stats | `test_G5_C12_degrade_keeps_mean_and_std` |
| G5-C13 One v2 step | `test_G5_C13_v2_step_all_tasks`, `test_G5_C13_disabled_tasks_have_no_modules` |
| G5-C14 Transfer | `test_G5_C14_ssl_v2_checkpoint_into_picker[1, 4]` |
| G5-C15 Scale learning (slow) | `test_G5_C15_scale_learning` |
| G5-C16 Shortcut control (slow) | `test_G5_C16_no_scale_shortcut_on_noise` |
| G5-C17 Dense embeddings (slow) | `test_G5_C17_dense_embeddings_separate_species` |

32 fast test cases, 3 slow ones. T5.1 (`make_ssl_source_chunks`, `ScalePairDataset`) has no pass
criterion and is not tested; its signatures are left to the implementer.

## Interface contract assumed by the tests (where the spec leaves it open)

Arrays are `(Z, Y, X)`. `rng` is always a `numpy.random.Generator`; tests also seed `numpy.random`
(`RandomFourierDegradation` draws from it). Functions may return numpy arrays or torch tensors, with
or without leading singleton axes (the tests squeeze them).

- `make_scale_pair(source, centre, s_a, s_b, apix_ref, rng)`: `source` is a numpy float32 volume
  (128^3 or larger), `centre` a `(z, y, x)` tuple of floats in source voxels; the test only uses
  centres for which `centre ± 32·s` is inside the source. A view covers `64·s` source voxels, so one
  view voxel is `s` source voxels (G5-C6 checks this: fitted scale `1/s` within 3 %).
  The source centre must map to view coordinate 32 or 31.5 (both accepted, ± 0.5).
  `coord_maps` is opaque; it must not depend on the volume content (G5-C6 reuses the maps of the first
  call for volumes that differ only in content).
- `wedge_preserving_augment(view, target, rng) -> (view, target, flips)`: the outputs must be exact
  flips of the inputs (the tests detect the axes by exact comparison); `flips` is opaque and is passed
  unchanged to `corresponding_points`.
- `corresponding_points(n, coord_maps, flips_a, flips_b, rng) -> (pts_a, pts_b)`: `[n, 3]` float,
  **`(z, y, x)` in view voxel coordinates** (voxel `i` is at coordinate `i`), after the flips, inside
  `[-0.5, 63.5]`.
- `degrade(view, ssl_cfg, rng)`: the **Fourier part** of `D` (`RandomFourierDegradation`, applied
  with probability `degradation_p`, wedge `keep_angle` in `[keep_angle_min, keep_angle_max]`). G5-C12
  requires the mean and std to be kept within 2 %, which the intensity jitter `a·x + b`
  (`a ∈ [0.8, 1.2]`) cannot satisfy, so that jitter and the Gaussian noise belong to the rest of `D`
  inside `make_training_pair`, not to `degrade`. G5-C3 calls `degrade` with
  `use_fourier_aug=False, use_mw_aug=True, degradation_p=1` on unit-std wedged noise.
- `equalise_noise(view_a, view_b) -> (view_a, view_b)`.
- `make_training_pair(source=..., apix_ref=..., ssl_cfg=..., rng=...)` (keywords; other parameters
  optional) returns a dict with at least the spec's keys. `input_*`, `target_*`, `mask_*` squeeze to
  `(64, 64, 64)`; `mask_*` is bool-convertible with **True = masked**; `pts_*` are `[n_anchors, 3]`
  (same convention as `corresponding_points`); `target_log2`, `apix_*` are scalars with
  `log2(apix_a / apix_b) = target_log2`. Values must go through `torch.utils.data.default_collate`
  (contiguous arrays or tensors).
  * G5-C11 compares two calls with the same seeds, `degradation_p = 1` vs `0`: `target_a` and
    `target_b` must be **bitwise equal**. So (a) the random draws of `D` must not shift the draws that
    build the views and targets (e.g. draw `D` from a child generator spawned at a fixed point), and
    (b) the affine normalisation that the target shares with the input must be computed from the clean
    view (after polarity, before `D`), not from the degraded input.
  * Inversion: with `inversion_p = 1` vs `0` (same seeds) the targets are anti-correlated (r < -0.99),
    and input and target keep the same polarity.
  * Masked input voxels are exactly `0` after the normalisation.
- `masked_l1(pred, target, mask, visible_weight)`: bool `mask`, True = masked; scalar. The test uses a
  mask that covers exactly half of the voxels, so normalising by the voxel count of each part or by the
  total count both pass.
- `dense_info_nce(f_a, f_b, pts_a, pts_b, temperature)`: `f` is `[B, D, d, h, w]` covering the view's
  field of view; `pts` `[B, n, 3]` in view voxel coordinates (as above). Tests keep points in `[8, 28]`.
- `RelativeScaleHead(in_dim)`, `head(p_a, p_b)` with `p` `[B, in_dim]` (`in_dim = 8fs + 16fs`),
  output `[B]` or `[B, 1]`. Antisymmetry is checked in eval mode.
- `SSLConfig`: `None` defaults may be resolved at construction or by a `resolve()` (or `resolved()`)
  method returning the resolved config (or `None` after resolving in place). The `ValueError` for
  `version=2, use_rotation=True` may come from the constructor, from `resolve()`, or from the
  `SelfSupervisedModel` constructor. Setting `train.ssl.version = 2` on a default config after
  construction (as an override does) must still build a v2 model.
- v2 model (`SelfSupervisedModel` with `train.ssl.version = 2`): `model.model` is a
  `ScaleConditionedSwinUNETR`; no `rotation_head` attribute (or `None`); it trains on
  `default_collate` of `make_training_pair` dicts. A task whose loss weight is `0`
  (`contrast_weight`, `dense_weight`, `scale_weight`) has no head module (invariant 5).
  SSL v2 heads are instances of `ProjectionHead`, `DenseProjectionHead`, `RelativeScaleHead`.
- Transfer (C14): every tensor of the picker network is loaded from the SSL checkpoint, and the
  checkpoint tensors left behind are exactly the state of those head modules.

## The wedge metric for views (deviation from the literal criterion, G5-C3)

`utils.wedge.wedge_power_fraction` of a crop is not near 0 even for a perfect crop: the crop is not
periodic, and its spectrum is the source spectrum convolved with the crop window. Measured on wedged
white noise (60°): a plain 64^3 crop gives 0.016, a Fourier-resampled crop with the 0.9-Nyquist
low-pass 0.023-0.025, i.e. above the spec's 0.02 for any implementation. G5-C3 therefore measures
`make_scale_pair` views with `g5_helpers.view_wedge_fraction` (demean, Tukey(0.5) window, `kmin = 0.1`,
power beyond 70°, i.e. a 10° margin), threshold 0.02 as in the spec. Measured: correct views ~0.0015,
a 90° rotation ~0.16. Periodic volumes (augment and degrade tests, C2) use the plain metric.

## Golden (G5-C1)

No earlier gate stored an SSL v1 golden. `make_golden_g5_c1.py` was run in the `v2-gate-0-base`
worktree (commit ce6f7b0, unmodified):
`cd ../tomocpt-base && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python ../tomocpt/tests/gate5/make_golden_g5_c1.py --expect-root $PWD`.
Tiny `SelfSupervisedModel` (fs 12, dropout 0, `use_checkpoint=False`, CPU fp32, batch 2), weights
overwritten by `g5_common.reinit_by_name_` (per-name seeds, so they do not depend on constructor RNG use),
input `RandomState(51)` normal clipped to [-3, 3], `numpy.random` and torch seeded with 5 right before
`_step`. Golden: total 13.81668, rotation 1.24775, contrastive 1.17316, reconstruction 11.39577 (rtol 1e-5);
the eval-mode output sums are a weight cross-check. The test also requires the v1 parameter names to
be unchanged.

## Slow tests (G5-C15 – C17)

In-memory data (no chunking): 128^3 phantoms at 10 Å/px (`make_phantom`: 100 Å solid, 200 Å solid,
200 Å hollow, 300 Å solid, one 250 Å tube; wedge 60°, SNR 1) or wedged white noise, robust-normalised;
items from `make_training_pair`; `SelfSupervisedModel` v2 with default SSL settings, Lightning trainer
on GPU, bf16-mixed, lr 1e-3, batch 8, `TOMOCPT_G5_STEPS` (default 1500) steps, 12 loader workers.
One training on phantoms serves C15 and C17, a second on noise serves C16.
C15/C16 read the prediction with the model's single `RelativeScaleHead` on encoder features
(`swinViT` hidden[3], hidden[4], averaged) of `input_a` / `input_b`. C17 reads embeddings at the true
centres of six held-out 96^3 phantoms (FiLM-conditioned `decoder2` features computed from the public
submodules, and the `DenseProjectionHead` output if it accepts them; the better one counts),
leave-one-tomogram-out 5-NN, balanced accuracy.
**The thresholds are the spec's and were not calibrated** (nothing to train yet). The implementer should
run a seed sweep (`TOMOCPT_G5_SEED`) and report it; if a threshold is unreachable with a correct
implementation, write a test dispute.

## Results before implementation (HEAD 1a69661; the same on `v2-gate-0-base`)

`pytest tests/gate5 -m "not slow"`: **3 passed, 29 failed**, about 15 s. Passing: C1 (legacy lock) and the
two C2 tests (pure numpy cause check). Every other test fails with `ModuleNotFoundError`
(`tomocpt.networks.ssl`, `tomocpt.defaultConfigs.ssl_config`). The slow tests error the same way.

The view, loss, head and config tests (25 cases of `test_g5_wedge.py`, `test_g5_views.py`,
`test_g5_losses_heads.py`, `test_g5_config.py`) were checked against a throw-away reference
implementation in a scratch copy (not committed): all pass except the two C10 tests that need the v2
model, in about 26 s. C13, C14 and the slow tests could not be checked without a v2 model.
