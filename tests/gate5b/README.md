# Gate 5b tests (SSL v2 collapse fix and shared-noise shortcut)

Written by the Test Author from `.claude/agents/05b_AGENT_SSL_FIX.md` before the implementer.
Fast tier: `pytest tests/gate5b -m "not slow"` (CPU, ~45 s here). Slow tier: `pytest tests/gate5b -m slow` (CUDA).
New public names are imported inside the test functions, so collection works on code without them.
Gate 5 helpers (`tests/gate5/g5_helpers.py`, `tests/gate5/test_g5_learning.py`, frozen) are reused, not copied.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), `g5b_fit` (short CPU fits, once per session) |
| `g5b_helpers.py` | Fourier bands, correlation, flip search, queue-buffer lookup, `fit_ssl_v2` (n steps, gradient and `ProjectionHead`-output capture) |
| `test_g5b_config.py` | New `SSLConfig` fields and changed defaults |
| `test_g5b_v1_lock.py` | C1 |
| `test_g5b_views.py` | C2 |
| `test_g5b_losses.py` | C3, C4 (functions), `ProjectionHead` BatchNorm |
| `test_g5b_model.py` | C4 (queue on the model), C5 |
| `test_g5b_learning.py` | C6, C7, C8 (slow) |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| Config (spec section "Config") | `test_G5b_CFG_new_fields_and_changed_defaults[None, 2]`, `test_G5b_CFG_unchanged_gate5_fields`, `test_G5b_CFG_collapse_reg_none_accepted` |
| G5b-C1 SSL v1 unchanged | `test_G5b_C1_ssl_v1_step_matches_base_golden` (calls the frozen G5-C1 test), `test_G5b_C1_ssl_v1_model_has_no_gate5b_modules`; plus the whole earlier suite |
| G5b-C2 decorrelation | `test_G5b_C2_white_noise_views_decorrelate[0-4]`, `test_G5b_C2_structure_survives[0-2]`, `test_G5b_C2_cutoff_and_snr_are_the_config_values`, `test_G5b_C2_input_not_modified`, `test_G5b_C2_training_pair_inputs_decorrelated_at_s1[0-2]`, `test_G5b_C2_reconstruction_target_unchanged_bitwise` |
| G5b-C3 VICReg | `test_G5b_C3_variance_term_zero_when_every_std_at_least_one`, `test_G5b_C3_variance_term_positive_when_collapsed`, `test_G5b_C3_covariance_term_zero_for_decorrelated_dims`, `test_G5b_C3_gradients_finite_under_bf16`, `test_G5b_C3_projection_head_uses_batchnorm` |
| G5b-C4 queue | `test_G5b_C4_empty_queue_equals_nt_xent[0.0, 1e-4]`, `test_G5b_C4_queue_rows_are_extra_negatives`, `test_G5b_C4_gradients_finite_under_bf16`, `test_G5b_C4_queue_is_a_detached_buffer_capped_at_queue_size`, `test_G5b_C4_queue_is_fifo_detached_and_capped` |
| G5b-C5 one v2 step | `test_G5b_C5_every_parameter_gets_a_gradient_each_step`, `test_G5b_C5_dense_loss_reaches_decoder2` |
| G5b-C6 (slow) no collapse | `test_G5b_C6_contrastive_below_chance_and_no_collapse[seed 0, 1, 2]` |
| G5b-C7 (slow) species kNN | `test_G5b_C7_dense_embeddings_still_separate_species` |
| G5b-C8 (slow) C15/C16 | `test_G5b_C8_scale_learning_still_holds`, `test_G5b_C8_no_scale_shortcut_on_noise` |

33 fast cases, 6 slow ones.

## Interface contract assumed by the tests (where the spec leaves it open)

- `decorrelate_view_noise(view, ssl_cfg, rng)`: numpy float32 `(Z, Y, X)` view, `SSLConfig`,
  `numpy.random.Generator`; returns the decorrelated view (numpy or torch, singleton axes allowed),
  same shape, input not modified. Cutoff is `U(view_lowpass_min, view_lowpass_max)` x Nyquist with
  `|k|` isotropic in units of Nyquist (0.5 cycles/voxel); noise std from `U(view_noise_snr_min,
  view_noise_snr_max)` relative to the view's own std (power or amplitude SNR both pass).
  In `make_training_pair` it acts on the inputs only; changing its settings must leave `target_a`/
  `target_b` bitwise identical (so its random draws must not shift the draws that build views and
  targets, e.g. a child generator spawned at a fixed point, as Gate 5 does for D).
- `vicreg_terms(z) -> (variance_term, covariance_term)`, `z` `[N, D]`, two **unweighted** scalar
  tensors (a dict with one `*var*` and one `*cov*` key is also accepted). Variance: hinge
  `relu(1 - sqrt(var_d + eps))`, mean (or sum) over dimensions. Covariance: squared off-diagonal
  covariance entries, any positive normalisation.
- `nt_xent_with_queue(z_a, z_b, queue, temperature, l1_weight)` (positional): Gate 5
  `nt_xent(z_a, z_b, temperature, l1_weight)` plus the queue rows `[Q, D]` (`Q` may be 0) as extra
  negatives of all `2B` anchors (cosine similarity / temperature); the L1 term stays on `z` only.
  Checked against a float64 reference (rtol 1e-5).
- `ProjectionHead(in_dim, out_dim)` (Gate 5 constructor) contains a BatchNorm (`_BatchNorm`, so
  SyncBatchNorm passes).
- Queue on the model: exactly one floating 2-D registered buffer whose name contains `queue`
  (anywhere in the module tree), `[Q, D]` or `[D, Q]` (`D` = `SELF_SUPERVISED_EMBEDING_SIZE`),
  `Q <= queue_size`, not a parameter, `requires_grad=False`. Filled in **training** steps with copies
  of the `ProjectionHead` outputs (any subset of a step's rows, raw or L2-normalised; matched by
  cosine > 0.9999); oldest dropped first. With `queue_size = 4` and 3 steps of 2 pairs, the queue
  must hold 4 entries, none from step 1, all from steps 2-3, at least one from step 3. Pointer/count
  buffers are allowed (not 2-D floating). Persistence is not tested, **but** the frozen G5-C14 test
  requires that the SSL checkpoint keys the picker leaves behind are exactly the SSL head modules'
  state: a persistent queue must live inside a head module, or be `persistent=False`.
- C5 dense gradient: fit with `recon_weight = contrast_weight = scale_weight = 0`,
  `collapse_reg = "none"`; `decoder2` gradient norm > 0 with `dense_grad_scale = 1`, exactly 0 with
  `dense_grad_scale = 0` (control). `recon_weight = 0` must therefore be accepted.

## Deviation from the literal criterion (G5b-C2)

"Voxel-wise correlation < 0.1 on pure white noise" is unreachable by the spec's own design: both
views keep the source's white-noise content below the smaller cutoff, so the full-band correlation
is about `(c_min/c_max)^1.5 * snr/(1+snr)`. A throw-away reference implementation (scratch copy,
not committed) gave 0.20-0.42 (direct) and 0.20-0.32 (inside `make_training_pair`, s = 1). The
tests require **< 0.1 above 0.65 x Nyquist** (only independent noise lives there; reference
|r| < 0.005) and **< 0.7 over the full band** (before the fix: 0.97-0.99). Structure: correlation
of the <= 0.25 x Nyquist band > 0.7 (reference >= 0.97). See `.claude/reports/gate_05b_ta_questions.md`.

## Slow tests (G5b-C6 - C8)

Reuse `tests/gate5/test_g5_learning.py` (`train_ssl_v2`, phantom/noise sources, `predicted_ratios`,
`_knn_score`, `untrained_ssl_v2`) with the Gate 5b defaults. One phantom training per seed
(`TOMOCPT_G5B_SEEDS`, default `0,1,2`), `TOMOCPT_G5B_STEPS` (default 1000; Gate 5 used 1500); the
first seed's model serves C7 and C8/C15; one noise training serves C8/C16. C6 metrics are computed
on held-out phantom pairs through `trainer.validate` with forward hooks on the model's
`ProjectionHead` / `DenseProjectionHead` (definitions in the module docstring): global NT-Xent
without the queue vs `ln(2B-1)`, dense InfoNCE vs `ln(B * n_anchors)` (B = 16, n_anchors = 256),
spread of L2-normalised projections `sqrt(sum_d var_d)` > 0.1 (per-dim std would be ~0.06 even for
uniform embeddings in 256 dims), pooled `hidden[4]` `mean_c std / mean|p|` > 0.1.
C7 floor: the Gate 5 C17 value from `gate_05_report.json` metrics if recorded (key containing
`C17`), else 0.8 (none recorded today).
**Thresholds are the spec's and are not calibrated** (no implementation yet). Runtime: ~0.45 s
per step on the RTX A6000 (12 loader workers; 32 workers no faster), so ~8 min per training and
~32 min for the tier, over the ~15 min budget; lower `TOMOCPT_G5B_STEPS` or the seed list to trade
off.

## Results before implementation

Reference = clean `v2-dev` HEAD f5ede1f in a scratch worktree (this gate modifies Gate 5 code, so
`v2-gate-0-base`, which has no SSL v2, is not the reference). Note: while these tests were written,
the main working tree already contained uncommitted edits to `ssl_config.py`, `heads.py` and
`losses.py` (not by the TA); results on that tree are not the "before" state.

`pytest tests/gate5b -m "not slow"` on HEAD: **3 passed, 30 failed** (~12 s). Passing:
`test_G5b_C1_*` (both) and `test_G5b_CFG_unchanged_gate5_fields` (unchanged behaviour). Failing:
missing `decorrelate_view_noise`, `vicreg_terms`, `nt_xent_with_queue` (ImportError), missing config
fields (TypeError/AttributeError), no BatchNorm in `ProjectionHead`, no queue buffer, and
`test_G5b_C2_training_pair_inputs_decorrelated_at_s1` on the behaviour itself (input correlation
0.97-0.99 at s = 1, the shared-noise shortcut). `pytest tests/gate5 tests/gate5b -m "not slow"` on
HEAD: all 32 Gate 5 tests pass, collection is clean.

The C2 tests were also run against the throw-away reference `decorrelate_view_noise` (low-pass +
noise, child generator, inputs only): all 15 C2 cases pass.

Slow tier on HEAD (Gate 5 code, smoke run with `TOMOCPT_G5B_STEPS=20`): the machinery runs; C6
fails (global 1.04 x chance, dense 1.06 x chance), C7 fails on `dense_grad_scale == 0`, C8/C15 fails
(MAE 0.35 after 20 steps).

Full-length HEAD run (seed 0, 1000 steps, 8 min): C6 **fails on the dense criterion only**:
global NT-Xent 0.71 x chance, dense InfoNCE 0.95 x chance, global spread 0.67, `hidden[4]` ratio
0.66. So the Gate 5 code does **not** reproduce the CZII global collapse on the phantom set; on
phantoms C6 detects the Gate 5 failure through the dense term (head-only gradient). The global and
spread checks are regression guards here, not discriminators. C7 fails on `dense_grad_scale == 0`.

## Orchestrator rulings (human-approved, 31 Oct 2026)
- **C2 measurement:** accepted as written: white-noise views must correlate below 0.1 above
  0.65 x Nyquist and below 0.7 over the full band. The literal full-band < 0.1 is unreachable
  with the spec's design.
- **G5-C11 superseded:** `tests/gate5/test_g5_views.py::test_G5_C11_inversion_applies_to_view_and_target`
  (`corr(input, target) > 0.5`) pins pre-5b behaviour. It is deselected by `conftest.py` whenever
  tests/gate5b is collected, and replaced by `test_G5b_C1_superseded_G5_C11_polarity`
  (r > 0.1, both polarities; inversion still flips the targets).
- **Slow budget:** about 32 min (above the 15 min guideline) is accepted. Use `TOMOCPT_G5B_STEPS` and `TOMOCPT_G5B_SEEDS` to shorten it.
