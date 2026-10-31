# Gate 5b - Test Author questions and notes (for the orchestrator)

The TA kept going with the choices below (tests written; see `tests/gate5b/README.md`).

## Q1. G5b-C2 "correlation < 0.1 on pure white noise" is unreachable as literally stated
Both views keep the source's white-noise content below the smaller of their two cutoffs (0.35-0.6 x
Nyquist), so the full-band voxel correlation of a correct implementation is about
`(c_min/c_max)^1.5 * snr/(1+snr)`: 0.20-0.42 measured with a reference implementation (0.20-0.32
inside `make_training_pair` at s = 1). **Choice:** < 0.1 in the band above 0.65 x Nyquist (where the
shortcut lives; reference |r| < 0.005, before the fix ~1) and < 0.7 over the full band (before:
0.97-0.99). Please confirm.

## Q2. Frozen Gate 5 test conflicts with the Gate 5b design (test dispute candidate)
`tests/gate5/test_g5_views.py::test_G5_C11_inversion_applies_to_view_and_target` asserts
`corr(input_a, target_a) > 0.5` on visible voxels (source `make_source(11)`, `degradation_p = 0`).
With the spec's decorrelation (inputs low-passed at 0.35-0.6 x Nyquist plus noise at SNR 0.5-1, clean
target) a reference implementation gives **r = 0.29** (both polarities), so a correct Gate 5b
implementation fails this frozen test. Its purpose (input and target share polarity) still holds:
the sign is right and |r| is far from -1. Suggested ruling: list it in the Gate 5b report as a test
pinning superseded v2 behaviour (e.g. re-check with `r > 0.1`, or with `view_noise_snr_*` large and
`view_lowpass_*` = 1). No other Gate 5 test pins `dense_grad_scale` or `recon_weight`.

## Q3. Queue and the frozen G5-C14 transfer test
G5-C14 requires the SSL checkpoint keys left behind by the picker to be exactly the SSL head modules'
state. A persistent queue buffer registered on `SelfSupervisedModel` itself breaks it; it must live
inside a head module (e.g. `ProjectionHead`) or be `persistent=False`. Gate 5b tests accept either
(the buffer is found anywhere by name, containing `queue`).

## Q4. Signatures chosen where the spec is silent
- `vicreg_terms(z) -> (variance_term, covariance_term)`, unweighted (dict with `*var*`/`*cov*` keys
  also accepted).
- `nt_xent_with_queue(z_a, z_b, queue, temperature, l1_weight)`, positional, mirroring Gate 5
  `nt_xent(z_a, z_b, temperature, l1_weight)`; queue rows are extra negatives of all 2B anchors.
- C5 isolates the dense gradient with `recon_weight = 0` (must be accepted by the model).

## Q5. Slow-tier budget and calibration
3 phantom seeds (C6) + 1 noise training (C8/C16), 1000 steps each at ~0.45 s/step on the A6000 =
~32 min, over the ~15 min target (Gate 5 needed 1500 steps for C15/C17; fewer steps risk false
failures of C7/C8). Knobs: `TOMOCPT_G5B_STEPS`, `TOMOCPT_G5B_SEEDS`. Thresholds (0.8 x chance for
global NT-Xent with B = 16 and dense InfoNCE with 16 x 256 anchors; spread 0.1; hidden[4] ratio 0.1)
are the spec's, uncalibrated; the dense one is the most uncertain.

## Q6. The phantom set does not reproduce the CZII global collapse
Seed 0, 1000 steps on HEAD (Gate 5 code): global NT-Xent 0.71 x chance, spread 0.67, hidden[4] ratio
0.66 (all pass); dense InfoNCE 0.95 x chance (fails). C6 therefore fails on HEAD only via the dense
term. A collapse-provoking phantom variant would need a spec change; the TA kept the spec's set.

## Note
During TA work the main working tree had uncommitted edits to `tomocpt/defaultConfigs/ssl_config.py`,
`tomocpt/networks/ssl/heads.py`, `tomocpt/networks/ssl/losses.py` (not made by the TA). The "before"
results were taken on a clean worktree of HEAD f5ede1f.
