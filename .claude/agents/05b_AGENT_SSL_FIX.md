# 05b_AGENT_SSL_FIX.md — Gate 5b: fix SSL v2 collapse and the shared-noise shortcut

Added 31 Oct 2026 by human decision. Build before Gate 6.

## Evidence (CZII pretraining, `benchmarks/out/czii_bench_20261029_2237/finetune/pretrain_ssl_v2`, 75 epochs, batch 8)
- **Global contrastive collapsed.** Train/val NT-Xent flat at 2.71 / 2.69 for all 75 epochs;
  chance for 16 views is ln 15 = 2.708. On the checkpoint (val batch of 16): cosine similarity
  0.9999996 between every pair of embeddings (positives and negatives alike); the pooled
  `hidden[4]` varies by 0.004 across samples against a mean magnitude of 0.76: the deepest
  encoder stage outputs almost a constant. Reconstruction (weight 10) can be solved through the
  decoder's skip connections from shallower stages, so nothing forces `hidden[4]` to carry
  information.
- **Dense contrastive never trained the network.** Gate 5 set `ssl.dense_grad_scale = 0`
  (head on detached features) because with the full gradient the dense task matched noise:
  both views are cut from the same source chunk, so at a shared location they carry the same
  noise realisation (noise equalisation only adds noise to the quieter view). Dense loss 7.5
  against chance ln 2048 = 7.6.
- Scale and reconstruction learn (scale MAE 0.04 in log2; recon falls). Gate 5 tests (C15–C17)
  did not detect either failure: no criterion checked that contrastive losses fall below chance
  or that embeddings vary across samples.

## Design changes (SSL v2 only; SSL v1 unchanged; v2 defaults change, recorded as a human decision)
1. **Independent noise between views.** Each view gets its own random isotropic low-pass in
   Fourier space, cutoff `U(view_lowpass_min, view_lowpass_max)` × Nyquist (default 0.35–0.6),
   then independent white noise at `view_noise_snr` relative to the view's own std (default
   U(0.5, 1.0)), applied to the contrastive inputs before `D`. The reconstruction target stays
   the clean view (Gate 5 C11 rule: target after flips and polarity only). Then exact
   high-frequency noise no longer matches between views.
2. **Dense task trains the network again:** default `dense_grad_scale = 1.0`.
3. **Anti-collapse.** `ssl.collapse_reg` = `vicreg` (default) | `none`: VICReg variance and
   covariance terms (Bardes et al. 2022) on the global projections and on the dense anchor
   embeddings, `variance_weight` 25 × hinge on per-dimension std (target 1), `covariance_weight`
   1 × off-diagonal covariance; projector gets BatchNorm between its layers (DDP: SyncBatchNorm
   when multi-GPU). The NT-Xent / InfoNCE terms stay.
4. **More negatives.** A FIFO queue of past global projections (`queue_size`, default 1024,
   detached buffer) used as extra negatives in NT-Xent (MoCo-style, no momentum encoder).
5. **Rebalance.** Default `recon_weight` 10 → 1 (`recon_visible_weight` 0.1 kept), so the
   contrastive tasks are not drowned out. The architecture is unchanged (one decoder);
   rebalancing plus anti-collapse is the fix.
6. **Monitoring.** Log per epoch: std of normalised global embeddings across the batch, mean
   off-diagonal cosine similarity, dense embedding std, and each contrastive loss divided by its
   chance level (`*_vs_chance`). Warn (log) if the global std stays below 0.05 after 3 epochs.

## Allowed write paths
- `tomocpt/networks/selfSupervisedModel.py`, `tomocpt/networks/ssl/**`,
  `tomocpt/dataManager/ssl_views.py`, `tomocpt/defaultConfigs/ssl_config.py`
- `benchmarks/ssl_*.py`, `experiments/run_benchmarks.sh` (SSL epochs if needed)
- `.claude/reports/gate_05b_report.json`

## Config (this gate adds to `SSLConfig`)
`view_lowpass_min: float = 0.35`, `view_lowpass_max: float = 0.6`, `view_noise_snr_min: float = 0.5`,
`view_noise_snr_max: float = 1.0`, `collapse_reg: str = "vicreg"`, `variance_weight: float = 25.0`,
`covariance_weight: float = 1.0`, `queue_size: int = 1024`; changed defaults:
`dense_grad_scale: 1.0`, `recon_weight: 1.0`. Gate 5 frozen tests that pin the old defaults:
the TA checks them first; a conflicting frozen test goes to a test dispute (orchestrator rules).

## Public names
`tomocpt.networks.ssl.views.decorrelate_view_noise(view, ssl_cfg, rng)`,
`tomocpt.networks.ssl.losses.{vicreg_terms, nt_xent_with_queue}`, `tomocpt.networks.ssl.heads.ProjectionHead` (BatchNorm).

## Pass criteria
- **G5b-C1** SSL v1 and every earlier test unchanged (Gate 5 C1 golden), except Gate 5 tests
  that pin superseded v2 defaults (listed in the report with the ruling).
- **G5b-C2** `decorrelate_view_noise`: on pure white noise, two views of the same source after
  decorrelation have voxel-wise correlation < 0.1 at the shared location (before: ≈ 1 at s = 1);
  on a structured phantom, the low-passed structure correlation stays > 0.7; the
  reconstruction target is unchanged (bitwise).
- **G5b-C3** `vicreg_terms`: variance term is 0 when every dimension has std ≥ 1 and positive for
  collapsed embeddings (all equal); covariance term 0 for decorrelated dimensions; gradients
  finite under bf16.
- **G5b-C4** `nt_xent_with_queue`: equals `nt_xent` with an empty queue; the queue is FIFO,
  detached, a buffer (not a parameter), capped at `queue_size`.
- **G5b-C5** One v2 step: every parameter of enabled modules gets a gradient (DDP rule), with
  `dense_grad_scale = 1` the dense loss sends a non-zero gradient into `decoder2`.
- **G5b-C6** (slow) Phantom pretraining (tiny model, the Gate 5 phantom set), 3 seeds: global
  NT-Xent below 0.8 × chance and dense InfoNCE below 0.8 × chance on held-out phantoms at the
  end; std of normalised global embeddings across samples > 0.1 (no collapse); pooled
  `hidden[4]` std across samples > 0.1 × its mean magnitude.
- **G5b-C7** (slow) Gate 5 C17 (dense embeddings separate four species, kNN ≥ 0.8) still holds,
  and with the full dense gradient (shared-noise shortcut removed) kNN does not fall below the
  Gate 5 value.
- **G5b-C8** (slow) Gate 5 C15/C16 (scale learning, shortcut control) still hold.

## Acceptance for humans (not a CI gate)
Re-run the SSL step of `experiments/run_benchmarks.sh` on CZII (`STEPS=ssl`): the SSL v2
TensorBoard traces must show contrastive losses well below chance, then the fine-tuning arms
decide whether SSL v2 is the default initialisation (Gate 5 acceptance rule).
