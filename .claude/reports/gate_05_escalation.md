# Gate 5 escalation: changes outside the spec's allowed write paths

Each change is the smallest one I found that makes SSL v2 usable end to end. Please rule on each.

## 1. `tomocpt/defaultConfigs/train_config.py` (Gate 2 file): `TrainConfig.ssl`

**Change:** `from tomocpt.defaultConfigs.ssl_config import SSLConfig` plus one field after
`repeat_factor_threshold`:
```python
ssl: Annotated[SSLConfig, typer.Option(help="Self-supervised pretraining settings (mode=selfSupervised)")] = field(default_factory=SSLConfig)
```
**Why:** the spec defines `TrainConfig.ssl: SSLConfig` (Config section), and the tests read
`MainConfig().train.ssl` (G5-C10) and set `cfg.train.ssl` (C13, C14 and the slow tests). The field is
not in the Gate 5 paths. Default `SSLConfig()` is `version = 1`, so legacy behaviour is unchanged.
`backfill_missing_fields` gives older checkpoints the default, so they load as SSL v1.

## 2. `tomocpt/training/run.py` (Gate 3 and Gate 4 file): data module for SSL v2

**Change:** step 3 builds `dataManager.ssl_views.SSLPairData(chunks_dir, batch_size,
n_cpus_for_dataloading, ssl_cfg, persistent_workers=use_gpus)` when `mode == selfSupervised` and
`train.ssl.version == 2`. Otherwise it builds the same `Data(...)` call as before. Only the
indentation of the `Data(...)` call changed.
**Why:** SSL v2 trains on scale pairs (`make_training_pair`) built from the 128^3 source chunks.
The legacy `Data` module reads 64^3 v1 chunks and applies the v1 augmentation, which cannot produce
these items. Without this change `tomocpt train` cannot run SSL v2. The tests do not need it, because
they build their own data modules.

## 3. `tomocpt/training/helpers.py` outside the two self-supervised branches

Allowed: the self-supervised branches of `_create_model` and `_prepare_data_if_needed`. Also changed:

a. **New module-level helpers** used by those branches: `_ssl_version`,
   `_prepare_ssl_v2_data_if_needed` and `_is_ssl_v2_checkpoint`. They contain no logic for other modes.

b. **`_create_model`, picking branch, `use_v2_fine_tune`**: adds
   `or _is_ssl_v2_checkpoint(checkpoint_config)`. This is the transfer from a self-supervised source.
   An SSL v2 network is always a `ScaleConditionedSwinUNETR`. The legacy fallback
   (`BasePickingModel(model=pretrained_model)`) would hand a conditioned network to an unconditioned
   pipeline-v1 picker. Its `cond.*` parameters would get no gradient (a DDP crash under invariant 5),
   and the SSL heads would be dropped only implicitly. SSL v1 sources keep the legacy path unchanged.

c. **`_fine_tune_into_v2`** (2 lines plus a log line): if the target network has no `cond`, it drops
   the source's `cond.*` weights before `load_legacy_state_dict` (otherwise they are "unexpected keys").
   This only happens for an SSL v2 source going into an unconditioned picker. G5-C14 (conditioned
   picker, C = 1 and C = 4) does not go through it. I checked it with a smoke run: SSL v2 into the
   default v1 picker loads every picker tensor from the checkpoint.

## 4. `benchmarks/ssl_common.py` (new)

This file holds the code that `ssl_probe.py` and `ssl_finetune.py` share: CLI pretraining, the
embedding reader and the k-NN probe. CLAUDE.md Section 7 gives Gate 5 `benchmarks/ssl_*.py`, but the
spec lists only the two benchmark files, so I am noting it here.
