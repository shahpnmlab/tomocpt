# Gate 3 escalation: `TrainConfig.particles` needs a change in `train_config.py`

## What
Add the field `particles: ParticlesConfig` to `TrainConfig` in
`tomocpt/defaultConfigs/train_config.py` (4 lines: one import and one field).

## Where
`tomocpt/defaultConfigs/train_config.py`. CLAUDE.md Section 7 gives this file to Gate 2
(`data_pipeline_version`, `scale`, `chunk_sampling`, ...). Gate 3 may write only
`defaultConfigs/{network_config,particles_config}.py`.

## Why
- CLAUDE.md Section 6 and the Gate 3 spec add `train.particles: ParticlesConfig` in Gate 3,
  but the field has to be declared on `TrainConfig`, which lives in a file Gate 3 does not own.
- Two frozen tests set `cfg.train.particles.class_weights` (`tests/gate3/g3_helpers.py:53`):
  `test_G3_C4_v2_training_step_loss_is_the_multiclass_loss[False]` and `[True]`.
- The Gate 3 code reads the field with `safe_getattr(train, "particles", None)` and falls back to
  the `ParticlesConfig` defaults. So without the field, everything works with the defaults, but
  the user cannot set `train.particles.*`.

## Patch
The patch is also saved at
`/tmp/claude-1002/-home-pranav-Desktop-pranav-software-tomocpt/3c46820d-86da-401b-8d5d-f1446197e18c/scratchpad/train_config_escalation.patch`.

```diff
@@ -8,6 +8,7 @@
 from tomocpt.defaultConfigs.network_config import NetworkConfig
 from tomocpt.defaultConfigs.scale_config import ScaleConfig
+from tomocpt.defaultConfigs.particles_config import ParticlesConfig
@@ -263,6 +264,9 @@
     scale: Annotated[
         ScaleConfig, typer.Option(help="Scale policy of data pipeline v2")
     ] = field(default_factory=ScaleConfig)
+    particles: Annotated[
+        ParticlesConfig, typer.Option(help="Multi-class picking settings (pipeline v2)")
+    ] = field(default_factory=ParticlesConfig)
     chunk_sampling: Annotated[
```

## Evidence
- In the repo as it is now (patch not applied): `pytest tests/gate3 -m "not slow"` gives
  42 passed and 2 failed (the two tests above).
- In a scratch copy of the repo with the patch applied:
  - `pytest tests -m "not slow"`: 435 passed, 27 skipped;
  - `pytest tests/gate3 -m "not slow"`: 44 passed.
