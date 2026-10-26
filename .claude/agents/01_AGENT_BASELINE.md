# 01_AGENT_BASELINE.md — Gate 1: legacy lock, checkpoint loading, metrics

## Mission
Make the code base safe to extend. After this gate the legacy numbers and checkpoint
loading are locked by tests, the shared test helpers exist, and the picking metrics that
every later decision uses are available. **No numerical behaviour changes in this gate.**
No config fields are added here; each later gate adds its own (CLAUDE.md Section 1.9).

## Allowed write paths
- `tomocpt/configManager/configManager.py` (`safe_getattr` only)
- `tomocpt/metrics.py` (new; inside the package because Gate 4 calibration uses it at run time)
- `benchmarks/__init__.py` (new, empty)
- `pyproject.toml` (`[tool.pytest.ini_options]` only: register marker `slow`)
- `.gitignore` (add `tests/.cache/`, `benchmarks/out/`)
- `.claude/reports/gate_01_report.json`

## Tasks

### T1.1 `safe_getattr`
Export `tomocpt.configManager.configManager.safe_getattr(obj, name, default)`: returns
`getattr(obj, name)` if present, else `default`. Later gates use it to read checkpoint
configs that predate a field. Check `compare_dataclasses` and
`update_config_with_changed_values` for bare `getattr` on fields that can be missing and
guard them with it. Keep all other behaviour.

### T1.2 Dependency pins
Run the fast suite with the pinned `numpy==1.26.4` and `scipy==1.14.1`. Report any
failure; do not change the pins without a human decision.

### T1.3 Picking metrics (`tomocpt/metrics.py`)
- `match_points(pred_xyz, true_xyz, radius) -> (tp_pairs, fp_idx, fn_idx)`: greedy
  one-to-one matching by increasing distance, within `radius`.
- `prf_beta(tp, fp, fn, beta=4.0) -> dict(precision, recall, f1, fbeta)` (β = 4 is the
  CZII challenge metric). Zero denominators give 0, not NaN.
- `score_classes(pred: dict[str, array], true: dict[str, array], radii: dict[str, float],
  match_radius_frac=0.5, beta=4.0, weights=None) -> dict`: per class and macro (optionally
  weighted) mean.

### T1.4 Shared test helpers
The Test Author writes `tests/utils/` (see `TA_TEST_AUTHOR.md`). The implementer does not.

## Public names (tests import these)
- `tomocpt.configManager.configManager.{update_config, backfill_missing_fields, safe_getattr}`
- `tomocpt.metrics.{match_points, prf_beta, score_classes}`

## Pass criteria
- **G1-C1** Gaussian target golden: `generate_gaussian_sphere(R, 2R+1)` for R in
  {5, 10, 17} equals the `v2-gate-0-base` golden (atol 1e-7).
- **G1-C2** Loss golden: `BasePickingModel.loss(ŷ, y)` on fixed random tensors
  (soft labels from G1-C1 placed in a 64³ volume, B=2) equals the golden (rtol 1e-6).
- **G1-C3** Forward golden: tiny `MySwinUNETR` (`tests/utils/tiny.py`), seed 0, eval,
  CPU fp32: logits and `hidden[4]` equal the golden (atol 1e-5).
- **G1-C4** Legacy checkpoints: `BasePickingModel.load_from_checkpoint` on
  `legacy_checkpoint()` succeeds; `InferenceModel.load_from_checkpoint(strict=True)`
  succeeds; logits equal the legacy model's (atol 1e-6). Same for
  `legacy_ssl_checkpoint()` with `SelfSupervisedModel`, and SSL → picking through
  `training.helpers._create_model` loads the backbone unchanged.
- **G1-C5** Resume: `legacy_checkpoint()` resumes through `_create_model` and
  `trainer.fit(ckpt_path=...)` for one more epoch on CPU.
- **G1-C6** Distillation with a legacy teacher constructs and runs one training step on CPU.
- **G1-C7** Config: `PrepdataConfig()`, `TrainConfig()`, `InferConfig()` instantiate
  alone; `MainConfig()` has no `train.prepData`; `update_config` keeps target defaults for
  fields missing on the source; `backfill_missing_fields` adds every field that a legacy
  checkpoint lacks; `safe_getattr` returns the default for a missing field.
- **G1-C8** `build_training_transforms(AugmentationConfig())` is transform-wise
  identical to the `v2-gate-0-base` pipeline (types, order, parameters; contrast inversion
  at `constants.CONTRAST_INVERSION_PROB`).
- **G1-C9** `tomocpt init --output-path x.yaml` writes a file that loads back with
  `--config-file`.
- **G1-C10** Metrics: hand-made cases (exact match, one miss, one false positive, a point
  just outside the radius, two predictions competing for one truth, empty inputs) give the
  expected counts and scores.
- **G1-C11** (optional, skips without the env var) the Zenodo checkpoint loads in
  `predict` model loading.

## Report notes for the next gate
State that no numerical code changed. List the goldens and where they live.
