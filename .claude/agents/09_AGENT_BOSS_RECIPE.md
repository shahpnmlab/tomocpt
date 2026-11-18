# 09_AGENT_BOSS_RECIPE.md — Gate 9: training recipe for the DS-10490 "boss" run

Added 17 Nov 2026 by human request. The DS-10490 run trains one picker on 132 particle species plus
microtubule and actin, with the softmax head and the supervised dense contrastive loss with memory
bank (Gate 7). Four opt-in training features make that recipe stable. Every field defaults to OFF:
with defaults, LR, loss, memory-bank contents and sampling weights are identical to today
(invariant 1). None of the fields is a CLI flag (slim CLI): set them in the YAML or as `key=value`.

## Fields (names are fixed; tests import them)

| Path | Type / default | Meaning |
|---|---|---|
| `train.lr_warmup_steps` | int = `0` (off) | Linear LR warmup over the first N optimizer steps |
| `train.discriminative.contrastive_warmup_steps` | int = `0` (off) | Contrastive weight ramps 0 → `contrastive_weight` over N steps |
| `train.discriminative.bank_max_age` | int = `0` (no limit) | Memory-bank entries older than N steps are not used |
| `train.repeat_factor_max` | Optional[float] = `None` (no cap) | Upper limit of the per-class repeat factor `r_c` |

## Behaviour

### 1. LR warmup (`tomocpt/networks/baseModel.py`)
- Factor `s(step) = min(1, (step + 1) / N)` with `step = trainer.global_step` (optimizer steps).
- `BaseModel.optimizer_step` scales every param group's `lr` by `s` for that one step and restores
  the nominal value afterwards. The nominal values are what `configure_optimizers` set (differential
  LR groups when fine-tuning, `train.lr_scaling` already applied to `self.lr`), and what
  `ReduceLROnPlateau` / SWA change. The scheduler therefore never sees the warmup: it is not reset by
  it and a reduction during or after warmup composes multiplicatively.
- Resume continues from the restored `global_step` (no state of its own).
- `train.schedule_free=True` with `lr_warmup_steps > 0` raises ValueError in
  `configure_optimizers`: `RAdamScheduleFree` (schedulefree 1.4.1) has no `warmup_steps` argument;
  its RAdam rectification is its warmup, and scaling its `lr` would corrupt its averaging weights.
- Public helper: `tomocpt.networks.baseModel.linear_warmup_factor(step, n) -> float`.

### 2. Contrastive warmup (`tomocpt/networks/pickingModel.py`)
- `BasePickingModel.contrastive_weight_at(step) = contrastive_weight · min(1, step / N)` (N = 0:
  the constant weight). Step 0 gives weight 0; the contrastive term is still computed, so the head
  gets a (zero) gradient every step (invariant 5) and anchors keep going into the memory bank.

### 3. Memory-bank age limit (`tomocpt/training/discriminative.py`)
- `ContrastiveMemoryBank(n_classes, size, dim, max_age=0)`. A non-persistent buffer `stamp`
  (`[n_classes + 1, size]` int64, `-1` = unknown) records the global step of each entry.
- `push(embeddings, labels, step=None)` stamps the entries; `get(step=None)` returns every stored
  entry when `max_age == 0` or `step is None`, else only the entries with `step - stamp <= max_age`.
- The state_dict keys stay `queue`, `ptr`, `count` (stamps are not saved), so old and new
  checkpoints load with `strict=True` in both directions.
- After a resume the restored entries have unknown age (`-1`); the first `get` / `push` with a step
  stamps them with that step, so they are used for at most `max_age` more steps.

### 4. Repeat-factor cap (`tomocpt/dataManager/{dataloading,otf}.py`)
- `r_c = min(repeat_factor_max, max(1, sqrt(t / f_c)))` (a cap below 1 is a ValueError), in both
  paths: the chunk-file sampler (`repeat_factor_weights`, `Data`) and on-the-fly
  (`otf.candidate_weights`, `OnTheFlyData`).
- Helper `repeat_factors(frequencies, threshold, max_factor=None) -> r_c` (in `dataloading.py`,
  re-exported by `otf.py`). The data modules keep `class_frequencies` and `class_repeat_factors`
  and log the effective oversampling once at setup (`log_repeat_factors`: threshold, cap, number of
  capped classes, the ~10 largest `r_c` with name and `f_c`).

## Allowed write paths
`tomocpt/defaultConfigs/{train_config,discriminative_config}.py` (own fields),
`tomocpt/networks/{baseModel,pickingModel}.py`, `tomocpt/training/discriminative.py`,
`tomocpt/training/run.py` (`Data(...)` argument only), `tomocpt/dataManager/{dataloading,otf}.py`,
`tests/gate9/**`, `CLAUDE.md` (Sections 6, 7), `README.md`, `TODO_v2.md`.

## Pass criteria (`tests/gate9/`)
- **G9-C1** Defaults: fields at their defaults; optimizer LR, contrastive weight, bank contents and
  sampling weights unchanged.
- **G9-C2** Warmup LR at steps 0, N/2, N, N+5, one and two groups; ReduceLROnPlateau still reduces
  after warmup; schedule-free raises.
- **G9-C3** Contrastive ramp values; head parameters get gradients at step 0; bank fills at step 0.
- **G9-C4** Bank age exclusion; state_dict keys unchanged; old bank state loads strictly; restored
  entries are stamped at the first use.
- **G9-C5** `repeat_factor_max` caps `r_c` in both sampler paths.
- **G9-C6** Config backfill of the new fields (Gate 1 sweep).
