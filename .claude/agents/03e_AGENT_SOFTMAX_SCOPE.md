# Gate 3e — softmax scope: particles only, filaments independent (TODO 5.1, human request 15 Nov 2026)

Status: **approved 15 Nov 2026** (human); tests frozen, implemented. Read `CLAUDE.md` and `03d_AGENT_SOFTMAX_HEAD.md` first.

## 1. Why
The Gate 3d softmax head (`train.particles.class_head=softmax`) normalises over an implicit background
logit and **every** available channel, filaments included: `p_c = exp(l_c) / (1 + Σ_d exp(l_d))`, and the
targets are renormalised to `y / max(1, Σ_c y_c)`. That suits particles (two particles cannot fill one
voxel; it fixed the Gate 6 size confusion). Filaments differ:
- They overlap particles for real: DS-10490 has decorated filaments (`alpha-actinin_CH1_bound_to_F-actin`,
  `Myosin-actin_rigor_complex` on actin; complexes on microtubules). Under one softmax the shared voxels
  get split targets, so both channels learn a weaker signal there.
- A filament channel is a long tube that is traced, not a blob that is peak-picked; competition with
  every particle channel along its length may fragment the trace.

Hypothesis, not a finding: the C = 134 smoke run (3 epochs, actin F1 0.49) cannot tell. This gate adds
the option; a comparison run decides the default for DS-10490.

## 2. Config (own field only)
| Path | Type / default |
|---|---|
| `train.particles.softmax_scope` | `all` \| `particles` = `all` (today's behaviour) |

Only read when `class_head = softmax`. YAML / `key=value` only, no CLI flag (slim CLI rule).
Checkpoints without the field load as `all` (`safe_getattr`, backfill).

## 3. Behaviour with `softmax_scope = particles`
- Particle channels: softmax over the background logit and the **available particle** channels only;
  filament logits do not enter the denominator.
- Filament channels: independent `sigmoid(l_c)`.
- Targets: renormalise over particle channels only (`y_p / max(1, Σ_particles y)`); filament targets
  unchanged.
- Same rule in the loss (`_probabilities`), `predict_step`, `predict_step_with_offsets`, and anywhere
  else `class_probabilities(..., "softmax", ...)` is called. At predict, the per-level `classes`
  subset still limits the particle softmax to that level's particle channels.
- Unavailable channels stay 0 (as 3d). A model without filament channels gives exactly the
  `all` result (scope irrelevant).

## 4. Files
`tomocpt/training/multiclass_loss.py` (`class_probabilities`, `renormalise_targets`: an optional
`softmax_channels` bool mask), `tomocpt/networks/pickingModel.py` (passing the mask),
`tomocpt/defaultConfigs/particles_config.py` (field), `README.md`.

## 5. Tests (`tests/gate3e/`, written and frozen before the code)
- C1 default `all` is bitwise today's output (loss, predict probabilities) on a mixed particle +
  filament batch.
- C2 `particles`: filament channels equal `sigmoid(logits)`; particle channels equal the 3d softmax of
  the particle channels alone (filament logits changed → particle probabilities unchanged).
- C3 targets: particle renormalisation ignores filament targets; filament targets untouched.
- C4 no filament channels → `particles` equals `all` bitwise.
- C5 an old checkpoint (no field) loads and predicts as `all`; DDP rule holds (no new parameters).
- C6 predict with a level subset (`classes`) normalises over that level's particle channels only.

## 6. Comparison run (after the tests pass)
DS-10490 joint arm as the C = 134 smoke run (`benchmarks/out/smoke_c134_20261114/run.sh`), long enough to
converge (early stopping), `softmax_scope` `all` vs `particles`, same seed and split. Read: actin and
microtubule F1, particle F4 of the filament-bound species (classes 68, 116, and others on filaments) and
the mean particle F4. Cost: two full runs (~50 min/epoch at B = 4, or B = 16 after the large-C fixes).
