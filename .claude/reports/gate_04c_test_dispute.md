# Gate 4c test dispute: `tests/gate4b/test_g4b_ensemble.py::test_G4b_C5_one_member_equals_the_model[softmax]`

**Status:** RESOLVED 4 Nov 2026: human ruling "yes we dont want leaks"; option 1 applied (`_load` in `applied_main_config(None)`), test re-frozen.

## What fails
Second half of the test, the scale-conditioned checkpoint (`cond_ckpt`, sigmoid head):
`single = _load(cond_ckpt)` vs `ens = load_ensemble([cond_ckpt])`, max abs difference 0.43.

## Why
In the `[softmax]` case the test first loads a softmax copy of an `abc` checkpoint with the legacy
`InferenceModel.load_from_checkpoint` (no CLI snapshot). That load writes `class_head = softmax`
into the global `mainConfig.train` (CLAUDE.md 1.10, the leak TODO 3a.4 fixes for ensembles). The
next legacy load, the test's reference `single = _load(cond_ckpt)`, inherits it: the sigmoid
checkpoint is loaded as **softmax**. Before Gate 4c, `load_ensemble` leaked the same way, so both
sides were equally wrong and the test passed.

Gate 4c (spec D4) loads ensemble members with `load_member_isolated`: each member keeps its own
settings, so the ensemble member is (correctly) sigmoid and differs from the contaminated reference.

Reproduction (scratch script, 4 Nov 2026):
```
cond ckpt alone, legacy load:                         class_head sigmoid
legacy load of cond AFTER a softmax checkpoint:        class_head softmax   <- test reference
isolated load (Gate 4c) of cond after the same:        class_head sigmoid   <- ensemble member
```
`[sigmoid]` passes (no softmax load before). The test passes on the code before Gate 4c.

## Proposed ruling
The test's assertion (a one-member ensemble equals its model) stays; only its reference load must
not inherit an earlier load's settings. Options:
1. (recommended) Re-freeze the test with the reference loaded in a clean state: wrap each
   `single = _load(...)` in `applied_main_config(None)` (restores `mainConfig` after the load), or
   load the reference with `load_member_isolated`. No change to the code under test.
2. Keep legacy leaking behaviour in `load_ensemble` (drop D4 for `load_ensemble` only). Not
   recommended: ensembles of mixed checkpoints (TODO 3a.4, contrastive + plain) then fail.
