# Gate 8c test dispute: `test_G8c_C1_defaults_are_chunks` (14 Nov 2026)

**Test:** `tests/gate8/test_g8c1_otf.py::test_G8c_C1_defaults_are_chunks`, frozen at 8c.1.

**Why it no longer holds:** the human ruled on 14 Nov 2026 (in the 8c close-out session) to flip the
`TrainConfig.sampling` default, after the CZII 5-fold CV (weighted OOF F4 0.748 chunks vs 0.763
on-the-fly + GPU augmentation) and the DS-10490 actin run (F1 0.836 vs 0.810) were accepted.
The test asserts the opt-in defaults of 8c.1:
`mainConfig.train.sampling == "chunks"` and `OnTheFlyConfig().augment_device == "cpu"`.

**Implemented:** `train.sampling = "auto"` and `train.otf.augment_device = "auto"`.
- `auto` sampling (`training.helpers.resolve_sampling`) gives `on_the_fly` for pipeline v2 picking and
  `chunks` for pipeline v1, SSL, and a `chunks_dir` that already holds chunk files (reused, not
  refused). Pipeline v1 is unchanged, so invariant 1 (legacy equivalence) still holds.
- `auto` augment device (`dataManager.otf.resolve_augment_device`) gives `cuda` only when training on
  a GPU and every enabled augmentation has a GPU port; else `cpu`.
- New tests: `tests/gate8/test_g8c_auto_defaults.py` (G8c-D1, G8c-D2).

**Proposed change to the frozen test** (the only edit; the other C1 tests pass unchanged):
```python
def test_G8c_C1_defaults_are_chunks():   # name kept; the defaults are now "auto" (human, 14 Nov 2026)
    ...
    assert mainConfig.train.sampling == "auto"
    ...
            o.seed, o.val_seed, o.val_samples) == ("float16", "auto", "auto", None, True, 0.0, 42, 1234, None)
```
Waiting for the orchestrator/human to apply and re-freeze; the hash in `gate_status.json` is then
recorded from the updated file.
