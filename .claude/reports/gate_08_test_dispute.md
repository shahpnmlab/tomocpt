# Gate 8 follow-up test dispute (16 Nov 2026, branch `v2-feat/gpu-augment-ports`)

Task: port every CPU augmentation to the GPU data path (`tomocpt/dataManager/gpu_augment.py`) and
remove each ported field from `gpu_augment._UNPORTED` so that `train.otf.augment_device=auto` picks
cuda. After the port `_UNPORTED` is empty and `check_gpu_augmentation` never raises.

Three frozen Gate 8 tests assert the *old* behaviour, that these augmentations have no GPU port.
They contradict the task by construction; they were not edited.

| Test | Assertion | Why it now fails |
|---|---|---|
| `tests/gate8/test_g8c2_gpu_augment.py::test_G8c_C7_unported_options_name_the_cpu_path` | `GpuAugment(...)` with `use_mw_aug`, `use_fourier_aug`, `brightness_p` or `noise_p` raises `ValueError` matching `augment_device` | The fields are ported; `GpuAugment` accepts them. |
| `tests/gate8/test_g8c2_gpu_augment.py::test_G8c2_datamodule_refuses_unported_options` | `OnTheFlyData(augment_device="cuda")` with `use_mw_aug=True` raises `ValueError` | Same: the Fourier degradation is ported. |
| `tests/gate8/test_g8c_auto_defaults.py::test_G8c_D2_augment_device` | `aug.noise_p = 0.5  # no GPU port` then `resolve_augment_device("auto", aug, True) == "cpu"` | `noise_p` has a GPU port, so `auto` resolves to `cuda`. Every other line of the test still holds. |

Proposed resolution: re-freeze the three tests with the opposite expectation (they are replaced by
`tests/gate8/test_g8_gpu_fourier.py::test_G8F5_ported_fields_accepted`, which checks that every
formerly unported field is accepted by `check_gpu_augmentation` and `GpuAugment` and that `auto`
resolves to cuda), or drop the "unported" assertions. If a future augmentation is added without a
GPU port, the `_UNPORTED` mechanism still exists and the old tests' pattern applies to it.

No legacy (`tests/test_*.py`) or Gate 1 test conflicts with the v2 default: the design keeps the
`AugmentationConfig` dataclass defaults `use_mw_aug=False`, `use_fourier_aug=False` (asserted by
`tests/test_config_backfill.py` and `tests/gate1/test_g1_backfill_instance_fields.py`) and adds
`augmentation.fourier_auto=True`, resolved for pipeline-v2 picking in
`training.helpers.resolve_fourier_augmentation`.
