# TA_TEST_AUTHOR.md — Test author (runs before each implementer)

## Role
You turn the "Pass criteria" of one gate spec into pytest tests. You write the
tests **from the spec**, not from an implementation. The implementer does not
exist yet. The orchestrator freezes your files after review.

## Rules
1. Write only in `tests/gateN/` (and `tests/utils/` at Gate 1) and in
   `tests/golden/`, `tests/fixtures/`.
2. Import only the public names that the gate spec lists. If a name you need is not
   in the spec, write `.claude/reports/gate_0N_ta_questions.md` and stop.
3. One test per criterion at minimum. Name each test after its criterion id
   (for example `test_G2_C2_band_rule`).
4. Fast tests: CPU, fp32, `torch.manual_seed(0)`, `np.random.seed(0)`,
   `random.seed(0)`, no network access, total runtime per gate < 2 min.
5. Learning tests: mark `@pytest.mark.slow`. Use the tiny model
   (`tests/utils/tiny.py`). Use fixed seeds. Set thresholds with a margin: a
   correct implementation must pass in at least 9 of 10 seeds. State the
   margin in a comment.
6. Legacy goldens come from `v2-gate-0-base` code only. Write the generator script
   into `tests/golden/make_goldens_gateN.py`, run it in a worktree of
   `v2-gate-0-base`, and commit both the script and its output (`.json` or `.npz`).
7. Do not test private helpers. Do not assert on log text.
8. A test must fail on the legacy code if the criterion is new behaviour. Check this
   by running the test once on `v2-gate-0-base` before you hand over. Record the
   result in `tests/gateN/README.md` (which tests fail on base, and why).

## Shared helpers (Gate 1 only; frozen afterwards)
Create `tests/utils/` with:

### `tests/utils/wedge.py`
- `apply_missing_wedge(vol: np.ndarray, max_angle_deg: float = 60.0, tilt_axis: str = "y") -> np.ndarray`
  Arrays are `(Z, Y, X)`. Zero the Fourier coefficients outside the tilt range.
  For tilt axis `y`, a coefficient `(kz, ky, kx)` is measured if
  `|atan2(kz, kx)|` is within `max_angle_deg` of the `kx` axis (the missing region is a
  double wedge around `kz` in the `kz–kx` plane, constant along `ky`).
- `missing_wedge_mask(shape, max_angle_deg=60.0, tilt_axis="y") -> np.ndarray[bool]`
- `wedge_power_fraction(vol, max_angle_deg=60.0, tilt_axis="y", kmin=0.05) -> float`
  Fraction of Fourier power (excluding |k| < kmin·Nyquist) inside the **missing** region
  of the given orientation. Near 0 for a correctly wedged volume in that orientation.

### `tests/utils/phantoms.py`
- `make_phantom(shape=(64,128,128), apix=10.0, species=(...), tubes=(...), wedge_deg=60.0,
  snr=0.5, seed=0, allow_border=False) -> (volume, coords, polylines)`.
  - `species`: tuple of `(class_number, diameter_ang, count, hollow: bool)`. Solid or hollow
    spheres with dark contrast (density negative, as in cryo-ET), placed without overlap,
    away from borders unless `allow_border=True`.
  - `tubes`: tuple of `(class_number, diameter_ang, count)`. Hollow cylinders (wall about
    50 Å for 250 Å tubes, solid for diameters < 100 Å) along smooth random curves, with a
    controllable minimum curvature radius; they may cross.
  - Apply the wedge, then add white Gaussian noise for the requested SNR.
  - `coords: dict[int, np.ndarray]` per class, `(x, y, z)` in voxels (RELION order).
  - `polylines: dict[int, list[np.ndarray]]` per tube class, `(x, y, z)` in voxels.
  - Also return `masks: dict[int, np.ndarray]` with the exact tube masks (for Gate 6
    mask → centreline tests).
- `write_mrc(path, vol, apix)` and `write_star(path, coords_by_class, tomo_name, apix)`
  (RELION 3.1 style, `rlnCoordinateX/Y/Z`, `rlnMicrographName`, `rlnClassNumber`).
- `make_dataset(tmp_path, n_tomos=2, ...)` that writes tomograms and a STAR file ready for
  `prepare_data`. Default `n_tomos=2` (one tomogram cannot be split into train/val).

### `tests/utils/tiny.py`
- `tiny_swin_kwargs()` → `dict(img_size=64, in_channels=1, out_channels=1,
  feature_size=12, use_v2=True, use_checkpoint=False, drop_rate=0.0,
  attn_drop_rate=0.0, dropout_path_rate=0.0)`.
- `tiny_main_config(tmp_path)` → a `MainConfig` with tiny network settings,
  `batch_size=2`, `n_cpus_for_dataloading=0`, `use_gpus=False`, paths under `tmp_path`.
- `random_chunk_datamodule(n=2, n_classes=1, v2=False)` → a LightningDataModule of random
  batches (v1 keys, or the CLAUDE.md Section 5.1 keys when `v2=True`) for one-step fits.

### `tests/utils/legacy.py`
- `legacy_checkpoint(tmp_cache_dir) -> Path`. Builds (once, then caches in `tests/.cache/`,
  git-ignored) a tiny `BasePickingModel` checkpoint **with code from `v2-gate-0-base`**, by
  running `tests/fixtures/make_legacy_ckpt.py` in a worktree of that tag through
  `subprocess`. Save it with `pl.Trainer(max_epochs=1, accelerator="cpu", logger=False)`
  `.fit(...)` on random tensors and `trainer.save_checkpoint(...)`, so the pickled
  `MainConfig` is the legacy class and the optimizer state is real. If the worktree cannot
  be built, **fail** (do not skip).
- `legacy_ssl_checkpoint(...)`: the same for `SelfSupervisedModel`.
- Optional: if the environment variable `TOMOCPT_ZENODO_CKPT` points to the public
  pretrained checkpoint, tests also use it; otherwise those tests skip with a reason.

## Hand-over
Write `tests/gateN/README.md`: criterion → test name table, runtime of the fast
suite on your machine, tests that fail on `v2-gate-0-base` and why.
