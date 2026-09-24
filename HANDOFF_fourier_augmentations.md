# HANDOFF: Port membrain-seg Fourier augmentations into tomocpt

**Branch:** `feat/fourier-augmentations` (branched from `main`)
**Status:** Gates 0–E complete. Gate F (GPU box) outstanding.

> **Corrections applied after implementation.** Three things in the original plan turned out
> to be wrong once measured, and are corrected in place below, each marked
> **[CORRECTED]**: the DC-coefficient claim in Gate D3, the imaginary-residual tolerance in
> Gate D4, and — most importantly — visual criterion (b) in Gate E, which was stated
> backwards. Gate B's open question about the angle convention is resolved and marked
> **[RESOLVED]**. Read those before acting on the surrounding text.

---

## Context

tomocpt's training augmentation is three hardcoded TorchIO transforms in
`VolumeDatsetIO.get_dataset` (`tomocpt/dataManager/dataloading.py:70-82`):
`RandomAffine(degrees=45)`, `RandomElasticDeformation`, `RandomBlur`. No config exposure, no
cryoET-specific augmentation, no intensity augmentation at all.

We are porting the IsoNet-inspired missing-wedge augmentation described in the membrain-seg
paper, plus two things shipped alongside it that matter just as much.

### The reference

membrain-seg's `FourierAugmentations` branch, checked out at `membrain-seg/`
**for reference only**. It is gitignored, must never be committed, and **will not exist on
the GPU box**. Reference paths below are relative to `membrain-seg/src/membrain_seg/`.

The reference transform `MissingWedgeMaskAndFourierAmplitudeMatchingCombined`
(`segmentation/dataloading/fourier_augmentations.py:153`) does three things. All three must
be ported:

1. **Missing wedge** — zero a wedge of Fourier coefficients.
2. **Amplitude spectrum randomization** — synthesize a fake radially-averaged amplitude
   spectrum from a Gaussian-smoothed random walk, expand it to a 3D `rotational_kernel`,
   multiply it into the FFT. Simulates varied CTF/defocus/dose-filtering conditions. For a
   picking model trained across heterogeneous datasets this is arguably the bigger
   generalization win, and it is easy to overlook when focusing on "missing wedge".
3. **Localized application** — blend the result back as
   `img + kernel * (img_modified - img)` with a Gaussian kernel, so degradation covers a
   soft blob rather than the whole subvolume. `randomize()` forces
   `_do_sample_kernel = True` whenever the wedge fires, so **the wedge is always
   localized**: the network sees patches that are partly degraded and partly clean.
   Preserve this. It is deliberate, not incidental.

The paper's framing, for intent: subvolumes are randomly rotated, then an artificial wedge
is applied by masking Fourier coefficients, **while the ground-truth label is left
unaltered** — training the network to infer structure from context in wedge-impacted
regions.

### Hard constraints

- Chunks are 4D `(C=1, 64, 64, 64)` float32 `torch.Tensor` (`CHUNK_SIZE=64`), loaded as
  `tio.ScalarImage` / `tio.LabelMap` in `VolumeDatsetIO.__getitem__`.
- Targets are **continuous Gaussian spheres in [0,1]**
  (`tomocpt/labels/helpers.py:generate_gaussian_sphere`), not integer classes, despite the
  `tio.LabelMap` wrapper. Loss is weighted Huber + `gradient3d_loss` — a regression target.
  Do not assume nearest-neighbour label semantics anywhere.
- Inputs are zero-centred and hard-clipped at ±3 by `robust_normalization`
  (`tomocpt/dataManager/preprocessing.py`). The reference ends by re-normalizing to
  zero-mean/unit-std, which is a *different* distribution from what tomocpt feeds in.
  Re-apply the ±3 clip after the transform, behind a `reclip` flag defaulting to `True`.
- Transforms run on CPU in dataloader workers under `bf16-mixed`. Keep everything float32
  and let Lightning cast.
- **Never touch the label.** The reference relies on `keys=["image"]`. Subclassing
  `tio.IntensityTransform` gives the same guarantee structurally — TorchIO applies such
  transforms only to `ScalarImage` and skips `LabelMap`.

### Machine split

Development machine is an **M1 MacBook Pro: no CUDA**. But every unit test in Gate D is pure
CPU numpy/torch on 64³ arrays and **runs fine locally**, including the port-equivalence
tests. Run them locally. Only **Gate F** (end-to-end training) needs the GPU box.

---

## Gate 0 — Branch setup  ✅ DONE

Branch `feat/fourier-augmentations` created from `main`. `.gitignore` updated with
`membrain-seg/`, `.DS_Store`, `test.tomcat`.

Note: `AGENTS.md` and `config.yaml` are untracked in this tree. `config.yaml` is edited in
Gate C2 — decide whether it should be tracked before starting that gate.

Resolved: `config.yaml` is tracked, `AGENTS.md` is ignored (commit `01447e6`).
`augmentation_dump/`, the default Gate E output directory, was added to `.gitignore`.

**Gate:** `git status` shows `membrain-seg/` ignored, not untracked. ✅

---

## Gate A — Vendor the filter utilities  ✅ DONE

`rotational_kernel` and its dependency `hypotenuse_ndim` from
`tomo_preprocessing/matching_utils/filter_utils.py` are required by the amplitude
augmentation. These are DeePiCt-derived, **Apache-2.0**, and carry a copyright header.

Create `tomocpt/dataManager/filter_utils.py` containing `hypotenuse_ndim` and
`rotational_kernel` copied verbatim, **with the original copyright header preserved
intact**. Do not copy `radial_average` — the spectrum here is synthesized, not measured.
Add an attribution note to the repo's license documentation.

**Gate:** ✅ passes.

```bash
python -c "import tomocpt.dataManager.filter_utils, sys; assert not [m for m in sys.modules if 'membrain' in m]; print('ok')"
```

Delivered as `tomocpt/dataManager/filter_utils.py`, with attribution in a new top-level
`NOTICE` and a License section in `README.md`.

**Note for anyone importing the reference.** `import
membrain_seg.tomo_preprocessing.matching_utils.filter_utils` fails: the package `__init__`
chain pulls in SimpleITK, which is not installed. Load the reference file directly with
`importlib.util.spec_from_file_location` instead. Gate D's `membrain_ref` fixture does
exactly this and is the worked example.

---

## Gate B — Port the Fourier transform  ✅ DONE

Create `tomocpt/dataManager/fourier_augmentations.py`. Port `wedge_mask`, `get_line_plot`,
`sample_scalar`, `_generate_kernel`, `run_interpolation`, restructured as a
`tio.IntensityTransform` subclass `RandomFourierDegradation`.

Port the numerics **faithfully first**. The wedge geometry is a slope-based construction on
the index grid, equivalent to an angular threshold only because the FFT is `fftshift`ed
first. Do **not** rewrite it as an `atan2` formulation — Gate D1 proves equivalence against
the original, and that proof is worthless if the implementation was "improved" in transit.
Optimize only after D1 is green.

### Use production parameters, not class defaults

The class `__init__` defaults differ from what the actual call site passes
(`segmentation/dataloading/memseg_augmentation.py:282`). Port the **call-site** values as
tomocpt's defaults:

```
missing_angle_range = (45, 45)     missing_wedge_prob  = 0.5
smooth_sigma_range  = (2, 4)       amplitude_prob      = 0.5
step_sigma_range    = (0.1, 4)     sample_kernel_prob  = 0.5
offset_range        = (2, 10)      loc                 = (-0.5, 1.5)

scale = lambda x, y: np.exp(np.random.uniform(np.log(x[y]) * 0.75, np.log(x[y]) * 1.5))
```

That `scale` differs sharply from the class default (`np.log(x[y] // 6)`): it yields a
kernel comparable to or larger than the patch, so the blob softly covers most of a 64³ chunk
with a gentle falloff. The class default would give a much tighter blob and materially
different augmentation. **Getting this wrong produces code that runs and trains but does the
wrong thing** — the worst failure mode in this port.

### Bugs to fix, not carry over

All are latent for cubic even-sized patches, so equally latent at `CHUNK_SIZE=64` — but they
are traps if chunk size ever changes, and cost nothing to fix now:

- `fft_patch_to_real` (line 146) calls `fft.fftshift` where it should call `ifftshift`.
  Identical only for even-length axes. The reference even carries a comment questioning it.
- `wedge_mask` (lines 66-70) indexes the centre voxel with `x_coords.shape[2]` twice; the
  middle index should be `shape[1]`. Correct only for cubic input.
- `randomize` (lines 238, 243) tests `self.smooth_sigma_range` when setting `offset` and
  `missing_angle` — copy-paste from the block above. Harmless while all three are tuples,
  silently wrong the moment one is passed as a float.
- `normalize_and_fft_patch` mutates its argument in place (`patch -= patch.min()`), and
  `patch` is a view into the input tensor. Copy before transforming.

### Deliberate deviations

- **Taper the wedge boundary.** The reference uses a hard binary mask, producing Gibbs
  ringing along the wedge edge — a shortcut feature a picking model can latch onto. Add a
  raised-cosine angular ramp, **defaulting to `taper_width=0`** (exact reference behaviour)
  so Gate D1 can prove equivalence first. Enable it only after D1 passes.
- **Pin the angle convention.** **[RESOLVED]** The docstring was right and the name was
  wrong: the parameter counts *angles to keep*. Measured retained fractions on a 64³ grid —
  0°→0.000, 30°→0.288, 45°→0.508, 60°→0.712, 90°→1.000, monotonic throughout. It is the
  half-opening of the retained double wedge, measured from axis 0 in the (axis 0, axis 2)
  plane. The tomocpt parameter is therefore named **`keep_angle_range`**, defaulting to
  `(45., 45.)`. Asserted by `test_keep_angle_is_the_angle_retained`.

### Defect found during the port, deliberately NOT fixed

`wedge_mask` is built about `mean(arange(64)) == 31.5`, while `fftshift` places DC at index
32; on top of that the strict `<` / `>` comparisons break ties differently on the two
boundary lines. The result is that the boundary voxels — 64 per slice, ~1.6% — are not
Friedel-symmetric, so the inverse FFT is not exactly real and `np.real()` silently discards
an imaginary residual of **~4.9%**. That is far above float32 noise.

This is left as-is because fixing it changes `wedge_mask` and invalidates the Gate D1
equivalence proof, which is the trade this document warns against elsewhere. It is pinned by
`test_known_friedel_asymmetry_at_the_wedge_boundary`, whose docstring says that a future
correct fix should make that test fail and be replaced with a strict symmetry assertion.
See also the **[CORRECTED]** note in Gate D4.

**Gate:** ✅ passes. Module imports; `RandomFourierDegradation()(subject)` runs on a
synthetic `tio.Subject`, returns unchanged shape and dtype, and leaves the `LabelMap`
bit-identical.

---

## Gate C — Intensity transforms, config, plumbing  ✅ DONE

### C1. Intensity transforms

Port from `segmentation/dataloading/transforms.py` as `tio.IntensityTransform` subclasses in
`tomocpt/dataManager/intensity_augmentations.py`:

| New class | Reference | Line |
|---|---|---|
| `RandomBrightnessGradient` | `BrightnessGradientAdditiveTransform` | 330 |
| `RandomLocalGamma` | `LocalGammaTransform` | 386 |
| `RandomBrightness` | `RandomBrightnessTransformd` | 89 |
| `RandomContrast` | `RandomContrastTransformd` | 116 |

The Gaussian-kernel builder is shared with `RandomFourierDegradation._generate_kernel`.
Factor it into **one** helper rather than duplicating it three times as the reference does.

**Reuse TorchIO, do not port** — equivalents already exist and the pipeline is TorchIO
end-to-end:

- `SimulateLowResolutionTransform` → `tio.RandomAnisotropy`
- Gaussian noise → `tio.RandomNoise`
- Global gamma → `tio.RandomGamma`
- Smooth multiplicative field → `tio.RandomBiasField` (complementary to the *additive*
  `RandomBrightnessGradient`, not redundant with it)

### C2. Config

Add an `AugmentationConfig` dataclass to `tomocpt/defaultConfigs/train_config.py`, mirroring
the existing `OptimizerConfig` / `NetworkConfig` pattern, attached to `TrainConfig` via
`field(default_factory=AugmentationConfig)`. Use lowercase `Annotated[..., typer.Option]`
fields so they surface through Typer and `config.yaml` (uppercase fields are treated as
constants and are not exposed).

Mirror the reference's two independent switches (`use_mw_aug`, `use_fourier_aug`), the
probabilities and ranges from Gate B, and the existing affine/elastic/blur probabilities.

Defaults must **reproduce today's behaviour** for the three existing transforms
(`affine_p=0.8`, `affine_degrees=45`, elastic/blur under a `OneOf` at 0.75), with all new
augmentations **off by default**, so existing runs are not silently changed.

Add the matching block to `config.yaml`, which has no augmentation section today. Also fix
the stale `train.n_cpus_for_train` key there — `TrainConfig` defines
`n_cpus_for_preprocessing` / `n_cpus_for_dataloading`.

**Two further stale keys were found in `config.yaml`, beyond the one above.** Both predate
this branch:

- `train.restore_full_state` is not defined on `TrainConfig` and appears nowhere in the
  codebase. Removed.
- `infer.predictions_coord_format: relion` is **not a valid enum member** — the choices are
  `warp`, `relion_31`, `relion_50`, `imod`. This makes the whole file fail to merge against
  the schema, so it is a live bug independent of this port. **Left unfixed on purpose**:
  choosing between `relion_31` and `relion_50` is a real decision, not a typo fix. Resolve
  it before Gate F, or `config.yaml` cannot be loaded at all.

### C3. Plumbing

`get_dataset` reads nothing from config today. Give it an **explicit** augmentation-config
argument passed from `Data.setup()` — an explicit argument, *not* a `mainConfig` global
read, because it is a classmethod called from two places and implicit global state makes it
untestable.

Move pipeline construction into `build_training_transforms(cfg) -> tio.Compose`, leaving
`get_dataset` to just call it. Order follows the reference, where the Fourier transform sits
directly after the rotations and `AxesShuffle`:

1. `tio.RandomAffine` (image + label together)
2. `tio.OneOf({RandomElasticDeformation, RandomBlur})` — as today
3. `RandomFourierDegradation` — **after all spatial transforms**
4. intensity transforms

Step 3's position is the entire point: it makes the wedge land at a random orientation
relative to the specimen while the label stays geometrically correct. **Putting it earlier
silently destroys the augmentation** while still training normally.

Validation keeps `transform=None`.

**Gate:** ✅ both halves pass.

```bash
python -c "from tomocpt.mainConfig import mainConfig; print(mainConfig.train.augmentation)"
```

prints the new block; and at default settings the composed pipeline is transform-wise
identical to today's — same types, same order, same `probability` values, with
`RandomAffine.degrees` / `default_pad_value`, `RandomBlur.std_ranges` and the elastic
parameters all compared equal against a verbatim copy of the old hardcoded pipeline.

Two notes for whoever tunes this next:

- **The two Gaussian-blob widths are different on purpose.** `RandomFourierDegradation` uses
  the call-site `exp(U(log(n)*0.75, log(n)*1.5))` — a blob at or larger than the patch. The
  brightness-gradient and local-gamma call sites use the *tighter*
  `exp(U(log(n//6), log(n)))`. They look nearly identical and swapping them is invisible.
- **The intensity augmentations push data well outside ±3.** Measured up to ±72 with all of
  them at `p=1.0`, as brightness-gradient strength (±5) stacks on contrast (×2). That is
  reference behaviour — `reclip` guards only the Fourier transform — and they are all off by
  default, but tomocpt's inputs are hard-clipped at ±3, so these want tuning rather than
  adopting the membrain-seg values wholesale.

**In TorchIO 0.20.2 the probability attribute is `transform.probability`, not
`transform.p`.** `p=` is accepted as a constructor kwarg. Anything introspecting a composed
pipeline needs the former.

---

## Gate D — Tests (run on the M1; no GPU needed)  ✅ DONE

There is no pytest suite today (only `tomocpt/configManager/test_configManager.py`, and
pytest is not a declared dependency). Add `pytest` to dev dependencies in `pyproject.toml`
and create `tests/` with a `conftest.py`.

`tests/test_fourier_augmentations.py`:

1. **Port equivalence** — the highest-value test. Run the vendored `wedge_mask` and
   `rotational_kernel` against the membrain-seg originals imported from `membrain-seg/`, on
   identical seeded inputs, asserting array equality. This is the *only* check that the port
   is faithful, and it is only possible while both copies coexist in this tree. Mark it
   `@pytest.mark.skipif` on the absence of `membrain-seg/` so it degrades cleanly on the GPU
   box.
2. **Round-trip identity** — with both augs disabled, FFT→iFFT returns the input within
   float tolerance. Catches `fftshift`/`ifftshift` ordering errors, which are silent for
   even sizes and catastrophic otherwise. Repeat with an **odd-sized** volume to confirm the
   Gate B `ifftshift` fix.
3. **Wedge geometry** — mask is constant along the tilt axis (a wedge, not a cone),
   symmetric under `k -> -k` (required for a real-valued output), and DC is retained. Pin
   the keep-vs-remove angle convention here.

   **[CORRECTED] DC is not retained — the reference removes it.** `wedge_mask` sets the
   centre of the *removal* mask to 1 immediately before inverting it, so the DC coefficient
   is discarded, the inverse of what the surrounding code reads as intending. Ported
   faithfully behind a `zero_dc` flag defaulting to `True`. It is harmless in practice
   because the output is re-normalized to zero mean afterwards, which discards DC anyway.
   `test_dc_handling_matches_reference_and_is_recoverable` asserts both halves: removed
   under the default, retained under `zero_dc=False`.

   The forced DC voxel is also the *only* thing that breaks tilt-axis constancy, so the
   wedge-not-cone test runs with `zero_dc=False`. Full symmetry under `k -> -k` does not
   hold — see the correction to test 4 below.
4. **Real and finite** — no NaN/Inf; imaginary residual below float32 tolerance.

   **[CORRECTED] The imaginary residual is ~4.9%, not float32 noise.** Cause and reasoning
   in the Gate B section above. Asserting a float32 tolerance here would mean either failing
   or loosening the bound until it asserted nothing, so the test instead pins the defect
   inside a known range and documents that a correct fix should trip it. The output is still
   genuinely real and finite — `np.real()` discards the residual — and that part is asserted
   normally.
5. **Label untouched** — run the full `tio.Compose` on a `tio.Subject` and assert the
   `LabelMap` tensor is bit-identical before and after. This is the paper's core requirement
   and the thing most likely to break silently during the TorchIO restructure.
6. **No input mutation** — the input tensor is unchanged after the call (guards the in-place
   bug from Gate B).
7. **Shape/dtype invariance** — `(1,64,64,64)` float32 in, same out, for every transform.
8. **Determinism under seed** — same seed gives the same output twice.

**Gate:** ✅ `pytest tests/ -v` green on the M1 — **71 passed, 0 skipped**, including all 26
port-equivalence assertions. Delivered as `tests/conftest.py` and
`tests/test_fourier_augmentations.py`, with `pytest` added as a `dev` extra and
`[tool.pytest.ini_options]` in `pyproject.toml`.

Each guard was mutation-tested by reverting the corresponding fix and confirming the suite
catches it:

| reverted fix | result |
|---|---|
| `ifftshift` → `fftshift` | odd shapes fail, **even shapes still pass** — the silent bug, reproduced |
| copy-before-transform | 2 failures |
| `IntensityTransform` → `Transform` | label-identity fails, full-pipeline test with it |
| Fourier transform moved before the affine | ordering test fails |

With `membrain-seg/` moved aside to simulate the GPU box: **45 passed, 26 skipped**.

---

## Gate E — Visual check (M1, no GPU needed)  ✅ DONE (on phantoms; rerun on real chunks)

Script at `scripts/dump_augmented_chunks.py`: load a few real chunks, apply the pipeline with
`missing_wedge_prob=1.0`, save via `tio.ScalarImage(tensor=...).save(...)` to `.nii.gz`.
`--synthetic` runs it with phantoms when no data is staged, `--report` prints numeric
measurements, `--png` writes a montage of central slices.

**Gate:** open in a viewer and confirm:

- (a) directional smearing in a central slice;
- (b) **[CORRECTED — this criterion was stated backwards]** the smearing axis is **the same
  in every sample**, with the specimen rotated differently underneath it;
- (c) a visible soft boundary between degraded and clean regions from the kernel blend.

### [CORRECTED] Criterion (b) was inverted

The original text read: *"the smearing direction **differs between samples**, because the
affine rotated the volume first. If the direction is identical across samples, the transform
is running before the affine."* **That is exactly backwards, and following it literally would
lead you to "fix" a correct pipeline.**

The wedge is applied in the array's own frame at a fixed `keep_angle` of 45°. Under
*correct* ordering the smearing is therefore always along the same array axis. What the
preceding affine varies is the **specimen's orientation relative to that fixed wedge** —
which is the actual point of the augmentation, but is not the same statement as "the
smearing direction differs."

A smearing direction that visibly **rotates** between samples is the *failure* signal: it
means a spatial transform ran after the wedge and dragged the wedge around with it.

Measured, wedge only (no blend, no amplitude randomization), as residual power inside the
wedge-removed region:

| ordering | residual power |
|---|---|
| affine → wedge (correct) | 3–6% |
| wedge → affine (wrong) | 21–33% |

Through the **full** pipeline the separation is statistical rather than per-sample, because
the kernel blend deliberately reintroduces un-wedged content and the amplitude
randomization redistributes power. Over 20 phantoms: correct ordering medians **9.1%**
(range 1–31%), mis-ordered **24.3%** (range 2–41%). Compare medians over enough samples;
`--report` needs `-n 20` or more before it will offer a verdict.

**The exact ordering guarantee is asserted by
`test_fourier_degradation_runs_after_all_spatial_transforms`, which inspects the composed
pipeline directly.** Trust that over any runtime statistic.

A caution from building this: an initial orientation estimator scored *"no affine at all"*
higher than the correct pipeline, i.e. it was measuring its own noise. If you write a
geometric metric here, calibrate it against a deliberately mis-ordered pipeline before
believing it.

**Status:** (a) and (c) confirmed visually on synthetic phantoms; (b) confirmed numerically
as corrected above. **Still to do: rerun on real chunks** — there was no chunk data in the
tree. `python scripts/dump_augmented_chunks.py --chunks-dir <chunks_dir>/train --report --png`.

---

## Gate F — GPU box only

Push the branch and install on the GPU box. `membrain-seg/` will not come with it, so Gate D
test 1 will skip there — expected: **45 passed, 26 skipped**, confirmed locally by moving the
reference checkout aside.

### Settle these before starting

1. **Dependency pin mismatch.** The local env was repaired during Gate B (see below) and now
   runs `scipy 1.17.1` / `numpy 2.4.6`, but `pyproject.toml` still pins `scipy==1.14.1` and
   `numpy==1.26.4`. Gates B–E passed under the *upgraded* versions; the GPU box will install
   the *pinned* ones. Reconcile the two rather than discovering the divergence there.
2. **`config.yaml` will not load** until `infer.predictions_coord_format` is fixed — see the
   Gate C section.
3. `pytest` is a `dev` extra now: install with `pip install -e '.[dev]'`.

**Local env repair, for the record.** The `tomocpt` conda env's scipy could not import at
all: `libgfortran5 13.2.0` shipped `libgfortran.5.dylib` with the rpath `@loader_path`
listed twice, which recent macOS dyld rejects outright. `--force-reinstall` reproduced the
same broken build; upgrading to `libgfortran5 16.2.0` fixed it, pulling `scipy`,
`libopenblas`, `libgcc` and a `numpy` relink along with it. That is the origin of the pin
mismatch in item 1, and it is a macOS-only problem — it should not affect the GPU box.

1. `pytest tests/ -v` — everything except the port-equivalence test.
2. Short training run with augmentation **off** — confirm the loss curve matches a
   pre-change baseline, proving the refactor is behaviour-preserving. Defaults were verified
   transform-wise identical to the old pipeline in Gate C, so any divergence here is a real
   finding.
3. Short training run with `use_mw_aug=True, use_fourier_aug=True, missing_wedge_prob=1.0` —
   confirm it starts, loss decreases, step time has not regressed.
4. Run `tomocpt train --help` and confirm the `augmentation__*` options appear. This could
   not be exercised locally: `tomocpt.main` imports `dask`, which is absent from the dev
   env. The options were verified to surface by driving
   `ConfigManager._process_nested_dataclass` directly (35 of them), but the end-to-end CLI
   path is untested.

A 64³ FFT in CPU dataloader workers should be negligible. **Measure rather than assume**:
the reference rebuilds `wedge_mask` and the rotational kernel on *every single call*. If
step time regresses, caching those (keyed on shape and angle) is the first fix —
deliberately not done in Gate B so that Gate D1 compares against unmodified logic.

**Gate:** the augmented run trains stably for a few hundred steps with no throughput cliff.

---

## Deferred decisions

Deliberately left open, each pinned by a test that will fail if it is changed silently:

| Decision | Current state | Where |
|---|---|---|
| Wedge boundary taper | `taper_width=0`, exact reference behaviour | Gate B; D1 is green, so it can be enabled now |
| DC coefficient | Removed, matching the reference | Gate D3 |
| Friedel asymmetry / ~4.9% imaginary residual | Not fixed, to keep D1 valid | Gate B, D4 |
| `infer.predictions_coord_format` | Invalid enum value, unfixed | Gate C |
| `scipy` / `numpy` pins | Mismatched against the repaired env | Gate F |
| Intensity-augmentation ranges | membrain-seg values, exceed ±3, all off by default | Gate C |
