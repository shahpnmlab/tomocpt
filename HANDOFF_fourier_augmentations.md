# HANDOFF: Port membrain-seg Fourier augmentations into tomocpt

**Branch:** `feat/fourier-augmentations` (branched from `main`)
**Status:** Gate 0 complete. Gates A–F outstanding.

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

**Gate:** `git status` shows `membrain-seg/` ignored, not untracked. ✅

---

## Gate A — Vendor the filter utilities

`rotational_kernel` and its dependency `hypotenuse_ndim` from
`tomo_preprocessing/matching_utils/filter_utils.py` are required by the amplitude
augmentation. These are DeePiCt-derived, **Apache-2.0**, and carry a copyright header.

Create `tomocpt/dataManager/filter_utils.py` containing `hypotenuse_ndim` and
`rotational_kernel` copied verbatim, **with the original copyright header preserved
intact**. Do not copy `radial_average` — the spectrum here is synthesized, not measured.
Add an attribution note to the repo's license documentation.

**Gate:**

```bash
python -c "import tomocpt.dataManager.filter_utils, sys; assert not [m for m in sys.modules if 'membrain' in m]; print('ok')"
```

---

## Gate B — Port the Fourier transform

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
- **Pin the angle convention.** The parameter is named `missing_angle`, but the `wedge_mask`
  docstring defines it as *angles to keep* (90 keeps everything, 0 keeps nothing), and
  production passes `(45, 45)`. Do not guess — establish the true semantics empirically in
  Gate D3 and give the tomocpt parameter an unambiguous name.

**Gate:** module imports; `RandomFourierDegradation()(subject)` runs on a synthetic
`tio.Subject` without error and returns unchanged shape and dtype.

---

## Gate C — Intensity transforms, config, plumbing

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

**Gate:**

```bash
python -c "from tomocpt.mainConfig import mainConfig; print(mainConfig.train.augmentation)"
```

prints the new block; and with all new probabilities at 0 the composed pipeline is
transform-wise identical to today's.

---

## Gate D — Tests (run on the M1; no GPU needed)

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
4. **Real and finite** — no NaN/Inf; imaginary residual below float32 tolerance.
5. **Label untouched** — run the full `tio.Compose` on a `tio.Subject` and assert the
   `LabelMap` tensor is bit-identical before and after. This is the paper's core requirement
   and the thing most likely to break silently during the TorchIO restructure.
6. **No input mutation** — the input tensor is unchanged after the call (guards the in-place
   bug from Gate B).
7. **Shape/dtype invariance** — `(1,64,64,64)` float32 in, same out, for every transform.
8. **Determinism under seed** — same seed gives the same output twice.

**Gate:** `pytest tests/ -v` green on the M1, **including test 1**. Do not proceed to Gate E
with test 1 failing or skipped locally — it is the whole basis for trusting the port.

---

## Gate E — Visual check (M1, no GPU needed)

Script at `scripts/dump_augmented_chunks.py`: load a few real chunks, apply the pipeline with
`missing_wedge_prob=1.0`, save via `tio.ScalarImage(tensor=...).save(...)` to `.nii.gz`.

**Gate:** open in a viewer and confirm:

- (a) directional smearing in a central slice;
- (b) the smearing direction **differs between samples**, because the affine rotated the
  volume first;
- (c) a visible soft boundary between degraded and clean regions from the kernel blend.

If the direction is identical across samples, the transform is running before the affine —
re-check the Gate C3 ordering.

---

## Gate F — GPU box only

Push the branch and install on the GPU box. `membrain-seg/` will not come with it, so Gate D
test 1 will skip there — expected.

1. `pytest tests/ -v` — everything except the port-equivalence test.
2. Short training run with augmentation **off** — confirm the loss curve matches a
   pre-change baseline, proving the refactor is behaviour-preserving.
3. Short training run with `use_mw_aug=True, use_fourier_aug=True, missing_wedge_prob=1.0` —
   confirm it starts, loss decreases, step time has not regressed.

A 64³ FFT in CPU dataloader workers should be negligible. **Measure rather than assume**:
the reference rebuilds `wedge_mask` and the rotational kernel on *every single call*. If
step time regresses, caching those (keyed on shape and angle) is the first fix —
deliberately not done in Gate B so that Gate D1 compares against unmodified logic.

**Gate:** the augmented run trains stably for a few hundred steps with no throughput cliff.
