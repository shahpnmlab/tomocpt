#!/usr/bin/env python
"""Dump augmented chunks to .nii.gz for visual inspection.

Gate E of the Fourier-augmentation port. Applies the training pipeline with
``missing_wedge_prob=1.0`` to a handful of chunks and writes the original and
augmented volumes side by side, so the augmentation can be checked by eye in a
viewer.

What to look for, opening ``sampleNN_augmented.nii.gz`` next to
``sampleNN_original.nii.gz``:

(a) **Directional smearing** in a central slice - structure stretched along one
    axis, the signature of the zeroed Fourier wedge.
(b) **The same smearing axis in every sample, with the specimen rotated
    differently underneath it.** This is worth stating carefully, because the
    intuitive expectation is the opposite. The wedge is applied in the array's
    own frame at a fixed ``keep_angle`` of 45 degrees, so the smearing is always
    along the same array axis. What the preceding affine varies is the
    *specimen's* orientation relative to that fixed wedge - which is exactly the
    point of the augmentation, and is visible as differently-oriented structure
    inside a consistently-oriented smear.

    A smearing direction that visibly *rotates* between samples is the failure
    signal: it means the affine is running after the wedge and dragging the
    wedge around with it. ``--report`` measures this directly; see
    :func:`wedge_residual_power`.
(c) **A soft boundary** between degraded and clean regions, from the Gaussian
    kernel blend. The degradation covers a blob, not the whole subvolume.

``--report`` additionally measures (b) and (c) numerically, which is useful when
no viewer is to hand and as a regression check.

Examples
--------
::

    python scripts/dump_augmented_chunks.py --chunks-dir /path/to/chunks/train
    python scripts/dump_augmented_chunks.py --synthetic --report
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchio as tio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tomocpt.dataManager.dataloading import VolumeDatsetIO, build_training_transforms
from tomocpt.dataManager.fourier_augmentations import wedge_mask
from tomocpt.defaultConfigs.train_config import AugmentationConfig


def make_synthetic_chunk(seed: int, shape=(64, 64, 64)) -> torch.Tensor:
    """Build a phantom with enough structure to show directional smearing.

    Real chunks are preferable; this exists so the script runs at all on a
    machine with no data staged, and so the numeric report has something with a
    broad, isotropic spectrum to work on.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)

    volume = rng.standard_normal(shape).astype(np.float32) * 0.3
    for _ in range(25):
        cz, cy, cx = rng.integers(8, min(shape) - 8, size=3)
        radius = rng.uniform(2.5, 5.0)
        r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume += 2.0 * np.exp(-r2 / (2 * radius**2)).astype(np.float32)

    volume -= volume.mean()
    std = volume.std()
    if std > 0:
        volume /= std
    return torch.from_numpy(np.clip(volume, -3.0, 3.0)[None])


def load_chunks(chunks_dir: Path, n: int, return_labels: bool):
    """Load up to ``n`` real chunk/label pairs from a prepared chunks directory."""
    pairs = VolumeDatsetIO._get_filepaths(str(chunks_dir), return_labels)
    if not pairs:
        raise FileNotFoundError(
            f"No chunks found under {chunks_dir}. Point --chunks-dir at a "
            f"prepared split (e.g. <chunks_dir>/train), or pass --synthetic."
        )
    for vol_path, label_path in pairs[:n]:
        subject = tio.Subject(
            input_data=tio.ScalarImage(vol_path),
            target_data=tio.LabelMap(label_path),
        )
        yield subject, Path(vol_path).stem


def wedge_residual_power(augmented: np.ndarray, keep: np.ndarray) -> float:
    """Fraction of output power still sitting inside the removed wedge.

    This is the reliable ordering check. The wedge is axis-aligned in the
    array's frame, so if it was the last geometric operation applied, the region
    it zeroed stays empty and this number is small - a few percent, left over
    from the kernel blend, the renormalization and float round-trip.

    If a spatial transform runs *after* the wedge, it rotates the wedge away
    from its axis-aligned position and interpolation fills the gap back in, so
    this number rises sharply.

    Read it as a population statistic, not a per-sample test. In isolation
    (wedge only, no blend, no amplitude randomization) the separation is clean:
    ~3-6% correct versus ~21-33% mis-ordered. Through the full pipeline the
    kernel blend deliberately reintroduces un-wedged content and the amplitude
    randomization redistributes power, so per-sample values overlap - measured
    over 20 phantoms, correct ordering gives a median of 9% (range 1-31%) and
    mis-ordered gives 24% (range 2-41%). Compare medians over enough samples.

    The *exact* ordering guarantee is asserted structurally by
    ``test_fourier_degradation_runs_after_all_spatial_transforms`` in the test
    suite; this is a runtime sanity check on top of it.

    Parameters
    ----------
    augmented : np.ndarray
        The augmented volume.
    keep : np.ndarray
        The keep mask the transform applies, from :func:`wedge_mask`.

    Returns
    -------
    float
        Residual power fraction inside the removed region.
    """
    power = np.abs(np.fft.fftshift(np.fft.fftn(augmented))) ** 2
    total = power.sum()
    if total <= 0:
        return float("nan")
    return float(power[~keep].sum() / total)


def blend_locality(original: np.ndarray, augmented: np.ndarray) -> float:
    """Ratio of most-changed to least-changed region, as weak evidence for (c).

    A ratio of exactly 1 would mean the degradation is perfectly uniform, i.e.
    no blend kernel at all. Anything above that means the change is graded
    across the subvolume.

    Expect a *modest* number, not a dramatic one. The production ``scale``
    deliberately yields a kernel comparable to or larger than the patch, so the
    blob covers most of a 64^3 chunk with a gentle falloff rather than forming a
    tight spot. Structureless input depresses the ratio further, since there is
    little signal for the wedge to remove unevenly. The eye is much better at
    spotting the soft boundary than this statistic is - treat a low value as a
    prompt to look at the montage, not as a failure.
    """
    difference = np.abs(augmented - original)
    blocks = difference.reshape(4, 16, 4, 16, 4, 16).mean(axis=(1, 3, 5))
    lo = max(float(blocks.min()), 1e-8)
    return float(blocks.max()) / lo


def write_montage(samples, path: Path) -> None:
    """Write central slices, original above augmented, for checking (a) and (c)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 7.4))
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(3, 1)

    mid = samples[0][1].shape[0] // 2
    for col, (stem, original, augmented) in enumerate(samples):
        difference = np.abs(augmented - original)
        for row, (image, label, cmap) in enumerate(
            [
                (original[mid], "original", "gray"),
                (augmented[mid], "augmented", "gray"),
                (difference[mid], "|difference|", "magma"),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(image, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0:
                ax.set_title(stem.split("_")[0], fontsize=8)
    fig.suptitle(
        "(a) smearing in row 2   (b) SAME smear axis per column, specimen rotated "
        "under it   (c) soft blob boundary in row 3",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks-dir", type=Path, help="Prepared chunks split, e.g. <chunks_dir>/train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic phantoms instead of real chunks")
    parser.add_argument("--output-dir", type=Path, default=Path("augmentation_dump"))
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taper-width", type=float, default=0.0, help="Wedge boundary taper in degrees")
    parser.add_argument("--keep-angle", type=float, default=45.0, help="Half-opening angle of the retained wedge")
    parser.add_argument(
        "--contrast-inversion-p",
        type=float,
        default=0.0,
        help="Probability of flipping contrast polarity; 0.5 gives an even mix",
    )
    parser.add_argument("--no-labels", action="store_true", help="Load selfSup labels instead of supervised")
    parser.add_argument("--report", action="store_true", help="Also print numeric measurements of (a)-(c)")
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write montage.png with central slices side by side, for checking (a) without a viewer",
    )
    args = parser.parse_args()

    if not args.chunks_dir and not args.synthetic:
        parser.error("pass --chunks-dir, or --synthetic to run without data")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = AugmentationConfig(
        use_mw_aug=True,
        use_fourier_aug=True,
        missing_wedge_prob=1.0,
        amplitude_prob=1.0,
        taper_width=args.taper_width,
        keep_angle_min=args.keep_angle,
        keep_angle_max=args.keep_angle,
        contrast_inversion_p=args.contrast_inversion_p,
    )
    pipeline = build_training_transforms(cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        sources = (
            (
                tio.Subject(
                    input_data=tio.ScalarImage(tensor=make_synthetic_chunk(args.seed + i)),
                    target_data=tio.LabelMap(tensor=torch.zeros(1, 64, 64, 64)),
                ),
                f"synthetic{i:02d}",
            )
            for i in range(args.num_samples)
        )
    else:
        sources = load_chunks(args.chunks_dir, args.num_samples, not args.no_labels)

    rows = []
    montage = []
    for index, (subject, name) in enumerate(sources):
        original = subject["input_data"].data.clone()
        augmented_subject = pipeline(subject)
        augmented = augmented_subject["input_data"].data

        stem = f"sample{index:02d}_{name}"
        tio.ScalarImage(tensor=original).save(args.output_dir / f"{stem}_original.nii.gz")
        tio.ScalarImage(tensor=augmented).save(args.output_dir / f"{stem}_augmented.nii.gz")
        tio.ScalarImage(tensor=augmented_subject["target_data"].data).save(
            args.output_dir / f"{stem}_label.nii.gz"
        )

        orig_np = original[0].numpy()
        aug_np = augmented[0].numpy()
        if args.png:
            montage.append((stem, orig_np, aug_np))
        if args.report:
            keep = wedge_mask(orig_np.shape, args.keep_angle, taper_width=args.taper_width)
            rows.append(
                (
                    stem,
                    wedge_residual_power(aug_np, keep),
                    blend_locality(orig_np, aug_np),
                )
            )

    print(f"Wrote {args.output_dir.resolve()}")

    if args.png and montage:
        write_montage(montage, args.output_dir / "montage.png")
        print(f"Wrote {(args.output_dir / 'montage.png').resolve()}")

    if args.report and rows:
        print()
        print(f"{'sample':<30}{'wedge residual':>16}{'locality':>11}")
        for stem, residual, locality in rows:
            print(f"{stem:<30}{residual:15.1%}{locality:11.1f}")

        median = float(np.median([row[1] for row in rows]))
        if len(rows) < 10:
            verdict = "inconclusive - use -n 20 or more; this is a population statistic"
        elif median < 0.15:
            verdict = "consistent with the wedge running after every spatial transform"
        else:
            verdict = "HIGH - is the wedge running BEFORE the affine? (see --help)"
        print()
        print(f"(b) median wedge residual power: {median:.1%}  -> {verdict}")
        print("    exact ordering is asserted by the pytest suite, not by this number")

        localities = [row[2] for row in rows]
        verdict = (
            "graded"
            if max(localities) > 1.15
            else "near-uniform - check the montage before concluding anything"
        )
        print(
            f"(c) max blend locality ratio: {max(localities):.1f}x  -> {verdict} "
            f"(broad by design; confirm the soft boundary by eye)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
