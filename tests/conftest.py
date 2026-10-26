"""Shared fixtures for the tomocpt test suite.

The most valuable fixture here is :func:`membrain_ref`, which loads membrain-seg's
original implementations so the vendored port can be compared against them
directly. That comparison is only possible while both copies coexist in the
working tree: ``membrain-seg/`` is gitignored and will not exist on the GPU box,
where these tests skip cleanly.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMBRAIN_ROOT = REPO_ROOT / "membrain-seg" / "src" / "membrain_seg"

#: Skip marker for tests that need the membrain-seg reference checkout.
requires_membrain = pytest.mark.skipif(
    not MEMBRAIN_ROOT.is_dir(),
    reason="membrain-seg/ reference checkout not present (expected on the GPU box)",
)

CHUNK_SHAPE = (64, 64, 64)


def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Import a single source file as a module, bypassing its package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def membrain_ref():
    """membrain-seg's original ``filter_utils`` and ``fourier_augmentations``.

    The reference files are loaded directly by path rather than imported as
    ``membrain_seg.*``. Importing the package proper executes
    ``membrain_seg/tomo_preprocessing/__init__.py``, which pulls in SimpleITK and
    the rest of membrain-seg's dependency tree; none of that is needed to compare
    a pair of pure-numpy functions, and requiring it would make the comparison
    impossible to run.

    The two intra-package imports the reference module performs are satisfied
    with the genuine sources (``rotational_kernel``) or a minimal stub
    (``sample_scalar``), so the ported functions themselves are the real thing.
    """
    if not MEMBRAIN_ROOT.is_dir():
        pytest.skip("membrain-seg/ reference checkout not present")

    ref_filter_utils = _load_module_from_path(
        "_membrain_ref_filter_utils",
        MEMBRAIN_ROOT / "tomo_preprocessing" / "matching_utils" / "filter_utils.py",
    )

    saved = {k: v for k, v in sys.modules.items() if k.startswith("membrain_seg")}
    try:
        for name in [
            "membrain_seg",
            "membrain_seg.segmentation",
            "membrain_seg.segmentation.dataloading",
            "membrain_seg.tomo_preprocessing",
            "membrain_seg.tomo_preprocessing.matching_utils",
        ]:
            sys.modules[name] = types.ModuleType(name)

        transforms_stub = types.ModuleType(
            "membrain_seg.segmentation.dataloading.transforms"
        )

        def sample_scalar(value, *args):
            if isinstance(value, (tuple, list)):
                return np.random.uniform(value[0], value[1])
            elif callable(value):
                return value(*args)
            return value

        transforms_stub.sample_scalar = sample_scalar
        sys.modules["membrain_seg.segmentation.dataloading.transforms"] = transforms_stub
        sys.modules[
            "membrain_seg.tomo_preprocessing.matching_utils.filter_utils"
        ] = ref_filter_utils

        ref_fourier = _load_module_from_path(
            "_membrain_ref_fourier",
            MEMBRAIN_ROOT / "segmentation" / "dataloading" / "fourier_augmentations.py",
        )
    finally:
        for name in list(sys.modules):
            if name.startswith("membrain_seg"):
                del sys.modules[name]
        sys.modules.update(saved)

    return types.SimpleNamespace(
        wedge_mask=ref_fourier.wedge_mask,
        get_line_plot=ref_fourier.get_line_plot,
        rotational_kernel=ref_filter_utils.rotational_kernel,
        hypotenuse_ndim=ref_filter_utils.hypotenuse_ndim,
    )


@pytest.fixture
def rng():
    """A seeded numpy generator, so failures are reproducible."""
    return np.random.default_rng(20240924)


@pytest.fixture
def image_tensor(rng):
    """A ``(1, 64, 64, 64)`` float32 chunk in tomocpt's input distribution.

    Zero-centred and hard-clipped at +/- 3, matching ``robust_normalization``.
    """
    data = rng.standard_normal(CHUNK_SHAPE).astype(np.float32)
    data = np.clip(data, -3.0, 3.0)
    return torch.from_numpy(data[None])


@pytest.fixture
def label_tensor():
    """A ``(1, 64, 64, 64)`` float32 target.

    Continuous Gaussian spheres in ``[0, 1]``, per
    :func:`tomocpt.labels.helpers.generate_gaussian_sphere` - a regression
    target, despite being carried in a :class:`torchio.LabelMap`. Nothing here
    may assume nearest-neighbour label semantics.
    """
    zz, yy, xx = np.mgrid[0:64, 0:64, 0:64].astype(np.float32)
    label = np.zeros(CHUNK_SHAPE, dtype=np.float32)
    for cz, cy, cx in [(20, 20, 20), (44, 30, 40), (32, 50, 18)]:
        r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        label = np.maximum(label, np.exp(-r2 / (2 * 4.0**2)))
    return torch.from_numpy(label[None])


@pytest.fixture
def subject(image_tensor, label_tensor):
    """A fresh :class:`torchio.Subject` shaped like one training sample."""
    return tio.Subject(
        input_data=tio.ScalarImage(tensor=image_tensor.clone()),
        target_data=tio.LabelMap(tensor=label_tensor.clone()),
    )
