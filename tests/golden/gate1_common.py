"""Inputs and serialisers shared by ``make_goldens_gate1.py`` and ``tests/gate1``.

Pure numpy / stdlib (no ``tomocpt`` import), so the generator running in the
``v2-gate-0-base`` worktree and the tests running on HEAD build bit-identical inputs.
Loaded by file path (see ``tests/gate1/conftest.py``).
"""

import numpy as np

SPHERE_RADII = (5, 10, 17)
CHUNK = 64

#: (sample, radius, (z, y, x) box corner) for the loss-golden target.
LOSS_SPHERES = (
    (0, 5, (14, 14, 14)),
    (0, 10, (30, 28, 36)),
    (0, 17, (10, 26, 2)),
    (1, 17, (16, 16, 16)),
    (1, 5, (50, 4, 40)),
)


def loss_inputs(spheres):
    """``(y_pred, y_true)`` float32 arrays of shape ``[2, 1, 64, 64, 64]``.

    ``spheres`` maps radius -> ``generate_gaussian_sphere(R, 2R+1)`` (the G1-C1 golden).
    ``y_pred`` is ``sigmoid`` of seeded normal logits (``RandomState(0)``).
    """
    logits = np.random.RandomState(0).standard_normal((2, 1, CHUNK, CHUNK, CHUNK))
    y_pred = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    y_true = np.zeros((2, 1, CHUNK, CHUNK, CHUNK), dtype=np.float32)
    for b, r, (z, y, x) in LOSS_SPHERES:
        s = np.asarray(spheres[r], dtype=np.float32)
        n = s.shape[0]
        view = y_true[b, 0, z:z + n, y:y + n, x:x + n]
        np.maximum(view, s, out=view)
    return y_pred, y_true


def forward_input():
    """Fixed ``[1, 1, 64, 64, 64]`` float32 input for the forward golden (``RandomState(1)``)."""
    return np.random.RandomState(1).standard_normal((1, 1, CHUNK, CHUNK, CHUNK)).astype(np.float32)


_SIMPLE = (bool, int, float, str, type(None))


def _simple(v):
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (tuple, list)):
        return all(_simple(i) for i in v)
    return False


def _norm(v):
    if isinstance(v, (tuple, list)):
        return [_norm(i) for i in v]
    if isinstance(v, float) and v.is_integer():
        return v  # keep floats as floats; json keeps 1.0 vs 1 apart only by type
    return v


def describe_transform(t):
    """JSON-serialisable description of a TorchIO transform tree: class name, top-level
    package (``torchio`` / ``tomocpt``), every public attribute of simple type, and children
    (``Compose.transforms`` in order; ``OneOf.transforms_dict`` as ``[child, weight]`` pairs)."""
    cls = type(t)
    d = {
        "type": cls.__qualname__,
        "package": cls.__module__.split(".")[0],
        "params": {k: _norm(v) for k, v in sorted(vars(t).items())
                   if not k.startswith("_") and k not in ("transforms", "transforms_dict") and _simple(v)},
    }
    if hasattr(t, "transforms_dict"):
        d["choices"] = [[describe_transform(c), float(w)] for c, w in t.transforms_dict.items()]
    elif hasattr(t, "transforms"):
        d["children"] = [describe_transform(c) for c in t.transforms]
    return d
