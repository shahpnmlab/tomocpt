"""Shared by ``make_goldens_gate3b.py`` and ``tests/gate3b``: deterministic loss inputs for G3b-C1.

Each case is a pipeline-v2 loss call on a 32^3 cube: soft targets drawn with
``labels.helpers.render_targets`` (particles of different classes overlap), a sigmoid of a
seeded normal as ``y_pred``, partial availability and per-sample diameters.
"""

import numpy as np
import torch

SIZE = 32

#: name -> (seed, d_vox [B][C], class_available [B][C], class_weights or None)
CASES = {
    "c1": (11, [[8.0], [8.0], [12.0]], [[True], [False], [True]], None),
    "c4": (12, [[6.0, 9.0, 12.0, 16.0], [7.0, 10.0, 13.0, 15.0]],
           [[True, False, True, True], [True, True, False, True]], None),
    "c4w": (13, [[6.0, 9.0, 12.0, 16.0], [8.0, 8.0, 14.0, 14.0]],
            [[True, True, True, False], [False, True, True, True]], [1.0, 2.0, 0.5, 3.0]),
}


def case_inputs(name, n_per_class=3):
    """``(y_pred, y_true, class_available, class_d_vox, class_weights)`` (float32 / bool)."""
    from tomocpt.labels.helpers import render_targets

    seed, d_vox, avail, weights = CASES[name]
    rng = np.random.RandomState(seed)
    B, C = len(d_vox), len(d_vox[0])
    y_true = np.zeros((B, C, SIZE, SIZE, SIZE), dtype=np.float32)
    for b in range(B):
        pts = []
        for c in range(C):
            d = d_vox[b][c]
            for _ in range(n_per_class):
                z, y, x = rng.uniform(d / 4, SIZE - d / 4, size=3)
                pts.append([z, y, x, c, d])
        y_true[b] = render_targets(np.asarray(pts, dtype=np.float64), (SIZE,) * 3, C, "clip")
    g = torch.Generator().manual_seed(seed)
    y_pred = torch.sigmoid(torch.randn((B, C, SIZE, SIZE, SIZE), generator=g, dtype=torch.float32))
    cw = None if weights is None else torch.tensor(weights, dtype=torch.float32)
    return (y_pred, torch.from_numpy(y_true), torch.tensor(avail, dtype=torch.bool),
            torch.tensor(d_vox, dtype=torch.float32), cw)


def loss_and_grad(loss_fn, name, checkpoint, **kw):
    """Run ``loss_fn`` (``multiclass_picking_loss``) on case ``name``; ``(loss, grad of y_pred)``."""
    y_pred, y_true, avail, d_vox, cw = case_inputs(name)
    yp = y_pred.clone().requires_grad_(True)
    loss = loss_fn(yp, y_true, avail, d_vox, class_weights=cw, checkpoint=checkpoint, **kw)
    loss.backward()
    return float(loss.detach()), yp.grad.detach().clone()
