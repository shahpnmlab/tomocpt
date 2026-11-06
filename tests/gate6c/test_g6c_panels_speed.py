"""Gate 6c addendum: chunk panels with signal, metric speed (spec 06c_AGENT_FILAMENT_DRIVER.md)."""

import time

import numpy as np
import pytest
import torch


def test_G6c_C7_best_slice_panel():
    from tomocpt.networks.pickingModel import MIN_PANEL_VOXELS, best_slice_panel

    x = torch.rand(3, 1, 8, 16, 20)
    y = torch.zeros(3, 2, 8, 16, 20)
    y_pred = torch.full_like(y, 0.25)
    y[2, 1, 5, 2:12, 3:9] = 0.9   # sample 2, class 1, slice 5
    y[1, 1, 1, 0:2, 0:2] = 0.9    # a smaller patch elsewhere
    panel = best_slice_panel(x, y, y_pred, 1)
    assert panel.ndim == 3 and panel.shape[0] == 1
    assert panel.shape[1] == 16 and panel.shape[2] >= 3 * 20
    # middle tile is the target of sample 2, slice 5, unscaled; right tile the prediction
    w = 20
    gap = (panel.shape[2] - 3 * w) // 2
    target = panel[0, :, w + gap:2 * w + gap]
    assert torch.equal(target, y[2, 1, 5].float())
    pred = panel[0, :, 2 * (w + gap):]
    assert torch.allclose(pred, torch.full_like(pred, 0.25))
    left = panel[0, :, :w]
    assert left.min() >= 0 and left.max() <= 1
    assert best_slice_panel(x, y, y_pred, 0) is None
    tiny = torch.zeros_like(y)
    tiny[0, 0, 0, 0, :MIN_PANEL_VOXELS - 1] = 1.0
    assert best_slice_panel(x, tiny, y_pred, 0) is None


def test_G6c_C8_metric_speed_and_values():
    from tomocpt.metrics import _point_to_polylines, centerline_scores

    rng = np.random.default_rng(1)
    true = [np.cumsum(rng.normal(0, 0.5, (201, 3)), axis=0) + rng.uniform(0, 400, 3) for _ in range(200)]
    pred = [t + rng.normal(0, 0.8, t.shape) for t in true]
    t0 = time.perf_counter()
    s = centerline_scores(pred, true, 2.0)
    assert time.perf_counter() - t0 < 5.0
    assert s["precision"] > 0.9 and s["recall"] > 0.9
    # exact on a sample of points (the full brute force is what this speeds up)
    from tomocpt.metrics import _within_tolerance

    pts = np.concatenate(pred)[::37]
    assert np.array_equal(_within_tolerance(pts, true, 0.3, 0.25), _point_to_polylines(pts, true) <= 0.3)
