"""TensorBoard chunk panels: gray input, target and prediction in turbo on a fixed [0, 1] scale."""
import torch
from matplotlib import colormaps

from tomocpt.networks.pickingModel import best_slice_panel, colour_panel


def test_colour_panel_turbo_tiles_and_gray_input():
    x = torch.rand(1, 1, 4, 10, 12)
    y = torch.zeros(1, 1, 4, 10, 12)
    y[0, 0, 2, 2:8, 2:8] = 1.0
    y_pred = torch.full_like(y, 0.25)
    panel = best_slice_panel(x, y, y_pred, 0)
    rgb = colour_panel(panel)
    w = 12
    assert rgb.shape == (3, 10, 3 * w + 2)
    left = rgb[:, :, :w]
    assert torch.equal(left[0], left[1]) and torch.equal(left[1], left[2])  # gray input
    turbo = lambda v: torch.tensor(colormaps["turbo"](round(v * 255))[:3]).float()
    assert torch.allclose(rgb[:, 5, w + 1 + 5], turbo(1.0))      # target 1
    assert torch.allclose(rgb[:, 0, w + 1], turbo(0.0))          # target 0
    assert torch.allclose(rgb[:, 0, 2 * w + 2], turbo(0.25))     # prediction 0.25, unscaled
    assert torch.all(rgb[:, :, w] == 1.0) and torch.all(rgb[:, :, 2 * w + 1] == 1.0)  # white gaps
