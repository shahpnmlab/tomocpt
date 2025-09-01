import torch
import torch.nn as nn
import kornia

def gradient3d_loss(ypred, ytrue, order=2):
    ypred = kornia.filters.spatial_gradient3d(ypred, mode="diff", order=order)
    ytrue = kornia.filters.spatial_gradient3d(ytrue, mode="diff", order=order)
    return nn.functional.huber_loss(ypred, ytrue, reduction="none")

