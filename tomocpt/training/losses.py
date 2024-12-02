import torch
import torch.nn as nn
from torch.nn.functional import softmax, one_hot
import kornia

#A list of losses can be found at https://arxiv.org/pdf/2006.14822.pdf

def dice_loss(sigmoidOut, ytrue, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        sigmoidOut: a tensor of shape [B, 1, H, W, D].
        ytrue: a tensor of shape [B, 1, H, W, D].
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = sigmoidOut.shape[1]
    ytrue = ytrue.to(torch.int64)

    if num_classes == 1:
        diagonal = torch.eye(num_classes + 1).to(ytrue.device)
        true_1_hot = diagonal[ytrue.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, -1, 1, 2, 3).float()
        true_1_hot_f = true_1_hot[:, 0:1, ...]
        true_1_hot_s = true_1_hot[:, 1:2, ...]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        neg_prob = 1 - sigmoidOut
        probas = torch.cat([sigmoidOut, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(ytrue.device)[ytrue.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, -1, 1, 2, 3).float()
        probas = sigmoidOut
    true_1_hot = true_1_hot.type(sigmoidOut.type())
    dims = (0,) + tuple(range(2, ytrue.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def gradient3d_loss(ypred, ytrue, order=2):
    ypred = kornia.filters.spatial_gradient3d(ypred, mode="diff", order=order)
    ytrue = kornia.filters.spatial_gradient3d(ytrue, mode="diff", order=order)
    return nn.functional.huber_loss(ypred, ytrue, reduction="none")

def get_one_hot(y_true, n_classes):
    y_true = y_true.to(torch.int64)
    y_true = one_hot(y_true, num_classes=n_classes)
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    y_true = y_true.to(torch.int8)
    return y_true


def boundary_loss(y_pred, y_true, dtm, smooth=1e-6, class_weights=None):
        # Compute region based loss

        # Prepare inputs
        y_true = get_one_hot(y_true, 1) #y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        if class_weights is None:
            class_weights = torch.sum(y_true, dim=(-3,-2-1))
            class_weights = 1. / (torch.square(class_weights) + 1.)
        else:
            class_weights = class_weights

        # Compute boundary loss
        # Flip each one-hot encoded class
        y_worst = torch.square(1.0 - y_true)

        num = torch.sum(torch.square(dtm * (y_worst - y_pred)), axis=(-3,-2,-1))
        num *= class_weights

        den = torch.sum(torch.square(dtm * (y_worst - y_true)), axis=(-3,-2,-1))
        den *= class_weights
        den += smooth

        boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        boundary_loss = torch.mean(boundary_loss)
        boundary_loss = 1. - boundary_loss

        return boundary_loss