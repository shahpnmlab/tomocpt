import torch
import torch.nn as nn
import kornia

def gradient3d_loss(ypred, ytrue, order=2):
    ypred = kornia.filters.spatial_gradient3d(ypred, mode="diff", order=order)
    ytrue = kornia.filters.spatial_gradient3d(ytrue, mode="diff", order=order)
    return nn.functional.huber_loss(ypred, ytrue, reduction="none")

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Calculates the knowledge distillation loss for sigmoid-based outputs.
    This uses the Kullback-Leibler divergence between the softened probability
    distributions of the student and teacher.

    Args:
        student_logits: Raw outputs (logits) from the student model.
        teacher_logits: Raw outputs (logits) from the teacher model.
        temperature: A value to soften the probability distributions. Higher
                     values result in softer distributions.

    Returns:
        The distillation loss.
    """
    # The standard distillation loss is T^2 * KL(p_teacher_soft || p_student_soft).
    # For sigmoid outputs, this is equivalent to binary cross-entropy between the
    # student's soft predictions and the teacher's soft targets.
    soft_teacher_targets = torch.sigmoid(teacher_logits / temperature)

    # We use BCEWithLogitsLoss for numerical stability, which takes logits as input.
    # The student input is student_logits / temperature.
    loss = nn.functional.binary_cross_entropy_with_logits(
        student_logits / temperature,
        soft_teacher_targets,
        reduction='mean'
    )

    # Scale by T^2, as in the original paper, to preserve the gradient magnitude.
    return loss * (temperature ** 2)