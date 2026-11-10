"""Learning rate and EMA decay that follow the effective batch size (Gate 8g).

``train.lr_scaling``: ``none`` | ``sqrt`` | ``linear``; reference batch
``train.lr_reference_batch`` (the batch the recipe's lr was tuned at, 16).
``k = batch_size x devices / reference``:

- ``sqrt``: ``lr * sqrt(k)``. Krizhevsky (2014), "One weird trick for parallelizing convolutional
  neural networks"; Malladi et al. (2022), "On the SDEs and Scaling Rules for Adaptive Gradient
  Algorithms" (the rule for Adam-family optimisers such as RAdam).
- ``linear``: ``lr * k``. Goyal et al. (2017), "Accurate, Large Minibatch SGD" (derived for SGD,
  used there with a warm-up).
- EMA decay (both rules): ``decay ** k``, Busbridge et al. (2023), "How to Scale Your EMA": the
  average then spans the same number of samples.

Epochs are not changed (Goyal et al. keep the epoch budget). ``k = 1`` returns the inputs unchanged.
"""
import logging
import math

RULES = ("none", "sqrt", "linear")


def _rule(rule) -> str:
    r = str(getattr(rule, "value", rule))
    if r not in RULES:
        raise ValueError(f"train.lr_scaling must be one of {RULES}, got '{r}'")
    return r


def scale_factor(batch_size: int, devices: int, reference_batch: int) -> float:
    """``k`` = effective batch (``batch_size x devices``) / ``reference_batch``."""
    if int(reference_batch) <= 0 or int(batch_size) <= 0 or int(devices) <= 0:
        raise ValueError("batch_size, devices and lr_reference_batch must be positive")
    return int(batch_size) * int(devices) / int(reference_batch)


def scaled_lr(lr: float, k: float, rule) -> float:
    r = _rule(rule)
    if r == "none" or k == 1.0:
        return lr
    return float(lr) * (math.sqrt(k) if r == "sqrt" else k)


def scaled_ema_decay(decay: float, k: float, rule) -> float:
    r = _rule(rule)
    if r == "none" or k == 1.0:
        return decay
    return float(decay) ** k


def training_devices(config) -> int:
    """Processes the trainer will run (the ``devices`` of ``training.helpers._setup_trainer``)."""
    from tomocpt.utils import accelerator_selector

    return int(accelerator_selector(use_cuda=config.use_gpus, n_cpus=config.N_CPUS_IF_NO_GPU)[1])


def batch_scale(config):
    """``(k, rule)`` for a train config (``k = 1`` when scaling is off)."""
    from tomocpt.configManager.configManager import safe_getattr

    rule = _rule(safe_getattr(config, "lr_scaling", "none"))
    if rule == "none":
        return 1.0, rule
    k = scale_factor(config.batch_size, training_devices(config), int(safe_getattr(config, "lr_reference_batch", 16)))
    return k, rule


def config_lr(config) -> float:
    """``train.optimizer.lr`` scaled for the effective batch (logged when it changes)."""
    k, rule = batch_scale(config)
    lr = scaled_lr(config.optimizer.lr, k, rule)
    if lr != config.optimizer.lr:
        logging.info(f"lr_scaling={rule}: effective batch / reference = {k:g}, lr {config.optimizer.lr:g} -> {lr:g}")
    return lr


def config_ema_decay(config, decay: float) -> float:
    k, rule = batch_scale(config)
    out = scaled_ema_decay(decay, k, rule)
    if out != decay:
        logging.info(f"lr_scaling={rule}: EMA decay {decay:g} -> {out:.6g} (k = {k:g})")
    return out
