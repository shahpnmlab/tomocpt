import pytorch_lightning as pl
import logging

class LRVerificationCallback(pl.Callback):
    """
    Verifies and logs the learning rates for each parameter group in the optimizer(s)
    at the start of training. This will only log from the main process.
    """
    def on_train_start(self, trainer, pl_module):
        # Do nothing if this is not the main process
        if not trainer.is_global_zero:
            return

        logging.info("="*40)
        logging.info("   Verifying Final Optimizer Configuration   ")
        logging.info("="*40)
        
        # Check if optimizers have been configured
        if not trainer.optimizers:
            logging.warning("No optimizers found in trainer.")
            return

        for i, optimizer in enumerate(trainer.optimizers):
            logging.info(f"--> Optimizer {i}: {optimizer.__class__.__name__}")
            
            # Check for parameter groups
            if not optimizer.param_groups:
                logging.warning(f"  Optimizer {i} has no parameter groups.")
                continue

            for j, param_group in enumerate(optimizer.param_groups):
                # Count parameters in this group
                param_count = sum(p.numel() for p in param_group['params'] if p.requires_grad)
                # Get the learning rate for this group
                lr = param_group['lr']
                logging.info(f"  - Group {j}: {param_count:,} parameters, Learning Rate = {lr:.2e}")
        logging.info("="*40)


from pytorch_lightning.callbacks import EarlyStopping as _EarlyStopping


class RelativeEarlyStopping(_EarlyStopping):
    """Early stopping (mode ``min``) whose improvement test is relative (Gate 4h): an epoch counts
    as progress only if ``current < best * (1 - rel_delta)``, so the threshold follows the loss
    scale. Lightning's state dict is unchanged (resume restores the best value and wait count)."""

    def __init__(self, monitor: str, rel_delta: float, patience: int = 10, verbose: bool = False, **kwargs):
        if not 0.0 < float(rel_delta) < 1.0:
            raise ValueError(f"rel_delta must be in (0, 1), got {rel_delta}")
        super().__init__(monitor=monitor, min_delta=0.0, patience=patience, verbose=verbose, mode="min", **kwargs)
        self.rel_delta = float(rel_delta)

    def _evaluate_stopping_criteria(self, current):
        import torch

        best = self.best_score
        # Lightning tests ``current - min_delta < best``; min_delta is stored negated for mode min
        self.min_delta = -self.rel_delta * float(best) if torch.isfinite(best) else 0.0
        return super()._evaluate_stopping_criteria(current)
