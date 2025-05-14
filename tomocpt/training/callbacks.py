import pytorch_lightning as pl
import logging

class LRVerificationCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        logging.info("=== Verifying Actual Optimizer Configuration ===")
        for i, optimizer in enumerate(trainer.optimizers):
            logging.info(f"Optimizer {i}: {optimizer.__class__.__name__}")
            for j, group in enumerate(optimizer.param_groups):
                params_count = sum(p.numel() for p in group['params'])
                logging.info(f"  Group {j}: {params_count} parameters, lr={group['lr']}")
