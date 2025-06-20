import pytorch_lightning as pl
import logging

class LRVerificationCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        # The key change: Do nothing if this is not the main process
        if not trainer.is_global_zero:
            return
        logging.info("=== Verifying Actual Optimizer Configuration ===")
        for i, optimizer in enumerate(trainer.optimizers):
            logging.info(f"Optimizer {i}: {optimizer.__class__.__name__}")
