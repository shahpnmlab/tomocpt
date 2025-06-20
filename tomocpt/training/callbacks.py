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
