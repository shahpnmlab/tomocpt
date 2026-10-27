import importlib
import logging

import hydra
import pytorch_lightning as pl
import torch

from tomocpt import constants
from tomocpt.configManager.configManager import (
    backfill_missing_fields,
    safe_getattr,
    update_config,
    update_config_with_changed_values,
)
from tomocpt.defaultConfigs.train_config import TrainingModes
from tomocpt.mainConfig import mainConfig, _mainConfigNoChanges

from tomocpt.logger import get_logger

logging = get_logger()

class BaseModel(pl.LightningModule):

    def set_default_args(self, lr: float | None):
        self.lr = lr if lr is not None else self.hparams.config.train.optimizer.lr


    def __init__(self, lr: float | None = None,
             config=None,
             constants_dict=None,
             train_continue=None):
        super(BaseModel, self).__init__()
        if config is None:
            config = mainConfig
        else:
            # A checkpoint written before a config field existed unpickles without
            # that attribute, so fill in current defaults before walking the config.
            # Without this, loading an older checkpoint raises AttributeError.
            config = backfill_missing_fields(config, _mainConfigNoChanges)
            # We first take the config in the checkpoint, and overwrite changes that come from the CLI
            updated_checkpoint_config = update_config_with_changed_values(target=config, originaConfig=_mainConfigNoChanges, configAfterCli=mainConfig)
            # Then, inject the updated checkpoint config into the whole config system at train level
            update_config(mainConfig.train, source=updated_checkpoint_config.train) # We update all the training details according to the checkpoint
            config = updated_checkpoint_config
        if constants_dict is None:
            constants_dict = {k: getattr(constants, k) for k in dir(constants) if not k.startswith("__")}
        else:
            for k, v in constants_dict.items():
                setattr(constants, k, v)

        self.save_hyperparameters(
            dict(constants_dict=constants_dict,
                 config=config,
                 train_continue=train_continue
                 )
        )
        self.set_default_args(lr=lr)

        self.patch_size = self.hparams.config.train.CHUNK_SIZE
        self.desired_particle_pixel_size = self.hparams.config.prepData.desired_particle_pixel_size


    def build_model(self, **kwargs):
        network_config = self.hparams.config.train.network
        # Default only: SelfSupervisedModel passes require_labels=True itself, because it
        # needs MySwinUNETR's (logits, hidden) tuple even though the mode is selfSupervised.
        kwargs.setdefault("require_labels", self.hparams.config.train.mode == TrainingModes.picking)
        if self.hparams.config.train.mode == TrainingModes.picking and self.is_pipeline_v2():
            # One output channel per class-table row (CLAUDE.md invariant 7). A v1 model keeps
            # one channel even when several single-class datasets are configured.
            kwargs.setdefault("out_channels", len(self.class_table()))
        model_name, model = network_config.build_model(img_size=self.patch_size, **kwargs)
        return model_name, model

    def is_pipeline_v2(self) -> bool:
        """True when this model was configured for data pipeline v2."""
        return int(safe_getattr(self.hparams.config.train, "data_pipeline_version", 1) or 1) == 2

    def class_table(self):
        """The class table (channel c = row c), rebuilt from the model's own config."""
        from tomocpt.labels.multiclass import class_table_from_config
        return class_table_from_config(self.hparams.config.prepData)

    def scale_conditioning(self) -> bool:
        """True when the network is FiLM-conditioned on the working voxel size. Once the
        network is built, the network itself decides (a model built from the global config
        shares that object, which can change after construction)."""
        network = self.__dict__.get("_modules", {}).get("model")
        if network is not None:
            return getattr(network, "cond", None) is not None
        return bool(safe_getattr(self.hparams.config.train.network, "scale_conditioning", False))

    def forward_and_zero_edges(self, x):
        pred = self.forward(x)
        pred[:, :, 0:2, 0:2, 0:2] *= 0
        pred[:, :, -3:-1, -3:-1, -3:-1, ] *= 0
        return pred

    def forward(self, x):

        if isinstance(x, (tuple, list)):
            out = self.model(x[0])
        else:
            out = self.model(x)

        return out

    def on_validation_epoch_end(self):
        """Log val_loss_smooth, an EMA of the epoch's val_loss. The LR scheduler and early
        stopping monitor it, so every model must log it, not only the picking model."""
        current_val_loss = self.trainer.callback_metrics.get("val_loss")
        if current_val_loss is None:
            return

        previous = getattr(self, "val_loss_ema", None)
        alpha = getattr(self, "ema_alpha", 0.3)
        self.val_loss_ema = current_val_loss if previous is None else alpha * current_val_loss + (1 - alpha) * previous

        self.log("val_loss_smooth", self.val_loss_ema, prog_bar=True, sync_dist=True)

    def _get_model_parameters_for_optimizer(self):
        # Check if we're fine-tuning based on train_continue in hparams
        is_finetuning = self.hparams.get('train_continue') is not None

        # If the model is a SwinUNETR model with differentiated LR method
        if hasattr(self.model, '_get_model_parameters_for_optimizer'):
            parameters = list(self.model._get_model_parameters_for_optimizer(different_lrs=is_finetuning))

            # The model's groups cover only swinViT, encoder* and decoder*. Everything else
            # that trains (the SwinUNETR `out` layer, and the self-supervised heads, which live
            # on this module rather than on self.model) would otherwise never be optimised.
            grouped_params = [p for g in parameters for p in g["params"]] if is_finetuning else parameters
            grouped = {id(p) for p in grouped_params}
            rest = [p for p in self.parameters() if p.requires_grad and id(p) not in grouped]

            if rest and not self._resume_needs_legacy_groups(parameters, is_finetuning):
                if is_finetuning:
                    parameters.append({"params": rest, "lr_mult": 1.0})  # Task head learns fully
                else:
                    parameters.extend(rest)
            return parameters

        # Otherwise return all parameters with default learning rate
        return self.parameters()

    def _resume_needs_legacy_groups(self, legacy_parameters, is_finetuning) -> bool:
        """True when resuming a run whose optimizer was built before `out` and the SSL heads
        were optimised. Its saved optimizer state fits only the old grouping, so that run
        keeps the old grouping until it ends."""
        saved_sizes = getattr(self, "_saved_optimizer_group_sizes", None)
        if saved_sizes is None:
            return False
        legacy_sizes = [len(g["params"]) for g in legacy_parameters] if is_finetuning \
            else [len(legacy_parameters)]
        if saved_sizes != legacy_sizes:
            return False
        logging.warning(
            "Resuming a run whose optimizer predates the fix that trains the output layer and "
            "the self-supervised heads; keeping its original parameter groups for this run."
        )
        return True

    def on_load_checkpoint(self, checkpoint):
        # Only a resume (trainer.fit(ckpt_path=...)) restores the optimizer state, and only
        # then is a trainer attached; load_from_checkpoint (fine-tuning) has none.
        if self._trainer is not None and checkpoint.get("optimizer_states"):
            param_groups = checkpoint["optimizer_states"][0]["param_groups"]
            self._saved_optimizer_group_sizes = [len(g["params"]) for g in param_groups]

    def configure_optimizers(self):
        train_config = self.hparams.config.train
        if self.global_rank == 0:
            logging.info(f"optimizer:{train_config.optimizer}")

        parameters = self._get_model_parameters_for_optimizer()

        # Check if we got parameter groups with lr_mult
        if isinstance(parameters, list) and all(isinstance(p, dict) and 'params' in p for p in parameters):
            # For parameter groups with lr_mult
            base_lr = self.lr
            param_groups = []

            for param_group in parameters:
                new_group = {}
                # Ensure params is a list of tensors
                if 'params' in param_group and param_group['params']:
                    new_group['params'] = param_group['params']

                    # Apply learning rate multiplier if present
                    if 'lr_mult' in param_group:
                        new_group['lr'] = base_lr * param_group['lr_mult']

                    # Copy other parameters
                    for k, v in param_group.items():
                        if k not in ['params', 'lr_mult'] and k not in new_group:
                            new_group[k] = v

                    param_groups.append(new_group)

            # Direct instantiation without using hydra
            opt_config = train_config.optimizer
            if opt_config._target_ == "torch.optim.RAdam":
                opt = torch.optim.RAdam(
                    param_groups,
                    lr=self.lr,
                    betas=opt_config.betas,
                    weight_decay=opt_config.weight_decay
                )
            else:
                # For other optimizers, construct manually if needed
                opt_class = hydra.utils.get_class(opt_config._target_)
                opt = opt_class(param_groups)
        else:
            # Standard parameter initialization
            opt = hydra.utils.instantiate(train_config.optimizer, params=parameters)

        conf = {
            'optimizer': opt,
        }

        conf.update({
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=train_config.FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS,
                cooldown=max(1, train_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4),
                patience=int(1.5 * train_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS)
            ),
            'monitor': 'val_loss_smooth'
        })
        return conf


def verify_different_lrs(model):

    # First, check the groups returned by _get_model_parameters_for_optimizer
    if hasattr(model, '_get_model_parameters_for_optimizer'):
        params = model._get_model_parameters_for_optimizer()
        if isinstance(params, list) and all(isinstance(p, dict) and 'params' in p for p in params):
            # Only log from rank 0 to avoid duplicate output
            if hasattr(model, 'global_rank') and model.global_rank == 0:
                logging.info("USING DIFFERENT LEARNING RATES")
                for i, group in enumerate(params):
                    params_count = sum(p.numel() for p in group['params'])
                    lr_mult = group.get('lr_mult', 1.0)
                    logging.info(f"Group {i}: {params_count} parameters, lr_mult={lr_mult}")

                    # Print some parameter names from this group (max 5)
                    if hasattr(model, 'named_parameters'):
                        param_names = []
                        for name, param in model.named_parameters():
                            if any(p is param for p in group['params']):
                                param_names.append(name)
                                if len(param_names) >= 5:
                                    break
                        logging.info(f"  Sample parameters: {', '.join(param_names)}")
            elif not hasattr(model, 'global_rank'):
                # Fallback for when model doesn't have global_rank attribute
                logging.info("USING DIFFERENT LEARNING RATES")
                for i, group in enumerate(params):
                    params_count = sum(p.numel() for p in group['params'])
                    lr_mult = group.get('lr_mult', 1.0)
                    logging.info(f"Group {i}: {params_count} parameters, lr_mult={lr_mult}")

                    # Print some parameter names from this group (max 5)
                    if hasattr(model, 'named_parameters'):
                        param_names = []
                        for name, param in model.named_parameters():
                            if any(p is param for p in group['params']):
                                param_names.append(name)
                                if len(param_names) >= 5:
                                    break
                        logging.info(f"  Sample parameters: {', '.join(param_names)}")
        else:
            # Only log from rank 0 to avoid duplicate output
            if hasattr(model, 'global_rank') and model.global_rank == 0:
                logging.info("USING SINGLE LEARNING RATE FOR ALL PARAMETERS")
                params_count = sum(p.numel() for p in model.parameters())
                logging.info(f"Total parameters: {params_count}")
            elif not hasattr(model, 'global_rank'):
                # Fallback for when model doesn't have global_rank attribute
                logging.info("USING SINGLE LEARNING RATE FOR ALL PARAMETERS")
                params_count = sum(p.numel() for p in model.parameters())
                logging.info(f"Total parameters: {params_count}")