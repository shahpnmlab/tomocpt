import importlib
import logging

import hydra
import pytorch_lightning as pl
import torch

from tomocpt import constants
from tomocpt.configManager.configManager import update_config, update_config_with_changed_values
from tomocpt.mainConfig import mainConfig, _mainConfigNoChanges


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
        model_name, model = network_config.build_model(img_size=self.patch_size, **kwargs)
        return model_name, model

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

    def _get_model_parameters_for_optimizer(self):
        # Check if we're fine-tuning based on train_continue in hparams
        is_finetuning = self.hparams.get('train_continue') is not None

        # If the model is a SwinUNETR model with differentiated LR method
        if hasattr(self.model, '_get_model_parameters_for_optimizer'):
            return self.model._get_model_parameters_for_optimizer(different_lrs=is_finetuning)

        # Otherwise return all parameters with default learning rate
        return self.parameters()

    def configure_optimizers(self):
        train_config = self.hparams.config.train
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
                verbose=True,
                factor=train_config.FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS,
                cooldown=max(1, train_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4),
                patience=int(1.5 * train_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS)
            ),
            'monitor': 'val_loss'
        })
        return conf


def verify_different_lrs(model):
    """Print learning rate details for parameter groups."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info("=== Verifying Parameter Groups for Different Learning Rates ===")

    # First, check the groups returned by _get_model_parameters_for_optimizer
    if hasattr(model, '_get_model_parameters_for_optimizer'):
        params = model._get_model_parameters_for_optimizer()
        if isinstance(params, list) and all(isinstance(p, dict) and 'params' in p for p in params):
            logger.info("USING DIFFERENT LEARNING RATES")
            for i, group in enumerate(params):
                params_count = sum(p.numel() for p in group['params'])
                lr_mult = group.get('lr_mult', 1.0)
                logger.info(f"Group {i}: {params_count} parameters, lr_mult={lr_mult}")

                # Print some parameter names from this group (max 5)
                if hasattr(model, 'named_parameters'):
                    param_names = []
                    for name, param in model.named_parameters():
                        if any(p is param for p in group['params']):
                            param_names.append(name)
                            if len(param_names) >= 5:
                                break
                    logger.info(f"  Sample parameters: {', '.join(param_names)}")
        else:
            logger.info("USING SINGLE LEARNING RATE FOR ALL PARAMETERS")
            params_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {params_count}")
