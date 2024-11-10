"""
Modified from https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/Pretrain/

"""
from typing import Optional

import numpy as np
from numpy.random import randint
import torch
import torchio as tio
import torchvision
from torch import nn
from torch.nn import functional as F
from tomocpt import constants
from tomocpt.networks.baseModel import BaseModel
from tomocpt.networks.pickingModel import train_config
from tomocpt.networks.swinunetr import MySwinUNETR
from tomocpt.networks.unet import Unet
from tomocpt.mainConfig import mainConfig
network_config = mainConfig.network

class SelfSupervisedModel(BaseModel):
    def set_default_args(self, lr: float | None,
                         num_levels: int | None):

        self.lr = lr if lr is not None else network_config.learning_rate
        self.num_levels = num_levels if num_levels is not None else network_config.NUM_LEVELS

    def __init__(self, lr: float | None = None,
                 num_levels: int | None = None, model: Optional[BaseModel] = None):

        self.set_default_args(lr = lr,
                              num_levels = num_levels)
        self.model = model

        super(SelfSupervisedModel, self).__init__()

        n_voxels = network_config.CHUNK_SIZE

        MODEL_TYPES = {
            "UNET": lambda: Unet(
                first_layer_out_channels=network_config.FIRST_LAYER_OUT_CHANNELS,
                last_activation="linear",
                stride_conv_instead_pooling=True,
                num_levels=num_levels
            ),
            "SWINUNETR": lambda: MySwinUNETR(
                img_size=(network_config.CHUNK_SIZE, network_config.CHUNK_SIZE, network_config.CHUNK_SIZE),
                in_channels=1,
                out_channels=1,
                feature_size=network_config.SWINUNETR_FEAT_SIZE,
                use_v2=True,
                use_checkpoint=True,
                drop_rate=network_config.DROP_RATE,
                attn_drop_rate=network_config.ATTN_DROP_RATE,
                dropout_path_rate=network_config.DROPOUT_PATH_RATE,
            )
        }

        if model is None:
            self.model_name = network_config.model_type.value
            model_constructor = MODEL_TYPES.get(self.model_name.upper())
            if model_constructor:
                self.model = model_constructor()
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
        else:
            self.model_name = model.model_name
            self.model = model.model


        exampleX = torch.rand(train_config.batch_size, 1, n_voxels, n_voxels, n_voxels)
        out, hid = self.model(exampleX)
        batchSize, nchan, s, _, _ = hid.shape
        assert batchSize > 1
        self.model_name = network_config.model_type
        print(f"SelfSupervisedModel using {self.model_name}")

        def generateHead(outDims, bias=False):
            return nn.Sequential(
                nn.Conv3d(in_channels=nchan, out_channels=nchan, kernel_size=1, padding="same", bias=bias),
                nn.AdaptiveAvgPool3d(output_size=1), nn.Flatten(),
                nn.Linear(nchan, outDims, bias=bias)
            )

        contrastiveEmbSize = 256 #TODO: Check if this needs to go into network_config
        self.rotation_head = generateHead(4, bias=True)
        self.contrastive_head = generateHead(contrastiveEmbSize, bias=False)

        self.lr = lr
        self.loss_function = LossPretrain()

    def resolve_batch(self, batch):
        return batch["input_data"][tio.DATA]

    def forward(self, x):
        out, hid = self.model(x)
        rotp = self.rotation_head(hid)
        contrp = self.contrastive_head(hid)
        return rotp, contrp, out

    def _step(self, batch, batch_idx):

        x = self.resolve_batch(batch)

        x1, rot1 = rot_rand(x)
        x2, rot2 = rot_rand(x)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)

        rot1_p, contrastive1_p, rec_x1 = self(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = self(x2_augment)
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        loss, losses_tasks = self.loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

        return loss, losses_tasks, (x, x1, x2), (contrastive1_p, contrastive2_p), imgs_recon

    def training_step(self, batch, batch_idx):
        loss, losses_tasks, (x, x1, x2), (contrastive1_p, contrastive2_p), imgs_recon = self._step(batch, batch_idx)
        # TODO: if on_step=True, reconsider sync_dist
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        self.log('loss_rot', losses_tasks[0], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log('loss_contrastive', losses_tasks[1], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log('loss_inpainting', losses_tasks[2], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])

        if batch_idx == 0:
            tensorboard = self.logger.experiment
            size = x.shape[2]
            # print(x.shape)

            grid1 = torchvision.utils.make_grid(x[:, :1, size // 2, :, :])
            grid2 = torchvision.utils.make_grid(x1[:, :1, size // 2, :, :])
            grid3 = torchvision.utils.make_grid(x2[:, :1, size // 2, :, :])
            grid4 = torchvision.utils.make_grid(x[:, :1, size // 2, :, :])

            tensorboard.add_image("input_train", grid1, global_step=self.current_epoch)
            tensorboard.add_image("aug1_train", grid2, global_step=self.current_epoch)
            tensorboard.add_image("aug2_train", grid3, global_step=self.current_epoch)
            tensorboard.add_image("recons_train", grid4, global_step=self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, losses_tasks, (x, x1, x2), (contrastive1_p, contrastive2_p), imgs_recon = self._step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        self.log('val_loss_rot', losses_tasks[0], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log('val_loss_contrastive', losses_tasks[1], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log('val_loss_inpainting', losses_tasks[2], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])

        if batch_idx == 0:
            tensorboard = self.logger.experiment
            size = x.shape[2]
            # print(x.shape)
            grid1 = torchvision.utils.make_grid(x[:, :1, size // 2, :, :])
            grid2 = torchvision.utils.make_grid(x1[:, :1, size // 2, :, :])
            grid3 = torchvision.utils.make_grid(x2[:, :1, size // 2, :, :])
            grid4 = torchvision.utils.make_grid(x[:, :1, size // 2, :, :])
            tensorboard.add_image("input_val", grid1, global_step=self.current_epoch)
            tensorboard.add_image("aug1_val", grid2, global_step=self.current_epoch)
            tensorboard.add_image("aug2_val", grid3, global_step=self.current_epoch)
            tensorboard.add_image("recons_val", grid4, global_step=self.current_epoch)

        # emb1 = contrastive1_p.unsqueeze(1).repeat(1, 10, 1)
        # emb2 = contrastive1_p.unsqueeze(1).repeat(1, 10, 1)
        # tensorboard = self.logger.experiment
        # grid1 = torchvision.utils.make_grid(emb1.unsqueeze(1), nrow=1, normalize=True, padding=5)
        # tensorboard.add_image(f"emb1_val", grid1, global_step=self.current_epoch)
        # grid2 = torchvision.utils.make_grid(emb2.unsqueeze(1), nrow=1, normalize=True, padding=5)
        # tensorboard.add_image(f"emb2_val", grid2, global_step=self.current_epoch)

        return loss

    def predict_step(self, batch):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.99),
                                weight_decay=1e-8) #decoupled_weight_decay=True)

        conf = {
            'optimizer': opt,
        }

        conf.update({
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,
                                                                       factor=network_config.FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS,
                                                                       cooldown=max(1,
                                                                                    network_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4),
                                                                       patience=int(
                                                                           1.5 * network_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS)),
            'monitor': 'val_loss'
        })
        return conf


def patch_rand_drop(x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=x.device
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                    torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def rot_rand(x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = x_s.device
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(x_aug[i], x_aug[idx_rnd])
    return x_aug


class ContrastLoss(nn.Module):
    is_differentiable: Optional[bool] = True

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, temperature = network_config.CONTRAST_LOSS_TEMPERATURE,
                 l1_embeddings_w = network_config.CONTRAST_LOSS_L1_EMB_REGULARIZATION):
        """
        Low temperature penalizes much more embeddings that are the same
        """
        super().__init__()
        self.temp = temperature
        self.l1_embeddings_w = l1_embeddings_w

    def forward(self, x_i, x_j):
        z = torch.cat([x_i, x_j], dim=0)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.einsum("id,jd->ij", z_norm, z_norm)
        # sim_ij = torch.diag(sim_matrix, x_i.shape[0])
        # sim_ji = torch.diag(sim_matrix, -x_i.shape[0])

        sim_matrix[torch.eye(sim_matrix.shape[0], dtype=torch.bool)] = float("-inf")

        # positives = torch.cat([sim_ij, sim_ji], dim=0)
        # negatives = sim_matrix[~torch.eye(sim_matrix.shape[0], dtype=bool)].reshape(z.shape[0], -1)

        # logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        # # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        labels = torch.arange(x_i.shape[0], dtype=torch.long).to(sim_matrix.device)
        labels = torch.tile(labels, (2,))
        labels[0:x_i.shape[0]] += x_i.shape[0]
        loss = F.cross_entropy(sim_matrix / self.temp, labels, reduction="mean")

        return loss + self.l1_embeddings_w * (z.abs().sum())


class LossPretrain(torch.nn.Module):
    def __init__(self, rotLossW=1.0, contrasLossW=1.0, reconsLossW=10.0):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.contrast_loss = ContrastLoss()
        self.rotLossW = rotLossW
        self.contrasLossW = contrasLossW
        self.reconsLossW = reconsLossW

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.rotLossW * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.contrasLossW * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.reconsLossW * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)


if __name__ == "__main__":
    lFun = ContrastLoss(temperature=0.1)
    print(lFun(torch.rand(50, 3), torch.rand(50, 3)))
    print(lFun(torch.tensor([[1, 0, 0], [1, 0, 1.], [1, 1, 1]]), torch.tensor([[1, 0, 0], [1, 0, 1.], [1, 1, 1]])))
    print(lFun(torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 1, 1]]), torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 1, 1]])))
    print(lFun(torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 0, 0.]]), torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 0, 0.]])))

    network_config.model_type = "SwinUNETR"  # "UNET" #"SwinUNETR"
    model = SelfSupervisedModel()

    chunk_size = network_config.CHUNK_SIZE
    batchSize = 3
    x = torch.rand(batchSize, 1, chunk_size, chunk_size, chunk_size)
    out = model(x)
    print([_out.shape for _out in out])
    batch = {"input_data": {tio.DATA: x}}
    out = model._step(batch, batch_idx=0)

    print()
