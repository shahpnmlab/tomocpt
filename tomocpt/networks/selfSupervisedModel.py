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
from tomocpt.mainConfig import mainConfig
from tomocpt.configManager.configManager import safe_getattr
from tomocpt.defaultConfigs.ssl_config import SSLConfig

class SelfSupervisedModel(BaseModel):


    def __init__(self, lr: float | None = None,
                 num_levels: int | None = None, model: Optional[BaseModel] = None, *args, **kwargs):

        # num_levels is unused and must not be forwarded: BaseModel's second positional
        # argument is `config`, which load_from_checkpoint also passes by keyword.
        super(SelfSupervisedModel, self).__init__(lr, *args, **kwargs)

        # train.ssl (Gate 5); a checkpoint without it is SSL v1. resolved() raises ValueError
        # for version 2 with use_rotation=True.
        self.ssl_cfg = safe_getattr(self.hparams.config.train, "ssl", None) or SSLConfig()
        self.ssl_cfg = self.ssl_cfg.resolved()
        if self.ssl_cfg.version == 2:
            self._init_v2(lr, model)
            return

        n_voxels = mainConfig.train.CHUNK_SIZE

        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model


        exampleX = torch.rand(self.hparams.config.train.batch_size, 1, n_voxels, n_voxels, n_voxels)
        out, hid = self.model(exampleX)
        batchSize, nchan, s, _, _ = hid.shape
        assert batchSize > 1
        print(f"SelfSupervisedModel using {self.model_name}")

        def generateHead(outDims, bias=False):
            return nn.Sequential(
                nn.Conv3d(in_channels=nchan, out_channels=nchan, kernel_size=1, padding="same", bias=bias),
                nn.AdaptiveAvgPool3d(output_size=1), nn.Flatten(),
                nn.Linear(nchan, outDims, bias=bias)
            )

        contrastiveEmbSize = self.hparams.config.train.network.SELF_SUPERVISED_EMBEDING_SIZE
        self.rotation_head = generateHead(4, bias=True)
        self.contrastive_head = generateHead(contrastiveEmbSize, bias=False)

        self.lr = lr
        self.loss_function = LossPretrain()

    def build_model(self):
        if self.is_ssl_v2():
            return self._build_model_v2()
        return super().build_model(require_labels=True)

    def is_ssl_v2(self) -> bool:
        cfg = self.__dict__.get("ssl_cfg")
        return cfg is not None and cfg.version == 2

    # ----------------------------------------------------------------------------------
    # SSL v2 (Gate 5): scale pairs, masked reconstruction of the clean view, global and
    # dense contrastive learning, relative scale. The heads live on this module, so a
    # transfer to the picker drops exactly them.
    # ----------------------------------------------------------------------------------

    def _build_model_v2(self):
        """The Gate 3 ``ScaleConditionedSwinUNETR`` with one output channel (reconstruction),
        whatever ``train.network.scale_conditioning`` says: the decoder receives ``apix_view``."""
        import copy

        from tomocpt.defaultConfigs.network_config import ModelType

        network_config = copy.copy(self.hparams.config.train.network)
        if ModelType(network_config.model_type) != ModelType.SwinUNETR:
            raise ValueError("train.ssl.version=2 needs train.network.model_type=SwinUNETR")
        network_config.scale_conditioning = True
        return network_config.build_model(img_size=self.patch_size, require_labels=True, out_channels=1)

    def _init_v2(self, lr, model):
        from tomocpt.networks.ssl.heads import DenseProjectionHead, ProjectionHead, RelativeScaleHead
        from tomocpt.networks.swinunetr import ScaleConditionedSwinUNETR

        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model
        if not isinstance(self.model, ScaleConditionedSwinUNETR):
            raise ValueError("SSL v2 needs a ScaleConditionedSwinUNETR network")
        fs = self.model.out.conv.conv.in_channels
        ssl = self.ssl_cfg
        network_config = self.hparams.config.train.network
        self.rotation_head = None
        # Invariant 5: a task with weight 0 has no module.
        self.contrastive_head = ProjectionHead(16 * fs, network_config.SELF_SUPERVISED_EMBEDING_SIZE) \
            if ssl.contrast_weight > 0 else None
        self.dense_head = DenseProjectionHead(fs, int(ssl.dense_dim)) if ssl.dense_weight > 0 else None
        self.scale_head = RelativeScaleHead(8 * fs + 16 * fs) if ssl.scale_weight > 0 else None
        self.contrast_temperature = float(network_config.CONTRAST_LOSS_TEMPERATURE)
        self.contrast_l1 = float(network_config.CONTRAST_LOSS_L1_EMB_REGULARIZATION)
        self.dense_temperature = 0.1
        self.lr = lr
        print(f"SelfSupervisedModel (SSL v2) using {self.model_name}")

    def _forward_v2(self, x, apix):
        """``(reconstruction logits, swinViT hidden states, decoder2 output)``; the decoder is
        FiLM-conditioned on ``apix`` as in ``ScaleConditionedSwinUNETR.forward``."""
        net = self.model
        e = net.cond.scale_embed(apix)
        film = net.cond.film
        hs = net.swinViT(x, net.normalize)
        enc0 = net.encoder1(x)
        enc1 = net.encoder2(hs[0])
        enc2 = net.encoder3(hs[1])
        enc3 = net.encoder4(hs[2])
        dec4 = film[0](net.encoder10(hs[4]), e)
        dec3 = film[1](net.decoder5(dec4, hs[3]), e)
        dec2 = film[2](net.decoder4(dec3, enc3), e)
        dec1 = film[3](net.decoder3(dec2, enc2), e)
        dec0 = film[4](net.decoder2(dec1, enc1), e)
        out = film[5](net.decoder1(dec0, enc0), e)
        return net.out(out), hs, dec0

    @staticmethod
    def _vol5(t):
        t = torch.as_tensor(t)
        return t.unsqueeze(1) if t.ndim == 4 else t

    def _step_v2(self, batch):
        from tomocpt.networks.ssl.losses import dense_info_nce, masked_l1, nt_xent

        ssl = self.ssl_cfg
        xa, xb = self._vol5(batch["input_a"]).float(), self._vol5(batch["input_b"]).float()
        n = xa.shape[0]
        x = torch.cat([xa, xb], dim=0)
        apix = torch.cat([batch["apix_a"].reshape(-1), batch["apix_b"].reshape(-1)]).float().reshape(-1, 1)
        target = torch.cat([self._vol5(batch["target_a"]), self._vol5(batch["target_b"])], dim=0).float()
        mask = torch.cat([self._vol5(batch["mask_a"]), self._vol5(batch["mask_b"])], dim=0).bool()

        recon, hs, dec0 = self._forward_v2(x, apix)
        losses = {"recon": float(ssl.recon_weight) * masked_l1(recon, target, mask, ssl.recon_visible_weight)}
        extras = {}
        if self.contrastive_head is not None:
            z = self.contrastive_head(hs[4].mean(dim=(2, 3, 4)))
            losses["contrastive"] = float(ssl.contrast_weight) * nt_xent(
                z[:n], z[n:], self.contrast_temperature, self.contrast_l1)
        if self.dense_head is not None:
            f = self.dense_head(_scale_gradient(dec0, float(ssl.dense_grad_scale)))
            losses["dense"] = float(ssl.dense_weight) * dense_info_nce(
                f[:n], f[n:], batch["pts_a"], batch["pts_b"], self.dense_temperature, view_size=xa.shape[-1])
        if self.scale_head is not None:
            p = torch.cat([hs[3].mean(dim=(2, 3, 4)), hs[4].mean(dim=(2, 3, 4))], dim=1).float()
            r = self.scale_head(p[:n], p[n:]).float()
            t = batch["target_log2"].reshape(-1).float()
            losses["scale"] = float(ssl.scale_weight) * F.smooth_l1_loss(r, t, beta=0.1)
            extras["scale_mae"] = (r - t).abs().mean().detach()
        total = sum(losses.values())
        return total, losses, extras, (xa, batch["target_a"], recon[:n])

    def _log_v2(self, prefix, batch, batch_idx):
        total, losses, extras, (xa, ta, ra) = self._step_v2(batch)
        n = xa.shape[0]
        self.log(f"{prefix}loss", total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=n)
        for name, value in losses.items():
            self.log(f"{prefix}loss_{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=n)
        for name, value in extras.items():
            self.log(f"{prefix}{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=n)
        experiment = getattr(self.logger, "experiment", None) if self.logger is not None else None
        if batch_idx == 0 and hasattr(experiment, "add_image"):
            tag = "train" if prefix == "" else "val"
            size = xa.shape[2]
            with torch.no_grad():
                for name, v in (("input", xa), ("target", self._vol5(ta)), ("recons", ra)):
                    grid = torchvision.utils.make_grid(v[:, :1, size // 2].detach().float().cpu())
                    experiment.add_image(f"{name}_{tag}", grid, global_step=self.current_epoch)
        return total

    def resolve_batch(self, batch):
        return batch["input_data"] #[tio.DATA] Seems to be not needed in newer versions

    def forward(self, x, apix=None):
        if self.is_ssl_v2():
            return self._forward_v2(x, apix)
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
        if self.is_ssl_v2():
            return self._log_v2("", batch, batch_idx)
        loss, losses_tasks, (x, x1, x2), (contrastive1_p, contrastive2_p), imgs_recon = self._step(batch, batch_idx) #if on_step=True, reconsider sync_dist
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
            # Reconstruction of view 1 (rec_x1, the first half of imgs_recon); its target is x1.
            grid4 = torchvision.utils.make_grid(imgs_recon[:x1.shape[0], :1, size // 2, :, :])

            tensorboard.add_image("input_train", grid1, global_step=self.current_epoch)
            tensorboard.add_image("aug1_train", grid2, global_step=self.current_epoch)
            tensorboard.add_image("aug2_train", grid3, global_step=self.current_epoch)
            tensorboard.add_image("recons_train", grid4, global_step=self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_ssl_v2():
            return self._log_v2("val_", batch, batch_idx)

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
            # Reconstruction of view 1 (rec_x1, the first half of imgs_recon); its target is x1.
            grid4 = torchvision.utils.make_grid(imgs_recon[:x1.shape[0], :1, size // 2, :, :])
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



def _scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward; the gradient flowing back into ``x`` is multiplied by ``scale``.

    SSL v2 uses it for the dense contrastive task (``train.ssl.dense_grad_scale``, default 0):
    with the full gradient, dense instance discrimination between two views that share the
    source noise drives ``decoder2`` towards instance-specific noise codes and destroys the
    species separation (Gate 5 report); with 0 the dense head learns on the reconstruction
    features and the network is shaped by the other tasks."""
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return x.detach() + scale * (x - x.detach())


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

    def __init__(self, temperature, l1_embeddings_w):
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
    def __init__(self, rotLossW=1.0, contrasLossW=1.0, reconsLossW=10.0): #TODO: Pranav, you can change here the weight of each term of the self supervised loss
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.contrast_loss = ContrastLoss(temperature=mainConfig.train.network.CONTRAST_LOSS_TEMPERATURE,
                                          l1_embeddings_w=mainConfig.train.network.CONTRAST_LOSS_L1_EMB_REGULARIZATION)
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
    lFun = ContrastLoss(temperature=0.1, l1_embeddings_w=2)
    print(lFun(torch.rand(50, 3), torch.rand(50, 3)))
    print(lFun(torch.tensor([[1, 0, 0], [1, 0, 1.], [1, 1, 1]]), torch.tensor([[1, 0, 0], [1, 0, 1.], [1, 1, 1]])))
    print(lFun(torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 1, 1]]), torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 1, 1]])))
    print(lFun(torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 0, 0.]]), torch.tensor([[1, 0, 0], [1, 0, 0.], [1, 0, 0.]])))

    network_config = mainConfig.train.network

    network_config.model_type = "SwinUNETR"  # "UNET" #"SwinUNETR"
    model = SelfSupervisedModel()

    chunk_size = mainConfig.train.CHUNK_SIZE
    batchSize = 3
    x = torch.rand(batchSize, 1, chunk_size, chunk_size, chunk_size)
    out = model(x)
    print([_out.shape for _out in out])
    batch = {"input_data": {tio.DATA: x}}
    out = model._step(batch, batch_idx=0)

    print()
