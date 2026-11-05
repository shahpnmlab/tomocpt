import torch
import torchvision
from torch import nn

from tomocpt.configManager.configManager import safe_getattr
from tomocpt.networks.baseModel import BaseModel
from tomocpt.training.losses import gradient3d_loss
from tomocpt.training.multiclass_loss import (CLASS_HEADS, class_probabilities, multiclass_picking_loss,
                                              renormalise_targets)
from tomocpt.logger import get_logger

logging = get_logger()


def parse_class_weights(text: str) -> dict:
    """Parse ``train.particles.class_weights`` (``"1:1.0,2:2.0"``) into ``{class number: weight}``."""
    weights = {}
    for entry in (e.strip() for e in str(text).split(",")):
        if not entry:
            continue
        key, sep, value = entry.partition(":")
        try:
            if not sep:
                raise ValueError
            number, weight = int(key.strip()), float(value.strip())
        except ValueError:
            raise ValueError(f"Malformed class_weights entry '{entry}': expected '<class number>:<weight>'")
        if weight < 0:
            raise ValueError(f"class_weights entry '{entry}': the weight must be >= 0")
        if number in weights:
            raise ValueError(f"Duplicate class {number} in class_weights")
        weights[number] = weight
    return weights


class BasePickingModel(BaseModel):

    def __init__(
            self, lr: float | None = None, model=None, config_dict=None,
            ema_alpha: float = 0.3, *args, **kwargs
    ):
        # EMA Alpha (smoothing factor):
        # ema_alpha=0.3: 30% current loss + 70% historical average = quick adaptation
        # ema_alpha=0.1: 10% current loss + 90% historical average = heavy smoothing
        # Higher alpha = more responsive to recent validation losses
        # Lower alpha = more noise filtering, slower to detect real changes

        # Remove config_dict from kwargs before passing to parent
        if "config_dict" in kwargs:
            config_dict = kwargs.pop("config_dict")

        super(BasePickingModel, self).__init__(lr, *args, **kwargs)

        # Store config if needed
        self.config = config_dict
        self.ema_alpha = ema_alpha  # Smoothing factor for EMA
        self.val_loss_ema = None

        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model

        self.lr = lr
        if self.class_head() not in CLASS_HEADS:
            raise ValueError(f"train.particles.class_head must be one of {CLASS_HEADS}, got {self.class_head()!r}")
        self.contrastive_head = None
        self._build_discriminative()
        self.offset_head = None
        self._build_offset_head()

    # ------------------------------------------------------------------ Gate 7
    def _disc(self, name, default):
        """``train.discriminative.<name>`` of this model's (checkpoint) config, or ``default``."""
        disc = safe_getattr(self.hparams.config.train, "discriminative", None)
        return safe_getattr(disc, name, default) if disc is not None else default

    def contrastive_weight(self) -> float:
        return float(self._disc("contrastive_weight", 0.0) or 0.0)

    def value_loss(self) -> str:
        v = self._disc("value_loss", "huber")
        return str(getattr(v, "value", v))

    def _build_discriminative(self):
        """Create the contrastive head and memory bank when ``contrastive_weight > 0`` (only then:
        invariant 5). The head is a 1x1x1 projection of the ``decoder2`` output."""
        if self.value_loss() not in ("huber", "focal"):
            raise ValueError(f"train.discriminative.value_loss must be 'huber' or 'focal', got {self.value_loss()!r}")
        if self.contrastive_weight() <= 0:
            return
        if not self.is_pipeline_v2():
            raise ValueError("train.discriminative.contrastive_weight > 0 needs data pipeline v2")
        from tomocpt.networks.ssl.heads import DenseProjectionHead
        from tomocpt.training.discriminative import ContrastiveMemoryBank

        fs = self.model.out.conv.conv.in_channels  # decoder2 output channels = feature_size (CLAUDE.md 1.4)
        dim = int(self._disc("embed_dim", 64))
        self.contrastive_head = DenseProjectionHead(fs, dim)
        self.contrastive_bank = ContrastiveMemoryBank(len(self.class_table()), int(self._disc("bank_size", 1024)), dim)

    # ------------------------------------------------------------------ Gate 4f
    def _particles(self, name, default):
        """``train.particles.<name>`` of this model's (checkpoint) config, or ``default``."""
        particles = safe_getattr(self.hparams.config.train, "particles", None)
        return safe_getattr(particles, name, default) if particles is not None else default

    def offset_weight(self) -> float:
        return float(self._particles("offset_weight", 0.0) or 0.0)

    def has_offset_head(self) -> bool:
        return self.offset_head is not None

    def _build_offset_head(self):
        """Offset head (``train.particles.offset_weight > 0`` only: invariant 5) on the full-resolution
        ``decoder1`` output: 3 channels ``(dz, dy, dx)`` in working voxels, last layer zero-initialised
        (invariant 6)."""
        if self.offset_weight() <= 0:
            return
        if not self.is_pipeline_v2():
            raise ValueError("train.particles.offset_weight > 0 needs data pipeline v2")
        fs = self.model.out.conv.conv.in_channels  # decoder1 output channels = feature_size
        last = nn.Conv3d(16, 3, kernel_size=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.offset_head = nn.Sequential(nn.Conv3d(fs, 16, kernel_size=3, padding=1), nn.GELU(), last)

    def _forward_all(self, x, apix=None):
        """``(logits, decoder2, decoder1)`` in one network pass."""
        x = x[0] if isinstance(x, (tuple, list)) else x
        if apix is None or not self.scale_conditioning():
            out = self.model.forward_features(x, with_decoder1=True)
        else:
            out = self.model.forward_features(x, apix=apix, with_decoder1=True)
        return out[0], out[2], out[3]

    def _offset_term(self, offsets, y, batch):
        from tomocpt.training.offsets import offset_loss, offset_targets

        target, mask = offset_targets(y, batch["class_d_vox"], batch["class_available"],
                                      [not f for f in self._is_filament()])
        return offset_loss(offsets, target, mask)

    def predict_step_with_offsets(self, batch, apix=None, classes=None):
        """``(probabilities as predict_step, offsets [B, 3, D, H, W] in working voxels)``."""
        if self.offset_head is None:
            raise ValueError("this model has no offset head (train.particles.offset_weight was 0)")
        use_apix = apix if self.scale_conditioning() else None
        logits, _, dec1 = self._forward_all(batch, apix=use_apix)
        if self.class_head() == "softmax":
            probs = class_probabilities(logits, "softmax", classes)
        else:
            probs = nn.functional.sigmoid(logits)
        return probs, self.offset_head(dec1)

    def _is_filament(self):
        from tomocpt.labels.multiclass import FILAMENT

        return [str(getattr(r.kind, "value", r.kind)) == FILAMENT for r in self.class_table()]

    def _forward_features(self, x, apix=None):
        """``(logits, decoder2 features)``; logits as ``forward``."""
        x = x[0] if isinstance(x, (tuple, list)) else x
        if apix is None or not self.scale_conditioning():
            out = self.model.forward_features(x)
        else:
            out = self.model.forward_features(x, apix=apix)
        return out[0], out[2]

    def contrastive_loss(self, features, y_pred, batch, update_bank: bool = True):
        """Supervised dense contrastive loss (Gate 7, option a) on the ``decoder2`` features.

        Anchors from the batch's dense target, hard negatives ranked by ``y_pred`` (detached), mask
        negatives from ``batch["negative_mask"]``; SupCon against the batch and the memory bank,
        in float32. The head is always in the graph (DDP)."""
        from tomocpt.training.discriminative import sample_anchors, sample_hard_negatives, supcon_loss

        target = batch["target_data"]
        target = target.data if hasattr(target, "data") and not torch.is_tensor(target) else target
        with torch.autocast(device_type=features.device.type, enabled=False):
            emb_map = self.contrastive_head(features.float())  # [B, E, d, h, w]
            a_idx, a_lab = sample_anchors(
                target, batch["class_available"], batch["class_d_vox"], self._is_filament(),
                min_target=float(self._disc("anchor_min_target", 0.5)),
                max_per_class=int(self._disc("anchors_per_class", 32)))
            n_idx, _, _ = sample_hard_negatives(
                y_pred.detach(), target, batch["class_d_vox"], int(self._disc("hard_negatives", 64)),
                min_score=float(self._disc("hard_negative_min_score", 0.3)),
                max_score=float(self._disc("hard_negative_max_score", 0.9)),
                exclusion_radius=float(self._disc("exclusion_radius", 1.0)),
                negative_mask=batch.get("negative_mask"), n_mask=int(self._disc("mask_negatives", 64)))
            idx = torch.cat([a_idx, n_idx]).to(emb_map.device)
            labels = torch.cat([a_lab, torch.full((len(n_idx),), -1, dtype=torch.long)]).to(emb_map.device)
            # target voxel -> feature voxel
            scale = torch.tensor([s_f / s_t for s_f, s_t in zip(emb_map.shape[2:], target.shape[2:])],
                                 device=emb_map.device)
            fz = (idx[:, 1:].float() * scale).floor().long()
            emb = emb_map[idx[:, 0], :, fz[:, 0], fz[:, 1], fz[:, 2]]  # [N, E]
            bank_e, bank_l = self.contrastive_bank.get()
            loss = supcon_loss(emb, labels, bank_e, bank_l, float(self._disc("temperature", 0.1)))
            loss = loss + 0.0 * emb_map.sum()
            if update_bank and len(emb):
                self.contrastive_bank.push(nn.functional.normalize(emb.detach(), dim=1), labels)
        return loss, int((labels >= 0).sum())

    def class_head(self) -> str:
        """``train.particles.class_head`` of this model's (checkpoint) config; ``sigmoid`` if absent."""
        particles = safe_getattr(self.hparams.config.train, "particles", None)
        return str(safe_getattr(particles, "class_head", "sigmoid")) if particles is not None else "sigmoid"

    def _probabilities(self, logits, y, batch):
        """``(y_pred, y)`` for the loss: sigmoid, or the softmax head on a v2 batch (renormalised targets)."""
        if self.class_head() == "softmax" and self._is_v2_batch(batch):
            return class_probabilities(logits, "softmax", batch["class_available"]), renormalise_targets(y)
        return nn.functional.sigmoid(logits), y

    def setup(self, stage: str) -> None:
        """Called after the model is moved to the correct device."""
        if self.global_rank == 0:
            logging.info(f"PickingModel using {self.model_name}")

    def loss(self, y_pred, y_true):
        eps = 1
        num_positive = torch.sum(y_true, dim=[1, 2, 3, 4])
        num_negative = y_true.numel() / y_true.shape[0] - num_positive

        # Compute weights
        # Adding eps to the denominator to avoid division by zero
        positive_weights = (y_true.numel() / y_true.shape[0]) / (num_positive + eps)
        negative_weights = (y_true.numel() / y_true.shape[0]) / (num_negative + eps)

        # Reshape weights to be broadcastable
        positive_weights = positive_weights.view(y_true.shape[0], 1, 1, 1, 1)
        negative_weights = negative_weights.view(y_true.shape[0], 1, 1, 1, 1)

        # Create importance mask
        importance_mask = torch.where(
            y_true > 0, positive_weights, 2 * negative_weights
        )

        # Compute MSE
        # loss = importance_mask * (y_pred - y_true) ** 2
        loss = importance_mask * nn.functional.huber_loss(
            y_pred, y_true, reduction="none"
        )
        _loss = 10 * gradient3d_loss(y_pred, y_true).sum(2)
        loss += _loss
        loss = loss.mean()
        return loss

    def resolve_batch(self, batch):
        x = batch["input_data"].data
        y = batch["target_data"].data
        # y = (y>0).float()
        return x, y

    def forward(self, x, apix=None):
        """
        :param x: input volume ``[B, 1, D, H, W]``
        :param apix: working voxel size in Å (float or ``[B, 1]`` tensor). A scale-conditioned
            network uses it; ``None`` runs the unconditioned computation. An unconditioned
            network ignores it (its output does not depend on the voxel size).
        :return: logits ``[B, C, D, H, W]``
        """
        if apix is None or not self.scale_conditioning():
            out = super().forward(x)
        else:
            out = self.model(x[0] if isinstance(x, (tuple, list)) else x, apix=apix)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _batch_apix(self, batch):
        """``batch["apix_work"]`` for a scale-conditioned model, else ``None``."""
        if not self.scale_conditioning():
            return None
        if "apix_work" not in batch:
            raise ValueError(
                "train.network.scale_conditioning is on, but the batch has no 'apix_work' "
                "(scale conditioning needs data pipeline v2)"
            )
        return batch["apix_work"]

    @staticmethod
    def _is_v2_batch(batch) -> bool:
        return "class_available" in batch and "class_d_vox" in batch

    def _class_weights(self):
        """``train.particles.class_weights`` mapped to channels by class number (default 1.0),
        or ``None`` when unset."""
        particles = safe_getattr(self.hparams.config.train, "particles", None)
        text = safe_getattr(particles, "class_weights", None) if particles is not None else None
        if not text:
            return None
        if getattr(self, "_class_weights_cache", None) is None or self._class_weights_cache[0] != text:
            by_number = parse_class_weights(text)
            weights = [float(by_number.get(row.number, 1.0)) for row in self.class_table()]
            self._class_weights_cache = (text, weights)
        return self._class_weights_cache[1]

    def multiclass_loss(self, y_pred, y, batch):
        """Pipeline-v2 loss (CLAUDE.md Section 5.4) for a v2 batch."""
        particles = safe_getattr(self.hparams.config.train, "particles", None)
        use_checkpoint = bool(safe_getattr(particles, "loss_checkpoint", True)) if particles is not None else True
        kw = {}
        if self.value_loss() == "focal":
            # soft focal penalty on particle channels, Huber on filament channels (1 Nov 2026)
            kw = dict(value_loss="focal", focal_channels=[not f for f in self._is_filament()],
                      focal_gamma=float(self._disc("focal_alpha", 2.0)))
        core = float(self._particles("core_weight", 0.0) or 0.0)
        if core:
            kw.update(core_weight=core, core_power=float(self._particles("core_power", 4.0)),
                      core_channels=[not f for f in self._is_filament()])
        return multiclass_picking_loss(
            y_pred, y, batch["class_available"], batch["class_d_vox"],
            class_weights=self._class_weights(), checkpoint=use_checkpoint,
            cross_class_neg_weight=float(safe_getattr(particles, "cross_class_neg_weight", 0.0))
            if particles is not None else 0.0,
            **kw,
        )

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)
        y_raw = y
        contrastive = self.contrastive_head is not None and self._is_v2_batch(batch)
        offsets = self.offset_head is not None and self._is_v2_batch(batch)
        if offsets:
            logits, features, dec1 = self._forward_all(x, apix=self._batch_apix(batch))
        elif contrastive:
            logits, features = self._forward_features(x, apix=self._batch_apix(batch))
        else:
            logits = self(x, apix=self._batch_apix(batch))
        y_pred, y = self._probabilities(logits, y, batch)
        if self._is_v2_batch(batch):
            loss = self.multiclass_loss(y_pred, y, batch)
        else:
            loss = self.loss(y_pred, y)
        if contrastive:
            loss_c, n_anchors = self.contrastive_loss(features, y_pred, batch)
            loss = loss + self.contrastive_weight() * loss_c
            self.log("loss_contrastive", loss_c.detach(), on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=x.shape[0])
        if offsets:
            loss_o = self._offset_term(self.offset_head(dec1), y_raw, batch)
            loss = loss + self.offset_weight() * loss_o
            self.log("loss_offset", loss_o.detach(), on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=x.shape[0])

        self.log(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )  # if on_step=True, reconsider sync_dist

        if batch_idx == 0 and self.global_rank == 0:
            self._log_images(x, y, y_pred, "train")
        return loss


    def validation_step(self, batch, batch_idx):
            x, y = self.resolve_batch(batch)
            y_raw = y
            offsets = self.offset_head is not None and self._is_v2_batch(batch)
            if offsets:
                logits, _, dec1 = self._forward_all(x, apix=self._batch_apix(batch))
            else:
                logits = self(x, apix=self._batch_apix(batch))
            y_pred, y = self._probabilities(logits, y, batch)
            if self._is_v2_batch(batch):
                loss = self.multiclass_loss(y_pred, y, batch)
                self._log_per_class_val_loss(y_pred, y, batch)
            else:
                loss = self.loss(y_pred, y)
            if offsets:
                loss_o = self._offset_term(self.offset_head(dec1), y_raw, batch)
                loss = loss + self.offset_weight() * loss_o
                self.log("val_loss_offset", loss_o, on_step=False, on_epoch=True, sync_dist=True)

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if batch_idx == 0 and self.global_rank == 0:
                self._log_images(x, y, y_pred, "val")
                
            return loss

    def _log_per_class_val_loss(self, y_pred, y, batch):
        """Log ``val_loss_<class name>`` for every class available in this batch."""
        available = batch["class_available"].to(torch.bool)
        d_vox = batch["class_d_vox"]
        for c, row in enumerate(self.class_table()):
            n_available = int(available[:, c].sum())
            if n_available == 0:
                continue
            loss_c = multiclass_picking_loss(
                y_pred[:, c:c + 1], y[:, c:c + 1], available[:, c:c + 1], d_vox[:, c:c + 1],
                checkpoint=False,
            )
            self.log(f"val_loss_{row.name}", loss_c, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=n_available)

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0, apix=None, classes=None):
        """Per-class probabilities; with the softmax head they are normalised over ``classes``
        (channel indices, default all)."""
        logits = self(batch, apix=apix)
        if self.class_head() == "softmax":
            return class_probabilities(logits, "softmax", classes)
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        y_pred = nn.functional.sigmoid(logits)
        return y_pred

    def _log_images(self, x, y, y_pred, prefix):
        """Central slices of input, target and prediction; one target/prediction panel per
        class (tags keep the legacy names for a single-class model)."""
        tensorboard = self.logger.experiment
        size = y_pred.shape[2]
        grid_params = {
            "padding": 1,  # Smaller gap between images
            "pad_value": 1.0,  # White borders
        }

        def add(tag, volume):
            grid = torchvision.utils.make_grid(
                volume[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
            )
            tensorboard.add_image(tag, grid, global_step=self.current_epoch)

        add("input_val", x)
        n_classes = y_pred.shape[1]
        if n_classes == 1:
            add("label_val", y)
            add("prediction_val", y_pred)
            return
        names = [row.name for row in self.class_table()] if self.is_pipeline_v2() else []
        for c in range(n_classes):
            suffix = names[c] if c < len(names) else str(c)
            add(f"label_val_{suffix}", y[:, c:c + 1])
            add(f"prediction_val_{suffix}", y_pred[:, c:c + 1])


if __name__ == "__main__":
    model = BasePickingModel()
    batch_size = 3
    from tomocpt.mainConfig import mainConfig

    train_config = mainConfig.train
    batch = dict(
        input_data=dict(
            data=torch.randn(batch_size, 1, *(train_config.CHUNK_SIZE,) * 3)
        ),
        target_data=dict(
            data=torch.randn(batch_size, 1, *(train_config.CHUNK_SIZE,) * 3)
        ),
    )

    out = model.predict_step(batch["input_data"]["data"], 0, dataloader_idx=0)
    out = model.training_step(batch, 0)
