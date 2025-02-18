import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Sequence
import copy
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock


class IntegratedContinualSwinUNETR(SwinUNETR):
    """
    Integrated version of ContinualSwinUNETR that maintains compatibility with MONAI
    while adding continual learning capabilities.
    """

    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            feature_size: int = 48,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            temperature: float = 2.0,
            **kwargs
    ):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            **kwargs
        )

        self.previous_model = None
        self.temperature = temperature

        # Initialize feature adaptors
        self.feature_adaptors = nn.ModuleDict()
        self._initialize_feature_adaptors()

    def _initialize_feature_adaptors(self):
        """Initialize feature adaptation layers for continual learning"""

        def get_channels(module):
            for m in module.modules():
                if isinstance(m, nn.Conv3d):
                    return m.out_channels
            return None

        # Create adaptors for encoder outputs
        adaptor_configs = [
            ('enc1', self.encoder1),
            ('enc2', self.encoder2),
            ('enc3', self.encoder3)
        ]

        for name, module in adaptor_configs:
            channels = get_channels(module)
            if channels:
                self.feature_adaptors[name] = nn.Sequential(
                    nn.Conv3d(channels, channels, 1),
                    nn.BatchNorm3d(channels),
                    nn.ReLU(),
                    nn.Dropout3d(p=0.1)
                )

    def load_previous_weights(self, weights_path: str):
        """Load weights from previous model and initialize as teacher"""
        # Load state dict
        state_dict = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Create a copy of current model as previous model
        self.previous_model = copy.deepcopy(self)
        missing_keys = self.previous_model.load_state_dict(state_dict, strict=False)
        if missing_keys.missing_keys:
            print(f"Warning: Missing keys in previous model: {missing_keys.missing_keys}")

        # Freeze previous model
        self.previous_model.eval()
        for param in self.previous_model.parameters():
            param.requires_grad = False

    def freeze_backbone(self, freeze_norm_layers: bool = True):
        """Freeze SwinViT backbone layers with option to handle norm layers"""
        for name, param in self.swinViT.named_parameters():
            if not freeze_norm_layers and "norm" in name:
                continue
            param.requires_grad = False

    def get_layer_groups(self) -> List[Dict]:
        """Group parameters for different learning rates with adaptive scaling"""
        # Calculate base multipliers based on layer depth
        encoder_mults = torch.linspace(0.2, 0.4, 5)  # Progressive scaling for encoders
        decoder_mults = torch.linspace(0.5, 1.0, 5)  # Progressive scaling for decoders

        groups = [
            {"params": self.swinViT.parameters(), "lr_mult": 0.1},  # Backbone - slowest
        ]

        # Add encoder groups with progressive multipliers
        encoders = [
            self.encoder1, self.encoder2, self.encoder3,
            self.encoder4, self.encoder10
        ]
        for i, encoder in enumerate(encoders):
            groups.append({
                "params": encoder.parameters(),
                "lr_mult": float(encoder_mults[i])
            })

        # Add decoder groups with progressive multipliers
        decoders = [
            self.decoder5, self.decoder4, self.decoder3,
            self.decoder2, self.decoder1
        ]
        for i, decoder in enumerate(decoders):
            groups.append({
                "params": decoder.parameters(),
                "lr_mult": float(decoder_mults[i])
            })

        # Add feature adaptor groups
        for adaptor in self.feature_adaptors.values():
            groups.append({
                "params": adaptor.parameters(),
                "lr_mult": 1.0  # Allow adaptors to learn quickly
            })

        # Output layer learns fastest
        groups.append({"params": self.out.parameters(), "lr_mult": 1.0})

        return groups

    def forward(self, x_in, normalize: bool = True):
        """
        Forward pass with support for both standard and continual learning modes.

        Args:
            x_in: Input tensor
            normalize: Whether to normalize intermediate features

        Returns:
            During training with previous model:
                (logits, hidden_features, prev_logits)
            During training without previous model:
                (logits, hidden_features)
            During inference:
                logits
        """
        hidden_states_out = self.swinViT(x_in, normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        # Apply feature adaptation if available
        if self.feature_adaptors and self.training:
            enc0 = self.feature_adaptors["enc1"](enc0)
            enc1 = self.feature_adaptors["enc2"](enc1)
            enc2 = self.feature_adaptors["enc3"](enc2)

        # Decoder path
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        if self.training:
            if self.previous_model is not None:
                with torch.no_grad():
                    prev_outputs = self.previous_model(x_in)
                    # Handle both tuple and direct outputs from previous model
                    prev_logits = prev_outputs[0] if isinstance(prev_outputs, tuple) else prev_outputs
                return logits, hidden_states_out[4], prev_logits
            return logits, hidden_states_out[4]

        return logits

    def compute_distillation_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss between student and teacher.

        Args:
            student_logits: Predictions from current model
            teacher_logits: Predictions from previous model
            alpha: Weight for distillation loss

        Returns:
            Weighted distillation loss
        """
        # Soften probability distributions
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # Compute KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_probs,
            reduction="batchmean"
        ) * (self.temperature ** 2)

        return kd_loss * alpha

    def load_from(self, weights):
        """
        Load weights with compatibility for both MONAI and custom checkpoints.
        Extends MONAI's load_from with support for feature adaptors.
        """
        # First load weights using MONAI's method
        super().load_from(weights)

        # Then handle any feature adaptor weights if present
        state_dict = weights["state_dict"] if "state_dict" in weights else weights
        if any("feature_adaptors" in k for k in state_dict.keys()):
            adaptor_dict = {k: v for k, v in state_dict.items()
                            if "feature_adaptors" in k}
            self.load_state_dict(adaptor_dict, strict=False)