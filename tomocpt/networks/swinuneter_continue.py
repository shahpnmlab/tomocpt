import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import copy
from tomocpt.networks.swinunetr import MySwinUNETR


def get_conv_channels(model, module_name):
    """Helper function to find the number of channels in a module's Conv3d layers"""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and module_name in name:
            conv_layers.append(module)
    if not conv_layers:
        return None
    return conv_layers[-1].out_channels


class ContinualSwinUNETR(MySwinUNETR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_model = None
        self.temperature = 2.0

        # Get channel information from conv layers
        def get_channels(module):
            for m in module.modules():
                if isinstance(m, torch.nn.Conv3d):
                    return m.out_channels
            return None

        # Initialize feature adaptors after getting channel info
        self.feature_adaptors = nn.ModuleDict(
            {
                "enc1": nn.Sequential(
                    nn.Conv3d(
                        get_channels(self.encoder1), get_channels(self.encoder1), 1
                    ),
                    nn.BatchNorm3d(get_channels(self.encoder1)),
                    nn.ReLU(),
                    nn.Dropout3d(p=0.1),
                ),
                "enc2": nn.Sequential(
                    nn.Conv3d(
                        get_channels(self.encoder2), get_channels(self.encoder2), 1
                    ),
                    nn.BatchNorm3d(get_channels(self.encoder2)),
                    nn.ReLU(),
                    nn.Dropout3d(p=0.1),
                ),
                "enc3": nn.Sequential(
                    nn.Conv3d(
                        get_channels(self.encoder3), get_channels(self.encoder3), 1
                    ),
                    nn.BatchNorm3d(get_channels(self.encoder3)),
                    nn.ReLU(),
                    nn.Dropout3d(p=0.1),
                ),
            }
        )

    def load_previous_weights(self, weights_path: str):
        """Load weights from previous model and initialize as teacher"""
        # Load state dict
        state_dict = torch.load(weights_path, map_location="cpu")

        # Create a copy of current model as previous model
        self.previous_model = copy.deepcopy(self)
        self.previous_model.load_state_dict(state_dict)
        self.previous_model.eval()

        # Freeze previous model
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
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.encoder10,
        ]
        for i, encoder in enumerate(encoders):
            groups.append(
                {"params": encoder.parameters(), "lr_mult": float(encoder_mults[i])}
            )

        # Add decoder groups with progressive multipliers
        decoders = [
            self.decoder5,
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
        ]
        for i, decoder in enumerate(decoders):
            groups.append(
                {"params": decoder.parameters(), "lr_mult": float(decoder_mults[i])}
            )

        # Add feature adaptor groups if present
        if self.feature_adaptors is not None:
            for adaptor in self.feature_adaptors.values():
                groups.append(
                    {
                        "params": adaptor.parameters(),
                        "lr_mult": 1.0,  # Allow adaptors to learn quickly
                    }
                )

        # Output layer learns fastest
        groups.append({"params": self.out.parameters(), "lr_mult": 1.0})

        return groups

    def forward(self, x_in):
        # Get outputs from current model
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        # Apply feature adaptation if available
        if self.feature_adaptors is not None:
            enc0 = self.feature_adaptors["enc1"](enc0)
            enc1 = self.feature_adaptors["enc2"](enc1)
            enc2 = self.feature_adaptors["enc3"](enc2)

        # Continue with decoder path
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        if self.training and self.previous_model is not None:
            with torch.no_grad():
                prev_logits, prev_features = self.previous_model(x_in)
            return logits, hidden_states_out[4], prev_logits

        return logits, hidden_states_out[4]

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss between student and teacher"""
        # Soften probability distributions
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # Compute KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_probs,
            reduction="batchmean",
        ) * (self.temperature**2)

        return kd_loss * alpha
