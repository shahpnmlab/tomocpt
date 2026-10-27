import torch
from monai.networks.nets import SwinUNETR

from tomocpt.logger import get_logger

logging = get_logger()


def _get_model_parameters_for_optimizer(self, different_lrs):
    vit_params = []
    encoder_params = []
    decoder_params = []

    if different_lrs:
        # Collect ViT parameters (including norm layers)
        for name, param in self.swinViT.named_parameters():
            vit_params.append(param)

        # Collect encoder and decoder parameters
        for name, param in self.named_parameters():
            # Skip parameters already in ViT
            if any(p is param for p in vit_params):
                continue

            if "encoder" in name:
                encoder_params.append(param)
            elif "decoder" in name:
                decoder_params.append(param)

        # Create parameter groups with a more gradual learning rate decrease
        parameters = []
        if vit_params:
            parameters.append({"params": vit_params, "lr_mult": 0.01})  # Fine-tune backbone slowly
        if encoder_params:
            parameters.append({"params": encoder_params, "lr_mult": 0.1})  # Mid-level features adapt moderately
        if decoder_params:
            parameters.append({"params": decoder_params, "lr_mult": 1.0})  # Task-specific parts learn fully

        return parameters

    return self.parameters()

SwinUNETR._get_model_parameters_for_optimizer = _get_model_parameters_for_optimizer

class MySwinUNETR(SwinUNETR):
    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits, hidden_states_out[4]


class ScaleConditionedSwinUNETR(MySwinUNETR):
    """MySwinUNETR with FiLM conditioning on the working voxel size in the decoder path.

    Six FiLM points, after ``encoder10`` (16fs), ``decoder5`` (8fs), ``decoder4`` (4fs),
    ``decoder3`` (2fs), ``decoder2`` (fs) and ``decoder1`` (fs). The Swin encoder stays
    unconditioned. All new modules live under ``self.cond``; they start at zero, so legacy
    weights give the legacy output. ``forward(x, apix=None)`` is exactly ``MySwinUNETR.forward``.
    """

    def __init__(self, *args, cond_dim: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        from tomocpt.networks.conditioning import FiLM, ScaleEmbedding

        fs = self.out.conv.conv.in_channels  # decoder1 output = feature_size
        self.cond = torch.nn.Module()
        self.cond.scale_embed = ScaleEmbedding(cond_dim)
        self.cond.film = torch.nn.ModuleList(
            [FiLM(c, cond_dim) for c in (16 * fs, 8 * fs, 4 * fs, 2 * fs, fs, fs)]
        )

    def forward(self, x_in, apix=None):
        """``apix``: None, a float, or a tensor ``[B]`` / ``[B, 1]`` in Å per working voxel."""
        if apix is None:
            return super().forward(x_in)
        e = self.cond.scale_embed(apix)
        film = self.cond.film
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = film[0](self.encoder10(hidden_states_out[4]), e)
        dec3 = film[1](self.decoder5(dec4, hidden_states_out[3]), e)
        dec2 = film[2](self.decoder4(dec3, enc3), e)
        dec1 = film[3](self.decoder3(dec2, enc2), e)
        dec0 = film[4](self.decoder2(dec1, enc1), e)
        out = film[5](self.decoder1(dec0, enc0), e)
        logits = self.out(out)
        return logits, hidden_states_out[4]


def load_legacy_state_dict(model, state_dict):
    """Load a state dict written without conditioning modules (``strict=False``).

    Returns ``(missing, unexpected)``. Raises ``RuntimeError`` if a key is unexpected or a
    missing key is not a conditioning key (``cond.*``)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith("cond.")]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Cannot load legacy weights: unexpected keys {list(unexpected)}, "
            f"missing non-conditioning keys {bad_missing}"
        )
    return missing, unexpected


def load_single_class_into_multiclass(model, state_dict) -> None:
    """Warm-start a C-channel model from a single-class state dict.

    The one-channel ``out`` layer is copied into every channel (each class starts as the
    old picker); everything else loads as in ``load_legacy_state_dict``."""
    state_dict = dict(state_dict)
    target = model.state_dict()
    n_channels = None
    for key in [k for k in state_dict if k.startswith("out.")]:
        old = state_dict[key]
        new_shape = target[key].shape
        if old.shape[0] == 1 and new_shape[0] > 1:
            state_dict[key] = old.expand(new_shape[0], *old.shape[1:]).clone()
            n_channels = new_shape[0]
    load_legacy_state_dict(model, state_dict)
    if n_channels is not None:
        logging.info(f"Single-class weights loaded; output channel copied into {n_channels} class channels.")
    else:
        logging.info("Single-class weights loaded.")


def _test():
    n_voxels = 64
    model = SwinUNETR(
        img_size=(n_voxels, n_voxels, n_voxels),
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=False,
    )

    model = torch.nn.Sequential(model, torch.nn.Sigmoid())
    indata = torch.rand(8, 4, n_voxels, n_voxels, n_voxels)
    # model = model.cuda()
    # indata = indata.cuda()
    out = model(indata)
    print(out.shape)

    model = MySwinUNETR(
        img_size=(n_voxels, n_voxels, n_voxels),
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=False,
    )
    out, hid = model(indata)
    print(out.shape, hid.shape)


if __name__ == "__main__":
    _test()
