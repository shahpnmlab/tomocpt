from monai.networks.nets import SwinUNETR
import torch


def _get_model_parameters_for_optimizer(self, different_lrs):
    vit_params_freeze = []
    vit_params_standard = []

    encoder_params_freeze = []
    encoder_params_standard = []
    decoder_params = []
    if different_lrs:
        for name, param in self.swinViT.named_parameters():
            if "norm" in name:
                vit_params_standard.append(param)
                continue
            vit_params_freeze.append(param)

        for name, param in dict(self.named_children()).items():
            if name.startswith("encoder"):
                if "norm" in name:
                    encoder_params_standard.append(param)
                    continue
                encoder_params_freeze.append(param)
            elif name.startswith("decoder"):
                decoder_params.append(param)
        parameters = [
            {"params": vit_params_freeze, "lr_mult": 0.0},
            {"params": vit_params_standard, "lr_mult": 1.0},
            {"params": encoder_params_freeze, "lr_mult": 0.1},
            {"params": encoder_params_standard, "lr_mult": 1.0},
            {"params": decoder_params, "lr_mult": 1.0},
        ]
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


def _test():
    n_voxels = 64
    model = SwinUNETR(img_size=(n_voxels, n_voxels, n_voxels),
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

    model = MySwinUNETR(img_size=(n_voxels, n_voxels, n_voxels),
                      in_channels=4,
                      out_channels=3,
                      feature_size=24,
                      use_checkpoint=False)
    out, hid = model(indata)
    print(out.shape, hid.shape)


if __name__ == "__main__":
    _test()