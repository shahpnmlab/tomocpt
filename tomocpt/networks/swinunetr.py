from monai.networks.nets import SwinUNETR
import torch

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


def test():
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


    #TODO: downsample to 10A/pxl

if __name__ == "__main__":
    test()