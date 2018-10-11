import torch
import torch.nn as nn
import enum
import torch.nn.functional as F

from module.naive_encoder import FlexEncoder
from module.navie_decoder import FlexDecoder, UpsampleType


class DeepUnet(nn.Module):
    def __init__(self,
                 num_classes,
                 use_softmax=False,
                 encoder_batchnorm=False,
                 decoder_batchnorm=False):
        super(DeepUnet, self).__init__()

        self.encoder = FlexEncoder(block_dims=(8, 16, 32, 64, 128, 256, 512, 1024),
                                   use_batchnorm=encoder_batchnorm)
        self.decoder = FlexDecoder(
            num_classes=num_classes,
            in_max_channel=1024,
            block_dims=(512, 256, 128, 64, 32, 16, 8),
            upsample_type=UpsampleType.Deconv,
            use_batchnorm=decoder_batchnorm,
            use_softmax=use_softmax
        )

    def forward(self, inputs):
        x = inputs
        feat_list = self.encoder(x)

        ret = self.decoder(feat_list)

        return ret


if __name__ == '__main__':
    model = DeepUnet(1)
    im = torch.ones((1, 3, 512, 512), dtype=torch.float32)

    o = model(im)

    print(o['logit'].shape)
