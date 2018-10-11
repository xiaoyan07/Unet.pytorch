import torch
import torch.nn as nn


class FlexEncoder(nn.Module):
    def __init__(self,
                 block_dims=(8, 16, 32, 64, 128, 256, 512, 1024),
                 use_batchnorm=False,
                 ):
        super(FlexEncoder, self).__init__()
        in_channel = 3
        self.block_list = nn.ModuleList()
        self.maxpool2d = nn.MaxPool2d(2)
        for idx, out_dim in enumerate(block_dims):
            if use_batchnorm:
                block = nn.Sequential(
                    nn.Conv2d(in_channel, out_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channel, out_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True),
                )
            in_channel = out_dim
            self.block_list.append(block)

    def forward(self, inputs):
        x = inputs
        feat_list = []
        for block in self.block_list:
            x = block(x)
            feat_list.append(x)
            x = self.maxpool2d(x)

        return feat_list
