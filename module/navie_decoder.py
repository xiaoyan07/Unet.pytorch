import torch
import torch.nn as nn
import enum
import torch.nn.functional as F


class UpsampleType(enum.Enum):
    Bilinear = 0
    Deconv = 1


class FlexDecoder(nn.Module):
    def __init__(self,
                 num_classes,
                 in_max_channel=1024,
                 block_dims=(512, 256, 128, 64, 32, 16, 8),
                 upsample_type=UpsampleType.Deconv,
                 use_batchnorm=False,
                 use_softmax=False):
        super(FlexDecoder, self).__init__()
        self.upsample2d_type = upsample_type
        self.use_softmax = use_softmax

        if upsample_type == UpsampleType.Bilinear:
            self.upsample2d = nn.UpsamplingBilinear2d(scale_factor=2.)
        elif upsample_type == UpsampleType.Deconv:
            self.upsample2d = nn.ModuleList()

        in_channel = in_max_channel
        self.block_list = nn.ModuleList()

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
            if upsample_type == UpsampleType.Deconv:
                upsample_op = nn.Sequential(
                    nn.ConvTranspose2d(in_channel, out_dim, 4, 2, 1, bias=False),
                    nn.ReLU(inplace=True)
                )
                self.upsample2d.append(upsample_op)

            in_channel = out_dim
            self.block_list.append(block)

        self.cls_pred_conv = nn.Conv2d(in_channel, num_classes, 1, bias=True)

    def forward(self, inputs):
        feat_list = inputs

        feat_list.reverse()
        x_i = feat_list[0]
        for idx, feat in enumerate(feat_list[:-1]):
            x_i_before = feat_list[idx + 1]
            if self.upsample2d_type == UpsampleType.Deconv:
                p_i = self.upsample2d[idx](x_i)
            elif self.upsample2d_type == UpsampleType.Bilinear:
                p_i = self.upsample2d(x_i)
            else:
                raise NotImplementedError()

            concat_i = torch.cat([p_i, x_i_before], dim=1)

            out_i = self.block_list[idx](concat_i)

            x_i = out_i

        logit = self.cls_pred_conv(x_i)
        if self.use_softmax:
            prob = F.softmax(logit, dim=1)
        else:
            prob = F.sigmoid(logit)
        ret = {
            'logit': logit,
            'prob': prob
        }
        return ret
