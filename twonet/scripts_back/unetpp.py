# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from backbone import resnext50_32x4d, ResNet, Bottleneck

# Model architecture


class GlobalWeightedRankPooling2d(nn.Module):
    r"""Implements a global weighted rank pooling as defined on 2D inputs as defined in Qiuquiang Kong et Al. "A
        joint separation-classification model for sound event detection of weakly labelled data". In: CoRR
        abs/1711.03037 (2017). arXiv: 1711.03037. URL:http://arxiv.org/abs/1711.03037.
        Input shape: (N, C, H, W)
        Output shape: (N, C)
        sorted = sorted_in_descending_order(input)
        output[n, c, 0, 0] = (1/Z) * sum {j=0 to H*W} sorted[j] * d_c**j
        Z = sum {j=0 to H*W} (d_c**j)
        d_c is parametrizing the weights.
    """

    @classmethod
    def default_config(cls):
        r"""The pooling depend on the parameter dc.
        Returns:
            A dictionary with the default value for the parameter dc.
        """

        return {"dc": 0.999}

    def __init__(self, config):
        r"""Constructor: initializes the parameter dc.
        Args:
            config (dict): Expected to contain an entry for the key 'dc' with a floating point number in
            [0.0, 1.0] as value.
        """

        super(GlobalWeightedRankPooling2d, self).__init__()
        self.dc = config["dc"]

    def forward(self, x):
        # Flatten the height and width axis
        x = x.view(x.shape[0], x.shape[1], -1)
        # Sort each input[n, c, :] in descending order
        x, _ = torch.sort(x, dim=2, descending=True)
        # Build a tensor for the weights: dc**(j-1)
        weights = torch.tensor([self.dc ** j for j in range(x.shape[2])], device=x.device)
        # Compute the term Z.
        norm_term = torch.sum(weights)
        # Compute global weighted rank pooling for each channel.
        y = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        for c in range(x.shape[1]):
            y[:, c] = torch.sum(x[:, c, :] * weights, dim=1) / norm_term
        return y

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.GlobalWeightedRankPooling = GlobalWeightedRankPooling2d({"dc": 0.999})

    def forward(self, x):
        # y = self.GlobalWeightedRankPooling(x).unsqueeze(2).unsqueeze(3)
        # y = self.conv_du(y)
        # return x * y
        return x

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.InstanceNorm2d(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.InstanceNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(0.2,inplace=True)):
        super(VGGBlock, self).__init__()
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels),
            )
        # self.d1conv = ConvOffset2D(in_channels)

    def forward(self, x, flag=0):

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.dropout2d(out)
        out = self.act_func(out)


        return out

class Clstm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Clstm, self).__init__()
        # self.clstm = ConvLSTM(input_channels=in_channels, hidden_channels=[128, 64, out_channels], kernel_size=3, step=5,
        #                 effective_step=[2, 4, 8])
        self.conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        out = self.clstm(x)
        out = self.conv(out[0][0])
        return out


class Add_fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Add_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        x = self.conv1(x)
        assert x.shape==y.shape
        x = x + y
        x = self.conv2(x)
        return x



class NestedUNet(nn.Module):
    def __init__(self, input_channels=1,
    nb_filter = [32, 64, 128, 256, 512],
    layers=[3,4,6,3]):
        super().__init__()
        self.deepsupervision = False




        backbone_num = [64, 128, 256, 512]

        self.dropout2d = nn.Dropout2d(p=0.05)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(32, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.ca0 = CALayer(nb_filter[0]+nb_filter[1])
        self.ca1 = CALayer(nb_filter[1]+nb_filter[2])
        self.ca2 = CALayer(nb_filter[2]+nb_filter[3])
        self.ca3 = CALayer(nb_filter[3]+nb_filter[4])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.ca4 = CALayer(nb_filter[0]*2+nb_filter[1])
        self.ca5 = CALayer(nb_filter[1]*2+nb_filter[2])
        self.ca6 = CALayer(nb_filter[2]*2+nb_filter[3])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.ca7 = CALayer(nb_filter[0]*3+nb_filter[1])
        self.ca8 = CALayer(nb_filter[1]*3+nb_filter[2])


        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.last = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.out_2 = nn.Softmax(dim=1)

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 2, kernel_size=1)

        self.ResNet = ResNet(input_channels, Bottleneck, layers, nb_filter)

        # N FUSION MODULE

        self.fusion1 = Add_fusion(nb_filter[0], nb_filter[1])
        self.fusion2 = Add_fusion(nb_filter[1], nb_filter[2])
        self.fusion3 = Add_fusion(nb_filter[2], nb_filter[3])
        self.ac1 = nn.Conv2d(nb_filter[3], nb_filter[2], 1)
        self.ac2 = nn.Conv2d(nb_filter[2], nb_filter[1], 1)
        self.ac3 = nn.Conv2d(nb_filter[1], nb_filter[0], 1)

        self.fusion_conv = nn.Conv2d(nb_filter[0], 2, kernel_size=1)

        self.c1 = nn.Conv2d(nb_filter[1], 2, 1)
        self.c2 = nn.Conv2d(nb_filter[2], 2, 1)
        self.c3 = nn.Conv2d(nb_filter[3], 2, 1)


    def forward(self, input):
        x_shape = input.shape
        input = F.upsample(input, size=((input.size(2) // 32 * 32), (input.size(3) // 32 * 32) ) , mode='bilinear')
        x0, x1, x2, x3, x4 = self.ResNet(input)


        del input
        x0_0 = self.conv0_0(x0)
        # x1_0 = self.conv1_0(self.pool(x0_0))

        x0_1 = self.conv0_1(self.ca0(torch.cat([x0_0, self.up(x1)], 1))) # torch.Size([1, 32, 576, 544])

        # x2_0 = self.conv2_0(self.pool(x1))
        x1_1 = self.conv1_1(self.ca1(torch.cat([x1, self.up(x2)], 1))) #
        x0_2 = self.conv0_2(self.ca4(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))) # torch.Size([1, 32, 576, 544])

        # x3_0 = self.conv3_0(self.pool(x2))
        x2_1 = self.conv2_1(self.ca2(torch.cat([x2, self.up(x3)], 1)))
        x1_2 = self.conv1_2(self.ca5(torch.cat([x1, x1_1, self.up(x2_1)], 1)))
        x0_3 = self.conv0_3(self.ca7(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)))  # torch.Size([1, 32, 576, 544])

        # x4_0 = self.conv4_0(self.pool(x3))
        x3_1 = self.conv3_1(self.ca3(torch.cat([x3, self.up(x4)], 1)))


        x2_2 = self.conv2_2(self.ca6(torch.cat([x2, x2_1, self.up(x3_1)], 1)))


        x1_3 = self.conv1_3(self.ca8(torch.cat([x1, x1_1, x1_2, self.up(x2_2)], 1)))


        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)) # torch.Size([1, 32, 576, 544])




        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)


            output = output1+output2+output3+output4
            output = self.last(output)
            output = F.upsample(output, size=(x_shape[2], x_shape[3]), mode='bilinear')
            return output

        else:
            # out = self.final(x0_4)
            # output =self.out_2(out)
            out = F.upsample(x0_4, size=(x_shape[2], x_shape[3]), mode='bilinear')
            return out