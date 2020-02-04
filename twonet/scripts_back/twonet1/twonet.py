from torch import nn
from torch.nn import functional as F
import torch
from unetpp import NestedUNet

class Dual_net(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        self.middle_planes = 32
        self.net1 = NestedUNet(input_channels=input_channels)
        self.out1 = self._out_block1()

        self.convert = nn.Sequential(
            nn.Conv2d(self.middle_planes, self.middle_planes, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.middle_planes),
            nn.ReLU(inplace=True)
        )

        self.net2 = NestedUNet(input_channels=self.middle_planes)
        self.out2 = self._out_block2()
        self.block1 = self._block()

        self.final = nn.Conv2d(2*self.middle_planes, 2, kernel_size=3, padding=1)



    def _block(self):
        return nn.Sequential(
            nn.Conv2d(self.middle_planes, self.middle_planes, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.middle_planes),
            nn.ReLU(inplace=True)
        )

    def _out_block1(self):
        return nn.Conv2d(self.middle_planes, 2, kernel_size=3, padding=1)

    def _out_block2(self):
        return nn.Conv2d(self.middle_planes, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.net1(x)
        out1 = self.out1(x1)  # first output point

        x2 = self.net2(x1)
        out2 = self.out2(x2)  # second output point


        out3 = torch.cat([x1, x2], dim=1)
        out3 = self.final(out3)
        return [out1, out2, out3]
