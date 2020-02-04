from torch import nn
from torch.nn import functional as F
import torch
import unetpp 
import unetpp2

class Dual_net(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        self.middle_planes = 16
        self.net1 = unetpp.NestedUNet(input_channels=1,
                                                nb_filter = [16, 32, 64, 128, 256],
                                                layers=[12,8,6,3])
        self.out1 = self._out_block1()

        self.net2 = unetpp2.NestedUNet(input_channels=self.middle_planes,
                                                nb_filter = [16, 32, 64, 128, 256],
                                                layers=[3,4,6,3])
        self.out2 = self._out_block2()
        self.block1 = self._block()
        
        
        self.a1 = self._out()
        self.a2 = self._out()
        self.a3 = self._out()
        self.a4 = self._out()

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

    def _out(self):
        return nn.Sequential(
            nn.Conv2d(2*self.middle_planes, 2*self.middle_planes, kernel_size=3, padding=1),
            nn.InstanceNorm2d(2*self.middle_planes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x1 = self.net1(x)
        out1 = self.out1(x1)  # first output point

        x2,_,_,_ = self.net2(x1)
        out2 = self.out2(x2)  # second output point

        out3 = torch.cat([x1, x2], dim=1)
       
        
        out3 = self.final(out3)
        return out1, out2, out3
