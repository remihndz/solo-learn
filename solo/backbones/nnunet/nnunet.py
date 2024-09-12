# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from timm https://github.com/rwightman/pytorch-image-models/blob/master/timm/


import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.plans_handling import plans_handler
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights

class BasennUNet(nn.Module):
    def __init__(self, pathToPlan:str, pathToWeights:str=None, **kwargs):
        """
        kwargs should include at least:
        - input_channels: int
            The number of input channels
        - output_channels: int
            The number of output channels
        """
        super().__init__()

        planManager = plans_handler.PlansManager(pathToPlan)
        confDict = planManager.get_configuration('2d')
        self.FullNetwork = get_network_from_plans(
            confDict.network_arch_class_name,
            confDict.network_arch_init_kwargs,
            confDict.network_arch_init_kwargs_req_import,
            deep_supervision=False, **kwargs) 

        if pathToWeights is not None:
            load_pretrained_weights(self.FullNetwork, pathToWeights)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, x):
        return self.FullNetwork.encoder.forward(x)
            
@register_model
def nnUNet(pathToPlan:str, pathToWeights:str=None, **kwargs):
    """
    Note: here the network is called `encoder' because the `forward'
    method is that of the UNet's encoder module. However, the 
    `encoder' variable contains the whole network. 
    The whole UNet architecture can be accessed with: ```encoder.FullNetwork```
    """
    encoder = BasennUNet(pathToPlan, pathToWeights, **kwargs)
    return encoder
