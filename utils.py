"""
Building blocks of Resnet function
Reference: MaskRCNN implementation of pytorch
PR~
"""

import torch
import torch.nn as nn
from torch import Tensor

"""
Frozen batch normalisation module
"""
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


"""
Bottleneck layers
Bottleneck layer 0 -> Input and output channel size is different, used as first module in every layer
Bottleneck layer   -> Input and output channel size is same
"""
    
class Bottleneck_layer0(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, inter_channel: int, stride: int = 1):
        super(Bottleneck_layer0, self).__init__()
       
        self.conv1 = nn.Conv2d(input_channel, inter_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(inter_channel)
        self.conv2 = nn.Conv2d(inter_channel, inter_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(inter_channel)
        self.conv3 = nn.Conv2d(inter_channel, output_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = FrozenBatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
                                        FrozenBatchNorm2d(output_channel))
        

    def forward(self, x: Tensor) -> Tensor:
        # block1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # block2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # block3
        out = self.conv3(out)
        out = self.bn3(out)
        # concat with identity (Main contribution of resnet allows deeper network avoiding vanishing gradient problem)
        out += self.downsample(x)
        out = self.relu(out)
        return out

class Bottleneck_layer(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, inter_channel: int, stride: int = 1):
        super(Bottleneck_layer, self).__init__()
       
        self.conv1 = nn.Conv2d(input_channel, inter_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(inter_channel)
        self.conv2 = nn.Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(inter_channel)
        self.conv3 = nn.Conv2d(inter_channel, output_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = FrozenBatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x: Tensor) -> Tensor:
        # block1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # block2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # block3
        out = self.conv3(out)
        out = self.bn3(out)
        # concat with identity
        out += x
        out = self.relu(out)
        return out