"""
Resnet50_naive class
PR~
"""

import torch
import torch.nn as nn
from torch import Tensor
from utils import FrozenBatchNorm2d

"""
Class Resnet50_naive have detailed representation of every function in sequential steps, helps beginer to get hold of architecture
output: Features from all intermediate layers and final fc layer(classification task)
"""

class Resnet_50_naive(nn.Module):
    def __init__(self):
        super(Resnet_50_naive,self).__init__()

        #stem
        self.stem_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_norm1 = FrozenBatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #layer_1  ---> 3 bottle neck
        ## Bottle neck 0
        self.layer1_b0_conv1= nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.layer1_b0_norm1 = FrozenBatchNorm2d(64)
        self.layer1_b0_conv2= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_b0_norm2 = FrozenBatchNorm2d(64)
        self.layer1_b0_conv3= nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_b0_norm3 = FrozenBatchNorm2d(256)
        self.layer1_b0_relu3 = nn.ReLU(inplace=True)
        self.layer1_b0_dwn_conv4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_b0_dwn_norm5 = FrozenBatchNorm2d(256)
        ## Bottle neck 1
        self.layer1_b1_conv1= nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.layer1_b1_norm1 = FrozenBatchNorm2d(64)
        self.layer1_b1_conv2= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_b1_norm2 = FrozenBatchNorm2d(64)
        self.layer1_b1_conv3= nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_b1_norm3 = FrozenBatchNorm2d(256)
        self.layer1_b1_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 2
        self.layer1_b2_conv1= nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.layer1_b2_norm1 = FrozenBatchNorm2d(64)
        self.layer1_b2_conv2= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_b2_norm2 = FrozenBatchNorm2d(64)
        self.layer1_b2_conv3= nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_b2_norm3 = FrozenBatchNorm2d(256)
        self.layer1_b2_relu3 = nn.ReLU(inplace=True)

        #layer_2 --> 4 bottleneck
        ## Bottle neck 0
        self.layer2_b0_conv1= nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.layer2_b0_norm1 = FrozenBatchNorm2d(128)
        self.layer2_b0_conv2= nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_b0_norm2 = FrozenBatchNorm2d(128)
        self.layer2_b0_conv3= nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.layer2_b0_norm3 = FrozenBatchNorm2d(512)
        self.layer2_b0_relu3 = nn.ReLU(inplace=True)
        self.layer2_b0_dwn_conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.layer2_b0_dwn_norm5 = FrozenBatchNorm2d(512)
        ## Bottle neck 1
        self.layer2_b1_conv1= nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.layer2_b1_norm1 = FrozenBatchNorm2d(128)
        self.layer2_b1_conv2= nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_b1_norm2 = FrozenBatchNorm2d(128)
        self.layer2_b1_conv3= nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.layer2_b1_norm3 = FrozenBatchNorm2d(512)
        self.layer2_b1_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 2
        self.layer2_b2_conv1= nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.layer2_b2_norm1 = FrozenBatchNorm2d(128)
        self.layer2_b2_conv2= nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_b2_norm2 = FrozenBatchNorm2d(128)
        self.layer2_b2_conv3= nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.layer2_b2_norm3 = FrozenBatchNorm2d(512)
        self.layer2_b2_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 3
        self.layer2_b3_conv1= nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.layer2_b3_norm1 = FrozenBatchNorm2d(128)
        self.layer2_b3_conv2= nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_b3_norm2 = FrozenBatchNorm2d(128)
        self.layer2_b3_conv3= nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.layer2_b3_norm3 = FrozenBatchNorm2d(512)
        self.layer2_b3_relu3 = nn.ReLU(inplace=True)

        #layer_3 --> 6 bottleneck
        ## Bottle neck 0
        self.layer3_b0_conv1= nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.layer3_b0_norm1 = FrozenBatchNorm2d(256)
        self.layer3_b0_conv2= nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_b0_norm2 = FrozenBatchNorm2d(256)
        self.layer3_b0_conv3= nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.layer3_b0_norm3 = FrozenBatchNorm2d(1024)
        self.layer3_b0_relu3 = nn.ReLU(inplace=True)
        self.layer3_b0_dwn_conv4 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.layer3_b0_dwn_norm5 = FrozenBatchNorm2d(1024)
        ## Bottle neck 1
        self.layer3_b1_conv1= nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.layer3_b1_norm1 = FrozenBatchNorm2d(256)
        self.layer3_b1_conv2= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_b1_norm2 = FrozenBatchNorm2d(256)
        self.layer3_b1_conv3= nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.layer3_b1_norm3 = FrozenBatchNorm2d(1024)
        self.layer3_b1_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 2
        self.layer3_b2_conv1= nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.layer3_b2_norm1 = FrozenBatchNorm2d(256)
        self.layer3_b2_conv2= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_b2_norm2 = FrozenBatchNorm2d(256)
        self.layer3_b2_conv3= nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.layer3_b2_norm3 = FrozenBatchNorm2d(1024)
        self.layer3_b2_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 3
        self.layer3_b3_conv1= nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.layer3_b3_norm1 = FrozenBatchNorm2d(256)
        self.layer3_b3_conv2= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_b3_norm2 = FrozenBatchNorm2d(256)
        self.layer3_b3_conv3= nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.layer3_b3_norm3 = FrozenBatchNorm2d(1024)
        self.layer3_b3_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 4
        self.layer3_b4_conv1= nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.layer3_b4_norm1 = FrozenBatchNorm2d(256)
        self.layer3_b4_conv2= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_b4_norm2 = FrozenBatchNorm2d(256)
        self.layer3_b4_conv3= nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.layer3_b4_norm3 = FrozenBatchNorm2d(1024)
        self.layer3_b4_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 5
        self.layer3_b5_conv1= nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.layer3_b5_norm1 = FrozenBatchNorm2d(256)
        self.layer3_b5_conv2= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_b5_norm2 = FrozenBatchNorm2d(256)
        self.layer3_b5_conv3= nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.layer3_b5_norm3 = FrozenBatchNorm2d(1024)
        self.layer3_b5_relu3 = nn.ReLU(inplace=True)

        #layer_4 --> 3 bottleneck
        ## Bottle neck 0
        self.layer4_b0_conv1= nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.layer4_b0_norm1 = FrozenBatchNorm2d(512)
        self.layer4_b0_conv2= nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_b0_norm2 = FrozenBatchNorm2d(512)
        self.layer4_b0_conv3= nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.layer4_b0_norm3 = FrozenBatchNorm2d(2048)
        self.layer4_b0_relu3 = nn.ReLU(inplace=True)
        self.layer4_b0_dwn_conv4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.layer4_b0_dwn_norm5 = FrozenBatchNorm2d(2048)
        ## Bottle neck 1
        self.layer4_b1_conv1= nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.layer4_b1_norm1 = FrozenBatchNorm2d(512)
        self.layer4_b1_conv2= nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_b1_norm2 = FrozenBatchNorm2d(512)
        self.layer4_b1_conv3= nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.layer4_b1_norm3 = FrozenBatchNorm2d(2048)
        self.layer4_b1_relu3 = nn.ReLU(inplace=True)
        ## Bottle neck 2
        self.layer4_b2_conv1= nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.layer4_b2_norm1 = FrozenBatchNorm2d(512)
        self.layer4_b2_conv2= nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_b2_norm2 = FrozenBatchNorm2d(512)
        self.layer4_b2_conv3= nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.layer4_b2_norm3 = FrozenBatchNorm2d(2048)
        self.layer4_b2_relu3 = nn.ReLU(inplace=True)

        # For classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1000) #1000 is number of classes and # 4 is expansion layers

        # stem
        self.stem = nn.Sequential(self.stem_conv1,self.stem_norm1,self.stem_relu)

        # layer 1 of resnet
        self.layer1_downsample = nn.Sequential(self.layer1_b0_dwn_conv4,
                                              self.layer1_b0_dwn_norm5)
        self.layer1_b0 = nn.Sequential(self.layer1_b0_conv1,
                                       self.layer1_b0_norm1,
                                       self.layer1_b0_conv2,
                                       self.layer1_b0_norm2,
                                       self.layer1_b0_conv3,
                                       self.layer1_b0_norm3,
                                       self.layer1_b0_relu3)
        self.layer1_b1 = nn.Sequential(self.layer1_b1_conv1,
                                       self.layer1_b1_norm1,
                                       self.layer1_b1_conv2,
                                       self.layer1_b1_norm2,
                                       self.layer1_b1_conv3,
                                       self.layer1_b1_norm3,
                                       self.layer1_b1_relu3)
        self.layer1_b2 = nn.Sequential(self.layer1_b2_conv1,
                                       self.layer1_b2_norm1,
                                       self.layer1_b2_conv2,
                                       self.layer1_b2_norm2,
                                       self.layer1_b2_conv3,
                                       self.layer1_b2_norm3,
                                       self.layer1_b2_relu3)

        # layer 2 of resnet
        self.layer2_downsample = nn.Sequential(self.layer2_b0_dwn_conv4,
                                              self.layer2_b0_dwn_norm5)
        self.layer2_b0 = nn.Sequential(self.layer2_b0_conv1,
                                       self.layer2_b0_norm1,
                                       self.layer2_b0_conv2,
                                       self.layer2_b0_norm2,
                                       self.layer2_b0_conv3,
                                       self.layer2_b0_norm3,
                                       self.layer2_b0_relu3)
        self.layer2_b1 = nn.Sequential(self.layer2_b1_conv1,
                                       self.layer2_b1_norm1,
                                       self.layer2_b1_conv2,
                                       self.layer2_b1_norm2,
                                       self.layer2_b1_conv3,
                                       self.layer2_b1_norm3,
                                       self.layer2_b1_relu3)
        self.layer2_b2 = nn.Sequential(self.layer2_b2_conv1,
                                       self.layer2_b2_norm1,
                                       self.layer2_b2_conv2,
                                       self.layer2_b2_norm2,
                                       self.layer2_b2_conv3,
                                       self.layer2_b2_norm3,
                                       self.layer2_b2_relu3)
        self.layer2_b3 = nn.Sequential(self.layer2_b3_conv1,
                                       self.layer2_b3_norm1,
                                       self.layer2_b3_conv2,
                                       self.layer2_b3_norm2,
                                       self.layer2_b3_conv3,
                                       self.layer2_b3_norm3,
                                       self.layer2_b3_relu3)
        # layer 3 of resnet
        self.layer3_downsample = nn.Sequential(self.layer3_b0_dwn_conv4,
                                              self.layer3_b0_dwn_norm5)
        self.layer3_b0 = nn.Sequential(self.layer3_b0_conv1,
                                       self.layer3_b0_norm1,
                                       self.layer3_b0_conv2,
                                       self.layer3_b0_norm2,
                                       self.layer3_b0_conv3,
                                       self.layer3_b0_norm3,
                                       self.layer3_b0_relu3)
        self.layer3_b1 = nn.Sequential(self.layer3_b1_conv1,
                                       self.layer3_b1_norm1,
                                       self.layer3_b1_conv2,
                                       self.layer3_b1_norm2,
                                       self.layer3_b1_conv3,
                                       self.layer3_b1_norm3,
                                       self.layer3_b1_relu3)
        self.layer3_b2 = nn.Sequential(self.layer3_b2_conv1,
                                       self.layer3_b2_norm1,
                                       self.layer3_b2_conv2,
                                       self.layer3_b2_norm2,
                                       self.layer3_b2_conv3,
                                       self.layer3_b2_norm3,
                                       self.layer3_b2_relu3)
        self.layer3_b3 = nn.Sequential(self.layer3_b3_conv1,
                                       self.layer3_b3_norm1,
                                       self.layer3_b3_conv2,
                                       self.layer3_b3_norm2,
                                       self.layer3_b3_conv3,
                                       self.layer3_b3_norm3,
                                       self.layer3_b3_relu3)
        self.layer3_b4 = nn.Sequential(self.layer3_b4_conv1,
                                       self.layer3_b4_norm1,
                                       self.layer3_b4_conv2,
                                       self.layer3_b4_norm2,
                                       self.layer3_b4_conv3,
                                       self.layer3_b4_norm3,
                                       self.layer3_b4_relu3)
        self.layer3_b5 = nn.Sequential(self.layer3_b5_conv1,
                                       self.layer3_b5_norm1,
                                       self.layer3_b5_conv2,
                                       self.layer3_b5_norm2,
                                       self.layer3_b5_conv3,
                                       self.layer3_b5_norm3,
                                       self.layer3_b5_relu3)
        # layer 4 of resnet
        self.layer4_downsample = nn.Sequential(self.layer4_b0_dwn_conv4,
                                              self.layer4_b0_dwn_norm5)
        self.layer4_b0 = nn.Sequential(self.layer4_b0_conv1,
                                       self.layer4_b0_norm1,
                                       self.layer4_b0_conv2,
                                       self.layer4_b0_norm2,
                                       self.layer4_b0_conv3,
                                       self.layer4_b0_norm3,
                                       self.layer4_b0_relu3)
        self.layer4_b1 = nn.Sequential(self.layer4_b1_conv1,
                                       self.layer4_b1_norm1,
                                       self.layer4_b1_conv2,
                                       self.layer4_b1_norm2,
                                       self.layer4_b1_conv3,
                                       self.layer4_b1_norm3,
                                       self.layer4_b1_relu3)
        self.layer4_b2 = nn.Sequential(self.layer4_b2_conv1,
                                       self.layer4_b2_norm1,
                                       self.layer4_b2_conv2,
                                       self.layer4_b2_norm2,
                                       self.layer4_b2_conv3,
                                       self.layer4_b2_norm3,
                                       self.layer4_b2_relu3)
    
    def forward(self,x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stem_maxpool(x)

        out_l1 = self.layer1_b0(x)
        out_l1 = out_l1 + self.layer1_downsample(x)
        out_l1 = self.layer1_b1(out_l1)
        out_l1 = self.layer1_b2(out_l1)

        out_l2 = self.layer2_b0(out_l1)
        out_l2 = out_l2 + self.layer2_downsample(out_l1)
        out_l2 = self.layer2_b1(out_l2)
        out_l2 = self.layer2_b2(out_l2)
        out_l2 = self.layer2_b3(out_l2)

        out_l3 = self.layer3_b0(out_l2)
        out_l3 = out_l3 + self.layer3_downsample(out_l2)
        out_l2 = self.layer3_b1(out_l3)
        out_l3 = self.layer3_b2(out_l3)
        out_l3 = self.layer3_b3(out_l3)
        out_l3 = self.layer3_b4(out_l3)
        out_l3 = self.layer3_b5(out_l3)

        out_l4 = self.layer4_b0(out_l3)
        out_l4 = out_l4+ self.layer4_downsample(out_l3)
        out_l4 = self.layer4_b1(out_l4)
        out_l4 = self.layer4_b2(out_l4)

        net_output = self.avgpool(out_l4)
        net_output = torch.flatten(net_output, 1)
        net_output = self.fc(net_output)

        return net_output, out_l1, out_l2, out_l3, out_l4 