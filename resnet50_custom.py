"""
Resnet50_custom class
PR~
"""

from utils import *


"""
Class Resnet50 have predefined channel size based on original implementation
output: Features from all intermediate layers and final fc layer(classification task)
"""

class Resnet_50_custom(nn.Module):
    def __init__(self):
        super(Resnet_50_custom,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(*nn.ModuleList([Bottleneck_layer0(64,256,64,1),Bottleneck_layer(256,256,64,1),
                                Bottleneck_layer(256,256,64,1)]))
        self.layer2 = nn.Sequential(*nn.ModuleList([Bottleneck_layer0(256,512,128,2),Bottleneck_layer(512,512,128,1),
                                Bottleneck_layer(512,512,128,1),Bottleneck_layer(512,512,128,1)]))
        self.layer3 = nn.Sequential(*nn.ModuleList([Bottleneck_layer0(512,1024,256,2),Bottleneck_layer(1024,1024,256,1),
                  Bottleneck_layer(1024,1024,256,1),Bottleneck_layer(1024,1024,256,1),
                  Bottleneck_layer(1024,1024,256,1),Bottleneck_layer(1024,1024,256,1)]))
        self.layer4 = nn.Sequential(*nn.ModuleList([Bottleneck_layer0(1024,2048,512,2),Bottleneck_layer(2048,2048,512,1),
                  Bottleneck_layer(2048,2048,512,1)]))

        # For classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1000) #1000 is number of classes and # 4 is expansion layers
    
    def forward(self, x: Tensor) -> Tensor:
        # stem function 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #output of layer 1
        out_l1 = self.layer1(x) 
        #output of layer 2
        out_l2 = self.layer2(out_l1)
        #output of layer 3
        out_l3 = self.layer3(out_l2)
        #output of layer 4
        out_l4 = self.layer4(out_l3)
        # For classification task
        out = self.avgpool(out_l4)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out, out_l1, out_l2, out_l3, out_l4