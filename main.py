"""
Resnet implementation based on pytorch
PR~
"""
import torch
from torchsummary import summary
from resnet50_custom import Resnet_50_custom
from resnet50_naive import Resnet_50_naive

def main(model_name):

    if model_name == "custom":
        model = Resnet_50_custom()
    else:
        model = Resnet_50_naive()

    gpu_available = torch.cuda.is_available() 
    if gpu_available:
        model = model.cuda()

    """
    Summary has some bug to represent naive model
    https://discuss.pytorch.org/t/repeated-model-layers-real-or-torchsummary-bug/26489/5
    """
    if model_name == "custom":
        summary(model, input_size=(3, 224, 224)) # random input size 


if __name__ == "__main__":

    """
    Two resnet50 class is created..
    1. resnet_50_naive --> Naive method - For detailed understanding, 
                      doesnot support loading of pretrained weights due 
                      to variation in the module name
    2. resnet_50_custom --> based on recent implementaion by pytorch
    """
    main("custom") # use "naive" for using naive resnet50 implementation, "Resnet_50_custom" is recommended