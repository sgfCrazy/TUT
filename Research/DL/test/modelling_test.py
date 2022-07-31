import torch
from Research.DL.core.modeling.backbone.resnet import ResNet




def backbone_resnet_test():
    input_data = torch.rand([1, 3, 224, 224])
    resnet = ResNet(None, None)

    out = resnet(input_data)



if __name__ == '__main__':
    backbone_resnet_test()