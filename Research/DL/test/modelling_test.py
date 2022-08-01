import torch
from Research.DL.core.modeling.backbone.resnet import ResNet
from Research.DL.core.modeling.backbone.resnet import BottleneckBlock, BasicStemBlock
from Research.DL.core.common.shape_spec import ShapeSpec


def backbone_resnet_test():
    input_data = torch.rand([1, 3, 224, 224])
    resnet = ResNet(BottleneckBlock, [3, 4, 6, 3], ShapeSpec(channels=3, height=224, width=224, stride=1))

    out = resnet(input_data)



if __name__ == '__main__':
    backbone_resnet_test()