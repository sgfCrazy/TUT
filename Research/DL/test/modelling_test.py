import torch
from Research.DL.core.modeling.backbone.resnet import ResNet
from Research.DL.core.modeling.backbone.resnet import BottleneckBlock, BasicStemBlock
from Research.DL.core.common.shape_spec import ShapeSpec


def backbone_resnet_test():

    input_data = torch.rand([1, 3, 224, 224])

    resnet18 = ResNet(BasicStemBlock, [2, 2, 2, 2], ShapeSpec(channels=3, height=224, width=224, stride=1))
    # resnet34 = ResNet(BasicStemBlock, [3, 4, 6, 3], ShapeSpec(channels=3, height=224, width=224, stride=1))
    # resnet50 = ResNet(BottleneckBlock, [3, 4, 6, 3], ShapeSpec(channels=3, height=224, width=224, stride=1))
    # resnet101 = ResNet(BottleneckBlock, [3, 4, 23, 3], ShapeSpec(channels=3, height=224, width=224, stride=1))
    # resnet152 = ResNet(BottleneckBlock, [3, 8, 36, 3], ShapeSpec(channels=3, height=224, width=224, stride=1))
    print(resnet18)
    out = resnet18(input_data)
    # out = resnet34(input_data)
    # out = resnet50(input_data)
    # out = resnet101(input_data)
    # out = resnet152(input_data)




if __name__ == '__main__':
    backbone_resnet_test()