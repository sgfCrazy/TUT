# from ...wrappers.model import Module
from typing import Union, Type, List
from ...common.shape_spec import ShapeSpec
from torch import nn
from .backbone import Backbone

from torchvision.models import resnet

"""
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

"""


class BasicStemBlock(nn.Module):
    # 2层 3*3 stride=1 的卷积
    expansion: int = 1

    def __init__(self, in_channels, out_channels, use_1x1: bool = False, downsample: bool = False):
        """
            use_1x1: 是否使用1x1卷积，每层的第一个block时需要使用1x1卷积统一维度。
            downsample：是否进行降采样，第二个layer因为前面有了一个最大池化进行了降采样，所以不用再降采样，而其他层则需要进行降采样，即尺寸减半。
        """
        super(BasicStemBlock, self).__init__()

        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2 if downsample else 1
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ) if use_1x1 else None

        self.stride = 2 if downsample else 1

    def forward(self, input_data):

        identity = input_data

        out = self.conv1(input_data)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        if self.conv_1x1:
            identity = self.conv_1x1(identity)

        out += identity
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion: int = 4  # 该block的维度扩张

    def __init__(self, in_channels, out_channels, use_1x1: bool = False, downsample: bool = False):
        super(BottleneckBlock, self).__init__()

        # out_channels = mid_channels * self.expansion
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=2 if downsample else 1
        )
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=mid_channels)

        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2 if downsample else 1
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ) if use_1x1 else None

        self.stride = 2 if downsample else 1

    def forward(self, input_data):
        identity = input_data
        out = self.conv1(input_data)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.conv_1x1:
            identity = self.conv_1x1(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(Backbone):

    def __init__(self,
                 block: Type[Union[BasicStemBlock, BottleneckBlock]],
                 block_nums: List[int],
                 input_shape: ShapeSpec = None):
        super().__init__()

        self.block = block
        assert len(block_nums) == 4, f"block_nums: {block_nums}  的长度必须为4！"
        self.block_nums = block_nums
        self.in_channels = input_shape.channels

        self._make_layers()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self):
        self.layer_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2,
                                 padding=3)  # b 64 112 112

        self.in_channels = 64

        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_stage(mid_channels=64, nums_duplicate=self.block_nums[0], layer_num=2)
        )

        self.layer_3 = nn.Sequential(
            self._make_stage(mid_channels=128, nums_duplicate=self.block_nums[1], layer_num=3)
        )

        self.layer_4 = nn.Sequential(
            self._make_stage(mid_channels=256, nums_duplicate=self.block_nums[2], layer_num=4)
        )

        self.layer_5 = nn.Sequential(
            self._make_stage(mid_channels=512, nums_duplicate=self.block_nums[3], layer_num=5)
        )

    def _make_stage(self, mid_channels, nums_duplicate: int, layer_num: int):

        stage = nn.Sequential()

        out_channels = mid_channels * self.block.expansion

        for block_i in range(nums_duplicate):
            layer_2 = (layer_num == 2)
            use_1x1 = (block_i == 0)  # 每一层的第一个block要进行对齐
            downsample = (block_i == 0 and not layer_2)  # 除了第二层的第一个block外，其余层的第一个block需要进行降采样

            stage.add_module(f'block_{block_i}',
                             self.block(self.in_channels, out_channels, use_1x1, downsample))

            self.in_channels = out_channels  # 下一个block的input_channels应该为上一个block的out_channels

        return stage

    def forward(self, input_data):
        # assert input_data.shape == (1, 3, 224, 224)
        out = self.layer_1(input_data)
        print(out.shape)
        # assert out.shape == (1, 64, 112, 112)
        out = self.layer_2(out)
        print(out.shape)
        # assert out.shape == (1, 256, 56, 56)
        out = self.layer_3(out)
        print(out.shape)
        # assert out.shape == (1, 512, 28, 28)
        out = self.layer_4(out)
        print(out.shape)
        # assert out.shape == (1, 1024, 14, 14)
        out = self.layer_5(out)
        print(out.shape)
        # assert out.shape == (1, 2048, 7, 7)
        return out