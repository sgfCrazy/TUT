from abc import ABCMeta, abstractmethod
import torch.nn as nn
from ...common.shape_spec import ShapeSpec


__all__ = ["Backbone"]


#  metaclass=ABCMeta  加上这个后，子类必须实现被@abstractmethod修饰的函数，类似于接口、虚函数等概念。

class Backbone(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input_data):
        pass

    @property
    def size_divisibility(self):
        return 0

    @property
    def _output_features(self):
        return self._output_features if self._output_features else []

    @property
    def _output_feature_strides(self):
        return self._output_feature_strides if self._output_feature_strides else {}.setdefault(None)

    @property
    def _output_feature_channels(self):
        return self._output_feature_channels if self._output_feature_channels else {}.setdefault(None)

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._output_feature_channels[name], stride=self._output_feature_strides[name])
            for name in self._output_features
        }
