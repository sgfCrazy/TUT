from torch import nn
from ..common.shape_spec import ShapeSpec


class Module(nn.Module):
    def __int__(self, cfg, input_shape: ShapeSpec = None):
        super(Module, self).__int__()

    @property
    def output_shape(self) -> ShapeSpec:
        raise NotImplementedError
