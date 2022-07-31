from ...utils.registry import Registry
from ...common.shape_spec import ShapeSpec

BACKBONE_REGISTRY = Registry("BACKBONE")


BACKBONE_REGISTRY.__doc__ = """

"""


def build_backbone(cfg, input_shape):

    backbone_name = cfg.MODEL.BACKBONE.NAME

    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    return backbone
