from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):

    # 参数列表中单独出现 * ， 表示 *　之后的参数必须以关键字形式进行传参
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)

