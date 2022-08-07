import cv2
import numpy as np
from abc import ABCMeta, abstractmethod

class Image(metaclass=ABCMeta):
    """
    一个样本对应的图片数据
    """

    def __init__(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass


class VOCImage(Image):

    def __init__(self, image_abspath, image_transform):
        super(VOCImage, self).__init__()

        self.image_abspath = image_abspath
        self.image_transform = image_transform
        self.data = self.read()

    def read(self):
        # 能够读取中文路径，c,h,w BGR
        image = cv2.imdecode(np.fromfile(self.image_abspath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image_transform:
            self.image_transform(image)
        return image

    def write(self):
        # TODO
        pass
