import cv2
import numpy as np


class Image:
    """
    一个样本对应的图片数据
    """

    def __init__(self):
        pass

    def read(self):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError


class ObjectDetectionImage(Image):
    def __init__(self, image_abspath, image_transform=None):
        super(ObjectDetectionImage, self).__init__()

        self.image_abspath = image_abspath
        self.image_transform = image_transform

    def read(self):
        # 能够读取中文路径，c,h,w BGR
        image = cv2.imdecode(np.fromfile(self.image_abspath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image_transform:
            self.image_transform(image)

        self.data = image
        self.height, self.width, self.channels = self.data.shape

        return self

    def write(self):
        pass


class VOCImage(ObjectDetectionImage):

    def __init__(self, image_abspath, image_transform=None):
        super(VOCImage, self).__init__(image_abspath, image_transform)

    def to_generic_image(self):
        odi = ObjectDetectionImage(self.image_abspath, self.image_transform)
        odi.data = self.data
        odi.width = self.width
        odi.height = self.height
        odi.channels = self.channels
        return odi


class YOLOImage(ObjectDetectionImage):

    def __init__(self, image_abspath, image_transform=None):
        super(YOLOImage, self).__init__(image_abspath, image_transform)

    def to_generic_image(self):
        odi = ObjectDetectionImage(self.image_abspath, self.image_transform)
        # 是否使用data的引用 TODO
        odi.data = self.data
        odi.width = self.width
        odi.height = self.height
        odi.channels = self.channels
        return odi
