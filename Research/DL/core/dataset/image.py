import cv2
import numpy as np
from pathlib import Path


class Image:
    """
    一个样本对应的图片数据
    """

    def __init__(self):
        pass

    def read(self, image_abspath, image_transform=None):
        raise NotImplementedError

    def write(self, image_abspath):
        raise NotImplementedError


class ObjectDetectionImage(Image):
    def __init__(self):
        super(ObjectDetectionImage, self).__init__()
        self.image_abspath = None
        self.image_transform = None
        self.data = None
        self.height, self.width, self.channels = None, None, None

    def read(self, image_abspath, image_transform=None):
        self.image_abspath = image_abspath
        self.image_transform = image_transform
        # 能够读取中文路径，c,h,w BGR
        image = cv2.imdecode(np.fromfile(self.image_abspath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image_transform:
            self.image_transform(image)

        self.data = image
        self.height, self.width, self.channels = self.data.shape

        return self

    def write(self, image_abspath):
        cv2.imencode(Path(image_abspath).suffix, self.data)[1].tofile(image_abspath)


    def to_generic_image(self):
        odi = ObjectDetectionImage()
        odi.image_abspath = self.image_abspath
        odi.image_transform = self.image_transform
        odi.data = self.data
        odi.width = self.width
        odi.height = self.height
        odi.channels = self.channels
        return odi


class VOCImage(ObjectDetectionImage):

    def __init__(self):
        super(VOCImage, self).__init__()




class YOLOImage(ObjectDetectionImage):

    def __init__(self):
        super(YOLOImage, self).__init__()


class COCOImage(ObjectDetectionImage):

    def __init__(self):
        super(COCOImage, self).__init__()


