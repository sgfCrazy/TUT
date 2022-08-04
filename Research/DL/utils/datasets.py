

class DatasetReader:
    pass


class DatasetWriter:
    pass


# 目标检测类的图像数据集，最本质是images和annos，还有分割方式
class ODDatasetTransfer:  # 目标检测数据集转换
    def __init__(self, ):

        self.images = None
        self.annos = None

        pass

    def check(self):
        pass

    def read(self):
        pass

    def write(self):
        pass

    # def split(self):
    #     pass









class VOCReader(DatasetReader):
    def __init__(self):
        pass


class COCOReader(DatasetReader):
    def __init__(self):
        pass


class VOCWriter(DatasetWriter):
    pass


class COCOWriter(DatasetWriter):
    pass