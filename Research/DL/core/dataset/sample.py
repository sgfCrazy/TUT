import copy
from .image import Image
from .annotation import Annotation


class Sample:
    def __init__(self, sample_id):
        self.sample_id = sample_id


class ObjectDetectionSample(Sample):

    def __init__(self, sample_id, image: Image, anno: Annotation):
        super(ObjectDetectionSample, self).__init__(sample_id=sample_id)
        self.image: Image = image
        self.anno: ObjectDetectionSample = anno

# class VOCObjectDetectionSample(ObjectDetectionSample):
#     def __init__(self, sample_id, image: Image, anno: Annotation):
#         super(VOCObjectDetectionSample, self).__init__(sample_id, image, anno)
#
#     def to_generic_sample(self):
#
#
#
#         pass
