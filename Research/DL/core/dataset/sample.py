from image import Image
from annotation import Annotation


class Sample:

    def __init__(self, sample_id):
        self.sample_id = sample_id




class ObjectDetectionSample(Sample):

    def __init__(self, sample_id, image:Image, anno:Annotation):
        super(ObjectDetectionSample, self).__init__(sample_id=sample_id)
        self.sample_id = sample_id
        self.image = image
        self.anno = anno

