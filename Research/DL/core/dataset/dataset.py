from torch.utils.data import Dataset
from .sample import ObjectDetectionSample, Sample
from typing import List
from .image import VOCImage
from .annotation import VOCObjectDetectionAnnotation

from pathlib import Path
import logging
from abc import ABCMeta, abstractmethod
import json

logger = logging.getLogger(__name__)


class ObjectDetectionDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, dataset_dirname):
        self.dataset_dirname = dataset_dirname

        self.samples: List[ObjectDetectionSample] = None

        self.train_samples: List[ObjectDetectionSample] = None
        self.eval_samples: List[ObjectDetectionSample] = None
        self.test_samples: List[ObjectDetectionSample] = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @abstractmethod
    def read(self, *args, **kwargs) -> List[Sample]:
        pass

    @abstractmethod
    def write(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def to_generic_dataset(self):
    #     pass


class COCOObjectDetectionDataset(ObjectDetectionDataset):
    def __init__(self, dataset_dirname):
        super(COCOObjectDetectionDataset, self).__init__(dataset_dirname)

    def _write(self, new_dataset_dirname):



        pass

    def write(self, *args, **kwargs):
        self._write(args[0])


class YOLOObjectDetectionDataset(ObjectDetectionDataset):
    def __init__(self, dataset_dirname):
        super(YOLOObjectDetectionDataset, self).__init__(dataset_dirname)


class VOCObjectDetectionDataset(ObjectDetectionDataset):

    def __init__(self, dataset_dirname, image_transform=None, anno_transform=None):
        super(VOCObjectDetectionDataset, self).__init__(dataset_dirname)

        # 图像的转换
        self.image_transform = image_transform
        # 标签的转换
        self.anno_transform = anno_transform

        # self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径
        self.images_dirname = Path(self.dataset_dirname, 'JPEGImages')
        self.annos_dirname = Path(self.dataset_dirname, 'Annotations')
        self.split_txt_dirname = Path(self.dataset_dirname, 'ImageSets', 'Main')

        train_txt_abspath = Path(self.split_txt_dirname, 'train.txt')
        val_txt_abspath = Path(self.split_txt_dirname, 'val.txt')
        test_txt_abspath = Path(self.split_txt_dirname, 'test.txt')
        self.trainval_txt_abspath = Path(self.split_txt_dirname, 'val.txt')

        self.train_samples: List[ObjectDetectionSample] = self.read(train_txt_abspath)
        self.eval_samples: List[ObjectDetectionSample] = self.read(val_txt_abspath)
        self.test_samples: List[ObjectDetectionSample] = self.read(test_txt_abspath)

        # 默认为 train 模式
        self.mode = "train"
        self.samples: List[ObjectDetectionSample] = self.train_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _get_samples_path(self, txt_abspath):
        """
        根据voc数据集中的txt文件获取sample_id和相应的图片和标签的绝对路径
        """
        samples_path = []

        with open(txt_abspath, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            sample_id = line.strip()
            if not sample_id:
                logger.warning(f"{txt_abspath} : {line_num}行 为空白字符！")
                continue

            sample_id = Path(sample_id).stem
            image_abspath = Path(self.images_dirname, sample_id + '.jpg')
            anno_abspath = Path(self.annos_dirname, sample_id + '.xml')

            if not (Path(image_abspath).exists() and Path(anno_abspath).exists()):
                logger.error(f"{txt_abspath} : {line_num}行 {sample_id} 指向的图片或标签文件不存在！")
                continue
            else:
                samples_path.append((sample_id, image_abspath, anno_abspath))

        return samples_path

    def _read_image(self, image_abspath) -> VOCImage:

        return VOCImage(image_abspath, self.image_transform).read()

    def _read_anno(self, anno_abspath) -> VOCObjectDetectionAnnotation:
        return VOCObjectDetectionAnnotation(anno_abspath, self.anno_transform).read()

    def read(self, *args, **kwargs):
        return self._read(args[0])

    def _read(self, txt_abspath) -> List[ObjectDetectionSample]:
        """
        读取数据集
        """

        samples_path = self._get_samples_path(txt_abspath)
        samples = []

        for sample_id, image_abspath, anno_abspath in samples_path:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

        return samples

    def write(self, *args, **kwargs):
        """
        将数据集写到新的目录下
        """
        dataset, type, new_dataset_dirname = kwargs['dataset'], kwargs['type'], kwargs['new_dataset_dirname']
        self._write(dataset, new_dataset_dirname)

    def _write(self, new_dataset_dirname):
        # TODO
        pass

    def train(self):
        self.mode = "train"
        self.samples = self.train_samples

    def eval(self):
        self.mode = "eval"
        self.samples = self.eval_samples

    def test(self):
        self.mode = "test"
        self.samples = self.test_samples

    # def to_generic_dataset(self):
    #     """
    #     将self.sampes 转换成标准sample  即只有name 和 标注框
    #     """
    #
    #     generic_dataset = ObjectDetectionDataset()
    #
    #     new_train_samples = self.train_samples
    #     pass


class ObjectDetectionDatasetTransfer:

    def __init__(self, dataset: ObjectDetectionDataset):
        self.dataset = dataset

        self.dataset_dirname = self.dataset.dataset_dirname
        self.train_samples, self.eval_samples, self.test_samples = self.to_generic_dataset()

    def _to_generic_samples(self, samples):
        train_samples = []
        for sample in samples:
            oba = sample.anno.to_generic_anno()
            obi = sample.image.to_generic_image()

            new_sample = ObjectDetectionSample(self.dataset.samples, obi, oba)
            train_samples.append(new_sample)
        return train_samples

    def to_generic_dataset(self):
        train_samples = self._to_generic_samples(self.dataset.train_samples)
        eval_samples = self._to_generic_samples(self.dataset.eval_samples)
        test_samples = self._to_generic_samples(self.dataset.test_samples)

        return train_samples, eval_samples, test_samples


    def to_coco(self, dataset_dirname):
        for sample in self.train_samples:
            sample.anno

        pass

    def to_voc(self, ):
        pass

    def to_yolo(self, ):



        pass
