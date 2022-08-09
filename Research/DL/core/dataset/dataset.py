from torch.utils.data import Dataset
from .sample import ObjectDetectionSample, Sample
from typing import List
from .image import VOCImage, YOLOImage
from .annotation import VOCObjectDetectionAnnotation, YOLOObjectDetectionAnnotation

from pathlib import Path
import logging
from abc import ABCMeta, abstractmethod
import json

logger = logging.getLogger(__name__)


class ObjectDetectionDataset(Dataset):

    def __init__(self, image_transform=None, anno_transform=None):
        super(ObjectDetectionDataset, self).__init__()
        self.dataset_dirname = None
        self.train_samples: List[ObjectDetectionSample] = None
        self.eval_samples: List[ObjectDetectionSample] = None
        self.test_samples: List[ObjectDetectionSample] = None

        # 图像的转换
        self.image_transform = image_transform
        # 标签的转换
        self.anno_transform = anno_transform

        self.mode_dict = {
            'train': self.train,
            'eval': self.eval,
            'test': self.test
        }

        self.mode = "train"
        self.mode_dict[self.mode]()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def read(self, dataset_dirname) -> List[Sample]:
        self.dataset_dirname = dataset_dirname

    def write(self, *args, **kwargs):
        pass

    def to_generic_dataset(self, new_dataset_dirname=None):
        odd = ObjectDetectionDataset()
        odd.dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname

        def _to_generic_samples(samples):
            new_samples = []
            for sample in samples:
                oba = sample.anno.to_generic_anno()
                obi = sample.image.to_generic_image()

                new_sample = ObjectDetectionSample(sample.sample_id, obi, oba)
                new_samples.append(new_sample)
            return new_samples

        odd.train_samples = _to_generic_samples(self.train_samples)
        odd.eval_samples = _to_generic_samples(self.eval_samples)
        odd.test_samples = _to_generic_samples(self.test_samples)
        odd.mode_dict[odd.mode]()

        return odd

    def to_yolo(self, new_dataset_dirname=None):
        yolo_odd = YOLOObjectDetectionDataset()
        yolo_odd.dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname

        samples_set = [self.train_samples, self.eval_samples, self.test_samples]
        new_samples_set = [yolo_odd.train_samples, yolo_odd.eval_samples, yolo_odd.test_samples]
        for i, samples in enumerate(samples_set):
            new_samples = []
            for sample in samples:
                # 图像转换
                new_image = sample.image
                # 标签转换
                new_anno = sample.anno
                ods = ObjectDetectionSample(sample_id=sample.sample_id, image=new_image, anno=new_anno)
                new_samples.append(ods)

            new_samples_set[i] = new_samples

    def train(self):
        self.mode = "train"
        self.samples = self.train_samples

    def eval(self):
        self.mode = "eval"
        self.samples = self.eval_samples

    def test(self):
        self.mode = "test"
        self.samples = self.test_samples


class COCOObjectDetectionDataset(ObjectDetectionDataset):
    def __init__(self):
        super(COCOObjectDetectionDataset, self).__init__()

    def _write(self, new_dataset_dirname):
        pass

    def write(self, *args, **kwargs):
        self._write(args[0])


class YOLOObjectDetectionDataset(ObjectDetectionDataset):
    def __init__(self, image_transform=None, anno_transform=None):
        super(YOLOObjectDetectionDataset, self).__init__(image_transform, anno_transform)

    def read(self, dataset_dirname, type_set=None):
        self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径
        self._read(type_set)

    def _read_anno(self, anno_abspath) -> YOLOObjectDetectionAnnotation:
        return YOLOObjectDetectionAnnotation(anno_abspath, self.anno_transform).read()

    def _read_image(self, image_abspath) -> YOLOImage:
        return YOLOImage(image_abspath, self.image_transform).read()

    def _get_samples_path(self, images_dirname, annos_dirname):
        # TODO 目前只适配了一个jpg

        annos_path = Path(annos_dirname).glob('*.txt')

        for anno_path in annos_path:
            sample_id = Path(anno_path).stem
            images_path = Path(images_dirname).glob(f'{sample_id}.jpg')[0]

        return

    def _get_samples(self, images_dirname, annos_dirname):
        samples_path = self._get_samples_path(images_dirname, annos_dirname)
        samples = []

        for sample_id, image_abspath, anno_abspath in samples_path:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

        return samples

    def _read(self, type_set):
        """
        读取数据集
        """
        if type_set is None:
            type_set = ['train', 'test', 'eval']
        for type in type_set:
            images_dirname = Path(self.dataset_dirname, 'images', '%s' % type)
            annos_dirname = Path(self.dataset_dirname, 'labels', '%s' % type)

            samples = self._get_samples(images_dirname, annos_dirname)
            if type == "train":
                self.train_samples = samples
            elif type == "eval":
                self.eval_samples = samples
            elif type == "test":
                self.test_samples = samples
            else:
                raise ValueError

        # 默认为 train 模式
        self.mode = "train"
        self.samples: List[ObjectDetectionSample] = self.train_samples


class VOCObjectDetectionDataset(ObjectDetectionDataset):

    def __init__(self, image_transform=None, anno_transform=None):
        super(VOCObjectDetectionDataset, self).__init__(image_transform, anno_transform)


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

    def read(self, dataset_dirname):
        self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径
        self._read()
        return self

    def _get_samples(self, txt_abspath):
        samples_path = self._get_samples_path(txt_abspath)
        samples = []

        for sample_id, image_abspath, anno_abspath in samples_path:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

        return samples

    def _read(self):
        """
        读取数据集
        """
        self.images_dirname = Path(self.dataset_dirname, 'JPEGImages')
        self.annos_dirname = Path(self.dataset_dirname, 'Annotations')
        self.split_txt_dirname = Path(self.dataset_dirname, 'ImageSets', 'Main')

        train_txt_abspath = Path(self.split_txt_dirname, 'train.txt')
        val_txt_abspath = Path(self.split_txt_dirname, 'val.txt')
        test_txt_abspath = Path(self.split_txt_dirname, 'test.txt')
        # trainval_txt_abspath = Path(self.split_txt_dirname, 'trainval.txt')

        # txts_abspath = [train_txt_abspath, val_txt_abspath, test_txt_abspath, trainval_txt_abspath]

        self.train_samples: List[ObjectDetectionSample] = self._get_samples(train_txt_abspath)
        self.eval_samples: List[ObjectDetectionSample] = self._get_samples(val_txt_abspath)
        self.test_samples: List[ObjectDetectionSample] = self._get_samples(test_txt_abspath)

        # 默认为 train 模式
        self.mode = "train"
        self.samples: List[ObjectDetectionSample] = self.train_samples

    def write(self, *args, **kwargs):
        """
        将数据集写到新的目录下
        """
        dataset, type, new_dataset_dirname = kwargs['dataset'], kwargs['type'], kwargs['new_dataset_dirname']
        self._write(dataset, new_dataset_dirname)

    def _write(self, new_dataset_dirname):
        # TODO
        pass


class ObjectDetectionDatasetTransfer:

    def __init__(self, dataset: ObjectDetectionDataset):
        self.dataset = dataset

        self.dataset_dirname = self.dataset.dataset_dirname
        # self.train_samples, self.eval_samples, self.test_samples = self.to_generic_dataset()

    # def to_generic_dataset(self):
    #     train_samples = self._to_generic_samples(self.dataset.train_samples)
    #     eval_samples = self._to_generic_samples(self.dataset.eval_samples)
    #     test_samples = self._to_generic_samples(self.dataset.test_samples)
    #
    #     return train_samples, eval_samples, test_samples

    def to_coco(self, dataset: ObjectDetectionDataset, dataset_dirname):
        odd = dataset.to_generic_dataset(dataset_dirname)

        pass

    def to_voc(self, ):
        pass

    def to_yolo(self, ):
        pass
