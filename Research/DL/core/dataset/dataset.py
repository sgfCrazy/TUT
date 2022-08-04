from torch.utils.data import Dataset
from sample import ObjectDetectionSample
from typing import List
from image import VOCImage
from annotation import VOCObjectDetectionAnnotation

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# class ObjectDetectionDataset(Dataset, metaclass=ABCMeta):
#
#     def __init__(self, root_fullpath):
#         pass
#         # self.root_fullpath = root_fullpath
#         # self.samples = self.read()
#
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         return self.samples[idx]
#
#     @abstractmethod
#     def read(self) -> List[Sample]:
#         pass
#
#     def add(self, sample: Sample):
#         self.samples.append(sample)


class COCOObjectDetectionDataset(Dataset):
    def __init__(self):
        super(COCOObjectDetectionDataset, self).__init__()


class YOLOObjectDetectionDataset(Dataset):
    def __init__(self):
        super(YOLOObjectDetectionDataset, self).__init__()


class VOCObjectDetectionDataset(Dataset):

    def __init__(self, dataset_dirname, image_transform=None, anno_transform=None):
        super(VOCObjectDetectionDataset, self).__init__()

        # 图像的转换
        self.image_transform = image_transform
        # 标签的转换
        self.anno_transform = anno_transform

        self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径
        self.images_dirname = Path(self.dataset_dirname, 'JPEGImages')
        self.annos_dirname = Path(self.dataset_dirname, 'Annotations')
        self.split_txt_dirname = Path(self.dataset_dirname, 'ImageSets', 'Main')

        train_txt_abspath = Path(self.split_txt_dirname, 'train.txt')
        val_txt_abspath = Path(self.split_txt_dirname, 'val.txt')
        test_txt_abspath = Path(self.split_txt_dirname, 'test.txt')
        self.trainval_txt_abspath = Path(self.split_txt_dirname, 'val.txt')

        self.train_samples = self.read(train_txt_abspath)
        self.eval_samples = self.read(val_txt_abspath)
        self.test_samples = self.read(test_txt_abspath)

        # 默认为 train 模式
        self.mode = "train"
        self.samples = self.train_samples

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

        for line_num, line in enumerate(1, lines):
            sample_id = line.strip()
            if sample_id:
                logger.warning(f"{txt_abspath} : {line_num}行 为空白字符！")
                continue

            sample_id = Path(sample_id).stem
            image_abspath = Path(self.images_dirname, sample_id, '.jpg')
            anno_abspath = Path(self.annos_dirname, sample_id, '.xml')

            if not (Path(image_abspath).exists() and Path(anno_abspath).exists()):
                logger.error(f"{txt_abspath} : {line_num}行 {sample_id} 指向的图片或标签文件不存在！")
                continue
            else:
                samples_path.append((sample_id, image_abspath, anno_abspath))

        return samples_path

    def _read_image(self, image_abspath) -> VOCImage:

        return VOCImage(image_abspath, self.image_transform)

    def _read_anno(self, anno_abspath) -> VOCObjectDetectionAnnotation:

        return VOCObjectDetectionAnnotation(anno_abspath, self.anno_transform)

    def read(self, txt_abspath) -> List[ObjectDetectionSample]:
        samples_path = self._get_samples_path(txt_abspath)

        samples = []

        for sample_id, image_abspath, anno_abspath in samples_path:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

        return samples

    def write(self, new_dataset_dirname):
        """
        将数据集写到新的目录下
        """

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


class DatasetTransfer:
    def __init__(self):
        pass

    pass
