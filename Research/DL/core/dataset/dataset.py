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
        self.classes_name = []
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
        self.classes_name_abspath = Path(self.dataset_dirname, 'label.txt')

    def write(self, new_dataset_dirname):
        pass

    def _read_classes_name(self, classes_name_abspath):
        classes_name = []
        with open(classes_name_abspath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                classes_name.append(line)
        return classes_name

    def _write_classes_name(self, classes_name_abspath):
        with open(classes_name_abspath, 'w') as f:
            for class_name in self.classes_name:
                f.write(class_name + "\n")

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
        odd.classes_name_abspath = self.classes_name_abspath
        odd.classes_name = self.classes_name
        return odd

    def to_yolo(self, new_dataset_dirname=None):
        yolo_odd = YOLOObjectDetectionDataset()
        yolo_odd.dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        yolo_odd.classes_name_abspath = self.classes_name_abspath
        yolo_odd.classes_name = self.classes_name

        def _to_yolo_samples(samples):
            new_samples = []
            for sample in samples:
                # 图像转换
                image = sample.image
                new_image = YOLOImage()
                new_image.image_transform = image.image_transform
                new_image.image_abspath = image.image_abspath
                new_image.width = image.width
                new_image.height = image.height
                new_image.channels = image.channels
                new_image.data = image.data

                # 标签转换
                anno = sample.anno
                new_anno = YOLOObjectDetectionAnnotation()
                new_anno.anno_abspath = anno.anno_abspath
                new_anno.anno_transform = anno.anno_transform
                new_anno.width = anno.width
                new_anno.height = anno.height
                new_anno.channels = anno.channels

                objects = anno.objects
                new_objects = []
                for object in objects:
                    new_object = {}

                    clas = object["clas"]
                    clas_id = object["clas_id"]
                    p1, p2, p3, p4 = object["box"][:4]
                    xmin, ymin = p1
                    xmax, ymax = p3

                    x_center = (xmin + xmax) / 2 / new_anno.width
                    y_center = (ymin + ymax) / 2 / new_anno.height
                    w = (xmax - xmin) / new_anno.width
                    h = (ymax - ymin) / new_anno.height

                    new_object["clas_id"] = clas_id
                    new_object["clas"] = clas
                    new_object["x_center"] = x_center
                    new_object["y_center"] = y_center
                    new_object["w"] = w
                    new_object["h"] = h

                    new_objects.append(objects)

                new_anno.objects = new_objects

                ods = ObjectDetectionSample(sample_id=sample.sample_id, image=new_image, anno=new_anno)
                new_samples.append(ods)
            return samples

        yolo_odd.train_samples = _to_yolo_samples(self.train_samples)
        yolo_odd.eval_samples = _to_yolo_samples(self.eval_samples)
        yolo_odd.test_samples = _to_yolo_samples(self.test_samples)

        yolo_odd.train()

        return yolo_odd

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

    def read(self, dataset_dirname, classes_name_abspath=None, types=None):
        self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径
        self.classes_name_abspath = classes_name_abspath if classes_name_abspath else Path(self.dataset_dirname,
                                                                                           'label.txt')
        self.classes_name = self._read_classes_name(self.classes_name_abspath)
        self._read(types)
        return self

    def _read_anno(self, anno_abspath, image) -> YOLOObjectDetectionAnnotation:
        return YOLOObjectDetectionAnnotation().read(image, self.classes_name, anno_abspath, self.anno_transform)

    def _read_image(self, image_abspath) -> YOLOImage:
        return YOLOImage().read(image_abspath, self.image_transform)

    def _get_samples_path(self, images_dirname, annos_dirname):
        # TODO 目前只适配了一个jpg

        annos_abspath = Path(annos_dirname).glob('*.txt')
        samples_path = []
        for anno_abspath in annos_abspath:
            sample_id = Path(anno_abspath).stem
            image_abspath = Path(images_dirname, f'{sample_id}.jpg')
            # TODO 这里是否进行检查
            assert anno_abspath.exists() and image_abspath.exists(), f"{sample_id} 路径不存在！"
            samples_path.append((sample_id, image_abspath, anno_abspath))

        return samples_path

    def _get_samples(self, images_dirname, annos_dirname):
        samples_path = self._get_samples_path(images_dirname, annos_dirname)
        samples = []

        for sample_id, image_abspath, anno_abspath in samples_path:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath, image)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

        return samples

    def _read(self, types):
        """
        读取数据集
        """
        if types is None:
            types = ['train', 'test', 'eval']
        for type in types:
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

        return VOCImage().read(image_abspath, self.image_transform)

    def _read_anno(self, anno_abspath) -> VOCObjectDetectionAnnotation:
        return VOCObjectDetectionAnnotation().read(anno_abspath, self.anno_transform)

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

        self.classes_name_abspath = Path(self.dataset_dirname, 'label.txt')
        self.classes_name = self._read_classes_name(self.classes_name_abspath)
        self.train_samples: List[ObjectDetectionSample] = self._get_samples(train_txt_abspath)
        self.eval_samples: List[ObjectDetectionSample] = self._get_samples(val_txt_abspath)
        self.test_samples: List[ObjectDetectionSample] = self._get_samples(test_txt_abspath)

        # 默认为 train 模式
        self.mode = "train"
        self.samples: List[ObjectDetectionSample] = self.train_samples

    def write(self, new_dataset_dirname=None):
        """
        将数据集写到新的目录下
        """
        new_dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        self._write(new_dataset_dirname)

    def _write(self, new_dataset_dirname):
        # TODO
        new_images_dirname = Path(new_dataset_dirname, 'JPEGImages').mkdir(parents=True, exist_ok=True)
        new_annos_dirname = Path(new_dataset_dirname, 'Annotations').mkdir(parents=True, exist_ok=True)
        new_split_txt_dirname = Path(new_dataset_dirname, 'ImageSets', 'Main').mkdir(parents=True, exist_ok=True)

        train_txt_abspath = Path(new_split_txt_dirname, 'train.txt')
        val_txt_abspath = Path(new_split_txt_dirname, 'val.txt')
        test_txt_abspath = Path(new_split_txt_dirname, 'test.txt')

        self.classes_name_abspath = Path(self.dataset_dirname, 'label.txt')
        # 写label.txt
        self._write_classes_name(self.classes_name_abspath)

        self._write_samples(self.train_samples, new_images_dirname, new_annos_dirname, train_txt_abspath)
        self._write_samples(self.eval_samples, new_images_dirname, new_annos_dirname, val_txt_abspath)
        self._write_samples(self.test_samples, new_images_dirname, new_annos_dirname, test_txt_abspath)

    def _write_samples(self, samples, images_dirname, annos_dirname, txt_abspath):

        for sample in samples:
            sample_id = sample.sample_id
            image = sample.image
            image_abspath = Path(images_dirname, f"{sample_id}.jpg")
            image.write(image_abspath)

            anno = sample.anno
            anno_abspath = Path(annos_dirname, f"{sample_id}.jpg")
            anno.write(anno_abspath)


            pass
        pass

    def _write_image(self):
        pass

    def _write_anno(self):
        pass

# class ObjectDetectionDatasetTransfer:
#
#     def __init__(self, dataset: ObjectDetectionDataset):
#         self.dataset = dataset
#
#         self.dataset_dirname = self.dataset.dataset_dirname
#         # self.train_samples, self.eval_samples, self.test_samples = self.to_generic_dataset()
#
#     # def to_generic_dataset(self):
#     #     train_samples = self._to_generic_samples(self.dataset.train_samples)
#     #     eval_samples = self._to_generic_samples(self.dataset.eval_samples)
#     #     test_samples = self._to_generic_samples(self.dataset.test_samples)
#     #
#     #     return train_samples, eval_samples, test_samples
#
#     def to_coco(self, dataset: ObjectDetectionDataset, dataset_dirname):
#         odd = dataset.to_generic_dataset(dataset_dirname)
#
#         pass
#
#     def to_voc(self, ):
#         pass
#
#     def to_yolo(self, new_dataset_dirname):
#         pass
