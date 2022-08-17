from torch.utils.data import Dataset
from .sample import ObjectDetectionSample, Sample
from typing import List
from .image import VOCImage, YOLOImage, COCOImage, Image, ObjectDetectionImage
from .annotation import VOCObjectDetectionAnnotation, YOLOObjectDetectionAnnotation, COCOObjectDetectionAnnotation

from pathlib import Path
from ..common.logger import *
import json
from tqdm import tqdm
import cv2

logger = Logger().logger(__name__)


# class ObjectDetectionDataset(Dataset):
class ObjectDetectionDataset:

    def __init__(self, image_transform=None, anno_transform=None):
        super(ObjectDetectionDataset, self).__init__()

        self.name = "Generic"

        self.dataset_dirname = None
        self.classes_name = []
        self.train_samples: List[ObjectDetectionSample] = None
        self.eval_samples: List[ObjectDetectionSample] = None
        self.test_samples: List[ObjectDetectionSample] = None

        self.types = ['train', 'test', 'val']

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

    def _read_image(self, image_abspath) -> ObjectDetectionImage:
        return ObjectDetectionImage().read(image_abspath, self.image_transform)

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
                new_anno.anno_abspath = Path(str(Path(anno.anno_abspath).parent), str(Path(anno.anno_abspath).stem),
                                             '.txt')
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

                    new_objects.append(new_object)

                new_anno.objects = new_objects

                ods = ObjectDetectionSample(sample_id=sample.sample_id, image=new_image, anno=new_anno)
                new_samples.append(ods)
            return new_samples

        yolo_odd.train_samples = _to_yolo_samples(self.train_samples)
        yolo_odd.eval_samples = _to_yolo_samples(self.eval_samples)
        yolo_odd.test_samples = _to_yolo_samples(self.test_samples)

        yolo_odd.train()

        return yolo_odd

    def to_voc(self, new_dataset_dirname=None):

        voc_odd = VOCObjectDetectionDataset()
        voc_odd.dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        voc_odd.classes_name_abspath = self.classes_name_abspath
        voc_odd.classes_name = self.classes_name

        def _to_voc_samples(samples):
            new_samples = []
            for sample in samples:
                # 图像转换
                image = sample.image
                new_image = VOCImage()
                new_image.image_transform = image.image_transform
                new_image.image_abspath = image.image_abspath
                new_image.width = image.width
                new_image.height = image.height
                new_image.channels = image.channels
                new_image.data = image.data

                # 标签转换
                anno = sample.anno
                new_anno = VOCObjectDetectionAnnotation(voc_odd.classes_name)
                new_anno.anno_abspath = Path(str(Path(anno.anno_abspath).parent), str(Path(anno.anno_abspath).stem),
                                             '.xml')
                new_anno.anno_transform = anno.anno_transform

                new_anno.size = {}
                new_anno.size['width'] = image.width
                new_anno.size['height'] = image.height
                new_anno.size['depth'] = image.channels

                new_anno.folder = str(Path(image.image_abspath).parent)

                new_anno.filename = str(Path(image.image_abspath).name)

                new_anno.source = {
                    'database': "database",
                    'annotation': "annotation",
                    'image': "image",
                    'flickrid': "flickrid",
                }

                new_anno.owner = {
                    'flickrid': "flickrid",
                    'name': "name"
                }

                new_anno.segmented = 0

                objects = anno.objects
                new_objects = []
                for object in objects:
                    new_object = {}
                    new_object["name"] = object["clas"]
                    new_object["pose"] = 0
                    new_object["truncated"] = 0
                    new_object["difficult"] = 0

                    bndbox = {}

                    p1, p2, p3, p4 = object["box"][:4]
                    xmin, ymin = p1
                    xmax, ymax = p3

                    bndbox['xmin'] = xmin
                    bndbox['ymin'] = ymin
                    bndbox['xmax'] = xmax
                    bndbox['ymax'] = ymax
                    new_object['bndbox'] = bndbox

                    new_objects.append(new_object)

                new_anno.objects = new_objects

                ods = ObjectDetectionSample(sample_id=sample.sample_id, image=new_image, anno=new_anno)
                new_samples.append(ods)
            return new_samples

        voc_odd.train_samples = _to_voc_samples(self.train_samples)
        voc_odd.eval_samples = _to_voc_samples(self.eval_samples)
        voc_odd.test_samples = _to_voc_samples(self.test_samples)

        voc_odd.train()

        return voc_odd

    def to_coco(self, new_dataset_dirname=None):
        coco_odd = COCOObjectDetectionDataset()
        coco_odd.dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        coco_odd.classes_name_abspath = self.classes_name_abspath
        coco_odd.classes_name = self.classes_name

        image_id = 1
        anno_id = 1
        coco_odd.categories = [{'supercategory': clas, 'id': self.classes_name.index(clas), 'name': clas} for clas in
                               self.classes_name]

        def _to_coco_samples(samples, image_id, anno_id):
            new_samples = []
            new_anno_dict = {
                "info": {},
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": coco_odd.categories
            }

            # start sample ------------------------------------
            for sample in samples:
                # 图像转换
                image = sample.image
                new_image = COCOImage()
                new_image.image_transform = image.image_transform
                new_image.image_abspath = image.image_abspath
                new_image.width = image.width
                new_image.height = image.height
                new_image.channels = image.channels
                new_image.data = image.data
                new_image.image_id = image_id

                new_anno_dict['images'].append({
                    "license": 0,
                    "file_name": str(Path(new_image.image_abspath).name),
                    "coco_url": "coco_url",
                    "height": new_image.height,
                    "width": new_image.width,
                    "data_captured": "data_captured",
                    "fickr_url": "fickr_url",
                    "id": image_id
                })

                # 标签转换
                anno = sample.anno
                new_anno = COCOObjectDetectionAnnotation(coco_odd.classes_name)
                new_anno.anno_abspath = Path(str(Path(anno.anno_abspath).parent), str(Path(anno.anno_abspath).stem),
                                             '.json')  # TODO
                new_anno.anno_transform = anno.anno_transform

                objects = anno.objects
                new_objects = []

                # start object ---------------------------------
                for object in objects:
                    new_object = {}

                    new_object['segmentation'] = [[1.0]]
                    new_object['area'] = 0
                    new_object['iscrowd'] = 0
                    new_object['image_id'] = image_id

                    p1, p2, p3, p4 = object["box"][:4]
                    xmin, ymin = p1
                    xmax, ymax = p3

                    new_object['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
                    new_object['category_name'] = object["clas"]
                    new_object['category_id'] = self.classes_name.index(object["clas"])
                    new_object['id'] = anno_id

                    new_objects.append(new_object)
                    anno_id += 1
                # end object ---------------------------------
                new_anno.objects = new_objects

                ods = ObjectDetectionSample(sample_id=sample.sample_id, image=new_image, anno=new_anno)
                new_samples.append(ods)
            # end sample --------------------------------------
            image_id += 1

            return new_samples, image_id, anno_id

        coco_odd.train_samples, image_id, anno_id = _to_coco_samples(self.train_samples, image_id, anno_id)
        coco_odd.eval_samples, image_id, anno_id = _to_coco_samples(self.eval_samples, image_id, anno_id)
        coco_odd.test_samples, image_id, anno_id = _to_coco_samples(self.test_samples, image_id, anno_id)

        coco_odd.train()

        return coco_odd

    def _visualized_samples(self, samples, save_folder):

        # 能够读取中文路径，c,h,w BGR
        font_scale = 0.8
        rect_thickness = 2
        text_thickness = 1
        # 速度：LINE_8>LINE_AA 美观：LINE_AA>LINE_8
        font = cv2.FONT_HERSHEY_TRIPLEX
        line_type = 8
        text_color = (255, 255, 255)
        rect_color = (255, 0, 0)

        pbar = tqdm(samples)

        for sample in pbar:
            image = sample.image.data
            image_filename = Path(sample.image.image_abspath).name
            for object in sample.anno.objects:
                clas_id = object['clas_id']
                clas = object['clas']
                p1, p3 = object['box'][0], object['box'][2]
                xmin, ymin = p1
                xmax, ymax = p3
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                text_size, baseline = cv2.getTextSize(clas, font, font_scale, text_thickness)

                txt_org_y = ymin - baseline
                txt_ymin = txt_org_y - text_size[1]  # 文本框 ymin
                txt_xmin = xmin  # 文本框 xmin
                txt_xmax = txt_xmin + text_size[0]  # 文本框 xmax
                txt_ymax = ymin  # 文本框 ymin

                if txt_ymin < 0:
                    txt_org_y = ymin + text_size[1]
                    txt_ymin = ymin  # 文本框 ymin
                    txt_xmin = xmin  # 文本框 xmin
                    txt_xmax = txt_xmin + text_size[0]  # 文本框 xmax
                    txt_ymax = txt_org_y + baseline  # 文本框 ymin
                    pass

                # 绘制文字包围框 thickness=-1 为填充
                cv2.rectangle(image, (txt_xmin, txt_ymin), (txt_xmax, txt_ymax), rect_color, -1)
                # 绘制文字
                cv2.putText(image, clas, (xmin, txt_org_y), font, font_scale, text_color, text_thickness,
                            lineType=line_type, bottomLeftOrigin=False)
                # 绘制标注框
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), rect_color, rect_thickness)

            new_image_abspath = str(Path(save_folder, image_filename))

            cv2.imwrite(new_image_abspath, image)

            pbar.set_description(f"{image_filename}")
        pass

    def visualized(self, save_folder):

        Path(save_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"可视化...")
        self._visualized_samples(self.samples, save_folder)

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
    def __init__(self, image_transform=None, anno_transform=None):
        super(COCOObjectDetectionDataset, self).__init__(image_transform, anno_transform)

        self.name = "COCO"

        self.categories = None
        self.info = []
        self.images = []

    def read(self, dataset_dirname, classes_name_abspath=None, types=None):
        self.dataset_dirname = dataset_dirname  # coco数据集的根目录的路径

        self.classes_name_abspath = classes_name_abspath if classes_name_abspath else Path(self.dataset_dirname,
                                                                                           'label.txt')
        self.classes_name = self._read_classes_name(self.classes_name_abspath)

        self.types = types if types else ['train', 'test', 'val']

        self._read()

        return self

    def _read(self):

        for type in self.types:
            json_abspath = Path(self.dataset_dirname, f'{type}.json')
            images_dirname = Path(self.dataset_dirname, '%s' % type)
            logger.info(f"{self.name} read {type} samples ...")

            samples = self._read_samples(json_abspath, images_dirname)

            if type == "train":
                self.train_samples = samples
            elif type == "val":
                self.eval_samples = samples
            elif type == "test":
                self.test_samples = samples
            else:
                raise ValueError

        # 默认为 train 模式
        self.mode_dict["train"]()

    def _read_samples(self, json_abspath, images_dirname):
        samples = []

        with open(json_abspath, 'r') as f:
            json_dict = json.load(f)

        annotations = json_dict['annotations']
        images_id_dict = {image['id']: image for image in json_dict['images']}
        images_filename_dict = {image['file_name']: image for image in json_dict['images']}
        categories_id_dict = {category['id']: category for category in json_dict['categories']}

        if self.categories is None:
            self.categories = json_dict['categories']  # TODO 判断合法性
        else:
            assert self.categories == json_dict['categories']

        def change_json():
            """
            将原始json转为每个样本对应一个json的形式
            """
            # new_json_dict = {}.setdefault({'image': None, 'anno': {}.setdefault(())})
            new_json_dict = {}

            for annotation in annotations:
                segmentation = annotation['segmentation']
                area = annotation['area']
                iscrowd = annotation['iscrowd']
                image_id = annotation['image_id']
                category_id = annotation['category_id']
                category = categories_id_dict[category_id]
                annotation['category_name'] = category['name']
                image = images_id_dict[image_id]

                sample_id = Path(image["file_name"]).stem
                if sample_id not in new_json_dict.keys():
                    new_json_dict[sample_id] = {"image": None, "anno": []}
                new_json_dict[sample_id]['image'] = image
                new_json_dict[sample_id]['anno'].append(annotation)

            return new_json_dict

        new_json_dict = change_json()

        pbar = tqdm(new_json_dict.items())

        for sample_id, sample_dict in pbar:
            image_abspath = Path(images_dirname, sample_id + ".jpg")
            anno_abspath = Path(images_dirname, sample_id + ".json")
            anno_dict = sample_dict['anno']

            image = self._read_image(image_abspath, anno_dict[0]["image_id"])
            anno = self._read_anno(anno_abspath, anno_dict)
            anno.width = image.width
            anno.height = image.height
            anno.channels = image.channels

            ods = ObjectDetectionSample(sample_id, image, anno)
            samples.append(ods)

            pbar.set_description(f"{sample_id}")

        return samples

    def _read_anno(self, anno_abspath, json_dict) -> COCOObjectDetectionAnnotation:
        return COCOObjectDetectionAnnotation(self.classes_name).read(anno_abspath, json_dict, self.anno_transform)

    def _read_image(self, image_abspath, image_id) -> COCOImage:
        return COCOImage().read(image_abspath, image_id, self.image_transform)

    def _write(self, new_dataset_dirname):

        for type in self.types:

            logger.info(f"{self.name} write {type} samples ...")

            if type == "train":
                samples = self.train_samples
            elif type == "val":
                samples = self.eval_samples
            elif type == "test":
                samples = self.test_samples
            else:
                raise ValueError

            new_images_dirname = Path(new_dataset_dirname, type)
            new_images_dirname.mkdir(parents=True, exist_ok=True)

            new_annos_abspath = Path(new_dataset_dirname, f'{type}.json')

            self._write_samples(samples, new_images_dirname, new_annos_abspath)
        pass

    def write(self, new_dataset_dirname):
        new_dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        Path(new_dataset_dirname).mkdir(parents=True, exist_ok=True)
        # 写label.txt
        new_classes_name_abspath = Path(new_dataset_dirname, 'label.txt')
        self._write_classes_name(new_classes_name_abspath)

        self._write(new_dataset_dirname)

    def _write_samples(self, samples, images_dirname, new_annos_abspath):
        """
        image{
        "id" : int,
        "width" : int,
        "height" : int,
        "file_name" : str,
        "license" : int,
        "flickr_url" : str,
        "coco_url" : str,
        "date_captured" : datetime,
        }
        """

        json_dict = {
            "info": {
                "description": "description",
                "url": "url",
                "version": "version",
                "year": 0,
                "contributor": "contributor",
                "data_created": "data_created"
            },
            "licenses": [
                {
                    "url": "url",
                    "id": 0,
                    "name": "name",
                }
            ],
            "images": [],
            "annotations": [],
            "categories": self.categories
        }

        pbar = tqdm(samples)
        for sample in pbar:
            sample_id = sample.sample_id
            image = sample.image
            image_abspath = Path(images_dirname, f"{sample_id}.jpg")
            image.write(image_abspath)
            image_id = image.image_id

            json_dict["images"].append(
                {
                    "id": image_id,
                    "width": image.width,
                    "height": image.height,
                    "file_name": f"{sample_id}.jpg",
                    "license": 0,
                    "flickr_url": "flickr_url",
                    "coco_url": "coco_url",
                    "date_captured": "date_captured"
                }
            )

            anno = sample.anno
            # anno_abspath = Path(annos_dirname, f"{sample_id}.txt")
            anno.write(json_dict)

            pbar.set_description(f"{sample_id}")

        # 写json
        with open(new_annos_abspath, 'w') as f:
            json.dump(json_dict, f)


class YOLOObjectDetectionDataset(ObjectDetectionDataset):
    def __init__(self, image_transform=None, anno_transform=None):
        super(YOLOObjectDetectionDataset, self).__init__(image_transform, anno_transform)
        self.name = "YOLO"

    def read(self, dataset_dirname, classes_name_abspath=None, types=None):
        self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径
        self.classes_name_abspath = classes_name_abspath if classes_name_abspath else Path(self.dataset_dirname,
                                                                                           'label.txt')
        self.classes_name = self._read_classes_name(self.classes_name_abspath)

        self.types = types if types else self.types

        self._read()
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

    def _read_samples(self, images_dirname, annos_dirname):
        samples_path = self._get_samples_path(images_dirname, annos_dirname)
        samples = []

        pbar = tqdm(samples_path)

        for sample_id, image_abspath, anno_abspath in pbar:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath, image)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

            pbar.set_description(f"{sample_id}")

        return samples

    def _read(self):
        """
        读取数据集
        """

        for type in self.types:

            logger.info(f"{self.name} read {type} samples")

            images_dirname = Path(self.dataset_dirname, 'images', '%s' % type)
            annos_dirname = Path(self.dataset_dirname, 'labels', '%s' % type)

            samples = self._read_samples(images_dirname, annos_dirname)

            if type == "train":
                self.train_samples = samples
            elif type == "val":
                self.eval_samples = samples
            elif type == "test":
                self.test_samples = samples
            else:
                raise ValueError

        # 默认为 train 模式
        self.mode_dict['train']()

    def write(self, new_dataset_dirname=None):
        """
        将数据集写到新的目录下
        """

        new_dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        Path(new_dataset_dirname).mkdir(parents=True, exist_ok=True)
        # 写label.txt
        new_classes_name_abspath = Path(new_dataset_dirname, 'label.txt')
        self._write_classes_name(new_classes_name_abspath)

        self._write(new_dataset_dirname)
        pass

    def _write(self, new_dataset_dirname):

        for type in self.types:
            logger.info(f"{self.name} write {type} samples ...")
            if type == "train":
                samples = self.train_samples
            elif type == "val":
                samples = self.eval_samples
            elif type == "test":
                samples = self.test_samples
            else:
                raise ValueError

            new_images_dirname = Path(new_dataset_dirname, type, 'images')
            new_annos_dirname = Path(new_dataset_dirname, type, 'labels')
            new_images_dirname.mkdir(parents=True, exist_ok=True)
            new_annos_dirname.mkdir(parents=True, exist_ok=True)

            self._write_samples(samples, new_images_dirname, new_annos_dirname)

    def _write_samples(self, samples, images_dirname, annos_dirname):
        pbar = tqdm(samples)
        for sample in pbar:
            sample_id = sample.sample_id
            image = sample.image
            image_abspath = Path(images_dirname, f"{sample_id}.jpg")
            image.write(image_abspath)

            anno = sample.anno
            anno_abspath = Path(annos_dirname, f"{sample_id}.txt")
            anno.write(anno_abspath)

            pbar.set_description(f"{sample_id}")


class VOCObjectDetectionDataset(ObjectDetectionDataset):

    def __init__(self, image_transform=None, anno_transform=None):
        super(VOCObjectDetectionDataset, self).__init__(image_transform, anno_transform)
        self.name = "VOC"

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
        return VOCObjectDetectionAnnotation(self.classes_name).read(anno_abspath, self.anno_transform)

    def read(self, dataset_dirname, classes_name_abspath=None, types=None):
        self.dataset_dirname = dataset_dirname  # voc数据集的根目录的路径

        self.classes_name_abspath = classes_name_abspath if classes_name_abspath else Path(self.dataset_dirname,
                                                                                           'label.txt')
        self.classes_name = self._read_classes_name(self.classes_name_abspath)

        self.types = types if types else self.types

        self._read()
        return self

    def _read_samples(self, txt_abspath):
        samples_path = self._get_samples_path(txt_abspath)
        samples = []

        pbar = tqdm(samples_path)

        for sample_id, image_abspath, anno_abspath in pbar:
            image = self._read_image(image_abspath)
            anno = self._read_anno(anno_abspath)

            sample = ObjectDetectionSample(sample_id, image=image, anno=anno)
            samples.append(sample)

            pbar.set_description(f"{sample_id}")

        return samples

    def _read(self):
        """
        读取数据集
        """
        self.images_dirname = Path(self.dataset_dirname, 'JPEGImages')
        self.annos_dirname = Path(self.dataset_dirname, 'Annotations')
        self.split_txt_dirname = Path(self.dataset_dirname, 'ImageSets', 'Main')

        for type in self.types:
            txt_abspath = Path(self.split_txt_dirname, f'{type}.txt')
            logger.info(f"{self.name} read {type} samples ...")
            samples = self._read_samples(txt_abspath)

            if type == "train":
                self.train_samples = samples
            elif type == "val":
                self.eval_samples = samples
            elif type == "test":
                self.test_samples = samples
            else:
                raise ValueError

        # 默认为 train 模式
        self.mode_dict["train"]()

    def write(self, new_dataset_dirname=None):
        """
        将数据集写到新的目录下
        """
        new_dataset_dirname = new_dataset_dirname if new_dataset_dirname else self.dataset_dirname
        Path(new_dataset_dirname).mkdir(parents=True, exist_ok=True)
        # 写label.txt
        new_classes_name_abspath = Path(new_dataset_dirname, 'label.txt')

        self._write_classes_name(new_classes_name_abspath)

        self._write(new_dataset_dirname)

    def _write(self, new_dataset_dirname):

        new_images_dirname = Path(new_dataset_dirname, 'JPEGImages')
        new_images_dirname.mkdir(parents=True, exist_ok=True)

        new_annos_dirname = Path(new_dataset_dirname, 'Annotations')
        new_annos_dirname.mkdir(parents=True, exist_ok=True)

        new_split_txt_dirname = Path(new_dataset_dirname, 'ImageSets', 'Main')
        new_split_txt_dirname.mkdir(parents=True, exist_ok=True)

        for type in self.types:
            txt_abspath = Path(new_split_txt_dirname, f'{type}.txt')
            logger.info(f"{self.name} write {type} samples ...")

            if type == "train":
                samples = self.train_samples
            elif type == "val":
                samples = self.eval_samples
            elif type == "test":
                samples = self.test_samples
            else:
                raise ValueError

            self._write_samples(samples, new_images_dirname, new_annos_dirname, txt_abspath)

        # 写trainval.txt  # TODO
        txt_abspath = Path(new_split_txt_dirname, 'trainval.txt')
        samples = self.train_samples + self.eval_samples
        images_dirname = new_images_dirname
        with open(txt_abspath, 'w') as f:
            pbar = tqdm(samples)
            for sample in pbar:
                sample_id = sample.sample_id
                image_abspath = Path(images_dirname, f"{sample_id}.jpg")
                f.write(f"{image_abspath}\n")

    def _write_samples(self, samples, images_dirname, annos_dirname, txt_abspath):

        with open(txt_abspath, 'w') as f:
            pbar = tqdm(samples)
            for sample in pbar:
                sample_id = sample.sample_id
                image = sample.image
                image_abspath = Path(images_dirname, f"{sample_id}.jpg")
                image.write(image_abspath)

                anno = sample.anno
                anno_abspath = Path(annos_dirname, f"{sample_id}.xml")
                anno.write(anno_abspath)

                pbar.set_description(f"{sample_id}")

                f.write(f"{image_abspath}\n")

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
