from Research.DL.core.dataset import VOCObjectDetectionDataset, YOLOObjectDetectionDataset
from Research.DL.core.dataset import ObjectDetectionDatasetTransfer


def test_voc():
    dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
    voc_odd = VOCObjectDetectionDataset().read(dataset_dirname)

    print()

def test_yolo():
    dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\coco128_yolo'
    yolo_odd = YOLOObjectDetectionDataset().read(dataset_dirname)
    print()


def test_transfer():

    dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'

    voc_odd = VOCObjectDetectionDataset()
    voc_odd.read(dataset_dirname)

    odd = voc_odd.to_generic_dataset()
    oddt = ObjectDetectionDatasetTransfer()

    pass

if __name__ == '__main__':
    test_yolo()