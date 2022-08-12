from Research.DL.core.dataset import VOCObjectDetectionDataset, YOLOObjectDetectionDataset, COCOObjectDetectionDataset
# from Research.DL.core.dataset import ObjectDetectionDatasetTransfer


def test_voc():
    dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\VOC2007'
    voc_odd = VOCObjectDetectionDataset().read(dataset_dirname)

    new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007'
    voc_odd.write(new_dataset_dirname)


def test_yolo():
    dataset_dirname = r'C:\Users\Songgf\Desktop\GIT\coco128'
    classes_name_abspath = r'C:\Users\Songgf\Desktop\GIT\label.txt'
    yolo_odd = YOLOObjectDetectionDataset().read(dataset_dirname, classes_name_abspath)
    # odd = yolo_odd.to_generic_dataset()

    # new_yolo_odd = odd.to_yolo()

    new_dataset_dirname = r'C:\Users\Songgf\Desktop\GIT\new_coco128'
    yolo_odd.write(new_dataset_dirname)
    print()


def test_coco():
    dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\COCO'
    classes_name_abspath = r'C:\Users\Songgf\Desktop\HyperAI\data\COCO\label.txt'
    coco_odd = COCOObjectDetectionDataset().read(dataset_dirname)



def test_transfer():

    # dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
    #
    # voc_odd = VOCObjectDetectionDataset()
    # voc_odd.read(dataset_dirname)
    #
    # odd = voc_odd.to_generic_dataset()
    # oddt = ObjectDetectionDatasetTransfer()

    pass

if __name__ == '__main__':
    # test_yolo()
    # test_voc()
    test_coco()