from Research.DL.core.dataset import VOCObjectDetectionDataset, YOLOObjectDetectionDataset, COCOObjectDetectionDataset


# from Research.DL.core.dataset import ObjectDetectionDatasetTransfer

platform = "self"
# platform = "zkhy"


def test_voc():

    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\VOC2007'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007_to_yolo'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007'
        visualized_folder = r'C:\Users\86158\Desktop\HyperDL\data\voc2007_1000\VOC2007_visualized'

        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_voc'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007'
        visualized_folder = r'C:\Users\86158\Desktop\HyperDL\data\new_voc\visualized'

    voc_odd = VOCObjectDetectionDataset().read(dataset_dirname)
    odd = voc_odd.to_generic_dataset()
    odd.visualized(visualized_folder)
    # yolo_odd = odd.to_yolo()
    # yolo_odd.write(yolo_dataset_dirname)


def test_voc_to_coco():

    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\VOC2007'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007_to_yolo'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007_to_yolo'
        coco_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC_to_COCO'

    voc_odd = VOCObjectDetectionDataset().read(dataset_dirname)
    odd = voc_odd.to_generic_dataset()
    # yolo_odd = odd.to_yolo()
    # yolo_odd.write(yolo_dataset_dirname)
    coco_odd = odd.to_coco()
    coco_odd.write(coco_dataset_dirname)

    coco_odd = COCOObjectDetectionDataset().read(coco_dataset_dirname)
    coco_odd.visualized()
    print()


def test_voc_to_yolo():

    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\VOC2007'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007_to_yolo'
        visualized_folder = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007_to_yolo_visualized'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007_to_yolo'

        visualized_folder = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007_to_yolo_visualized'

    voc_odd = VOCObjectDetectionDataset().read(dataset_dirname)
    odd = voc_odd.to_generic_dataset()
    yolo_odd = odd.to_yolo()
    yolo_odd.write(yolo_dataset_dirname)


    yolo_odd = YOLOObjectDetectionDataset().read(yolo_dataset_dirname)
    yolo_odd.to_generic_dataset().visualized(visualized_folder)
    print()



def test_yolo():

    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\GIT\coco128'
        classes_name_abspath = r'C:\Users\Songgf\Desktop\GIT\label.txt'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\GIT\new_coco128'
        visualized_folder = r'C:\Users\Songgf\Desktop\GIT\coco128_yolo_visualized'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\coco128_yolo'
        # classes_name_abspath = r'C:\Users\86158\Desktop\HyperDL\data\coco128_yolo\label.txt'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_coco128_yolo'
        visualized_folder = r'C:\Users\86158\Desktop\HyperDL\data\coco128_yolo_visualized'
    yolo_odd = YOLOObjectDetectionDataset().read(dataset_dirname)
    # odd = yolo_odd.to_generic_dataset()

    # new_yolo_odd = odd.to_yolo()

    # yolo_odd.write(new_dataset_dirname)

    odd = yolo_odd.to_generic_dataset()
    odd.visualized(visualized_folder)

    print()


def test_coco():
    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\COCO'
        classes_name_abspath = r'C:\Users\Songgf\Desktop\HyperAI\data\COCO\label.txt'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\new_COCO'
        yolo_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\new_COCO_to_YOLO'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\COCO'
        classes_name_abspath = r'C:\Users\86158\Desktop\HyperDL\data\COCO\label.txt'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\COCO\new_COCO'
        visualized_folder = r'C:\Users\86158\Desktop\HyperDL\data\COCO_visualized'

    coco_odd = COCOObjectDetectionDataset().read(dataset_dirname)
    # coco_odd.write(new_dataset_dirname)


    odd = coco_odd.to_generic_dataset()
    odd.visualized(visualized_folder)
    # yolo_odd = odd.to_yolo()
    # yolo_odd.write(yolo_dataset_dirname)
    print()


def test_coco_to_voc():
    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\COCO'
        classes_name_abspath = r'C:\Users\Songgf\Desktop\HyperAI\data\COCO\label.txt'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\new_COCO'
        yolo_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\new_COCO_to_YOLO'
        voc_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\new_COCO_to_VOC'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\COCO'
        classes_name_abspath = r'C:\Users\86158\Desktop\HyperDL\data\COCO\label.txt'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_COCO'
        yolo_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_COCO_to_YOLO'
        voc_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_COCO_to_VOC'

    coco_odd = COCOObjectDetectionDataset().read(dataset_dirname)
    # coco_odd.write(new_dataset_dirname)


    odd = coco_odd.to_generic_dataset()
    voc_odd = odd.to_voc()
    voc_odd.write(voc_dataset_dirname)

    voc_odd = VOCObjectDetectionDataset().read(voc_dataset_dirname)
    print()


def test_transfer():
    # dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
    #
    # voc_odd = VOCObjectDetectionDataset()
    # voc_odd.read(dataset_dirname)
    #
    # odd = voc_odd.to_generic_dataset()
    # oddt = ObjectDetectionDatasetTransfer()

    pass


def test_visualized():

    if platform == "zkhy":
        dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\VOC2007'
        new_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\Songgf\Desktop\HyperAI\data\voc2007_1000\new_VOC2007_to_yolo'
    else:
        dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'
        new_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007'
        yolo_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC2007_to_yolo'
        coco_dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\new_VOC_to_COCO'

        visualized_folder = r'C:\Users\86158\Desktop\HyperDL\data/VOC2007_visualized'

    voc_odd = VOCObjectDetectionDataset().read(dataset_dirname)
    odd = voc_odd.to_generic_dataset()
    odd.visualized(visualized_folder)



    pass

if __name__ == '__main__':
    # test_yolo()
    test_voc()
    # test_coco()
    # test_coco_to_voc()
    # test_voc_to_coco()
    # test_voc_to_yolo()
    # test_visualized()