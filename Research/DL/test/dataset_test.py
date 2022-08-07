from Research.DL.core.dataset import VOCObjectDetectionDataset
from Research.DL.core.dataset import ObjectDetectionDatasetTransfer


def test_voc():
    dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'

    voc_odd = VOCObjectDetectionDataset(dataset_dirname)

    print()

    pass

def test_transfer():


    dataset_dirname = r'C:\Users\86158\Desktop\HyperDL\data\VOC2007'

    voc_odd = VOCObjectDetectionDataset(dataset_dirname)
    oddf = ObjectDetectionDatasetTransfer(voc_odd)

    pass

if __name__ == '__main__':
    test_transfer()