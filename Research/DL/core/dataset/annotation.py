from xml.dom import minidom
import logging
from abc import ABCMeta, abstractmethod


class Annotation(metaclass=ABCMeta):
    def __init__(self):
        self.generic_anno = None

    # @abstractmethod
    # def to_generic_template(self):
    #     """
    #     子类自己去实现通用模板的转化
    #     """
    #     pass

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def read(self):
        pass


class ObjectDetectionAnnotation(Annotation):
    """
    一个目标检测样本对应的标签数据，包扩该样本的id和标签坐标
    """

    def __init__(self):
        super(ObjectDetectionAnnotation, self).__init__()
        self.anno_abspath = None
        self.height = None
        self.width = None
        self.channels = None
        self.objects = []  # 五个点的格式 [p1, p2, p3, p4, p1]


    def write(self):
        """

        """
        # TODO
        pass

    def read(self):
        """

        """
        # TODO
        pass

    def to_generic_template(self):
        pass


class VOCObjectDetectionAnnotation(ObjectDetectionAnnotation):

    def __init__(self, anno_abspath, anno_transform):
        super(VOCObjectDetectionAnnotation, self).__init__()

        self.anno_abspath = anno_abspath
        self.anno_transform = anno_transform

    def _get_nodes(self, parent_node, tag_name):
        """
        返回tag_name对应的所有node
        """

        nodes = parent_node.getElementsByTagName(tag_name)
        if nodes is None:
            logging.warning(f"{tag_name} 节点不存在!")
        return nodes

    def _get_single_node(self, parent_node, tag_name):
        """
        如果tag_name对应的节点只有一个时，调用此方法
        """
        node = self._get_nodes(parent_node, tag_name)
        return node if node is None else node[0]

    def _get_single_node_value(self, parent_node, tag_name):
        node = self._get_single_node(parent_node, tag_name)
        if node is None:
            return ""
        else:
            return node.data

    def read(self):
        # 1. 工厂方法， 返回一个dom对象
        dom = minidom.parse(self.anno_abspath)
        # 2. 获取根节点
        root_node = dom.documentElement
        # 3. 读取各个节点
        self.folder = self._get_single_node_value(root_node, 'folder')  # 文件夹路径
        self.filename = self._get_single_node_value(root_node, 'filename')

        source_node = self._get_single_node(root_node, 'source')
        self.source = {}
        if source_node is not None:
            self.source['database'] = self._get_single_node_value(source_node, 'database')
            self.source['annotation'] = self._get_single_node_value(source_node, 'annotation')
            self.source['image'] = self._get_single_node_value(source_node, 'image')

        size_node = self._get_single_node(root_node, 'size')
        self.size = {}
        if source_node is not None:
            self.size['width'] = self._get_single_node_value(size_node, 'width')
            self.size['height'] = self._get_single_node_value(size_node, 'height')
            self.size['depth'] = self._get_single_node_value(size_node, 'depth')

        self.segmented = self._get_single_node_value(root_node, 'segmented')

        objects_node = self._get_nodes(root_node, 'source')
        self.objects = []

        #  ---------------------- find objects start -----------------
        for object_node in objects_node:
            object = {}
            name = self._get_single_node_value(object_node, 'name')
            pose = self._get_single_node_value(object_node, 'pose')
            truncated = self._get_single_node_value(object_node, 'truncated')
            difficult = self._get_single_node_value(object_node, 'difficult')

            bndbox = {}
            bndbox_node = self._get_single_node(object_node, 'bndbox')
            xmin = self._get_single_node_value(bndbox_node, 'xmin')
            ymin = self._get_single_node_value(bndbox_node, 'ymin')
            xmax = self._get_single_node_value(bndbox_node, 'xmax')
            ymax = self._get_single_node_value(bndbox_node, 'ymax')

            bndbox['xmin'] = xmin
            bndbox['ymin'] = ymin
            bndbox['xmax'] = xmax
            bndbox['ymax'] = ymax

            object['name'] = name
            object['pose'] = pose
            object['truncated'] = truncated
            object['difficult'] = difficult
            object['bndbox'] = bndbox
        #  ---------------------- find objects end -----------------

    def write(self):
        # TODO
        pass

    def to_generic_template(self) -> ObjectDetectionAnnotation:
        """
        子类自己去实现通用模板的转化
        """
        oba = ObjectDetectionAnnotation()
        oba.anno_abspath = self.anno_abspath
        oba.height = self.size['height']
        oba.width = self.size['width']
        oba.channels = self.size['depth']
        for object in self.objects:
            name = object['name']
            xmin = object['bndbox']['xmin']
            ymin = object['bndbox']['ymin']
            xmax = object['bndbox']['xmax']
            ymax = object['bndbox']['ymax']

            oba_obj = {
                'name': name,
                'box': [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                    [xmin, ymin]
                ]
            }
            oba.objects.append(oba_obj)

        return oba
