from xml.dom import minidom
import logging


class Annotation:
    def __init__(self):

        self.id = None

        self.coor = None

    def to_generic_template(self):
        """
        子类自己去实现通用模板的转化
        """
        pass
    pass


class ObjectDetectionAnnotation(Annotation):
    """
    一个目标检测样本对应的标签数据，包扩该样本的id和标签坐标
    """

    def __init__(self):
        super(ObjectDetectionAnnotation, self).__init__()

        # self.sample_id = sample_id  # image_id == anno_id
        # self.coors = coordinate  # 通用的顺时针或逆时针坐标点的格式


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
            self.size['image'] = self._get_single_node_value(size_node, 'image')

        self.segmented = self._get_single_node_value(root_node, 'segmented')


        object_nodes = self._get_nodes(root_node, 'source')
        for






# class Annotations:
#     """
#         每张图片的anno
#     """
#
#     def __init__(self):
#         self.annos = {}  # key: id, value: anno 某个图片对应的所有标注
#         pass
#
#     def add(self, anno: Annotation):
#
#         self.annos[anno.id] = anno.coor
#         pass
