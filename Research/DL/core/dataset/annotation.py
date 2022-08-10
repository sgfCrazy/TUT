from xml.dom import minidom
import logging
from abc import ABCMeta, abstractmethod


class Annotation:
    def __init__(self):
        pass

    def write(self):
        raise NotImplementedError

    def read(self, anno_abspath, anno_transform=None):
        raise NotImplementedError

    # def to_generic_anno(self):
    #     raise NotImplementedError


class ObjectDetectionAnnotation(Annotation):
    """
    一个目标检测样本对应的标签数据，包扩该样本的id和标签坐标
    """

    def __init__(self):
        super(ObjectDetectionAnnotation, self).__init__()
        self.anno_abspath = None
        self.anno_transform = None

        self.height = None
        self.width = None
        self.channels = None

        """
            oba_obj = {
                'clas_id': clas_id,
                'clas': clas,
                'box': [  # 五个点的格式 [p1, p2, p3, p4, p1]
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                    [xmin, ymin]
                ]
            }
        """
        self.objects = []



    def write(self):
        """

        """
        # TODO
        pass

    def read(self, anno_abspath, anno_transform):
        """
        """
        # TODO
        pass


class VOCObjectDetectionAnnotation(ObjectDetectionAnnotation):

    def __init__(self):
        super(VOCObjectDetectionAnnotation, self).__init__()

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
        nodes = self._get_nodes(parent_node, tag_name)
        return nodes[0] if nodes else nodes

    def _get_single_node_value(self, parent_node, tag_name):
        node = self._get_single_node(parent_node, tag_name)
        if node is None:
            return ""
        else:
            return node.childNodes[0].data

    def read(self, anno_abspath, anno_transform=None):
        self.anno_abspath = anno_abspath
        self.anno_transform = anno_transform
        # 1. 工厂方法， 返回一个dom对象
        dom = minidom.parse(str(self.anno_abspath))
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
            self.size['width'] = int(self._get_single_node_value(size_node, 'width'))
            self.size['height'] = int(self._get_single_node_value(size_node, 'height'))
            self.size['depth'] = int(self._get_single_node_value(size_node, 'depth'))

        self.segmented = self._get_single_node_value(root_node, 'segmented')

        objects_node = self._get_nodes(root_node, 'object')
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

            bndbox['xmin'] = float(xmin)
            bndbox['ymin'] = float(ymin)
            bndbox['xmax'] = float(xmax)
            bndbox['ymax'] = float(ymax)

            object['name'] = name
            object['pose'] = pose
            object['truncated'] = int(truncated)
            object['difficult'] = int(difficult)
            object['bndbox'] = bndbox
            self.objects.append(object)
        #  ---------------------- find objects end -----------------

        return self

    def write(self, anno_abspath):
        # 1.创建DOM树对象
        dom = minidom.Document()
        # 2.创建根节点。每次都要用DOM对象来创建任何节点。
        root_node = dom.createElement('annotation')

        # 创建 folder 节点
        folder_node = dom.createElement('folder')
        folder_txt = dom.createTextNode(self.folder)
        folder_node.appendChild(folder_txt)

        # 创建 folder 节点
        filename_node = dom.createElement('filename')
        filename_txt = dom.createTextNode(self.filename)
        filename_node.appendChild(filename_txt)

        # 创建 source 节点
        source_node = dom.createElement('source')

        # 创建 database 节点

        filename_txt = dom.createTextNode(self.filename)
        filename_node.appendChild(filename_txt)



        # 3.用DOM对象添加根节点
        dom.appendChild(root_node)

        pass

    def to_generic_anno(self) -> ObjectDetectionAnnotation:
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


class YOLOObjectDetectionAnnotation(ObjectDetectionAnnotation):

    def __init__(self):
        super(YOLOObjectDetectionAnnotation, self).__init__()



    def read(self, image, classes_name, anno_abspath, anno_transform=None):

        # self.image = image
        self.height = image.height
        self.width = image.width
        self.channels = image.channels

        self.anno_abspath = anno_abspath
        self.anno_transform = anno_transform

        with open(self.anno_abspath, 'r') as f:
            lines = f.readlines()

        object = {}
        # read txt start---------------------------------------------------------
        for line in lines:
            line = line.strip()
            # 标注框的中心点坐标和宽高
            clas_id, x_center, y_center, w, h = line.split()

            object["clas_id"] = int(clas_id)
            object["clas"] = classes_name[object["clas_id"]]

            object["x_center"] = float(x_center)
            object["y_center"] = float(y_center)
            object["w"] = float(w)
            object["h"] = float(h)
            self.objects.append(object)
        # read txt end---------------------------------------------------------
        return self

    def write(self):
        # TODO
        pass

    def to_generic_anno(self) -> ObjectDetectionAnnotation:
        """
        子类自己去实现通用模板的转化
        """
        oba = ObjectDetectionAnnotation()
        oba.anno_abspath = self.anno_abspath
        oba.height = self.height
        oba.width = self.width
        oba.channels = self.channels

        for object in self.objects:
            clas_id = object['clas_id']
            clas = object['clas']
            x_center, y_center, w, h = object['x_center'], object['y_center'], object['w'], object['h']

            x_center, y_center, w, h = x_center * self.width, y_center * self.height, w * self.width, h * self.height

            xmin = x_center - w / 2
            ymin = x_center - h / 2
            xmax = x_center + w / 2
            ymax = x_center + h / 2

            oba_obj = {
                'clas_id': clas_id,
                'clas': clas,
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
