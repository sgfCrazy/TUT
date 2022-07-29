from collections import defaultdict
from ..common.history_buffer import HistoryBuffer
from contextlib import contextmanager
import logging
import datetime
import time
import torch
import json
import os

'''
@contextmanager
def file_open(path):
    try:
        f_obj = open(path,"w")
        yield f_obj
    except OSError:
        print("We had an error!")
    finally:
        print("Closing file")
        f_obj.close()

if __name__ == "__main__":
    with file_open("test/test.txt") as fobj:
        fobj.write("Testing context managers")
'''


_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    返回正在使用的 `EventStorage` 对象。如果没有正在使用的`EventStorage` 对象则抛出一个异常。
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class EventStorage:
    """
    提供指标存储功能的面向用户的类。将来，如果需要，我们可能会添加对 存储/日志 其他类型的数据的支持。
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): 起始迭代次数
        """
        self._history = defaultdict(HistoryBuffer)  # 当字典中没有的键第一次出现时，defaultdict 会将value自动设置为HistoryBuffer
        self._smoothing_hints = {}
        self._latest_scalars = {}  # 最后一个标量
        self._iter = start_iter  # 迭代次数
        self._current_prefix = ""  # 前缀
        self._vis_data = []  # 要进行可视化的图像

    def put_image(self, img_name, img_tensor):
        """
        将“img_tensor”添加到与“img_name”关联的“_vis_data”。

        Args:
            img_name (str): 要放入Tensorboard的图像的名称。
            img_tensor (torch.Tensor or numpy.array): img_tensor的值可以是[0, 1] (float32)或者是[0, 255] (uint8)。
                                                      img_tensor的shape必须为[channel, height, width]。
                                                      img_tensor的格式应为RGB。
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def clear_images(self):
        """
        删除所有要进行可视化的图像。这应该在图像写入Tensorboard后调用。
        """
        self._vis_data = []

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        向 HistoryBuffer 中添加一个与 name 关联起来的标量。

        Args:
            smoothing_hint (bool): 当添加的标量是噪声时是否进行提示。该提示可以通过 EventStorage.smoothing_hints 方法进行访问，一个writer可以忽略提示并且应用自定义的平滑规则。默认为 True。
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                    existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        添加多个标量
        Examples:
            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def history(self, name):
        """
            HistoryBuffer: 与name关联的HistoryBuffer
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
            返回所有的HistoryBuffer
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[name -> number]: 在当前迭代中添加的标量。
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        与 ：meth：'latest' 类似，但返回值要么是未平滑的原始最新值，要么是给定window_size的中位数，这取决于smoothing_hint是否为 True。
        这提供了其他writer可以使用的默认行为。
        """
        result = {}
        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].median(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: 如果name对应的bool为true，则该name对应的HistoryBuffer就需要进行平滑。
        """
        return self._smoothing_hints

    def step(self):
        """
        用户应在每次迭代开始时调用此函数，以通知storage新迭代的开始。然后，storage将能够将新数据与正确的迭代编号相关联。
        """
        self._iter += 1
        self._latest_scalars = {}

    @property
    def vis_data(self):
        return self._vis_data

    @property
    def iter(self):
        return self._iter

    @property
    def iteration(self):
        # 后向兼容
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            在此上下文中，所有的events被存储在 name 空间。
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix


class EventWriter:
    """
    writer的基类：从 EventStorage 中获取
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class TensorboardXWriter(EventWriter):
    """
    将所有的标量写入tensorboard文件
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """

        Args:
            log_dir (str): 保存目录
            window_size (int): 中位数平滑的窗口大小
            kwargs: 传递给 'torch.utils.tensorboard.SummaryWriter（...）' 的其他参数
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(k, v, storage.iter)

        if len(storage.vis_data) >= 1:
            for img_name, img, step_num in storage.vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class CommonMetricPrinter(EventWriter):
    """
    打印 **common** metrics 到终端，包括 iteration time, ETA, memory, all losses和学习率。
    如果想打印其他的值，请实现自己的printer。
    """

    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): 训练的最大迭代次数。用于计算预计到达时间。
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # 当最初几次迭代时或者没有使用SimpleTrainer时，data_time可能不存在
            data_time = None

        eta_string = "N/A"
        try:
            iter_time = storage.history("time").global_avg()  # 每个iter的时间
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)  # 剩余时间
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                # 估计迭代时间
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                        iteration - self._last_write[0]
                )
                # 估计剩余时间
                eta_seconds = estimate_iter_time * (self._max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0  # TODO 单位为M？
        else:
            max_mem_mb = None

        # 注意：max_mem由dev/parse_results.sh脚本得出
        self.logger.info(
            " eta: {eta}  iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=eta_string,
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.3f}".format(k, v.median(20))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class JSONWriter(EventWriter):
    """
    将标量写入json文件。
    它将标量保存为每行一个 json（而不是一个大 json），以便于解析。

    Examples parsing such a json file:

    .. code-block:: none

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): json文件路径。 如果文件存在，那么新数据将放入文件后面。
            window_size (int): 其“smoothing_hint”为 True 的标量的中值平滑的窗口大小。
        """
        self._file_handle = open(json_file, "a")
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        to_save = {"iteration": storage.iter}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()
