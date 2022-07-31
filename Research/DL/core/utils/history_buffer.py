from typing import List, Optional, Tuple
import numpy as np



class HistoryBuffer:
    """
    跟踪一系列标量值，并提供求中位数和全局平均值的接口。
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: 缓冲区的最大容量。当缓冲器的容量耗尽时，旧值将被删除。
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration)
        self._count: int = 0  #
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """
        在特定 iteration 添加一个新的标量值，若iteration未指定，则在缓冲区末尾添加。
        如果缓冲区的长度超过 self._max_length，最旧的元素将从缓冲区中删除。
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """
        返回最后添加的value
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """
        返回缓冲区的中位数。
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """
        返回缓冲区的平均值
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        """
        返回缓冲区中所有元素的平均值。请注意，这个平均值包括那些由于缓冲区存储有限而被删除的。
        """
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        """
        返回当前缓冲区的所有值。
        """
        return self._data