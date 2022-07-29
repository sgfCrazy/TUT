from time import perf_counter   # 计算两次调用之间的时间
from typing import Optional


class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        重置timer
        """
        self._start = perf_counter()  # 重置起始时间
        self._paused: Optional[float] = None  # 暂停时的时间
        self._total_paused = 0  # 暂停总用时
        self._count_start = 1  # 恢复计时器的次数

    def pause(self) -> None:
        """
        暂停timer
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()  # 暂停时的时间

    def is_paused(self) -> bool:
        """
        Returns:
            bool: timer是否处于暂停状态
        """
        return self._paused is not None

    def resume(self) -> None:
        """
        恢复timer
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        # pyre-fixme[58]: `-` is not supported for operand types `float` and
        #  `Optional[float]`.
        self._total_paused += perf_counter() - self._paused  # 计算暂停的总用时
        self._paused = None  # 取消暂停状态
        self._count_start += 1  # 恢复的次数

    def seconds(self) -> float:
        """
        Returns:
            (float): 自 启动/重置 以来的总秒数计时器，不包括计时器暂停的时间。
        """
        if self._paused is not None:  # 如果处于暂停状态，那么截至时间为最后一次暂停的时间
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()  # 当前时间
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Returns:
            (float): 每次 启动/重置 和 暂停 之间的平均秒数。
        """
        return self.seconds() / self._count_start
