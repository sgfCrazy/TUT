# -*- coding: utf-8 -*-

import datetime
import logging
import time
from Research.DL.core.utils.timer import Timer


__all__ = [
    "CallbackHook",
    "IterationTimer",
]


"""
Implement some common hooks.
"""


class HookBase:
    """
    hooks的基类
    hook是TrainerBase 的 回调方法

    每一个hook有4个待实现的方法。
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    注意:
        1. 在hook方法中，可以通过调用selr.trainer访问上下文更多的属性，如 iteration。
        2. 通常在 before_step 中能做的事在after_step中也能等效实现。惯例是before_step不做太多事情。

    属性:
        trainer:
        trainer: trainer 对象的弱引用。在hook被注册时由trainer设置。
    """

    def before_train(self):
        """
        在第一次迭代前被调用
        """
        pass

    def after_train(self):
        """
        在最后一次迭代后被调用
        """
        pass

    def before_step(self):
        """
        在每次迭代前调用
        """
        pass

    def after_step(self):
        """
        在每次迭代后调用
        """
        pass


class CallbackHook(HookBase):
    """
    使用由用户提供的callback函数创建hook
    """

    def __init__(self, *, before_train=None, after_train=None, before_step=None, after_step=None):
        """
        每个参数都是包含trainer参数的函数
        """
        self._before_train = before_train  # before_train(trainer)
        self._before_step = before_step
        self._after_step = after_step
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        # 这些函数可能包含trainer引用的闭包，因此释放这些函数，避免循环引用
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    """
    跟踪每次迭代花费的时间。
    Print a summary in the end of training.

    根据惯例，before_step hook所花时间应该可以忽略不计，因此IterationTimer hook可以放在hooks列表的开头。
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): 从计时中排除的迭代次数。
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()









