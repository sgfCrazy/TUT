# -*- coding: utf-8 -*-

import logging
import numpy as np
import time
import weakref
import torch
from ..utils import dist
import os
from ..utils.events import EventStorage
from .hooks import HookBase
from ..engine import hooks
from collections import OrderedDict
from ..utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter


__all__ = ["TrainerBase", "SimpleTrainer"]


class TrainerBase:
    """
    Trainer 的唯一假设为：训练在循环中进行，对于dataloader, optimizer, model等没有做出任何假设。
    Attributes:
        iter(int): 当前迭代次数.
        start_iter(int): 起始迭代次数。默认最小值为0。
        max_iter(int): 训练截至时的迭代次数。
        storage(EventStorage): 在训练过程中打开的事件存储。
    """

    def __init__(self):
        self._hooks = []


    def register_hooks(self, hooks):
        """
        将hook注册到trainer中，hooks将按照注册的顺序进行执行。
        Args:
            hooks (list[Optional[HookBase]]): hook列表
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # 为了避免循环引用，hook和trainer不能互相拥有彼此的引用
            # 这通常没什么问题，但当涉及到的对象包含__del__时，将有可能导致内存泄漏。
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)  # TODO  弱引用的原理，实现机制，使用场景？
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): 起始迭代次数，最大迭代次数
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    一个简单的trainer对于大多数通用任务。
    单成本 单优化器 单数据源 迭代优化。
    对于每次迭代，假设:

    1. 使用data_loader中的数据计算损失。
    2. 使用以上的loss计算梯度。
    3. 使用优化器更新模型。
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: 一个pytorch的Module，接收一个来自data_loader的data，返回一个losses的dict。
            data_loader: 一个可迭代的对象，包含被模型使用的数据。
            optimizer: 一个torch的优化器。
        """
        super().__init__()

        """
        在trainer中我们设置模型为训练模式。但是，训练处于评估模式的模型是合理的。
        如果您希望模型（或其子模块）在训练期间进行评估，您可以覆盖其train()方法。
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        实现上述标准训练逻辑。
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        你可以通过包装dataloader来实现一些自定义操作。
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        可以通过包装model实现自定义功能。
        """
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        # 检查是否有异常值
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        可以通过重写optimizer的 zero_grad() 实现自定义梯度计算等操作。
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        如果你想 clipping/scaling 梯度或其他处理操作，可以重写optimizer的 step() 方法。
        """
        self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): 评价指标dict
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # 收集所有工作线程中的指标以进行日志记录。这假设我们进行DDP式训练，目前detectron2中唯一受支持的方法。  # TODO
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # worker之间的data_time可能具有很高的差异。以所有worker中data_time的最大值作为本次step的用时。
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # 平均其他的指标
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }

            # 总损失
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

