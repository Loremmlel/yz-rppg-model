"""
模型池（ModelPool）：类似线程池的 rPPG Model 实例池。

设计目标：
  - 维护一批预热好的空闲模型实例，避免每次请求时重新加载权重。
  - 按需动态扩容：当空闲实例不足时自动创建新实例（不超过 max_size）。
  - 自动缩容：后台收缩线程定期检查，若空闲实例超过 min_size 且长时间未被使用，
    则销毁多余实例，节约内存与 GPU/CPU 资源。
  - 归还时自动重置模型上下文（清空历史帧缓冲），使其可被新会话复用。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import rppg

logger = logging.getLogger(__name__)


@dataclass
class _PooledModel:
    """池中的一个模型槽位."""
    model: rppg.Model
    # 最近一次归还到池子的时间（用于缩容判断）
    returned_at: float = field(default_factory=time.time)


class ModelPool:
    """
    线程安全的 rPPG 模型池。

    参数
    ----
    model_name : str
        传递给 rppg.Model 的模型名称。
    min_size : int
        池中保持的最小空闲实例数（即初始预热数量）。
    max_size : int
        池允许创建的最大总实例数（空闲 + 使用中）。
        为 0 表示不限制。
    idle_timeout : float
        空闲实例超过 min_size 部分，若超过此时长（秒）未被使用则销毁。
    shrink_interval : float
        后台收缩检查的周期（秒）。
    """

    def __init__(
        self,
        model_name: str,
        min_size: int = 2,
        max_size: int = 0,
        idle_timeout: float = 300.0,
        shrink_interval: float = 60.0,
    ):
        self._model_name = model_name
        self._min_size = max(0, min_size)
        self._max_size = max_size          # 0 = 不限制
        self._idle_timeout = idle_timeout
        self._shrink_interval = shrink_interval

        # 空闲模型队列
        self._idle: list[_PooledModel] = []
        # 当前在使用中的实例数量
        self._in_use: int = 0
        self._lock = threading.Condition(threading.Lock())

        self._closed = False

        # 后台收缩线程
        self._shrink_stop = threading.Event()
        self._shrink_thread = threading.Thread(
            target=self._shrink_loop,
            name="ModelPool-Shrink",
            daemon=True,
        )
        self._shrink_thread.start()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def prewarm(self, count: Optional[int] = None) -> int:
        """
        预热模型实例并放入空闲队列。

        若不指定 count，则预热至 min_size 个（已有的不重复创建）。
        返回实际新建的数量。
        """
        with self._lock:
            if self._closed:
                return 0
            target = self._min_size if count is None else count
            need = target - len(self._idle)

        if need <= 0:
            return 0

        created = 0
        for _ in range(need):
            model = self._create_model()
            if model is None:
                break
            with self._lock:
                if self._closed:
                    self._destroy_model(model)
                    break
                self._idle.append(_PooledModel(model=model))
            created += 1
            logger.info("模型池：预热实例 #%d 完成", created)

        return created

    def acquire(self, timeout: float = 30.0) -> rppg.Model:
        """
        从池中取出一个可用模型实例（阻塞直到拿到或超时）。

        若空闲队列为空且未达 max_size，会立即创建新实例；
        否则等待其他请求归还。

        Raises
        ------
        RuntimeError
            池已关闭或等待超时。
        """
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("ModelPool 已关闭")

                # 优先取空闲实例
                if self._idle:
                    slot = self._idle.pop()
                    self._in_use += 1
                    logger.debug("模型池：取出空闲实例，剩余空闲=%d，使用中=%d",
                                 len(self._idle), self._in_use)
                    return slot.model

                # 没有空闲，判断是否可以扩容
                total = len(self._idle) + self._in_use
                can_expand = (self._max_size == 0) or (total < self._max_size)

            if can_expand:
                # 在锁外创建，避免长时间持锁
                model = self._create_model()
                if model is not None:
                    with self._lock:
                        if self._closed:
                            self._destroy_model(model)
                            raise RuntimeError("ModelPool 已关闭")
                        self._in_use += 1
                    logger.info("模型池：扩容创建新实例，当前使用中=%d", self._in_use)
                    assert model is not None
                    return model
                # 创建失败，退化为等待空闲实例
                can_expand = False

            # 无法扩容，等待归还
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"ModelPool.acquire 超时（{timeout}s），"
                    f"空闲={len(self._idle)}，使用中={self._in_use}"
                )
            with self._lock:
                # 再次确认，避免 TOCTOU
                if self._idle:
                    continue
                self._lock.wait(timeout=min(remaining, 1.0))

        raise RuntimeError("ModelPool.acquire：不应到达此处")  # 仅供类型检查器使用

    def release(self, model: rppg.Model) -> None:
        """
        将模型实例归还到池中。

        归还前会自动重置模型上下文（清空历史帧缓冲），
        使其可被下一个会话复用。
        """
        self._reset_model(model)

        with self._lock:
            self._in_use = max(0, self._in_use - 1)
            if self._closed:
                # 池已关闭，直接销毁
                pass
            else:
                self._idle.append(_PooledModel(model=model))
                self._lock.notify_all()   # 唤醒可能正在等待的 acquire
                logger.debug("模型池：实例归还，空闲=%d，使用中=%d",
                             len(self._idle), self._in_use)
                return

        # 走到这里说明 _closed=True
        self._destroy_model(model)

    def shutdown(self) -> None:
        """关闭模型池，销毁所有空闲实例."""
        self._shrink_stop.set()

        with self._lock:
            self._closed = True
            idle_models = [s.model for s in self._idle]
            self._idle.clear()
            self._lock.notify_all()

        for model in idle_models:
            self._destroy_model(model)

        logger.info("模型池：已关闭，销毁 %d 个空闲实例", len(idle_models))

    @property
    def idle_count(self) -> int:
        """当前空闲实例数量."""
        with self._lock:
            return len(self._idle)

    @property
    def in_use_count(self) -> int:
        """当前使用中实例数量."""
        with self._lock:
            return self._in_use

    @property
    def total_count(self) -> int:
        """池中实例总数（空闲 + 使用中）."""
        with self._lock:
            return len(self._idle) + self._in_use

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _create_model(self) -> Optional[rppg.Model]:
        """创建并进入上下文的新 Model 实例，失败返回 None."""
        try:
            model = rppg.Model(self._model_name)
            model.__enter__()
            return model
        except Exception:
            logger.exception("模型池：创建模型实例失败")
            return None

    @staticmethod
    def _reset_model(model: rppg.Model) -> None:
        """
        重置模型内部状态，清空历史帧缓冲，使其可被新会话复用。

        具体属性名依赖 open-rppg 实现；若接口变更请同步修改此处。
        """
        try:
            # open-rppg 的 Model 支持 reset() 方法（若有）
            if hasattr(model, "reset"):
                model.reset()
                return

            # 回退方案：手动清空已知缓冲区
            with model.frame_lock:
                if hasattr(model, "face_buff"):
                    model.face_buff.clear()
                if hasattr(model, "ts_buff"):
                    model.ts_buff.clear()
        except Exception:
            logger.exception("模型池：重置模型状态失败")

    @staticmethod
    def _destroy_model(model: rppg.Model) -> None:
        """安全退出模型上下文（释放推理线程等资源）."""
        try:
            model.__exit__(None, None, None)
        except Exception:
            logger.exception("模型池：销毁模型实例时出错")

    def _shrink_loop(self) -> None:
        """
        后台收缩线程：定期将超过 min_size 且空闲时间过长的实例销毁。
        """
        while not self._shrink_stop.wait(self._shrink_interval):
            self._shrink_once()

    def _shrink_once(self) -> None:
        """执行一次收缩检查."""
        now = time.time()
        to_destroy: list[rppg.Model] = []

        with self._lock:
            if self._closed:
                return
            excess = len(self._idle) - self._min_size
            if excess <= 0:
                return

            # 按 returned_at 升序排列（最久未用的排前面），优先驱逐
            self._idle.sort(key=lambda s: s.returned_at)
            surviving: list[_PooledModel] = []
            removed = 0
            for slot in self._idle:
                idle_secs = now - slot.returned_at
                if removed < excess and idle_secs >= self._idle_timeout:
                    to_destroy.append(slot.model)
                    removed += 1
                else:
                    surviving.append(slot)
            self._idle = surviving

        if to_destroy:
            logger.info("模型池：收缩销毁 %d 个长期空闲实例（idle_timeout=%.0fs）",
                        len(to_destroy), self._idle_timeout)
            for model in to_destroy:
                self._destroy_model(model)




