"""
模型池（ModelPool）：固定容量的 rPPG Model 实例池。

设计目标：
  - 维护一批预热好的空闲模型实例，避免每次请求时重新加载权重。
  - 固定容量：始终保持 size 个实例，不动态扩容或缩容。
  - 不复用已使用模型：模型一旦被使用完毕（客户端断连），立即销毁，
    并异步创建一个全新实例补充到池中，确保池始终满容量。
"""

import logging
import threading
from typing import Optional

import rppg

logger = logging.getLogger(__name__)


class ModelPool:
    """
    线程安全的固定容量 rPPG 模型池。

    参数
    ----
    model_name : str
        传递给 rppg.Model 的模型名称。
    size : int
        池的固定容量（空闲实例数量上限）。
    """

    def __init__(
        self,
        model_name: str,
        size: int = 2,
    ):
        self._model_name = model_name
        self._size = max(1, size)

        # 空闲模型队列
        self._idle: list[rppg.Model] = []
        # 当前在使用中的实例数量
        self._in_use: int = 0
        self._lock = threading.Condition(threading.Lock())

        self._closed = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def prewarm(self, count: Optional[int] = None) -> int:
        """
        预热模型实例并放入空闲队列。

        若不指定 count，则预热至 size 个（已有的不重复创建）。
        返回实际新建的数量。
        """
        with self._lock:
            if self._closed:
                return 0
            target = self._size if count is None else count
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
                self._idle.append(model)
            created += 1
            logger.info("模型池：预热实例 #%d 完成", created)

        return created

    def acquire(self, timeout: float = 30.0) -> rppg.Model:
        """
        从池中取出一个可用模型实例（阻塞直到拿到或超时）。

        池容量固定，若暂时没有空闲实例则等待直到有实例被补充。

        Raises
        ------
        RuntimeError
            池已关闭或等待超时。
        """
        import time
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("ModelPool 已关闭")

                if self._idle:
                    model = self._idle.pop()
                    self._in_use += 1
                    logger.debug("模型池：取出空闲实例，剩余空闲=%d，使用中=%d",
                                 len(self._idle), self._in_use)
                    return model

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"ModelPool.acquire 超时（{timeout}s），"
                        f"空闲={len(self._idle)}，使用中={self._in_use}"
                    )
                self._lock.wait(timeout=min(remaining, 1.0))

    def discard(self, model: rppg.Model) -> None:
        """
        丢弃已使用的模型实例（销毁），并在后台异步创建一个新实例补充到池中。

        客户端断连后调用此方法，确保模型不被复用，
        同时保持池的固定容量。
        """
        with self._lock:
            self._in_use = max(0, self._in_use - 1)

        # 销毁旧实例
        self._destroy_model(model)
        logger.info("模型池：旧实例已销毁，启动后台线程补充新实例")

        # 异步补充新实例，避免阻塞调用方
        t = threading.Thread(
            target=self._replenish_one,
            name="ModelPool-Replenish",
            daemon=True,
        )
        t.start()

    def shutdown(self) -> None:
        """关闭模型池，销毁所有空闲实例."""
        with self._lock:
            self._closed = True
            idle_models = list(self._idle)
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

    def _replenish_one(self) -> None:
        """创建一个新的模型实例并放入空闲队列（供后台线程调用）."""
        model = self._create_model()
        if model is None:
            logger.error("模型池：补充新实例失败")
            return

        with self._lock:
            if self._closed:
                self._destroy_model(model)
                return
            self._idle.append(model)
            self._lock.notify_all()  # 唤醒可能正在等待的 acquire
            logger.info("模型池：新实例补充完成，当前空闲=%d，使用中=%d",
                        len(self._idle), self._in_use)

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
    def _destroy_model(model: rppg.Model) -> None:
        """安全退出模型上下文（释放推理线程等资源）."""
        try:
            model.__exit__(None, None, None)
        except Exception:
            logger.exception("模型池：销毁模型实例时出错")

