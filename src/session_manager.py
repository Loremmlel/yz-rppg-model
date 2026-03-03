"""
会话管理器：维护每个 session 的 rPPG Model 实例。

每个 session_id 映射到持久化 Model 上下文，用于累计历史帧以提高 HR/HRV 计算准确性。
内部通过 ModelPool 复用模型实例，避免每次请求重新加载权重。
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import rppg

from src.model_pool import ModelPool

logger = logging.getLogger(__name__)

# 默认使用的 rPPG 模型
DEFAULT_MODEL = "FacePhys.rlap"
# 会话空闲超过该时间（秒）将被清理
SESSION_TIMEOUT_SECONDS = 120


@dataclass
class SessionState:
    """保存单个会话的 rPPG Model 及同步对象."""
    session_id: str
    model: rppg.Model
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_active: float = field(default_factory=time.time)
    entered: bool = False


class SessionManager:
    """
    线程安全的会话管理器，按 session 维护 rPPG Model 实例。

    内部使用 ModelPool 进行模型复用：
      - 会话创建时从池中 acquire() 一个已预热的模型实例。
      - 会话结束（超时/手动关闭）时将模型 release() 回池，
        池会自动重置其上下文供下一个会话使用。
      - 池在长时间空闲时会自动收缩到 pool_min_size，节约资源。

    参数
    ----
    model_name : str
        rPPG 模型名称。
    timeout : float
        会话空闲超时时间（秒）。
    pool_min_size : int
        模型池最小空闲实例数（同时也是预热数量）。
    pool_max_size : int
        模型池允许的最大总实例数（0 = 不限制）。
    pool_idle_timeout : float
        超过 min_size 的空闲实例，闲置超过此时长（秒）后销毁。
    pool_shrink_interval : float
        模型池后台收缩检查周期（秒）。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        timeout: float = SESSION_TIMEOUT_SECONDS,
        pool_min_size: int = 2,
        pool_max_size: int = 0,
        pool_idle_timeout: float = 300.0,
        pool_shrink_interval: float = 60.0,
    ):
        self._sessions: dict[str, SessionState] = {}
        self._global_lock = threading.Lock()
        self._model_name = model_name
        self._timeout = timeout
        self._closed = False

        # 模型池（取代原先简单的 _warm_pool 列表）
        self._pool = ModelPool(
            model_name=model_name,
            min_size=pool_min_size,
            max_size=pool_max_size,
            idle_timeout=pool_idle_timeout,
            shrink_interval=pool_shrink_interval,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def prewarm(self, count: int) -> int:
        """
        预热指定数量的模型实例放入池中，返回实际新建数量。
        若 count <= 0，则预热至 pool_min_size。
        """
        if count <= 0:
            return self._pool.prewarm()
        return self._pool.prewarm(count)

    def get_or_create(self, session_id: str) -> SessionState:
        """获取指定 session_id 的 SessionState，不存在则从池中取模型创建."""
        with self._global_lock:
            if self._closed:
                raise RuntimeError("SessionManager is shut down")
            if session_id not in self._sessions:
                logger.info("创建新会话：%s", session_id)
                # 从模型池取出一个预热好的实例
                model = self._pool.acquire()
                state = SessionState(session_id=session_id, model=model, entered=True)
                self._sessions[session_id] = state
            state = self._sessions[session_id]
            state.last_active = time.time()
            return state

    def remove_session(self, session_id: str) -> None:
        """显式关闭并移除会话，将模型归还到池中."""
        with self._global_lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            self._return_session_model(state)

    def cleanup_expired(self) -> int:
        """清理空闲超时的会话，将其模型归还到池中，返回清理数量."""
        now = time.time()
        expired: list[SessionState] = []
        with self._global_lock:
            to_remove = [
                sid for sid, st in self._sessions.items()
                if now - st.last_active > self._timeout
            ]
            for sid in to_remove:
                expired.append(self._sessions.pop(sid))

        for state in expired:
            logger.info("清理过期会话：%s（空闲 %.0fs）",
                        state.session_id, now - state.last_active)
            self._return_session_model(state)

        return len(expired)

    def shutdown(self) -> None:
        """关闭所有会话并关闭模型池."""
        with self._global_lock:
            self._closed = True
            all_states = list(self._sessions.values())
            self._sessions.clear()

        for state in all_states:
            logger.info("关闭会话：%s", state.session_id)
            self._return_session_model(state)

        # 关闭模型池（销毁所有空闲实例）
        self._pool.shutdown()

    def active_session_ids(self) -> list[str]:
        """返回当前活跃的会话 ID 列表."""
        with self._global_lock:
            return list(self._sessions.keys())

    def pool_stats(self) -> dict:
        """返回模型池当前统计信息，便于监控接口展示."""
        return {
            "idle": self._pool.idle_count,
            "in_use": self._pool.in_use_count,
            "total": self._pool.total_count,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _return_session_model(self, state: SessionState) -> None:
        """将会话持有的模型归还到模型池（而非销毁）."""
        try:
            # release 内部会重置模型上下文，无需在此处理
            self._pool.release(state.model)
            state.entered = False
        except Exception:
            logger.exception("归还会话 %s 的模型时出错", state.session_id)
