"""
主入口：启动 gRPC 服务器（用于 rPPG 帧分析）
并可选启动 FastAPI HTTP 服务器（用于健康检查/调试）。
"""

import logging
import os
import signal
import sys
import threading
from concurrent import futures

import grpc
import uvicorn
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中，保证无论当前工作目录如何
# 都能通过 "src.generated.frame_analysis_pb2" 之类的路径导入。
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generated import frame_analysis_pb2_grpc as pb2_grpc
from src.grpc_servicer import FrameAnalysisServicer
from src.session_manager import SessionManager

# ---------------------------------------------------------------------------
# 配置（可通过环境变量覆盖）
# ---------------------------------------------------------------------------
GRPC_PORT            = int(os.getenv("GRPC_PORT", "50051"))
HTTP_PORT            = int(os.getenv("HTTP_PORT", "8000"))
GRPC_MAX_WORKERS     = int(os.getenv("GRPC_MAX_WORKERS", "8"))
RPPG_MODEL           = os.getenv("RPPG_MODEL", "FacePhys.rlap")
SESSION_TIMEOUT      = int(os.getenv("SESSION_TIMEOUT", "120"))

# --- 模型池参数 ---
# 固定容量：池始终维持该数量的预热实例，不动态扩容或缩容
POOL_SIZE            = int(os.getenv("RPPG_POOL_SIZE", "2"))

# 最大消息长度（默认 64 MB，足够传 30 帧 JPEG）
MAX_MESSAGE_LENGTH   = int(os.getenv("MAX_MESSAGE_LENGTH", str(64 * 1024 * 1024)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 会话管理器（含内置模型池，单例）
# ---------------------------------------------------------------------------
session_manager = SessionManager(
    model_name=RPPG_MODEL,
    timeout=SESSION_TIMEOUT,
    pool_size=POOL_SIZE,
)

# ---------------------------------------------------------------------------
# FastAPI 应用 – 轻量级 HTTP 接口
# ---------------------------------------------------------------------------
app = FastAPI(title="rPPG gRPC Service", version="1.0.0")


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}


@app.get("/sessions")
async def sessions():
    """返回当前活跃的会话 ID 列表。"""
    return {"active_sessions": session_manager.active_session_ids()}


@app.post("/sessions/{session_id}/close")
async def close_session(session_id: str):
    """手动关闭指定会话，模型将归还到池中。"""
    session_manager.remove_session(session_id)
    return {"status": "closed", "session_id": session_id}


@app.get("/pool")
async def pool_stats():
    """返回模型池当前状态（空闲 / 使用中 / 总数）。"""
    stats = session_manager.pool_stats()
    stats.update({
        "pool_size": POOL_SIZE,
    })
    return stats


# ---------------------------------------------------------------------------
# 后台：周期性清理过期会话
# ---------------------------------------------------------------------------
_cleanup_stop = threading.Event()


def _session_cleanup_loop(interval: float = 30.0):
    """定期清理过期会话，模型归还到模型池。"""
    while not _cleanup_stop.is_set():
        try:
            n = session_manager.cleanup_expired()
            if n:
                logger.info("已清理 %d 个过期会话", n)
        except Exception:
            logger.exception("会话清理时发生错误")
        _cleanup_stop.wait(interval)


# ---------------------------------------------------------------------------
# gRPC 服务器
# ---------------------------------------------------------------------------
def serve_grpc() -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=GRPC_MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    servicer = FrameAnalysisServicer(session_manager)
    pb2_grpc.add_FrameAnalysisServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info("gRPC 服务器已启动，端口 %d（workers=%d）", GRPC_PORT, GRPC_MAX_WORKERS)
    return server


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    # 0. 预热模型实例到 pool_size
    logger.info("模型池配置：固定容量=%d", POOL_SIZE)
    warmed = session_manager.prewarm(POOL_SIZE)
    logger.info("模型池预热完成，共 %d 个实例就绪", warmed)

    # 1. 启动 gRPC 服务器
    grpc_server = serve_grpc()

    # 2. 启动会话清理线程
    cleanup_thread = threading.Thread(target=_session_cleanup_loop, daemon=True)
    cleanup_thread.start()

    # 3. 使用后台线程启动 FastAPI/Uvicorn
    uvicorn_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": "0.0.0.0",
            "port": HTTP_PORT,
            "log_level": "info",
        },
        daemon=True,
    )
    uvicorn_thread.start()
    logger.info("HTTP 健康检查服务器已启动，端口 %d", HTTP_PORT)

    # 4. 等待退出信号
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        logger.info("收到信号 %s，正在关闭服务…", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    stop_event.wait()

    # 5. 优雅关闭
    logger.info("正在停止 gRPC 服务器…")
    grpc_server.stop(grace=5).wait()
    _cleanup_stop.set()
    session_manager.shutdown()
    logger.info("服务已完全关闭。")


if __name__ == "__main__":
    main()

