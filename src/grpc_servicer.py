"""
gRPC 服务端实现 FrameAnalysisService。

每次 AnalyzeFrames 调用流程：
  1. 按 session_id 获取（或创建）持久化会话。
  2. 解码 WEBP 帧并通过 update_face() 喂给 rPPG Model。
  3. 获取当前 HR/HRV 结果并以 JSON bytes 返回。
"""

import json
import logging
import time

import cv2
import grpc
import numpy as np

from src.generated import frame_analysis_pb2 as pb2
from src.generated import frame_analysis_pb2_grpc as pb2_grpc
from src.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _numpy_to_python(obj):
    """递归将 numpy 类型转换为原生 Python 类型，便于 JSON 序列化."""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class FrameAnalysisServicer(pb2_grpc.FrameAnalysisServiceServicer):
    """实现 FrameAnalysisService RPC."""

    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    def AnalyzeFrames(self, request: pb2.FrameBatchRequest, context: grpc.ServicerContext) -> pb2.FrameBatchResponse:
        session_id = request.session_id
        frames = request.frames

        if not session_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_id is required")
            return pb2.FrameBatchResponse()

        if not frames:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("frames list is empty")
            return pb2.FrameBatchResponse()

        try:
            state = self._session_manager.get_or_create(session_id)
        except RuntimeError as e:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
            return pb2.FrameBatchResponse()

        # 获取该会话的锁，保证同一会话的帧串行处理
        with state.lock:
            model = state.model

            for frame in frames:
                webp_bytes = frame.image_data
                ts_seconds = frame.timestamp_ms / 1000.0

                buf = np.frombuffer(webp_bytes, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

                if img is None:
                    logger.warning("Session %s: failed to decode WEBP frame at ts=%.3f", session_id, ts_seconds)
                    continue

                # BGR -> RGB（open-rppg 期望 RGB）
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 将人脸图像送入模型
                # 图像已是ROI人脸裁剪，可直接调用 update_face
                model.update_face(img_rgb, ts=ts_seconds)

            # 模型推理线程异步运行，稍作等待以处理新帧
            _wait_for_inference(model, timeout=2.0)

            # 收集结果
            result = model.hr()

        if result is None:
            result = {"hr": None, "SQI": None, "hrv": {}, "latency": 0}

        # 转换 numpy 类型并序列化为 JSON
        result = _numpy_to_python(result)
        payload = json.dumps(result, ensure_ascii=False).encode("utf-8")

        return pb2.FrameBatchResponse(
            session_id=session_id,
            payload=payload,
        )


def _wait_for_inference(model, timeout: float = 2.0):
    """
    等待模型推理线程消化完缓存帧，或直到超时。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with model.frame_lock:
            pending = len(model.face_buff)
        if pending == 0:
            break
        time.sleep(0.05)

