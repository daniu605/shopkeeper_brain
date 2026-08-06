"""
查询流程 FastAPI 路由（对接前端 chat.html）

接口契约（与 chat.html 前端一一对应）:
    POST  /query               提交查询；流式返回 task_id，非流式返回完整答案
    GET   /stream/{task_id}    SSE 长连接，持续推送 progress/delta/final 事件
    GET   /status/{task_id}    任务状态查询（轮询备用）
    GET   /history/{session_id} 加载会话历史（时间升序）
    DELETE /history/{session_id} 清空会话历史

架构:
    API 层（本文件）→ 业务层（service/）→ 查询图（query_process）
    service 实例通过 core/deps.py 的缓存单例 + Depends 注入
"""
import logging
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_history_service, get_query_service
from knowledge.schema.query_schema import HistoryResponse, QueryRequest
from knowledge.service.history_service import HistoryService
from knowledge.service.query_service import QueryService

# 查询服务日志器
logger = logging.getLogger("api.query")

# 前端静态页面目录（本文件位于 knowledge/api/ 下）
FRONT_DIR = Path(__file__).resolve().parent.parent / "front"


def register_router(app: FastAPI) -> None:
    """
    注册查询相关路由到 FastAPI 应用

    :param app: FastAPI 应用实例
    :return: None
    """

    @app.post("/query")
    async def query(
        req: QueryRequest,
        service: QueryService = Depends(get_query_service),  # DI 注入缓存单例
    ):
        """
        提交查询

        :param req: 查询请求（query / session_id / is_stream）
        :param service: 查询服务（Depends 注入）
        :return: 流式 → {message, session_id, task_id}；非流式 → 完整答案
        """
        logger.info(f"[session:{req.session_id}] 收到查询: {req.query[:50]}... (流式={req.is_stream})")
        result = service.submit_query(req.query, req.session_id or "", req.is_stream)
        logger.info(f"[session:{req.session_id}] 查询已响应: {result.get('task_id')}")
        return result

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        """
        SSE 长连接：持续推送 progress（节点进度）/ delta（流式增量）/ final（最终答案）

        :param task_id: 任务 ID
        :param request: FastAPI 请求（用于感知前端断开）
        :return: StreamingResponse（text/event-stream）
        """
        from knowledge.utils.sse_util import sse_generator  # 延迟导入

        logger.info(f"[task:{task_id}] SSE 长连接建立")
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲，保证流式即时性
            },
        )

    @app.get("/status/{task_id}")
    async def get_status(task_id: str):
        """
        任务状态查询（前端轮询备用）

        :param task_id: 任务 ID
        :return: status / done_list / running_list / durations / answer / error / image_urls
        """
        from knowledge.utils.task_util import get_task_info, get_task_results  # 延迟导入

        info = get_task_info(task_id)
        if not info["status"]:
            return {"task_id": task_id, "status": "not_found", "message": "任务不存在"}

        result = get_task_results(task_id)
        answer = result.get("answer", "")
        info["task_id"] = task_id
        info["result"] = result
        info["answer"] = answer
        info["error"] = result.get("error", "")
        info["image_urls"] = HistoryService.extract_image_urls(answer)
        return info

    @app.get("/history/{session_id}", response_model=HistoryResponse)
    async def get_history(
        session_id: str,
        service: HistoryService = Depends(get_history_service),  # DI 注入缓存单例
    ):
        """
        加载会话历史（时间升序）

        :param session_id: 会话 ID
        :param service: 历史服务（Depends 注入）
        :return: {"items": [{role, text, ts, image_urls}, ...]}
        """
        return {"items": service.get_history(session_id)}

    @app.delete("/history/{session_id}")
    async def delete_history(
        session_id: str,
        service: HistoryService = Depends(get_history_service),  # DI 注入缓存单例
    ):
        """
        清空会话历史

        :param session_id: 会话 ID
        :param service: 历史服务（Depends 注入）
        :return: {"message", "deleted"}
        """
        deleted = service.clear_history(session_id)
        logger.info(f"[session:{session_id}] 清空历史，删除 {deleted} 条")
        return {"message": "已清空", "deleted": deleted}


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例

    :return: FastAPI 应用
    """
    app = FastAPI(title="知识库查询服务")

    # CORS：允许前端跨域访问（开发期放开）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_router(app)

    # 托管前端静态页面：浏览器打开 http://127.0.0.1:8001/chat.html 即可使用
    # 注意：须在注册 API 路由之后再挂载，保证 /query、/stream 优先匹配
    if FRONT_DIR.exists():
        app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="front")

    return app


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8001, log_level="info")
