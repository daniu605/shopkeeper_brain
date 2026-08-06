"""查询业务层

负责提交查询、后台执行查询图、维护任务状态与 SSE 队列。

流程:
    - 流式（is_stream=True）: 建 SSE 队列 → 后台线程执行查询图 → 立即返回 task_id
    - 非流式（is_stream=False）: 同步执行查询图 → 返回完整答案
"""
import logging
import threading
import uuid
from typing import Any, Dict

logger = logging.getLogger("service.query")


class QueryService:
    """查询服务：提交查询 + 后台执行查询图 + 任务状态维护"""

    def submit_query(self, query: str, session_id: str, is_stream: bool) -> Dict[str, Any]:
        """
        提交查询

        :param query: 用户问题
        :param session_id: 会话 ID
        :param is_stream: 是否流式输出
        :return: 流式 → {message, session_id, task_id}；非流式 → {session_id, answer, ...}
        """
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[task:{task_id}] 提交查询: {query[:50]}... (流式={is_stream})")

        if is_stream:
            return self._submit_stream(task_id, query, session_id)
        return self._submit_sync(task_id, query, session_id)

    # ==================== 流式提交 ====================

    def _submit_stream(self, task_id: str, query: str, session_id: str) -> Dict[str, Any]:
        """
        流式提交：建 SSE 队列 + 开启后台线程执行查询图，立即返回 task_id

        :param task_id: 任务 ID
        :param query: 用户问题
        :param session_id: 会话 ID
        :return: {"message", "session_id", "task_id"}
        """
        from knowledge.utils.sse_util import create_sse_queue
        from knowledge.utils.task_util import TASK_STATUS_PROCESSING, add_done_task, update_task_status

        # 关键：先建 SSE 队列，后台线程才开始推事件（避免事件丢失）
        create_sse_queue(task_id)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        add_done_task(task_id, "query_submit")

        threading.Thread(
            target=self._run_query_background,
            args=(task_id, query, session_id),
            daemon=True,
        ).start()

        logger.info(f"[task:{task_id}] 流式查询已提交，后台执行中")
        return {"message": "查询已提交，正在处理", "session_id": session_id, "task_id": task_id}

    # ==================== 非流式提交（同步） ====================

    def _submit_sync(self, task_id: str, query: str, session_id: str) -> Dict[str, Any]:
        """
        非流式提交：同步执行整个查询图，返回完整答案

        :param task_id: 任务 ID
        :param query: 用户问题
        :param session_id: 会话 ID
        :return: {"session_id", "answer", "task_id", "done_list", "error", "image_urls", "message"}
        """
        from knowledge.utils.task_util import (
            TASK_STATUS_COMPLETED,
            TASK_STATUS_FAILED,
            TASK_STATUS_PROCESSING,
            get_done_task_list,
            update_task_status,
        )
        from knowledge.service.history_service import HistoryService

        try:
            from knowledge.processor.query_process.main_graph import query_app  # 延迟导入，避免服务启动时加载重量级依赖

            update_task_status(task_id, TASK_STATUS_PROCESSING)
            state = self._build_state(task_id, query, session_id, is_stream=False)
            result = query_app.invoke(state)
            answer = (result or {}).get("answer", "")

            update_task_status(task_id, TASK_STATUS_COMPLETED)
            logger.info(f"[task:{task_id}] 同步查询完成（{len(answer)} 字）")
            return {
                "session_id": session_id,
                "answer": answer,
                "task_id": task_id,
                "done_list": get_done_task_list(task_id),
                "error": "",
                "image_urls": HistoryService.extract_image_urls(answer),
                "message": "查询完成",
            }
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"[task:{task_id}] 同步查询失败: {e}", exc_info=True)
            return {
                "session_id": session_id,
                "answer": "",
                "task_id": task_id,
                "done_list": [],
                "error": str(e),
                "image_urls": [],
                "message": "查询失败",
            }

    # ==================== 后台执行 ====================

    def _run_query_background(self, task_id: str, query: str, session_id: str) -> None:
        """
        后台线程入口：执行查询图，更新任务状态并推送 SSE 事件

        :param task_id: 任务 ID
        :param query: 用户问题
        :param session_id: 会话 ID
        :return: None
        """
        try:
            from knowledge.processor.query_process.main_graph import query_app  # 延迟导入
            from knowledge.utils.sse_util import SSEEvent, push_sse_event
            from knowledge.utils.task_util import (
                TASK_STATUS_COMPLETED,
                TASK_STATUS_FAILED,
                get_done_task_list,
                set_task_result,
                update_task_status,
            )

            state = self._build_state(task_id, query, session_id, is_stream=True)
            result = query_app.invoke(state)
            answer = (result or {}).get("answer", "")

            update_task_status(task_id, TASK_STATUS_COMPLETED)
            set_task_result(task_id, "answer", answer)
            # 推送完成进度事件（前端 final 监听器会兜底渲染）
            push_sse_event(task_id, SSEEvent.PROGRESS, {
                "status": TASK_STATUS_COMPLETED,
                "done_list": get_done_task_list(task_id),
                "running_list": [],
            })
            logger.info(f"[task:{task_id}] 流式查询完成（{len(answer)} 字）")
        except Exception as e:
            logger.error(f"[task:{task_id}] 流式查询失败: {e}", exc_info=True)
            try:
                from knowledge.utils.sse_util import SSEEvent, push_sse_event
                from knowledge.utils.task_util import (
                    TASK_STATUS_FAILED,
                    set_task_result,
                    update_task_status,
                )

                update_task_status(task_id, TASK_STATUS_FAILED)
                set_task_result(task_id, "error", str(e))
                # 前端 final 监听器渲染该兜底文案
                push_sse_event(task_id, SSEEvent.FINAL, {"answer": f"抱歉，本次处理失败：{e}"})
                push_sse_event(task_id, SSEEvent.PROGRESS, {
                    "status": TASK_STATUS_FAILED,
                    "done_list": [],
                    "running_list": [],
                })
            except Exception:
                pass

    # ==================== 工具 ====================

    def _build_state(
        self,
        task_id: str,
        query: str,
        session_id: str,
        is_stream: bool,
    ) -> Dict[str, Any]:
        """
        构建查询图初始状态

        :param task_id: 任务 ID
        :param query: 用户问题
        :param session_id: 会话 ID
        :param is_stream: 是否流式
        :return: 查询图初始状态字典
        """
        return {
            "session_id": session_id,
            "task_id": task_id,
            "message_id": "",               # 用户消息由 answer_output 节点创建
            "original_query": query,
            "rewritten_query": "",          # 由 item_name_confirm 节点填充
            "item_names": [],
            "item_name_options": [],
            "history": [],                  # 由 item_name_confirm 节点加载并回填
            "answer": "",
            "is_stream": is_stream,
        }
