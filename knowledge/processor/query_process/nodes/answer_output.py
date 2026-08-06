"""
答案生成节点

职责:
    1. 校验并获取输入（历史对话 / reranked_docs / task_id / session_id / item_names / 重写与原始问题）
    2. 格式化 reranked_docs 为参考上下文（元数据 + content）
    3. 用 ANSWER_PROMPT 拼接提示词，调用 LLM 流式生成答案
    4. 通过 SSE 推送流式增量（DELTA）与最终答案（FINAL）
    5. 将用户提问与助手答案保存到 MongoDB（两条记录）
    6. 输出 state["answer"]

分支逻辑:
    - state["answer"] 已有值（如商品名确认节点直接给出的提示）→ 不走 LLM，
      直接流式输出该答案，并同样保存两条记录
    - state["answer"] 为空 → 走完整 LLM 流式生成流程

设计:
    - 防御性编程：字段缺失取默认值；重写问题缺失取原始问题
    - 降级策略：LLM 生成失败 → 返回兜底提示文案；保存失败 → 记日志不阻塞
    - 日志可观测性：每个步骤都有日志
"""
from typing import Any, Dict, List, Optional, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState | Dict[str, Any]:
        """
        答案生成主流程

        :param state: 查询图状态
        :return: 更新后的 state（answer 已填充，同时经 SSE 流式输出并落库）
        """
        # 1. 校验并获取输入（防御性取值）
        (
            session_id, task_id, question, item_names,
            history_text, rewritten_query, message_id, answer,
        ) = self._prepare(state)

        # 2. 分支：answer 已有值 → 直接流式输出（不走 LLM）
        if answer:
            self.log_step("步骤2", "answer 已有值，直接流式输出（不走 LLM）")
            self._stream_text(task_id, answer)
        else:
            # 3. 格式化 reranked_docs 为参考上下文
            docs = state.get("reranked_docs", [])
            self.log_step("步骤3", f"格式化参考上下文（{len(docs)} 条文档）")
            context = self._format_context(docs)

            # 4. 拼接 ANSWER_PROMPT 提示词
            prompt = self._build_prompt(context, history_text, item_names, question)
            self.log_step("步骤4", "调用 LLM 流式生成答案")

            # 5. LLM 流式生成（内部推 DELTA / FINAL 事件）
            answer = self._stream_generate(task_id, prompt)

        # 6. 保存用户提问 + 助手答案两条记录到 MongoDB
        self.log_step("步骤5", "保存对话记录（用户 + 助手）")
        self._save_messages(session_id, question, rewritten_query, item_names, answer, message_id)

        # 7. 输出 state
        state["answer"] = answer
        self.log_step("完成", f"答案生成完成（{len(answer)} 字）")
        return {"answer": answer}

    # ==================== 输入准备 ====================

    def _prepare(
        self,
        state: QueryGraphState,
    ) -> Tuple[str, str, str, List[str], str, str, str, str]:
        """
        校验并获取节点所需输入（防御性取值）

        :param state: 查询图状态
        :return: (session_id, task_id, question, item_names, history_text,
                  rewritten_query, message_id, answer)
        :raises StateFieldError: session_id 或 question（重写/原始均无）缺失
        """
        session_id = state.get("session_id")
        task_id = state.get("task_id", "")
        question = state.get("rewritten_query") or state.get("original_query")  # 重写缺失取原始
        item_names = state.get("item_names", [])
        rewritten_query = state.get("rewritten_query", "")
        message_id = state.get("message_id", "")
        answer = state.get("answer", "")

        if not session_id:
            raise StateFieldError(node_name=self.name, field_name="session_id", expected_type=str)
        if not question:
            raise StateFieldError(node_name=self.name, field_name="rewritten_query/original_query", expected_type=str)
        if not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 历史对话：state["history"] 为消息字典列表 → 格式化为文本（缺失取空）
        history_text = self._format_history(state.get("history", []))

        self.log_step("步骤1", f"输入就绪: session={session_id}, 商品={item_names}, answer_已有={bool(answer)}")
        return session_id, task_id, question, item_names, history_text, rewritten_query, message_id, answer

    def _format_history(self, messages: List[Dict[str, Any]]) -> str:
        """
        将历史消息列表格式化为 "role: text" 文本

        :param messages: 消息字典列表（[{role, text, ...}, ...]）
        :return: 格式化文本；无历史返回 "暂无历史对话"
        """
        if not messages:
            return "暂无历史对话"

        lines = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            text = msg.get("text", "")
            lines.append(f"{role}: {text}")
        return "\n".join(lines) if lines else "暂无历史对话"

    # ==================== 上下文与提示词 ====================

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        将 reranked_docs 格式化为参考上下文

        每个文档格式: [来源 | 标题 | 链接 | 切片ID]\ncontent: {内容}

        :param docs: 统一格式的文档列表（{source, title, url, chunk_id, content}）
        :return: 拼接后的参考上下文；无文档返回空串
        """
        if not docs:
            self.logger.warning("无参考文档，context 为空")
            return ""

        blocks = []
        for doc in docs:
            doc = doc if isinstance(doc, dict) else {}
            meta_parts = []
            if doc.get("source"):
                meta_parts.append(f"来源:{doc['source']}")
            if doc.get("title"):
                meta_parts.append(f"标题:{doc['title']}")
            if doc.get("url"):
                meta_parts.append(f"链接:{doc['url']}")
            if doc.get("chunk_id"):
                meta_parts.append(f"切片ID:{doc['chunk_id']}")
            meta_str = " | ".join(meta_parts) if meta_parts else "无元数据"
            blocks.append(f"[{meta_str}]\ncontent: {doc.get('content', '')}")

        return "\n\n".join(blocks)

    def _build_prompt(
        self,
        context: str,
        history_text: str,
        item_names: List[str],
        question: str,
    ) -> str:
        """
        拼接 ANSWER_PROMPT 提示词

        :param context: 格式化后的参考上下文
        :param history_text: 历史对话文本
        :param item_names: 商品名列表
        :param question: 用户问题（重写优先）
        :return: 完整提示词
        """
        item_names_text = "、".join(item_names) if item_names else "无"
        return ANSWER_PROMPT.format(
            context=context if context else "暂无参考内容",
            history=history_text,
            item_names=item_names_text,
            question=question,
        )

    # ==================== 流式生成 ====================

    def _stream_generate(self, task_id: str, prompt: str) -> str:
        """
        调用 LLM 流式生成答案，同时经 SSE 推送增量（DELTA）与最终答案（FINAL）

        降级策略: LLM 调用失败 → 记 error 日志，返回兜底提示文案（同样推送 FINAL）

        :param task_id: 任务 ID（SSE 队列标识）
        :param prompt: 完整提示词
        :return: 完整答案文本
        """
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入
            from knowledge.utils.sse_util import SSEEvent, push_sse_event

            llm_client = AIClients.get_llm_openai(False)  # 自由文本生成，不用 JSON 模式
            if not llm_client:
                self.logger.warning("LLM 客户端获取失败，返回兜底文案")
                fallback = "抱歉，答案生成服务暂时不可用，请稍后再试。"
                push_sse_event(task_id, SSEEvent.FINAL, {"answer": fallback})
                return fallback

            full_answer = ""
            for chunk in llm_client.stream([{"role": "user", "content": prompt}]):
                piece = (chunk.content or "") if hasattr(chunk, "content") else str(chunk)
                if not piece:
                    continue
                full_answer += piece
                # 流式推送增量
                push_sse_event(task_id, SSEEvent.DELTA, {"delta": piece})

            # 推送最终完整答案
            push_sse_event(task_id, SSEEvent.FINAL, {"answer": full_answer})
            self.logger.info(f"LLM 流式生成完成（{len(full_answer)} 字）")
            return full_answer
        except Exception as e:
            self.logger.error(f"LLM 流式生成失败（返回兜底文案）: {e}")
            fallback = "抱歉，答案生成失败，请稍后再试。"
            try:
                from knowledge.utils.sse_util import SSEEvent, push_sse_event
                push_sse_event(task_id, SSEEvent.FINAL, {"answer": fallback})
            except Exception:
                pass
            return fallback

    def _stream_text(self, task_id: str, text: str) -> None:
        """
        直接流式输出已有答案（不走 LLM）: 推送 DELTA（整体）与 FINAL

        :param task_id: 任务 ID
        :param text: 已有答案文本
        :return: None
        """
        try:
            from knowledge.utils.sse_util import SSEEvent, push_sse_event

            push_sse_event(task_id, SSEEvent.DELTA, {"delta": text})
            push_sse_event(task_id, SSEEvent.FINAL, {"answer": text})
            self.logger.info(f"直接流式输出已有答案（{len(text)} 字）")
        except Exception as e:
            self.logger.error(f"流式输出已有答案失败: {e}")

    # ==================== 历史保存 ====================

    def _save_messages(
        self,
        session_id: str,
        question: str,
        rewritten_query: str,
        item_names: List[str],
        answer: str,
        message_id: str,
    ) -> Tuple[str, str]:
        """
        保存用户提问 + 助手答案两条记录到 MongoDB

        用户记录: 有 message_id → 更新（补 rewritten_query/item_names）；无 → 插入
        助手记录: 始终插入新记录

        降级策略: 保存失败 → 记 error 日志，不阻塞主流程

        :param session_id: 会话 ID
        :param question: 用户问题（重写优先）
        :param rewritten_query: 重写后的问题
        :param item_names: 商品名列表
        :param answer: 助手答案
        :param message_id: 用户消息 ID（可能为空）
        :return: (用户记录 ID, 助手记录 ID)
        """
        try:
            from knowledge.utils.mongo_history_util import save_chat_message  # 延迟导入

            # 1. 用户提问记录（有 message_id → 更新，无 → 插入）
            user_message_id = save_chat_message(
                session_id=session_id,
                role="user",
                text=question,
                rewritten_query=rewritten_query,
                item_names=item_names,
                message_id=message_id or None,
            )

            # 2. 助手答案记录（始终插入新记录）
            assistant_message_id = save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                item_names=item_names,
            )

            self.logger.info(
                f"对话记录已保存: 用户={user_message_id}, 助手={assistant_message_id}"
            )
            return user_message_id, assistant_message_id
        except Exception as e:
            self.logger.error(f"对话记录保存失败（降级跳过，不影响答案返回）: {e}")
            return "", ""


# ================================================================== #
#                        测试入口                                     #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    from knowledge.utils.sse_util import create_sse_queue, get_sse_queue

    task_id = "test_answer_task"
    create_sse_queue(task_id)

    # 场景1: answer 已有值 → 直接流式输出（不调 LLM，快速可测）
    print("\n" + "=" * 60)
    print("【场景1】answer 已有值（不走 LLM）")
    state_1 = {
        "session_id": "test_session_answer_1",
        "task_id": task_id,
        "message_id": "",
        "original_query": "RS-12 数字万用表怎么测电压？",
        "rewritten_query": "RS PRO RS-12 数字万用表怎么测电压？",
        "item_names": ["RS PRO RS-12 数字万用表"],
        "history": [],
        "reranked_docs": [],
        "answer": "您想询问哪个商品？请选择：\n1. RS PRO RS-12 数字万用表",
    }
    node = AnswerOutputNode()
    result_1 = node(state_1)
    print(f"  answer: {result_1.get('answer')}")

    # 场景2: answer 为空 → 走 LLM 流式生成（需 LLM API + MongoDB，在 PyCharm 中运行）
    print("\n" + "=" * 60)
    print("【场景2】answer 为空（走 LLM 流式生成，需 LLM/MongoDB 环境）")
    state_2 = {
        "session_id": "test_session_answer_2",
        "task_id": task_id,
        "message_id": "",
        "original_query": "万用表如何测量电阻？",
        "rewritten_query": "RS PRO RS-12 数字万用表如何测量电阻？",
        "item_names": ["RS PRO RS-12 数字万用表"],
        "history": [],
        "reranked_docs": [
            {"chunk_id": 468071782111254662, "source": "local", "title": "电阻测量",
             "url": "", "content": "将功能转盘置于最高电阻Ω位置，黑色表笔插入COM端口..."},
            {"chunk_id": "", "source": "web", "title": "万用表电阻测量教程",
             "url": "https://example.com/meter-resistance", "content": "万用表测电阻步骤详解..."},
        ],
        "answer": "",
    }
    result_2 = node(state_2)
    print(f"  answer: {str(result_2.get('answer'))[:100]}...")

    # 打印 SSE 队列中的流式事件（验证流式输出）
    print("\n" + "=" * 60)
    print("SSE 队列事件（流式输出验证）:")
    q = get_sse_queue(task_id)
    events = []
    while not q.empty():
        msg = q.get()
        events.append(msg["event"])
        data = msg["data"]
        if msg["event"] == "delta":
            print(f"  [delta] {str(data.get('delta', ''))[:40]}...")
        else:
            print(f"  [{msg['event']}] {str(data.get('answer', ''))[:40]}...")
    print(f"  事件序列: {events}")
    print("=" * 60)
