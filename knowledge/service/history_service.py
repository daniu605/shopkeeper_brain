"""历史对话业务层

提供会话历史的查询与清空，供查询 API 层调用。
"""
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger("service.history")


class HistoryService:
    """历史对话服务：查询与清空 MongoDB 聊天记录"""

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话历史（时间升序，含图片链接提取）

        :param session_id: 会话 ID
        :return: [{"role", "text", "ts", "image_urls"}, ...]；异常降级返回 []
        """
        try:
            from knowledge.utils.mongo_history_util import get_recent_messages

            # util 返回降序（最新在前），历史展示需要升序 → 反转
            messages = get_recent_messages(session_id, limit=100)
            messages = list(reversed(messages))

            items = []
            for msg in messages:
                text = msg.get("text", "")
                items.append({
                    "role": msg.get("role", "unknown"),
                    "text": text,
                    "ts": float(msg.get("ts", 0)),
                    "image_urls": self.extract_image_urls(text),
                })
            logger.info(f"[session:{session_id}] 加载历史 {len(items)} 条")
            return items
        except Exception as e:
            logger.error(f"获取历史对话失败（降级返回空）: {e}")
            return []

    def clear_history(self, session_id: str) -> int:
        """
        清空会话历史

        :param session_id: 会话 ID
        :return: 删除的记录数；失败返回 0
        """
        try:
            from knowledge.utils.mongo_history_util import clear_history

            deleted = clear_history(session_id)
            logger.info(f"[session:{session_id}] 清空历史 {deleted} 条")
            return deleted
        except Exception as e:
            logger.error(f"清空历史失败（降级返回 0）: {e}")
            return 0

    @staticmethod
    def extract_image_urls(text: str) -> List[str]:
        """
        从文本中提取图片链接（.png/.jpg/.jpeg/.gif/.webp/.bmp/.svg）

        :param text: 答案或消息文本
        :return: 去重保序的图片链接列表
        """
        if not text:
            return []

        urls = re.findall(r"https?://[^\s)\]}]+", text)
        seen = set()
        result = []
        for url in urls:
            url = url.rstrip("，。；、,.;")
            if re.search(r"\.(png|jpe?g|gif|webp|bmp|svg)(\?|#|$)", url, re.IGNORECASE):
                if url not in seen:
                    seen.add(url)
                    result.append(url)
        return result
