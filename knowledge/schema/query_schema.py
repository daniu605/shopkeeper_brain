"""查询接口数据模型

定义前端 chat.html 与查询服务之间的请求/响应结构。
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式输出")


class QueryResponse(BaseModel):
    """非流式查询响应（is_stream=False 时同步返回完整答案）"""
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="完整答案")
    task_id: str = Field("", description="任务ID")
    done_list: List[str] = Field(default_factory=list, description="已完成节点列表")
    error: str = Field("", description="错误信息（空表示成功）")
    image_urls: List[str] = Field(default_factory=list, description="答案中的图片链接")
    message: str = Field("", description="提示消息")


class StreamSubmitResponse(BaseModel):
    """流式提交响应（is_stream=True 时立即返回，前端凭 task_id 建立 SSE 连接）"""
    message: str = Field(..., description="提示消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID")


class MessageItem(BaseModel):
    """历史对话单条消息"""
    role: str = Field(..., description="角色（user/assistant）")
    text: str = Field("", description="消息内容")
    ts: float = Field(0, description="时间戳（秒）")
    image_urls: List[str] = Field(default_factory=list, description="消息中的图片链接")


class HistoryResponse(BaseModel):
    """历史对话响应"""
    items: List[MessageItem] = Field(default_factory=list, description="历史消息列表（时间升序）")
