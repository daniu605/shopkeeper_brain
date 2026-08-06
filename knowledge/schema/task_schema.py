"""任务状态数据模型"""
from typing import List, Dict
from pydantic import BaseModel, Field


class TaskStatusResponse(BaseModel):
    """任务状态响应（GET /status/{task_id} 与前端轮询共用）"""
    status: str = Field(..., description="任务状态（processing/completed/failed）")
    done_list: List[str] = Field(default_factory=list, description="已完成节点列表")
    running_list: List[str] = Field(default_factory=list, description="正在运行节点列表")
    durations: Dict[str, float] = Field(default_factory=dict, description="各节点耗时(秒)")
    answer: str = Field("", description="完整答案（完成后返回）")
    error: str = Field("", description="错误信息（空表示成功）")
    image_urls: List[str] = Field(default_factory=list, description="答案中的图片链接")
