"""依赖注入（DI）与缓存单例

通过「双重检查锁」缓存 service 实例（全进程只造一份），
再经 FastAPI Depends 注入到路由 —— DI × 缓存 的组合用法。

用法:
    @app.post("/query")
    async def query(req: QueryRequest, service: QueryService = Depends(get_query_service)):
        ...
"""
import threading
from typing import Optional

# 缓存单例：查询服务
_query_service: Optional[object] = None
_query_service_lock = threading.Lock()

# 缓存单例：历史服务
_history_service: Optional[object] = None
_history_service_lock = threading.Lock()


def get_query_service():
    """
    获取查询服务（缓存单例，供 Depends 注入）

    双重检查锁：先无锁快速检查 → 无则加锁 → 加锁后再检查一次，
    防止多线程同时创建多个实例。

    :return: QueryService 单例
    """
    global _query_service
    if _query_service is None:
        with _query_service_lock:
            if _query_service is None:
                from knowledge.service.query_service import QueryService  # 延迟导入

                _query_service = QueryService()
    return _query_service


def get_history_service():
    """
    获取历史服务（缓存单例，供 Depends 注入）

    :return: HistoryService 单例
    """
    global _history_service
    if _history_service is None:
        with _history_service_lock:
            if _history_service is None:
                from knowledge.service.history_service import HistoryService  # 延迟导入

                _history_service = HistoryService()
    return _history_service
