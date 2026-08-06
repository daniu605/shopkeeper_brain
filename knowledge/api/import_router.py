"""
导入流程 FastAPI 路由

架构:
    - POST /upload : 接收上传文件 → 双保存(本地 temp + MinIO) → 开启后台线程跑导入流水线
                     → 立即返回 task_id（不阻塞，前端马上开始轮询）
    - GET  /status : 前端按 status_poll_interval 轮询该接口，获取任务各节点实时状态
                     （运行中节点 / 已完成节点 / 各节点耗时 / 总体状态）

设计要点:
    - 职责分离：保存文件 / 后台执行 / 路由处理 各自独立成函数
    - 外部服务(MinIO)调用包 try/except 兜底：失败只告警，不阻塞导入
    - 日志观测：api.import 日志器，每个关键步骤都有日志
    - 复用 task_util.py 的任务追踪机制（BaseNode 已自动更新节点状态）
"""
import io
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    get_task_info,
    get_task_results,
    set_task_result,
    update_task_status,
)

# 任务日志器（与节点日志 import.xxx 同级，统一格式）
logger = logging.getLogger("api.import")

# 本地工作目录：knowledge/temp（本文件位于 knowledge/api/ 下）
WORK_DIR = Path(__file__).resolve().parent.parent / "temp"


def _save_uploaded_file(filename: str, content: bytes, task_id: str, work_dir: Path) -> Tuple[str, str]:
    """
    双保存上传文件：本地 temp/{task_id}/ + MinIO 按时间分区

    MinIO 对象路径按上传时间分区（精确到小时）:
        {年}/{月}/{日}/{时}/{文件名}，例: 2026/08/01/10/万用表RS-12的使用.pdf

    :param filename: 原始文件名（会被清洗为纯文件名，防路径穿越）
    :param content: 文件二进制内容
    :param task_id: 任务 ID
    :param work_dir: 本地工作目录根路径
    :return: (本地文件完整路径, MinIO URL)；MinIO 保存失败时 URL 为空字符串
    """
    # 1. 清洗文件名，防止路径穿越（../ 之类）
    safe_name = Path(filename).name or "uploaded_file"

    # 2. 保存到本地（流水线依赖的关键副本）
    task_dir = work_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    local_path = task_dir / safe_name
    local_path.write_bytes(content)
    logger.info(f"[task:{task_id}] 本地保存成功: {local_path} ({len(content)} 字节)")

    # 3. 上传到 MinIO（备份副本；失败仅告警，不阻塞导入流程）
    try:
        from knowledge.utils.client.storage_clients import StorageClients

        config = get_config()
        minio_client = StorageClients.get_minio_client()
        # 按时间分区（精确到小时），例: 2026/08/01/10/万用表RS-12的使用.pdf
        time_prefix = datetime.now().strftime("%Y/%m/%d")
        object_name = f"import/{time_prefix}/{task_id}/{safe_name}"
        minio_client.put_object(
            bucket_name=config.minio_bucket,
            object_name=object_name,
            data=io.BytesIO(content),
            length=len(content),
        )
        url = f"{config.get_minio_base_url()}/{config.minio_bucket}/{object_name}"
        logger.info(f"[task:{task_id}] MinIO 保存成功: {url}")
        return str(local_path), url
    except Exception as e:
        logger.warning(f"[task:{task_id}] MinIO 保存失败（不影响本地导入）: {e}")
        return str(local_path), ""


def _run_import_background(task_id: str, import_file_path: str, file_dir: str) -> None:
    """
    后台线程入口：执行导入流水线并更新任务状态

    :param task_id: 任务 ID
    :param import_file_path: 文件本地完整路径
    :param file_dir: 输出目录（MinerU 转换产物将写在该目录下）
    :return: None
    """
    logger.info(f"[task:{task_id}] 后台线程启动，开始导入: {import_file_path}")
    try:
        # 延迟导入主图：避免 API 服务启动时加载整条流水线的重量级依赖
        from knowledge.processor.import_process.main_graph import run_import_graph

        final_state = run_import_graph(import_file_path, file_dir, task_id)

        update_task_status(task_id, TASK_STATUS_COMPLETED)
        item_name = (final_state or {}).get("item_name", "")
        if item_name:
            set_task_result(task_id, "item_name", item_name)
        logger.info(f"[task:{task_id}] 导入完成，识别商品名: {item_name}")
    except Exception as e:
        logger.error(f"[task:{task_id}] 导入失败: {e}", exc_info=True)
        update_task_status(task_id, TASK_STATUS_FAILED)
        set_task_result(task_id, "error", str(e))


def register_router(app: FastAPI) -> None:
    """
    注册导入相关路由到 FastAPI 应用

    :param app: FastAPI 应用实例
    :return: None
    """

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """
        上传文件并异步启动导入流程

        :param file: 上传的 PDF/MD 文件（multipart/form-data）
        :return: {"task_id", "status", "message"}，立即返回，不等待导入完成
        """
        # 1. 生成任务 ID
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[task:{task_id}] 收到上传文件: {file.filename}")

        # 2. 读取内容并双保存（本地 + MinIO）
        try:
            content = await file.read()
            local_path, minio_url = _save_uploaded_file(
                file.filename or "", content, task_id, WORK_DIR
            )
        except Exception as e:
            logger.error(f"[task:{task_id}] 文件保存失败: {e}")
            raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

        # 3. 初始化任务状态
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        add_done_task(task_id, "upload_file")

        # 4. 开启后台线程执行导入（不阻塞请求）
        file_dir = str(Path(local_path).parent)
        threading.Thread(
            target=_run_import_background,
            args=(task_id, local_path, file_dir),
            daemon=True,
        ).start()

        # 5. 立即返回，前端拿到 task_id 后开始轮询 /status
        return {
            "task_id": task_id,
            "status": TASK_STATUS_PROCESSING,
            "message": "文件已接收，导入进行中",
            "minio_url": minio_url or None,
        }

    @app.get("/status/{task_id}")
    @app.get("/status")
    async def get_status(
        task_id: str,
        # 依赖注入(DI)：框架自动调用 get_config() 工厂，把配置单例注入进来
        # get_config 是「双重检查锁单例」→ Depends(DI) × 单例缓存 的组合示例
        config: ImportConfig = Depends(get_config),
    ):
        """
        获取任务实时状态（前端轮询接口）

        支持两种访问形式（前端用路径参数版）:
            GET /status/{task_id}
            GET /status?task_id=xxx

        :param task_id: 任务 ID
        :param config: 配置对象（由框架通过 Depends 注入，无需手动调用 get_config）
        :return: 任务状态信息：status / running_list / done_list / durations / result / poll_interval
        """
        info = get_task_info(task_id)

        if not info["status"]:
            # 任务不存在（可能从未创建，或已被清理）
            return {"task_id": task_id, "status": "not_found", "message": "任务不存在"}

        info["task_id"] = task_id
        info["result"] = get_task_results(task_id)
        info["poll_interval"] = config.status_poll_interval  # 注入的配置对象，前端直接读取轮询间隔
        return info


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例

    :return: FastAPI 应用
    """
    app = FastAPI(title="知识库导入服务")

    # CORS：允许前端跨域访问（开发期放开，生产按需收紧）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_router(app)

    # 托管前端静态页面：浏览器打开 http://127.0.0.1:8000/import.html 即可使用
    # 注意：须在注册 API 路由之后再挂载，保证 /upload、/status 优先匹配
    front_dir = Path(__file__).resolve().parent.parent / "front"
    if front_dir.exists():
        app.mount("/", StaticFiles(directory=front_dir, html=True), name="front")

    return app


if __name__ == "__main__":
    from knowledge.processor.import_process.base import setup_logging

    setup_logging()
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8000, log_level="info")
