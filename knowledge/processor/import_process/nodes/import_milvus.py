"""
切片入库节点（Milvus）

职责:
    将向量化后的切片存入 Milvus 集合(kb_chunks_v1)，供查询流程做混合检索
    （稠密向量语义检索 COSINE + 稀疏向量关键词检索 IP），并把自动生成的主键
    回填到 Chunk.chunk_id。

处理流程（仓库比喻）:
    1. 校验上游切片（无 content/向量 → 日志标记并剔除）
    2. 连接 Milvus，按三情况处理集合：
       A 有旧库且 delete_flag=True → drop 重建
       B 空地（不存在/已 drop）   → create
       C 已存在且不拆             → 直接用
    3. 建库规划货架（schema）
    4. 建两套检索台（索引）：稠密 COSINE、稀疏 IP
    5. 放货（insert）后回填单号（chunk_id）

设计:
    职责分离 —— 3 个辅助类各管一段，统一由 Import_milvus_Node.process 调度。
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.ducment_split import Chunk
from knowledge.processor.import_process.exceptions import (
    StateFieldError,
    ValidationError,
    MilvusError,
)


# ==================== 类1: 校验 + 过滤有效切片 ====================

class MilvusChunkValidator:
    """
    切片入库 - 校验与过滤

    职责:
        - 校验 state 中 chunks 字段合法性
        - 逐条判定切片是否可入库（有 content + 稠密向量 + 稀疏向量）
        - 无效切片打日志标记并剔除，仅保留有效切片
    """

    def __init__(self, node_name: str, logger):
        """
        :param node_name: 节点名称（用于异常标记）
        :param logger: 日志记录器
        """
        self.node_name = node_name
        self.logger = logger

    def validate_and_filter(self, state: ImportGraphState) -> List[Chunk]:
        """
        入口：取出 chunks → 过滤出有效切片 → 全无效则抛异常

        :param state: 图状态
        :return: 可入库的有效切片列表
        :raises ValidationError: 过滤后没有任何有效切片
        """
        chunks = self._get_chunks(state)
        valid_chunks = self._filter_valid_chunks(chunks)

        if not valid_chunks:
            raise ValidationError(
                node_name=self.node_name,
                message="过滤后无有效切片可入库（全部缺少 content 或向量）",
            )

        self.logger.info(f"校验完成，{len(valid_chunks)}/{len(chunks)} 条切片可入库")
        return valid_chunks

    def _get_chunks(self, state: ImportGraphState) -> List[Chunk]:
        """
        校验 chunks 字段存在且为非空列表

        :param state: 图状态
        :return: 切片列表
        :raises StateFieldError: chunks 缺失或不是 list
        :raises ValidationError: chunks 为空列表
        """
        chunks = state.get("chunks")
        if chunks is None or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.node_name, field_name="chunks", expected_type=list)
        if len(chunks) == 0:
            raise ValidationError(node_name=self.node_name, message="chunks 为空列表，无内容可入库")
        return chunks

    def _filter_valid_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        逐条判定切片有效性，无效的日志标记并剔除

        :param chunks: 全部切片
        :return: 有效切片列表
        """
        valid = []
        for index, chunk in enumerate(chunks):
            reason = self._classify_invalid(chunk)
            if reason:
                # 日志标记无效切片（有迹可循）
                self.logger.warning(f"切片[{index}] 标题='{chunk.title}' 不可入库，原因: {reason}，已跳过")
                continue
            valid.append(chunk)
        return valid

    def _classify_invalid(self, chunk: Chunk) -> str:
        """
        判定单条切片是否无效及原因

        :param chunk: 单个切片
        :return: 无效原因字符串；返回 "" 表示有效
        """
        if not chunk.content or not chunk.content.strip():
            return "缺少文档内容(content)"
        if not chunk.dense_vector:
            return "缺少稠密向量(dense_vector)"
        if not chunk.sparse_vector:
            return "缺少稀疏向量(sparse_vector)"
        return ""


# ==================== 类2: 连接 + 集合管理（drop/create/use）+ schema + 索引 ====================

class ChunkCollectionManager:
    """
    切片入库 - Milvus 集合管理

    职责:
        - 连接 Milvus 客户端
        - 按 delete_flag 与集合存在情况决定 drop/create/use
        - 构建 schema（货架）与两套索引（检索台）
    """

    def __init__(self, node_name: str, logger, config: ImportConfig):
        """
        :param node_name: 节点名称
        :param logger: 日志记录器
        :param config: 导入配置（提供集合名、向量维度）
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config
        self.collection_name = config.chunks_collection

    def prepare_collection(self, delete_flag: bool):
        """
        入口：连接客户端 → 确保集合就绪 → 返回客户端

        :param delete_flag: 是否删除并重建集合
        :return: 已就绪的 MilvusClient
        """
        client = self._connect_client()
        self._ensure_collection(client, delete_flag)
        return client

    def _connect_client(self):
        """
        连接 Milvus 客户端（延迟导入，避免顶层加载重量级依赖）

        :return: MilvusClient 实例
        :raises MilvusError: 客户端连接失败
        """
        try:
            from knowledge.utils.client.storage_clients import StorageClients
            client = StorageClients.get_milvus_client()
            self.logger.info("Milvus 客户端连接成功")
            return client
        except Exception as e:
            self.logger.error(f"Milvus 客户端连接失败: {e}")
            raise MilvusError(node_name=self.node_name, message="Milvus 客户端连接失败", cause=e)

    def _ensure_collection(self, client, delete_flag: bool) -> None:
        """
        按三种情况确保集合就绪

        情况A: 集合存在且 delete_flag=True → drop 后 create（拆旧重建）
        情况C: 集合存在且 delete_flag=False → 直接用（不改动）
        情况B: 集合不存在（或刚被 drop） → create（新建）

        :param client: MilvusClient
        :param delete_flag: 是否拆旧重建
        :return: None
        """
        exists = client.has_collection(self.collection_name)

        if exists and delete_flag:
            # 情况A：拆了重建
            self.logger.info(f"集合 [{self.collection_name}] 已存在且 delete_flag=True，拆旧重建")
            self._drop_collection(client)
            self._create_collection(client)
        elif exists and not delete_flag:
            # 情况C：直接用
            self.logger.info(f"集合 [{self.collection_name}] 已存在且 delete_flag=False，直接使用")
        else:
            # 情况B：空地新建
            self.logger.info(f"集合 [{self.collection_name}] 不存在，开始新建")
            self._create_collection(client)

    def _drop_collection(self, client) -> None:
        """
        删除已存在的集合

        :param client: MilvusClient
        :return: None
        """
        client.drop_collection(self.collection_name)
        self.logger.info(f"集合 [{self.collection_name}] 已删除")

    def _create_collection(self, client) -> None:
        """
        创建集合（schema + 索引一并建立）

        :param client: MilvusClient
        :return: None
        :raises MilvusError: 创建失败
        """
        try:
            schema = self._build_schema(client)
            index_params = self._build_index_params(client)
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            self.logger.info(f"集合 [{self.collection_name}] 创建成功，schema 与索引已建立")
        except Exception as e:
            self.logger.error(f"集合 [{self.collection_name}] 创建失败: {e}")
            raise MilvusError(node_name=self.node_name, message="集合创建失败", cause=e)

    def _build_schema(self, client):
        """
        构建集合 schema（货架规划）

        字段: chunk_id(自增主键) / dense_vector / sparse_vector /
              content / item_name / title / file_title / parent_title / part

        :param client: MilvusClient
        :return: Milvus schema 对象
        """
        schema = client.create_schema()
        # 仓库编号：自动生成唯一整数主键
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64,
                         is_primary=True, auto_id=True)
        # 向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
                         dim=self.config.embedding_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 文本元数据字段
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.VARCHAR, max_length=100)
        return schema

    def _build_index_params(self, client):
        """
        构建两套索引（检索台）

        - 稠密向量: AUTOINDEX + COSINE（语义检索）
        - 稀疏向量: SPARSE_INVERTED_INDEX + IP（关键词检索）

        :param client: MilvusClient
        :return: Milvus index_params 对象
        """
        index_params = client.prepare_index_params()
        # 检索台1：稠密向量语义检索（余弦）
        index_params.add_index(field_name="dense_vector", index_name="dense_vector_index",
                               index_type="AUTOINDEX", metric_type="COSINE")
        # 检索台2：稀疏向量关键词检索（内积）
        index_params.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        return index_params


# ==================== 类3: 组装数据 + 插入 + 回填 chunk_id ====================

class ChunkInserter:
    """
    切片入库 - 数据写入

    职责:
        - 将有效切片组装为 Milvus 插入数据
        - 执行 insert 拿到自增主键
        - 把主键回填到对应 Chunk.chunk_id
    """

    def __init__(self, node_name: str, logger, config: ImportConfig):
        """
        :param node_name: 节点名称
        :param logger: 日志记录器
        :param config: 导入配置（提供集合名）
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config
        self.collection_name = config.chunks_collection

    def insert_and_backfill(self, client, chunks: List[Chunk]) -> int:
        """
        入口：组装 → 插入 → 回填，返回插入条数

        :param client: MilvusClient
        :param chunks: 有效切片列表
        :return: 成功插入的条数
        """
        data = self._build_insert_data(chunks)
        ids = self._do_insert(client, data)
        self._backfill_chunk_ids(chunks, ids)
        self.logger.info(f"成功插入 {len(ids)} 条切片到集合 [{self.collection_name}]")
        return len(ids)

    def _build_insert_data(self, chunks: List[Chunk]) -> List[Dict]:
        """
        将切片列表映射为 Milvus 插入数据（不含自增主键 chunk_id）

        :param chunks: 有效切片列表
        :return: 字典列表，每个字典对应一条记录
        """
        data = []
        for chunk in chunks:
            data.append({
                "dense_vector": chunk.dense_vector,
                "sparse_vector": chunk.sparse_vector,
                "content": chunk.content,
                "item_name": chunk.item_name,
                "title": chunk.title,
                "file_title": chunk.file_title,
                "parent_title": chunk.parent_title,
                "part": chunk.part,
            })
        return data

    def _do_insert(self, client, data: List[Dict]) -> List[int]:
        """
        执行插入并返回自增主键列表

        :param client: MilvusClient
        :param data: 插入数据
        :return: 自增主键列表（与 data 同序）
        :raises MilvusError: 插入失败
        """
        try:
            result = client.insert(collection_name=self.collection_name, data=data)
            ids = result.get("ids", [])
            return ids
        except Exception as e:
            self.logger.error(f"Milvus 插入失败: {e}")
            raise MilvusError(node_name=self.node_name, message="Milvus 插入失败", cause=e)

    def _backfill_chunk_ids(self, chunks: List[Chunk], ids: List[int]) -> None:
        """
        将自增主键按顺序回填到对应切片的 chunk_id

        :param chunks: 有效切片列表（与 ids 同序）
        :param ids: 自增主键列表
        :return: None（原地修改 chunk）
        """
        for chunk, chunk_id in zip(chunks, ids):
            chunk.chunk_id = chunk_id


# ==================== 调度节点 ====================

class Import_milvus_Node(BaseNode):
    name = 'import_milvus'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点统一调度入口

        :param state: 图状态，需包含 chunks（已向量化）、delete_flag（是否重建集合）
        :return: 更新后的 state（有效 chunk 已回填 chunk_id）
        """
        # 1. 校验 + 过滤有效切片
        self.log_step("步骤1", "校验并过滤有效切片")
        validator = MilvusChunkValidator(self.name, self.logger)
        valid_chunks = validator.validate_and_filter(state)

        # 2. 准备集合（连接 + drop/create/use + schema + 索引）
        delete_flag = state.get("delete_flag", False)
        self.log_step("步骤2", f"准备集合（delete_flag={delete_flag}）")
        manager = ChunkCollectionManager(self.name, self.logger, self.config)
        client = manager.prepare_collection(delete_flag)

        # 3. 插入 + 回填 chunk_id
        self.log_step("步骤3", "插入切片并回填 chunk_id")
        inserter = ChunkInserter(self.name, self.logger, self.config)
        inserted = inserter.insert_and_backfill(client, valid_chunks)

        self.log_step("完成", f"成功入库 {inserted} 条切片")
        return state


# ==================== 测试 ====================

# 将项目根目录加入 sys.path（本文件在 .../shopkeeper_brain/knowledge/processor/import_process/nodes/）
# .parent x5 = shopkeeper_brain（knowledge 包的父目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / "knowledge" / ".env")


if __name__ == '__main__':
    setup_logging()

    from knowledge.processor.import_process.nodes.ducment_split import Document_Split_Node
    from knowledge.processor.import_process.nodes.bge_embedding import Bge_Embedding_Node

    # 1. 切分测试文档得到 chunks
    test_md_path = r"E:\temp\万用表RS-12的使用\auto\万用表RS-12的使用.md"
    try:
        md_content = Path(test_md_path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"读取测试文件失败: {e}")
        sys.exit(1)

    split_state = {
        "file_title": "万用表RS-12的使用",
        "md_content": md_content,
    }
    split_state = Document_Split_Node()(split_state)

    # 2. 手动回填 item_name（模拟 item_name_recognition 节点输出）
    for chunk in split_state["chunks"]:
        chunk.item_name = "RS-12 数字万用表"

    # 3. 向量化（bge_embedding 节点）
    embed_state = {
        "item_name": "RS-12 数字万用表",
        "chunks": split_state["chunks"],
    }
    embed_state = Bge_Embedding_Node()(embed_state)

    # 4. 构建本节点 state（delete_flag=True 拆旧重建，便于反复测试）
    state = {
        "delete_flag": True,
        "chunks": embed_state["chunks"],
    }

    # 5. 执行入库
    node = Import_milvus_Node()
    result = node(state)

    # 6. 验证结果
    print("\n" + "=" * 60)
    chunks = result["chunks"]
    stored = [c for c in chunks if c.chunk_id != -1]
    print(f"切片总数: {len(chunks)}")
    print(f"已回填 chunk_id 的切片数: {len(stored)}")
    if stored:
        print(f"首个 chunk_id: {stored[0].chunk_id}")
        print(f"末个 chunk_id: {stored[-1].chunk_id}")
    print("=" * 60)
