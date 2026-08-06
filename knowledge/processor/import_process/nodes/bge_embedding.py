"""
文档切片向量化节点

职责:
    将上游节点传来的切片(Chunk)内容向量化，生成稠密+稀疏向量并写回 chunk，
    供下游 import_milvus 节点入库。

处理流程:
    1. 校验 chunks
    2. 切片内容 + 商品名 拼接成向量化文本
    3. 分批调用 BGE-M3 生成 dense + sparse 向量（批大小来自 config）
    4. 向量写回 chunk

降级策略:
    单批失败 → 重试 1 次 → 仍失败则跳过该批(记日志)，继续其它批次；
    仅当全部批次失败时才抛出 EmbeddingError。
"""
import time
from typing import List, Dict, Tuple, Optional

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.ducment_split import Chunk
from knowledge.processor.import_process.exceptions import (
    StateFieldError,
    ValidationError,
    EmbeddingError,
)


class Bge_Embedding_Node(BaseNode):
    name = 'bge_embedding'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点统一调度入口

        :param state: 图状态，需包含 chunks(切片列表)、item_name(商品名，可选)
        :return: 更新后的 state（每个 chunk 已回填 dense_vector / sparse_vector）
        """
        # 1. 校验切片
        self.log_step("步骤1", "校验切片")
        chunks = self._validate_chunks(state)
        default_item_name = state.get("item_name", "")

        # 2. 获取向量化模型客户端
        self.log_step("步骤2", "获取 BGE-M3 模型客户端")
        bge_model = self._get_embedding_model()

        # 3. 分批向量化 + 写回（含降级）
        batch_size = self.config.embedding_batch_size
        self.log_step("步骤3", f"分批向量化，批大小={batch_size}，共 {len(chunks)} 条切片")
        batches = self._split_batches(chunks, batch_size)
        success_count, failed_count = self._embed_all_batches(bge_model, batches, default_item_name)

        # 4. 汇总 + 兜底
        self.log_step("步骤4", "汇总向量化结果")
        self._finalize(success_count, failed_count)

        return state

    def _validate_chunks(self, state: ImportGraphState) -> List[Chunk]:
        """
        校验 state 中的切片是否合法

        :param state: 图状态
        :return: 合法的切片列表
        :raises StateFieldError: chunks 字段缺失或不是列表
        :raises ValidationError: chunks 为空列表
        """
        chunks = state.get("chunks")

        if chunks is None or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        if len(chunks) == 0:
            raise ValidationError(node_name=self.name, message="chunks 为空列表，无可向量化内容")

        self.logger.info(f"校验通过，共 {len(chunks)} 条切片待向量化")
        return chunks

    def _get_embedding_model(self):
        """
        获取 BGE-M3 向量化模型客户端（延迟导入，避免启动即加载 torch）

        :return: BGEM3EmbeddingFunction 实例
        :raises EmbeddingError: 模型客户端获取失败
        """
        try:
            from knowledge.utils.client.ai_clients import AIClients
            model = AIClients.get_bge_m3_client()
            self.logger.info("BGE-M3 模型客户端获取成功")
            return model
        except Exception as e:
            self.logger.error(f"BGE-M3 模型客户端获取失败: {e}")
            raise EmbeddingError(node_name=self.name, message="BGE-M3 模型客户端获取失败", cause=e)

    def _build_embedding_text(self, chunk: Chunk, default_item_name: str) -> str:
        """
        拼接单条切片的向量化文本（带商品名标签）

        格式: 商品名称：{item_name}\n\n{content}
        item_name 优先取 chunk.item_name，为空则回退到 default_item_name，
        两者都为空则仅使用 content。

        :param chunk: 单个切片
        :param default_item_name: 默认商品名（来自 state，作兜底）
        :return: 拼接后的向量化文本
        """
        item_name = chunk.item_name or default_item_name
        content = chunk.content or ""
        if item_name:
            return f"商品名称：{item_name}\n\n{content}"
        return content

    def _split_batches(self, chunks: List[Chunk], batch_size: int) -> List[List[Chunk]]:
        """
        将切片列表按批大小切分为若干批次

        :param chunks: 全部切片
        :param batch_size: 每批最大条数
        :return: 批次列表，每个元素是一批切片
        """
        if batch_size <= 0:
            batch_size = 1  # 防御：非法批大小退化为逐条处理
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        self.logger.info(f"共切分为 {len(batches)} 个批次")
        return batches

    def _embed_batch(self, bge_model, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        对一批文本调用模型生成混合向量（单次调用，不含重试）

        :param bge_model: BGE-M3 模型客户端
        :param texts: 本批待向量化的文本列表
        :return: (稠密向量列表, 稀疏向量列表)，与 texts 一一对应
        :raises Exception: 模型调用失败时向上抛出
        """
        from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

        # is_query=False → 走 encode_documents（入库场景）
        result = generate_bge_m3_hybrid_vectors(bge_model, texts, is_query=False)
        return result["dense"], result["sparse"]

    def _embed_batch_with_retry(self, bge_model, texts: List[str], batch_idx: int
                                ) -> Optional[Tuple[List[List[float]], List[Dict[int, float]]]]:
        """
        带重试的单批向量化（降级核心）

        首次失败后重试 1 次；仍失败则记 error 日志并返回 None，由上层跳过该批。

        :param bge_model: BGE-M3 模型客户端
        :param texts: 本批文本
        :param batch_idx: 批次序号（用于日志定位）
        :return: 成功返回 (稠密向量列表, 稀疏向量列表)；彻底失败返回 None
        """
        max_attempts = 2  # 首次 + 重试 1 次
        for attempt in range(1, max_attempts + 1):
            try:
                return self._embed_batch(bge_model, texts)
            except Exception as e:
                self.logger.warning(f"批次 {batch_idx} 第 {attempt} 次向量化失败: {e}")
                if attempt == max_attempts:
                    self.logger.error(f"批次 {batch_idx} 重试后仍失败，跳过该批（{len(texts)} 条）")
                    return None
                time.sleep(1)  # 重试前短暂等待
        return None

    def _write_vectors_to_chunks(self, batch_chunks: List[Chunk],
                                 dense_vectors: List[List[float]],
                                 sparse_vectors: List[Dict[int, float]]) -> None:
        """
        将向量逐个写回对应切片的 dense_vector / sparse_vector 字段

        :param batch_chunks: 本批切片
        :param dense_vectors: 本批稠密向量
        :param sparse_vectors: 本批稀疏向量
        :return: None（原地修改 chunk）
        """
        for chunk, dense, sparse in zip(batch_chunks, dense_vectors, sparse_vectors):
            chunk.dense_vector = dense
            chunk.sparse_vector = sparse

    def _embed_all_batches(self, bge_model, batches: List[List[Chunk]], default_item_name: str
                           ) -> Tuple[int, int]:
        """
        遍历所有批次执行向量化并写回（含降级）

        :param bge_model: BGE-M3 模型客户端
        :param batches: 批次列表
        :param default_item_name: 默认商品名（拼接兜底）
        :return: (成功向量化的切片数, 失败的切片数)
        """
        success_count = 0
        failed_count = 0

        for idx, batch in enumerate(batches, 1):
            # 构建本批向量化文本
            texts = [self._build_embedding_text(chunk, default_item_name) for chunk in batch]

            # 带重试向量化
            vectors = self._embed_batch_with_retry(bge_model, texts, idx)

            if vectors is None:
                # 降级：跳过该批
                failed_count += len(batch)
                continue

            dense_vectors, sparse_vectors = vectors
            # 写回 chunk
            self._write_vectors_to_chunks(batch, dense_vectors, sparse_vectors)
            success_count += len(batch)
            self.logger.info(f"批次 {idx}/{len(batches)} 完成，{len(batch)} 条切片已向量化")

        return success_count, failed_count

    def _finalize(self, success_count: int, failed_count: int) -> None:
        """
        汇总向量化结果并做兜底处理

        :param success_count: 成功条数
        :param failed_count: 失败条数
        :return: None
        :raises EmbeddingError: 全部失败时抛出，让上层感知
        """
        self.logger.info(f"向量化汇总：成功 {success_count} 条，失败 {failed_count} 条")

        if success_count == 0:
            raise EmbeddingError(node_name=self.name, message="所有批次向量化均失败，无可用向量")

        if failed_count > 0:
            self.logger.warning(f"有 {failed_count} 条切片未成功向量化，下游应跳过空向量切片")


# ==================== 测试 ====================
import sys
from pathlib import Path

# 将项目根目录加入 sys.path（本文件在 .../shopkeeper_brain/knowledge/processor/import_process/nodes/）
# .parent x5 = shopkeeper_brain（knowledge 包的父目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / "knowledge" / ".env")


if __name__ == '__main__':
    setup_logging()

    from knowledge.processor.import_process.nodes.ducment_split import Document_Split_Node

    # 1. 先用 Document_Split_Node 切分测试文档得到 chunks
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

    # 2. 手动回填 item_name（模拟 item_name_recognition 节点的输出）
    for chunk in split_state["chunks"]:
        chunk.item_name = "RS-12 数字万用表"

    # 3. 构建本节点 state
    state = {
        "item_name": "RS-12 数字万用表",
        "chunks": split_state["chunks"],
    }

    # 4. 执行向量化
    node = Bge_Embedding_Node()
    result = node(state)

    # 5. 验证结果
    print("\n" + "=" * 60)
    chunks = result["chunks"]
    print(f"切片总数: {len(chunks)}")
    ok = sum(1 for c in chunks if c.dense_vector and c.sparse_vector)
    print(f"已成功向量化的切片数: {ok}")
    if chunks:
        first = chunks[0]
        print(f"首个切片 dense_vector 维度: {len(first.dense_vector)}")
        print(f"首个切片 sparse_vector 非零项数: {len(first.sparse_vector)}")
    print("=" * 60)
