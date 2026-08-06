
import json
from json import JSONDecodeError
from typing import Dict, List, Tuple, Optional, Any
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE


class Item_Name_Recognition_Node(BaseNode):
    name = 'item_name_recognition'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建商品名识别上下文
        item_name_recognition_context = self._prepare_item_name_recognition_context(
            chunks, item_name_chunks_k, item_name_chunk_size
        )

        # 3. LLM商品名识别
        item_name = self._recognition_name(file_title, item_name_recognition_context)

        # 4. 向量化提取到商品名
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到milvus中
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector,
                            self.config.item_name_collection)

        # 6. 回填item_name信息
        self._fill_item_name(item_name, state, chunks)

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, List, int, int]:
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        item_name_chunks_k = self.config.item_name_chunk_k
        if not item_name_chunks_k or item_name_chunks_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        return file_title, chunks, item_name_chunks_k, item_name_chunk_size

    def _prepare_item_name_recognition_context(self, chunks, item_name_chunks_k, item_name_chunk_size) -> str:
        total = 0
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            # chunk 是 Chunk dataclass 对象，直接访问属性
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            context = f"【切片{index}】\n{chunk_content}"

            if total + len(context) > item_name_chunk_size:
                # 截断当前切片内容，尽量多装一些
                remaining = item_name_chunk_size - total
                if remaining > 0:
                    final_context.append(context[:remaining])
                break

            total += len(context)
            final_context.append(context)

        return "\n\n".join(final_context)

    def _recognition_name(self, file_title, item_name_recognition_context) -> str:
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入，避免顶层加载重量级依赖
            llm_client = AIClients.get_llm_openai(response_format=True)
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title, context=item_name_recognition_context
            )

            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            # 解析 JSON 响应
            item_name = self._parse_llm_response(llm_response.content)

            if not item_name or item_name == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            self.logger.info(f"LLM提取到商品名: {item_name}")
            return item_name
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _parse_llm_response(self, content: str) -> str:
        """解析LLM返回的JSON，提取item_name字段"""
        try:
            result = json.loads(content.strip())
            return result.get("item_name", "").strip()
        except JSONDecodeError as e:
            self.logger.warning(f"LLM响应JSON解析失败: {e}，原始内容: {content[:200]}")
            return ""

    def _embedding_item_name(self, item_name) -> Tuple[List[float], Dict[int, float]]:
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入
            bge_m3_client = AIClients.get_bge_m3_client()
            vector_result = bge_m3_client.encode_documents([item_name])

            dense_vector = vector_result['dense'][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))

            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None

    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return

        try:
            from knowledge.utils.client.storage_clients import StorageClients  # 延迟导入
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return

        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)

            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk.item_name = item_name  # Chunk dataclass 属性赋值

        state['item_name'] = item_name


from knowledge.processor.import_process.base import setup_logging

if __name__ == '__main__':
    setup_logging()

    # 1. 先用 Document_Split_Node 切分文档，得到 chunks
    from knowledge.processor.import_process.nodes.ducment_split import Document_Split_Node

    test_split_state = {
        "file_title": "万用表RS-12的使用",
        "md_content": "",
    }

    test_md_path = r"E:\temp\万用表RS-12的使用\auto\万用表RS-12的使用.md"
    try:
        from pathlib import Path
        test_split_state["md_content"] = Path(test_md_path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"读取测试文件失败: {e}")
        exit(1)

    split_node = Document_Split_Node()
    processed_state = split_node(test_split_state)

    # 2. 构建 Item_Name_Recognition_Node 的 state
    state = {
        "file_title": "万用表RS-12的使用",
        "chunks": processed_state["chunks"]
    }

    # 3. 实例化节点并调用
    node = Item_Name_Recognition_Node()
    result = node(state)

    # 4. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    if result.get('chunks'):
        print(f"首个chunk的item_name: {result['chunks'][0].item_name}")
