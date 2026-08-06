"""
HyDE 假设文档检索节点

职责:
    1. 校验 state（rewritten_query / item_names）
    2. 【HyDE 核心】LLM 根据重写问题生成一段「假设性技术文档」
    3. BGE-M3 将生成内容向量化（文档侧 is_query=False）
    4. Milvus 商品名过滤 + 混合检索（稠密余弦 + 稀疏内积）
    5. 整理结果写入 state["hyde_embedding_chunks"]

原理:
    HyDE（Hypothetical Document Embeddings）——先用 LLM 根据问题写一段
    「如果真有文档回答这个问题，它大概长什么样」，再用这段假设文档去向量检索。
    相比直接检索问题，假设文档包含更多关键词和上下文，能提高召回质量。
"""
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import (
    EmbeddingError,
    MilvusError,
    StateFieldError,
)
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE


class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState | Dict[str, Any]:
        """
        HyDE 检索主流程

        :param state: 查询图状态（需含 rewritten_query / item_names）
        :return: 更新后的 state（新增 hyde_embedding_chunks）
        """
        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. LLM 生成假设性技术文档（HyDE 核心环节）
        self.log_step("步骤2", "LLM 生成假设性技术文档")
        hyde_content = self._generate_hypothetical_doc(rewritten_query, item_names)

        # 3. 将生成内容向量化（文档侧）
        self.log_step("步骤3", "BGE-M3 向量化假设文档")
        vector = self._embed_content(hyde_content)

        # 4. Milvus 混合检索（按 item_names 过滤）
        self.log_step("步骤4", "Milvus 混合检索（item_names 过滤）")
        search_results = self._hybrid_search(vector, item_names)

        # 5. 更新 state
        state["hyde_embedding_chunks"] = search_results
        self.log_step("完成", f"HyDE 检索到 {len(search_results)} 条切片")
        return {"hyde_embedding_chunks": search_results}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """
        校验输入参数

        :param state: 查询流程状态字典
        :return: (rewritten_query, item_names)
        :raises StateFieldError: 必需字段缺失
        """
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query:
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        if not item_names:
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)
        self.log_step("步骤1", "参数校验成功")
        return rewritten_query, item_names

    def _generate_hypothetical_doc(self, rewritten_query: str, item_names: List[str]) -> str:
        """
        LLM 根据问题生成一段假设性技术文档（HyDE 核心环节）

        降级策略：LLM 调用失败/返回空 → 直接用重写问题兜底（退化为普通检索，不中断流程）

        :param rewritten_query: 重写后的查询问题
        :param item_names: 商品名列表（注入提示词帮助 LLM 聚焦）
        :return: 生成的假设文档文本
        """
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入

            llm_client = AIClients.get_llm_openai(False)  # 自由文本生成，不用 JSON 模式
            if not llm_client:
                self.logger.warning("LLM 客户端获取失败，用重写问题兜底")
                return rewritten_query

            user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
                item_names="、".join(item_names),
                rewritten_query=rewritten_query,
            )
            system_prompt = f"您是一位{item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
            response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])


            content = (response.content or "").strip()
            if not content:
                self.logger.warning("LLM 返回空内容，用重写问题兜底")
                return rewritten_query

            self.logger.info(f"HyDE 假设文档生成成功---{content[:80]}---（{len(content)} 字）")
            return content
        except Exception as e:
            self.logger.error(f"HyDE 假设文档生成失败（用重写问题兜底）: {e}")
            return rewritten_query

    def _embed_content(self, content: str) -> Dict[str, Any]:
        """
        用 BGE-M3 将生成内容向量化（稠密 + 稀疏）

        设计说明: HyDE 生成的是「假设性文档」，按 HyDE 原论文思路，
        用文档编码器(encode_documents, is_query=False)编码，
        使其与库中已用文档编码入库的切片处于同一语义空间。

        :param content: 假设文档文本
        :return: {"dense": [...], "sparse": {...}} 混合向量
        :raises EmbeddingError: 模型获取或向量化失败
        """
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入
            from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

            bge_m3_client = AIClients.get_bge_m3_client()
            return generate_bge_m3_hybrid_vectors(bge_m3_client, [content], is_query=False)
        except Exception as e:
            self.logger.error(f"HyDE 内容向量化失败: {e}")
            raise EmbeddingError(node_name=self.name, message=f"HyDE 内容向量化失败: {str(e)}", cause=e)

    def _hybrid_search(self, vector: Dict[str, Any], item_names: List[str]) -> List[Dict[str, Any]]:
        """
        在 Milvus 切片集合上执行混合检索（按 item_names 过滤）

        与 vector_search 的检索逻辑一致，区别在于检索数量用 hyde_search_limit。

        :param vector: 查询向量 {"dense", "sparse"}
        :param item_names: 商品名列表（过滤条件 expr）
        :return: 检索命中的切片列表（含 entity 与 distance）
        :raises MilvusError: 客户端获取或检索失败
        """
        try:
            from knowledge.utils.client.storage_clients import StorageClients  # 延迟导入
            from knowledge.utils.milvus_util import (
                create_hybrid_search_requests,
                execute_hybrid_search_query,
                item_names_filter,
            )

            milvus_client = StorageClients.get_milvus_client()

            # 1. 构建过滤表达式（item_names in [...]）
            filter_expr, filter_expr_params = item_names_filter(item_names)

            # 2. 创建混合检索请求（稠密 + 稀疏）
            search_requests = create_hybrid_search_requests(
                dense_vector=vector["dense"][0],
                sparse_vector=vector["sparse"][0],
                expr=filter_expr,
                expr_params=filter_expr_params
            )

            # 3. 执行混合检索（内部已用 WeightedRanker 融合）
            res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=search_requests,
                output_fields=["chunk_id", "item_name", "content", "title"]
            )

            # 4. 单查询 → 取第一组命中；无命中返回空列表
            hits = res[0] if res else []
            self.logger.info(f"HyDE 混合检索完成，命中 {len(hits)} 条切片")
            return hits
        except Exception as e:
            self.logger.error(f"Milvus 混合检索失败: {e}")
            raise MilvusError(node_name=self.name, message=f"Milvus 混合检索失败: {str(e)}", cause=e)


# ================================================================== #
#                        测试入口                                     #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    test_state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }

    hyde_search = HyDeSearchNode()
    result = hyde_search(test_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"HyDE 检索到 {len(chunks)} 条切片")
    for r in chunks:
        entity = r.get("entity", {})
        print(f"- [{entity.get('chunk_id')}] {entity.get('item_name')} | 得分 {r.get('distance', 0):.4f}")
        print(f"  {entity.get('content', '')[:80]}...")
