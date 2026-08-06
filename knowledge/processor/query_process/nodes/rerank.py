"""
Rerank 重排序节点

职责:
    1. 校验 state（rrf_chunks / web_search_docs / rewritten_query）
    2. 将本地切片与网页结果统一为通用格式（chunk_id/source/content/title/url）
    3. 用 BGE-Reranker 对每个 doc 与查询做相关性打分（normalize 后 0~1）
    4. 分数与 doc 配对 [doc, score]，按分数降序排序
    5. 动态断崖检测：找到第一处分数断崖，返回断崖前内容
       （至少 config.rerank_min_top_k，至多 config.rerank_max_top_k）
    6. 最终 doc 列表存入 state["reranked_docs"]

设计:
    - 防御性编程：重写问题缺失 → 取原始问题；字段取不到 → 空值
    - 降级策略：重排器打分失败 → 记 error 日志 + 该 doc 分数置 0
      （全 0 时稳定排序保持原始顺序：本地在前、网页在后）
    - 日志可观测性：每个步骤都有日志
"""
from typing import Any, Dict, List, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class RerankNode(BaseNode):
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState | Dict[str, Any]:
        """
        重排序主流程

        :param state: 查询图状态（需含 rrf_chunks / web_search_docs / rewritten_query）
        :return: 更新后的 state（新增 reranked_docs：重排后的 doc 列表）
        """
        # 1. 校验并获取输入
        query, rrf_chunks, web_docs = self._validate_state(state)

        # 2. 统一为通用 doc 格式
        self.log_step("步骤2", f"统一文档格式: 本地 {len(rrf_chunks)} 条, 网页 {len(web_docs)} 条")
        docs = self._unify_docs(rrf_chunks, web_docs)

        if not docs:
            self.logger.warning("无任何可重排的文档（本地与网页均为空）")
            state["reranked_docs"] = []
            return {"reranked_docs": []}

        # 3. BGE-Reranker 打分，配对 [doc, score]
        self.log_step("步骤3", f"BGE-Reranker 重排打分（{len(docs)} 条）")
        scored_docs = self._rerank(query, docs)

        # 4. 排序 + 动态断崖检测
        top_docs = self._sort_and_cliff(scored_docs)

        # 5. 更新 state
        state["reranked_docs"] = top_docs
        self.log_step("完成", f"重排完成，输出 {len(top_docs)} 条")
        return {"reranked_docs": top_docs}

    def _validate_state(
        self,
        state: QueryGraphState,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        校验并获取输入（防御性取值）

        查询文本: rewritten_query 缺失 → 取 original_query（降级默认值）

        :param state: 查询图状态
        :return: (query, rrf_chunks, web_search_docs)
        :raises StateFieldError: 查询文本两者皆无，或字段存在但不是列表
        """
        # 查询文本：优先重写问题，缺失则用原始问题
        query = state.get("rewritten_query") or state.get("original_query")
        if not query:
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query/original_query",
                expected_type=str,
            )

        rrf_chunks = state.get("rrf_chunks", [])
        web_docs = state.get("web_search_docs", [])

        if not isinstance(rrf_chunks, list):
            raise StateFieldError(node_name=self.name, field_name="rrf_chunks", expected_type=list)
        if not isinstance(web_docs, list):
            raise StateFieldError(node_name=self.name, field_name="web_search_docs", expected_type=list)

        self.log_step("步骤1", f"参数校验通过: 本地 {len(rrf_chunks)} 条, 网页 {len(web_docs)} 条")
        return query, rrf_chunks, web_docs

    def _unify_docs(
        self,
        rrf_chunks: List[Dict[str, Any]],
        web_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        把本地切片与网页结果统一为通用 doc 格式

        通用格式:
            {"chunk_id": str, "source": "local|web", "content": str, "title": str, "url": str}

        字段规则:
            - chunk_id: 仅本地切片有，网页取空
            - url: 仅网页有，本地取空
            - content: 本地取 entity.content，网页取 snippet；取不到 → 空串

        :param rrf_chunks: 本地切片（RRF 输出，entity 字典列表）
        :param web_docs: 网页搜索结果（snippet/title/url 字典列表）
        :return: 统一格式的 doc 列表（本地在前、网页在后）
        """
        docs: List[Dict[str, Any]] = []

        # 1. 本地切片
        for chunk in rrf_chunks:
            chunk = chunk if isinstance(chunk, dict) else {}
            docs.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "source": "local",
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "url": "",  # 本地无 url
            })

        # 2. 网页结果
        for web in web_docs:
            web = web if isinstance(web, dict) else {}
            docs.append({
                "chunk_id": "",  # 网页无 chunk_id
                "source": "web",
                "content": web.get("snippet", ""),  # 网页的内容在 snippet 字段
                "title": web.get("title", ""),
                "url": web.get("url", ""),
            })

        return docs

    def _rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        用 BGE-Reranker 对每个 doc 与查询打分，配对为 [doc, score]

        降级策略: 重排器获取/打分失败 → 记 error 日志，所有 doc 分数置 0
        （分数全 0 时，下游稳定排序保持原始顺序：本地在前、网页在后）

        :param query: 查询文本（重写问题，缺失时已降级为原始问题）
        :param docs: 统一格式的 doc 列表
        :return: [(doc, score), ...] 与 docs 同序
        """
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入，避免顶层加载重量级依赖

            reranker = AIClients.get_bge_m3_rerank_client()

            # FlagReranker: 输入 [[query, doc], ...]，normalize=True 输出 0~1
            pairs = [[query, doc["content"]] for doc in docs]
            scores = reranker.compute_score(pairs, normalize=True)

            # 防御：返回数量与 docs 不一致时，缺失部分补 0
            if not isinstance(scores, list) or len(scores) != len(docs):
                self.logger.warning(f"重排器返回分数数量异常（{len(scores) if scores else 0}/{len(docs)}），缺失补 0")
                score_list = list(scores) if isinstance(scores, list) else []
                score_list += [0.0] * (len(docs) - len(score_list))
                scores = score_list

            self.logger.info(f"重排打分完成: {[round(s, 4) for s in scores]}")
            return list(zip(docs, [float(s) for s in scores]))
        except Exception as e:
            # 降级：打分失败 → 全部 0 分（保持原始顺序）
            self.logger.error(f"BGE-Reranker 打分失败（降级为 0 分）: {e}")
            return [(doc, 0.0) for doc in docs]

    def _sort_and_cliff(self, scored_docs: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        """
        按分数降序排序 + 动态断崖检测 + 数量约束

        断崖规则:
            - 相邻两名分数差 > config.rerank_gap_abs → 视为断崖，取断崖前的内容
            - 无断崖 → 返回 min(config.rerank_max_top_k, 总数)
            - 数量约束: 至少 config.rerank_min_top_k，至多 config.rerank_max_top_k

        :param scored_docs: [(doc, score), ...]
        :return: 重排后的 doc 列表（分数降序）
        """
        max_top_k = self.config.rerank_max_top_k
        min_top_k = self.config.rerank_min_top_k
        gap_abs = self.config.rerank_gap_abs

        # 1. 按分数降序（稳定排序：同分保持原始顺序）
        ranked = sorted(scored_docs, key=lambda item: item[1], reverse=True)
        self.logger.info(f"重排打分完成: {[round(s, 4) for s in [item[1] for item in ranked]]}")

        # 2. 动态断崖检测：找到第一处「相邻分数差 > gap」的位置
        cut = len(ranked)  # 默认全量（无断崖）
        for i in range(1, len(ranked)):
            drop = ranked[i - 1][1] - ranked[i][1]
            if drop > gap_abs:
                cut = i  # 断崖在第 i 名之后 → 返回前 i 个
                self.logger.info(
                    f"检测到断崖: 第 {i} 名({ranked[i-1][1]:.4f}) → 第 {i+1} 名({ranked[i][1]:.4f})，差 {drop:.4f}"
                )
                break
        if cut == len(ranked):
            self.logger.info("未检测到断崖，按上限返回")

        # 3. 数量约束：至少 min_top_k，至多 max_top_k，且不超过总数
        cut = max(cut, min_top_k)
        cut = min(cut, max_top_k, len(ranked))

        top_docs = [item[0] for item in ranked[:cut]]
        self.logger.info(f"重排输出 {len(top_docs)} 条（断崖切点 {cut}，min={min_top_k}，max={max_top_k}）")
        return top_docs


# ================================================================== #
#                        测试入口                                     #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    # 本地切片（RRF 输出，entity 字典格式）
    rrf_chunks = [
        {"chunk_id": 468071782111254662, "item_name": "RS PRO RS-12 数字万用表",
         "content": "将功能转盘置于最高电阻Ω位置，黑色表笔插入COM端口，红色表笔插入Ω端口...", "title": "电阻测量"},
        {"chunk_id": 468071782111254661, "item_name": "RS PRO RS-12 数字万用表",
         "content": "将功能转盘置于V AC的位置，黑色表笔插入COM端口，红色表笔插入V端口...", "title": "交流电压测量"},
        {"chunk_id": 468071782111254679, "item_name": "RS PRO RS-12 数字万用表",
         "content": "", "title": ""},
    ]
    # 网页搜索结果
    web_docs = [
        {"snippet": "万用表测电阻步骤详解：先将表笔短路调零，选择合适量程...",
         "title": "万用表电阻测量教程",
         "url": "https://example.com/meter-resistance"},
        {"snippet": "深圳市东微智能科技股份有限公司 100-499人 计算机软件...",
         "title": "东微智能公司简介",
         "url": "https://www.zhipin.com/companys/1b76db03938f8a431nZ_0ti6Eg~~.html"},
    ]

    test_state = {
        "original_query": "万用表如何测量电阻？",
        "rewritten_query": "RS PRO RS-12 数字万用表如何测量电阻？",
        "rrf_chunks": rrf_chunks,
        "web_search_docs": web_docs,
    }

    rerank_node = RerankNode()
    result = rerank_node(test_state)

    print("\n" + "=" * 60)
    print(f"重排结果: {len(result.get('reranked_docs', []))} 条")
    for i, doc in enumerate(result.get("reranked_docs", []), 1):
        print(f"  {i}. [{doc.get('source')}] {doc.get('title')} | {doc.get('content', '')[:30]}...")
    print("=" * 60)


"""
2026-08-05 11:18:56 - query.rerank_node - INFO - --- rerank_node 开始 ---
2026-08-05 11:18:56 - query.rerank_node - INFO - [步骤1] 参数校验通过: 本地 3 条, 网页 2 条
2026-08-05 11:18:56 - query.rerank_node - INFO - [步骤2] 统一文档格式: 本地 3 条, 网页 2 条
2026-08-05 11:18:56 - query.rerank_node - INFO - [步骤3] BGE-Reranker 重排打分（5 条）
2026-08-05 11:19:09 - knowledge.utils.client.base - INFO - bge_m3_rerank客户端初始化成功
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.

============================================================
重排结果: 2 条
  1. [web] 万用表电阻测量教程 | 万用表测电阻步骤详解：先将表笔短路调零，选择合适量程......
  2. [local] 电阻测量 | 将功能转盘置于最高电阻Ω位置，黑色表笔插入COM端口，红色表...
============================================================
2026-08-05 11:19:12 - query.rerank_node - INFO - 重排打分完成: [0.9869, 0.4821, 0.0035, 0.9886, 0.0004]
2026-08-05 11:19:12 - query.rerank_node - INFO - 重排打分完成: [0.9886, 0.9869, 0.4821, 0.0035, 0.0004]
2026-08-05 11:19:12 - query.rerank_node - INFO - 检测到断崖: 第 2 名(0.9869) → 第 3 名(0.4821)，差 0.5049
2026-08-05 11:19:12 - query.rerank_node - INFO - 重排输出 2 条（断崖切点 2，min=2，max=5）
2026-08-05 11:19:12 - query.rerank_node - INFO - [完成] 重排完成，输出 2 条
2026-08-05 11:19:12 - query.rerank_node - INFO - --- rerank_node 完成 ---

"""
