"""
RRF（Reciprocal Rank Fusion）结果融合节点

职责:
    1. 校验 state（embedding_chunks / hyde_embedding_chunks）
    2. 从两路检索结果中提取 entity（获取不到取空）
    3. 按 RRF 公式融合打分：score = Σ 第i路权重 / (k + 第i路排名)
    4. 融合分数降序排序，截取前 rrf_max_results 名
    5. 将排名前五的 entity 存入 state["rrf_chunks"]

原理:
    RRF 不依赖各路的原始相似度分数（不同路的分数量纲不同，不可直接比较），
    只依赖「排名」——排名越靠前贡献越大；同一切片在多路中出现，贡献相加。
    两路权重（需求指定）: HyDE 路 0.3，向量路 0.5（权重大的路话语权更大）。
"""
from typing import Any, Dict, List, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = "rrf_node"

    # 两路检索的 RRF 权重（需求指定）
    HYDE_WEIGHT = 0.3       # HyDE 路权重
    EMBEDDING_WEIGHT = 0.5  # 向量路权重

    def process(self, state: QueryGraphState) -> QueryGraphState | Dict[str, Any]:
        """
        RRF 融合主流程

        :param state: 查询图状态（需含 embedding_chunks / hyde_embedding_chunks）
        :return: 更新后的 state（新增 rrf_chunks：排名前五的 entity 列表）
        """
        # 1. 校验并获取两路检索结果
        embedding_chunks, hyde_chunks = self._validate_state(state)

        # 2. RRF 融合（提取 entity + 打分）
        self.log_step("步骤2", f"RRF 融合: 向量路 {len(embedding_chunks)} 条, HyDE 路 {len(hyde_chunks)} 条")
        fused = self._fuse(embedding_chunks, hyde_chunks)

        # 3. 排序截取前 N 名
        top_entities = self._sort_and_truncate(fused)

        # 4. 更新 state
        state["rrf_chunks"] = top_entities
        self.log_step("完成", f"RRF 融合完成，输出 {len(top_entities)} 条")
        return {"rrf_chunks": top_entities}

    def _validate_state(self, state: QueryGraphState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        校验并获取两路检索结果（缺失/空 → 空列表）

        :param state: 查询图状态
        :return: (embedding_chunks, hyde_embedding_chunks)
        :raises StateFieldError: 字段存在但不是列表
        """
        embedding_chunks = state.get("embedding_chunks", [])
        hyde_chunks = state.get("hyde_embedding_chunks", [])

        if not isinstance(embedding_chunks, list):
            raise StateFieldError(node_name=self.name, field_name="embedding_chunks", expected_type=list)
        if not isinstance(hyde_chunks, list):
            raise StateFieldError(node_name=self.name, field_name="hyde_embedding_chunks", expected_type=list)

        self.log_step("步骤1", f"参数校验通过: 向量路 {len(embedding_chunks)} 条, HyDE 路 {len(hyde_chunks)} 条")
        return embedding_chunks, hyde_chunks

    def _fuse(
        self,
        embedding_chunks: List[Dict[str, Any]],
        hyde_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        RRF 融合打分：score[key] = Σ 第i路权重 / (k + 第i路排名)

        同一 chunk_id 在多路中出现时，分数相加（RRF 的核心：多路互证加权）。

        :param embedding_chunks: 向量路命中列表（Milvus 已按分数降序，列表顺序即排名）
        :param hyde_chunks: HyDE 路命中列表（同上）
        :return: {chunk_id: {"entity": dict, "score": float}} 融合结果
        """
        k = self.config.rrf_k
        result: Dict[str, Dict[str, Any]] = {}

        # 两路检索：((命中列表, 权重), 路名)
        paths = [
            (embedding_chunks, self.EMBEDDING_WEIGHT, "向量路"),
            (hyde_chunks, self.HYDE_WEIGHT, "HyDE 路"),
        ]

        for chunks, weight, path_name in paths:
            for rank, hit in enumerate(chunks, 1):  # rank 从 1 开始（列表顺序即排名）
                chunk_id, entity = self._extract_entity(hit, rank)

                # RRF 公式：权重 / (k + 排名)
                contribution = weight / (k + rank)

                entry = result.setdefault(chunk_id, {"entity": entity, "score": 0.0})
                entry["score"] += contribution  # 同切片两路都出现 → 贡献相加

                self.logger.debug(
                    f"[{path_name}] chunk {chunk_id} 排名 {rank} 贡献 {contribution:.6f}，累计 {entry['score']:.6f}"
                )

        self.logger.info(f"RRF 融合完成，去重后共 {len(result)} 个切片")
        return result

    def _extract_entity(self, hit: Dict[str, Any], rank: int) -> Tuple[str, Dict[str, Any]]:
        """
        从单条命中提取 chunk_id 与 entity（获取不到取空）

        :param hit: 单条检索命中（{"chunk_id", "distance", "entity": {...}}）
        :param rank: 该命中的排名（用于无 chunk_id 时的兜底键）
        :return: (chunk_id, entity 字典)
        """
        # entity 获取不到 → 空字典兜底
        entity = hit.get("entity") if isinstance(hit, dict) else None
        entity = entity if isinstance(entity, dict) else {}

        # chunk_id 优先取 entity 内，其次取命中顶层
        chunk_id = entity.get("chunk_id") or (hit.get("chunk_id") if isinstance(hit, dict) else None)
        if not chunk_id:
            chunk_id = f"unknown_{rank}"  # 兜底键：保证参与融合，但不会跨路合并
        return chunk_id, entity

    def _sort_and_truncate(self, fused: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按融合分数降序排序，截取前 rrf_max_results 名的 entity

        :param fused: 融合结果 {chunk_id: {"entity", "score"}}
        :return: 排名前 N 的 entity 列表（按分数降序）
        """
        limit = self.config.rrf_max_results
        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        top = ranked[:limit]

        # 日志输出排名与分数（观测性）
        for i, item in enumerate(top, 1):
            title = item["entity"].get("title", "")
            self.logger.info(f"RRF 第 {i} 名: score={item['score']:.4f} | title={title}")

        return [item["entity"] for item in top]


# ================================================================== #
#                        测试入口                                     #
# ================================================================== #

if __name__ == '__main__':
    import json
    setup_logging()

    from knowledge.processor.query_process.config import get_config

    # 用需求中的真实样例结构构造两路检索结果
    embedding_chunks = [
        {"chunk_id": 468071782111254662, "distance": 0.746,
         "entity": {"chunk_id": 468071782111254662, "item_name": "RS PRO RS-12 数字万用表", "content": "电阻测量内容...", "title": "电阻测量"}},
        {"chunk_id": 468071782111254679, "distance": 0.709,
         "entity": {"chunk_id": 468071782111254679, "item_name": "RS PRO RS-12 数字万用表", "content": "", "title": ""}},
        {"chunk_id": 468071782111254661, "distance": 0.709,
         "entity": {"chunk_id": 468071782111254661, "item_name": "RS PRO RS-12 数字万用表", "content": "交流电压测量内容...", "title": "交流电压测量"}},
        {"chunk_id": 468071782111254678, "distance": 0.707,
         "entity": {"chunk_id": 468071782111254678, "item_name": "RS PRO RS-12 数字万用表", "content": "电流测量内容...", "title": ""}},
        {"chunk_id": 468071782111254668, "distance": 0.698,
         "entity": {"chunk_id": 468071782111254668, "item_name": "RS PRO RS-12 数字万用表", "content": "规格表内容...", "title": ""}},
    ]
    # HyDE 路：与向量路有 2 个重叠切片（661 和 662），验证「多路分数相加」
    hyde_chunks = [
        {"chunk_id": 468071782111254661, "distance": 0.720,
         "entity": {"chunk_id": 468071782111254661, "item_name": "RS PRO RS-12 数字万用表", "content": "交流电压测量内容...", "title": "交流电压测量"}},
        {"chunk_id": 468071782111254662, "distance": 0.710,
         "entity": {"chunk_id": 468071782111254662, "item_name": "RS PRO RS-12 数字万用表", "content": "电阻测量内容...", "title": "电阻测量"}},
        {"chunk_id": 468071782111254600, "distance": 0.650,
         "entity": {"chunk_id": 468071782111254600, "item_name": "RS PRO RS-12 数字万用表", "content": "HyDE 独有切片...", "title": ""}},
    ]

    test_state = {
        "embedding_chunks": embedding_chunks,
        "hyde_embedding_chunks": hyde_chunks,
    }

    rrf_node = RrfNode()
    result = rrf_node(test_state)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"RRF 融合结果: {len(result.get('rrf_chunks', []))} 条（config 上限 {get_config().rrf_max_results}）")
    for i, entity in enumerate(result.get("rrf_chunks", []), 1):
        print(f"  {i}. chunk_id={entity.get('chunk_id')} | title={entity.get('title')}")
    print("=" * 60)



"""
{
  "rrf_chunks": [
    {
      "chunk_id": 468071782111254662,
      "item_name": "RS PRO RS-12 \u6570\u5b57\u4e07\u7528\u8868",
      "content": "\u7535\u963b\u6d4b\u91cf\u5185\u5bb9...",
      "title": "\u7535\u963b\u6d4b\u91cf"
    },
    {
      "chunk_id": 468071782111254661,
      "item_name": "RS PRO RS-12 \u6570\u5b57\u4e07\u7528\u8868",
      "content": "\u4ea4\u6d41\u7535\u538b\u6d4b\u91cf\u5185\u5bb9...",
      "title": "\u4ea4\u6d41\u7535\u538b\u6d4b\u91cf"
    },
    {
      "chunk_id": 468071782111254679,
      "item_name": "RS PRO RS-12 \u6570\u5b57\u4e07\u7528\u8868",
      "content": "",
      "title": ""
    },
    {
      "chunk_id": 468071782111254678,
      "item_name": "RS PRO RS-12 \u6570\u5b57\u4e07\u7528\u8868",
      "content": "\u7535\u6d41\u6d4b\u91cf\u5185\u5bb9...",
      "title": ""
    },
    {
      "chunk_id": 468071782111254668,
      "item_name": "RS PRO RS-12 \u6570\u5b57\u4e07\u7528\u8868",
      "content": "\u89c4\u683c\u8868\u5185\u5bb9...",
      "title": ""
    }
  ]
}

"""
