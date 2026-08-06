"""
重写问题进行向量检索节点

职责:
    1. 校验 state（rewritten_query / item_names）
    2. BGE-M3 将重写问题向量化（查询侧 is_query=True，存锁查钥）
    3. Milvus 商品名过滤 + 混合检索（稠密余弦 + 稀疏内积）
    4. 整理结果写入 state["embedding_chunks"]
"""
from typing import Any, Dict, List, Tuple

from modelscope.fileio.format.jsonplus import json_dumps

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import (
    EmbeddingError,
    MilvusError,
    StateFieldError,
)
from knowledge.processor.query_process.state import QueryGraphState


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState|Dict[str, Any]:
        """
        向量检索主流程

        :param state: 查询图状态（需含 rewritten_query / item_names）
        :return: 更新后的 state（新增 embedding_chunks）
        """
        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. 将重写问题向量化
        self.log_step("步骤2", "BGE-M3 向量化重写问题")
        vector = self._embed_query(rewritten_query)

        # 3. Milvus 混合检索（按 item_names 过滤）
        self.log_step("步骤3", "Milvus 混合检索（item_names 过滤）")
        search_results = self._hybrid_search(vector, item_names)

        # 4. 更新 state
        state["embedding_chunks"] = search_results
        self.log_step("完成", f"向量检索到 {len(search_results)} 条切片")
        return {"embedding_chunks": search_results}

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
        return rewritten_query, item_names

    def _embed_query(self, rewritten_query: str) -> Dict[str, Any]:
        """
        用 BGE-M3 将重写问题向量化（稠密 + 稀疏）

        :param rewritten_query: 重写后的查询问题
        :return: {"dense": [...], "sparse": {...}} 混合向量
        :raises EmbeddingError: 模型获取或向量化失败
        """
        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入，避免顶层加载重量级依赖
            from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

            bge_m3_client = AIClients.get_bge_m3_client()
            # 注意: 需传模型 + 文档列表（列表包裹），查询侧 is_query=True
            return generate_bge_m3_hybrid_vectors(bge_m3_client, [rewritten_query], is_query=True)
        except Exception as e:
            self.logger.error(f"问题向量化失败: {e}")
            raise EmbeddingError(node_name=self.name, message=f"问题向量化失败: {str(e)}", cause=e)

    def _hybrid_search(self, vector: Dict[str, Any], item_names: List[str]) -> List[Dict[str, Any]]:
        """
        在 Milvus 切片集合上执行混合检索（按 item_names 过滤）

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
                expr_params=filter_expr_params,
                limit=self.config.embedding_search_limit,
            )

            # 3. 执行混合检索（内部已用 WeightedRanker 融合）
            res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=search_requests,
                output_fields=["chunk_id", "item_name", "content", "title"],
            )

            # 4. 单查询 → 取第一组命中；无命中返回空列表
            hits = res[0] if res else []
            self.logger.info(f"混合检索完成，命中 {len(hits)} 条切片")
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

    vector_search = VectorSearchNode()
    result = vector_search(test_state)

    chunks = result.get("embedding_chunks", [])
    print(f"检索到 {len(chunks)} 条切片")
    print(chunks)



"""
[
  {
    "chunk_id": 468071782111254662,
    "distance": 0.7463973164558411,
    "entity": {
      "chunk_id": 468071782111254662,
      "item_name": "RS PRO RS-12 数字万用表",
      "content": "# 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。
      \n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口
      \n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。
      \n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。
      \n\n![数字万用表置于电阻档，红黑表笔接入被测元件两端，显示屏显示1.999Ω，处于电阻测量状态。]
      (http://192.168.249.129:9000/knowledge-base-files/万用表RS-12的使用/images
      /dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)",
      "title": "电阻测量"
    }
  },
  {
    "chunk_id": 468071782111254679,
    "distance": 0.7098905444145203,
    "entity": {
      "chunk_id": 468071782111254679,
      "item_name": "RS PRO RS-12 数字万用表",
      "content": "",
      "title": ""
    }
  },
  {
    "chunk_id": 468071782111254661,
    "distance": 0.7098851799964905,
    "entity": {
      "chunk_id": 468071782111254661,
      "item_name": "RS PRO RS-12 数字万用表",
      "content": "# 交流电压测量\n\n警告：谨防触电。
      \n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。
      因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。
      \n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。
      \n\n\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。
      \n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。
      \n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。",
      "title": "交流电压测量"
    }
  },
  {
    "chunk_id": 468071782111254678,
    "distance": 0.7079825401306152,
    "entity": {
      "chunk_id": 468071782111254678,
      "item_name": "RS PRO RS-12 数字万用表",
      "content": "注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n1. 将黑色表笔插入负极COM端口。\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n6. 接通电源。\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。",
      "title": ""
    }
  },
  {
    "chunk_id": 468071782111254668,
    "distance": 0.6980453729629517,
    "entity": {
      "chunk_id": 468071782111254668,
      "item_name": "RS PRO RS-12 数字万用表",
      "content": "200Vrms 用于200mV量程</td></tr><tr><td rowspan=1 colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA 250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td rowspan=1 colspan=1>10A 250V 快速熔断保险丝(最多每15分钟，\n需时30秒)</td></tr><tr><td rowspan=1 colspan=1>电阻,\n短路测试</td><td rowspan=1 colspan=1>250Vrms,\n最多15秒</td></tr></table>\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。",
      "title": ""
    }
  }
]
"""

