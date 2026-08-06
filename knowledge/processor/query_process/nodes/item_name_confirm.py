"""
商品名确认节点

职责（7 步）:
    1. 从 MongoDB 获取历史对话（时间升序）并格式化为 "role: text" 文本
    2. LLM 从问题+历史中提取商品名并改写问题（JSON 结构化输出 + 清洗解析）
    3. BGE-M3 向量化商品名 → Milvus 商品名集合混合检索
    4. 按相似度分数对齐到 confirmed / options
    5. 多商品确认时按分数差阈值过滤误判结果
    6. 决策路由：confirmed → 走后续检索；options → 询问用户；皆无 → 无法识别提示
    7. 商品名首次确认时回填到历史中 item_names 为空的记录

设计:
    职责分离 —— 4 个组件类各管一段（历史/提取/检索/对齐），
    统一由 ItemNameConfirmNode.process 调度。
"""
import json
import re
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import (
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT,
    ITEM_NAME_EXTRACT_TEMPLATE,
)


# ==================== 类1: MongoDB 历史管理 ====================

class ItemNameHistory:
    """
    商品名确认 - MongoDB 历史管理

    职责:
        - 从 MongoDB 获取会话历史（一次，时间升序）
        - 将记录格式化为 "role: text" 文本供 LLM 理解上下文
        - 商品名确认后做【纯内存】回填：只更新已取出的记录，不查库、不写库
    """

    def __init__(self, node_name: str, logger):
        """
        :param node_name: 节点名称（日志标识）
        :param logger: 日志记录器实例
        """
        self.node_name = node_name
        self.logger = logger

    def get_history_messages(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        从 MongoDB 获取会话历史记录（时间升序）

        :param session_id: 会话 ID
        :param limit: 最多取多少条
        :return: 历史记录字典列表（升序）；无历史或异常返回 []
        """
        try:
            from knowledge.utils.mongo_history_util import get_recent_messages

            # util 返回降序（最新在前），需求要求升序 → 反转
            messages = get_recent_messages(session_id, limit=limit)
            messages = list(reversed(messages))
            self.logger.info(f"历史对话加载完成: {len(messages)} 条")
            return messages
        except Exception as e:
            self.logger.error(f"获取历史对话失败（降级为空）: {e}")
            return []

    def format_history_text(self, messages: List[Dict]) -> str:
        """
        把历史记录格式化为 "role: text" 换行文本（供 LLM 理解上下文）

        :param messages: 历史记录字典列表（升序）
        :return: 格式化后的文本
        """
        lines = [f"{msg.get('role', 'unknown')}: {msg.get('text', '')}" for msg in messages]
        return "\n".join(lines)

    def backfill_item_names(self, messages: List[Dict], confirmed_names: List[str]) -> int:
        """
        纯内存回填：只更新【已取出的记录】中 item_names 为空的那些

        注意: 不改动 MongoDB 里的历史数据，也不重新查询；
              回填后的记录随 state["history"] 流转，供下游使用。

        :param messages: 已取出的历史记录列表（会原地修改）
        :param confirmed_names: 确认的商品名列表
        :return: 回填的记录数
        """
        if not messages or not confirmed_names:
            return 0

        updated = 0
        for msg in messages:
            if not msg.get("item_names"):
                msg["item_names"] = confirmed_names
                updated += 1

        self.logger.info(f"历史记录内存回填: {updated} 条（未改动 MongoDB 数据）")
        return updated


# ==================== 类2: LLM 商品名提取与问题改写 ====================

class ItemNameExtractor:
    """
    商品名确认 - LLM 商品名提取与问题改写

    职责:
        - 将历史上下文 + 用户问题注入提示词，调 LLM(JSON 模式) 提取商品名并改写问题
        - 清洗解析 LLM 响应（去代码围栏 → json.loads → 字段类型校验）
        - 降级：任何失败返回空商品名 + 原问题兜底
    """

    def __init__(self, node_name: str, logger):
        """
        :param node_name: 节点名称（日志标识）
        :param logger: 日志记录器实例
        """
        self.node_name = node_name
        self.logger = logger

    def extract_item_names(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        LLM 根据用户问题 + 历史对话提取商品名并改写问题

        :param original_query: 用户原始问题
        :param history_text: 历史对话文本（role: text 格式）
        :return: {"item_names": [...], "rewritten_query": "..."}；失败降级为原问题
        """
        # 降级兜底结果：商品名为空、问题保持原样
        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        try:
            from knowledge.utils.client.ai_clients import AIClients  # 延迟导入，避免顶层加载重量级依赖

            llm_client = AIClients.get_llm_openai(response_format=True)
            if not llm_client:
                self.logger.warning("LLM 客户端获取失败，降级返回空商品名")
                return result

            human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text if history_text else "暂无上下文",
                query=original_query,
            )
            llm_response = llm_client.invoke([
                {"role": "system", "content": ITEM_NAME_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": human_prompt},
            ])

            llm_content = llm_response.content.strip()
            if not llm_content:
                self.logger.warning("LLM 返回空内容，降级返回空商品名")
                return result

            parse_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parse_result.get("rewritten_query") or original_query
            result["item_names"] = parse_result.get("item_names") or []
            self.logger.info(
                f"LLM 提取商品名: {result['item_names']}，"
                f"改写问题: {str(result['rewritten_query'])[:60]}..."
            )
            return result
        except Exception as e:
            self.logger.error(f"LLM 提取商品名失败（降级为原问题）: {e}")
            return result

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        """
        清洗并解析 LLM 响应（去除 ```json 围栏 → json.loads → 字段类型校验）

        :param llm_response: LLM 原始返回
        :return: {"item_names": [...], "rewritten_query": "..."}
        :raises ValueError: JSON 解析失败或字段类型不符
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)

            raw_item_names = parsed_llm_result.get("item_names")
            clean_item_names = (
                [raw_item for raw_item in raw_item_names if raw_item.strip()]
                if isinstance(raw_item_names, list)
                else []
            )

            raw_rewritten_query = parsed_llm_result.get("rewritten_query")
            clean_rewritten_query = raw_rewritten_query.strip() if isinstance(raw_rewritten_query, str) else ""

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON 反序列化 LLM 输出失败: {e}")


# ==================== 类3: 向量化与混合检索 ====================

class ItemNameVectorizer:
    """
    商品名确认 - 向量化与混合检索

    职责:
        - 用 BGE-M3 将商品名向量化（查询侧 is_query=True，存锁查钥）
        - 在 Milvus 商品名集合上做混合检索（稠密余弦 + 稀疏内积）
        - 整理结果为 [{item_name, score}] 列表
    """

    def __init__(self, node_name: str, logger, config: QueryConfig):
        """
        :param node_name: 节点名称（日志标识）
        :param logger: 日志记录器实例
        :param config: 查询流程配置（集合名/权重/检索数量）
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config

    def search_item_names(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
        逐个商品名混合检索，返回带分数的商品列表

        :param item_names: LLM 提取的商品名列表
        :return: [{"item_name": str, "score": float}, ...]；单个失败跳过，全部失败返回 []
        """
        if not item_names:
            return []

        try:
            from knowledge.utils.client.ai_clients import AIClients
            from knowledge.utils.client.storage_clients import StorageClients
            from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
            from knowledge.utils.milvus_util import (
                create_hybrid_search_requests,
                execute_hybrid_search_query,
            )
        except Exception as e:
            self.logger.error(f"导入检索工具失败: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for name in item_names:
            try:
                # 1. BGE-M3 向量化（查询侧）
                bge_m3 = AIClients.get_bge_m3_client()
                vectors = generate_bge_m3_hybrid_vectors(bge_m3, [name], is_query=True)
                dense_vector = vectors["dense"][0]
                sparse_vector = vectors["sparse"][0]

                # 2. 构建混合检索请求（稠密+稀疏）
                search_requests = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=self.config.embedding_search_limit,
                )

                # 3. 执行混合检索
                milvus_client = StorageClients.get_milvus_client()
                hits = execute_hybrid_search_query(
                    milvus_client=milvus_client,
                    collection_name=self.config.item_name_collection,
                    search_requests=search_requests,
                    limit=self.config.embedding_search_limit,
                    output_fields=["item_name"],
                    ranker_weights=(
                        self.config.item_name_dense_weight,
                        self.config.item_name_sparse_weight,
                    ),
                )

                # 4. 整理结果
                for hit in hits[0] if hits else []:
                    results.append({
                        "item_name": hit["entity"].get("item_name", ""),
                        "score": float(hit.get("distance", 0.0)),
                    })
                self.logger.info(f"商品名 '{name}' 检索到 {len(hits[0]) if hits else 0} 条结果")
            except Exception as e:
                self.logger.warning(f"商品名 '{name}' 检索失败（跳过）: {e}")

        return results


# ==================== 类4: 评分对齐 ====================

class ItemNameAligner:
    """
    商品名确认 - 评分对齐

    职责:
        - 按相似度分数把检索结果分派到 confirmed / options
        - 多商品 confirmed 时按分数差阈值剔除误判结果

    评分规则（high=item_name_high_confidence(0.7), mid=item_name_mid_confidence(0.6)）:
        score > high:
            与任一 LLM 商品名精准确认（相等/互含）→ confirmed
            否则: 唯一一条 >high → confirmed；多条 → options
        mid ≤ score ≤ high → options（上限 item_name_max_options，去重）
        score < mid → 丢弃
    """

    def __init__(self, node_name: str, logger, config: QueryConfig):
        """
        :param node_name: 节点名称（日志标识）
        :param logger: 日志记录器实例
        :param config: 查询流程配置（置信度阈值/候选上限/分数差阈值）
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config

    def align(
        self,
        llm_item_names: List[str],
        search_results: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        """
        将检索结果对齐到 confirmed / options

        :param llm_item_names: LLM 提取的商品名（用于精准确认）
        :param search_results: 检索结果 [{"item_name", "score"}, ...]
        :return: (confirmed 商品名列表, options 候选列表)
        """
        high = self.config.item_name_high_confidence
        mid = self.config.item_name_mid_confidence
        max_options = self.config.item_name_max_options

        confirmed: List[Dict[str, Any]] = []  # 带分数的确认结果（需求5 需要分数）
        options: List[str] = []
        seen_options = set()

        # 预收集 >high 的结果，用于「是否只有一条」判断
        high_scores = [r for r in search_results if r["score"] > high]

        for result in search_results:
            name = result["item_name"]
            score = result["score"]

            if score >= high:
                # 分支一：高置信
                if self._is_exact_match(name, llm_item_names):
                    # 精准确认 → confirmed
                    self._append_confirmed(confirmed, name, score)
                elif len(high_scores) == 1:
                    # 唯一一条高置信 → confirmed
                    self._append_confirmed(confirmed, name, score)
                else:
                    # 多条高置信 → options
                    if name not in seen_options:
                        seen_options.add(name)
                        options.append(name)
            elif mid <= score < high:
                # 分支二：中置信 → options
                if name not in seen_options:
                    seen_options.add(name)
                    options.append(name)
            # 分支三：低置信 → 丢弃

        # 需求5: 多商品 confirmed 按分数差过滤误判
        if len(confirmed) > 1:
            confirmed = self._filter_by_score_gap(confirmed, self.config.item_name_score_gap)

        confirmed_names = [item["item_name"] for item in confirmed]
        options = options[:max_options]

        self.logger.info(f"评分对齐完成: confirmed={confirmed_names}, options={options}")
        return confirmed_names, options

    @staticmethod
    def _append_confirmed(confirmed_list: List[Dict[str, Any]], name: str, score: float) -> None:
        """向 confirmed 列表追加结果（按商品名去重）"""
        if not any(item["item_name"] == name for item in confirmed_list):
            confirmed_list.append({"item_name": name, "score": score})

    def _is_exact_match(self, matched_name: str, llm_names: List[str]) -> bool:
        """
        精准确认：归一化（去空白/小写）后相等或互相包含

        :param matched_name: 检索匹配到的商品名（如 "RS-12 数字万用表"）
        :param llm_names: LLM 提取的商品名列表（如 ["RS-12"]）
        :return: True 表示可精准确认
        """
        def _normalize(text: str) -> str:
            return re.sub(r"\s+", "", text).lower()

        norm_matched = _normalize(matched_name)
        if not norm_matched:
            return False

        for llm_name in llm_names:
            norm_llm = _normalize(llm_name)
            if not norm_llm:
                continue
            if norm_matched == norm_llm or norm_matched in norm_llm or norm_llm in norm_matched:
                return True
        return False

    def _filter_by_score_gap(
        self,
        confirmed_with_scores: List[Dict[str, Any]],
        gap: float,
    ) -> List[Dict[str, Any]]:
        """
        剔除与最高分差距超过阈值的误判结果

        :param confirmed_with_scores: [{"item_name", "score"}, ...]
        :param gap: 分数差阈值
        :return: 过滤后的结果
        """
        max_score = max(item["score"] for item in confirmed_with_scores)
        filtered = [item for item in confirmed_with_scores if max_score - item["score"] <= gap]
        self.logger.info(
            f"分数差过滤: {len(confirmed_with_scores)} → {len(filtered)}"
            f"（最高 {max_score:.3f}，阈值 {gap}）"
        )
        return filtered


# ==================== 调度节点 ====================

class ItemNameConfirmNode(BaseNode):
    """
    商品名确认节点

    流程:
        校验 → 历史获取 → LLM 提取/改写 → 向量化混合检索 → 评分对齐
        → 决策路由（confirmed/options/无法识别）→ 历史回填
    """
    name = 'item_name_confirm_node'

    def __init__(self, config: Optional[QueryConfig] = None):
        super().__init__(config)
        self._history = ItemNameHistory(self.name, self.logger)
        self._extractor = ItemNameExtractor(self.name, self.logger)
        self._vectorizer = ItemNameVectorizer(self.name, self.logger, self.config)
        self._aligner = ItemNameAligner(self.name, self.logger, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        商品名确认主流程

        :param state: 查询图状态（需含 original_query / session_id）
        :return: 更新后的状态（item_names / rewritten_query / item_name_options / answer）
        """
        # 1. 校验 state
        original_query = state.get("original_query")
        session_id = state.get("session_id")
        if not original_query:
            raise StateFieldError(node_name=self.name, field_name="original_query", expected_type=str)
        if not session_id:
            raise StateFieldError(node_name=self.name, field_name="session_id", expected_type=str)
        self.log_step("步骤1", "校验 state 通过")

        # 2. 历史对话（需求1）：取一次记录 → 格式化文本 → 存入 state["history"]
        self.log_step("步骤2", "获取历史对话")
        messages = self._history.get_history_messages(session_id)
        history_text = self._history.format_history_text(messages)
        state["history"] = messages

        # 3. LLM 提取商品名 + 问题改写（需求2）
        self.log_step("步骤3", "LLM 提取商品名与问题改写")
        clean_result = self._extractor.extract_item_names(original_query, history_text)
        item_names = clean_result.get("item_names", [])
        rewritten_query = clean_result.get("rewritten_query", original_query)

        # 4. 短路：无商品名 → 无法识别提示（需求6 分支三）
        if not item_names:
            self.log_step("步骤4", "未识别出商品名，返回无法识别提示")
            state["item_names"] = []
            state["rewritten_query"] = rewritten_query
            state["answer"] = "抱歉，我无法识别您询问的商品，请提供更明确的商品名称。"
            return state

        # 5. 向量化 + 混合检索（需求3）
        self.log_step("步骤5", f"向量化并混合检索 {len(item_names)} 个商品名")
        search_results = self._vectorizer.search_item_names(item_names)

        # 6. 评分对齐（需求4）
        self.log_step("步骤6", "评分对齐到 confirmed/options")
        confirmed, options = self._aligner.align(item_names, search_results)

        # 7. 决策路由（需求6）
        self.log_step("步骤7", "决策路由")
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query

        if confirmed:
            # 确认成功 → 不写 answer，走后续检索（main_graph 按 answer 路由）
            self.logger.info(f"商品名确认成功: {confirmed}，继续后续检索")
            # 8. 历史回填（需求7，纯内存：只改已取出的记录，不写 MongoDB）
            self.log_step("步骤8", "商品名确认，内存回填历史记录")
            #这里不需要再写state[history]，因为前面写了，列表是引用类型会根据引用的内容更新而更新
            self._history.backfill_item_names(messages, confirmed)
        elif options:
            # 模糊 → 询问用户选择
            state["item_name_options"] = options
            option_text = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(options))
            state["answer"] = f"您想询问哪个商品？请选择：\n{option_text}"
            self.logger.info(f"商品名模糊，提供候选: {options}")
        else:
            # 无法识别
            state["answer"] = "抱歉，我无法识别您询问的商品，请提供更明确的商品名称。"
            self.logger.warning("商品名确认失败（无 confirmed 无 options）")

        return state


# ==================== 测试 ====================

# 将项目根目录加入 sys.path（本文件在 .../shopkeeper_brain/knowledge/processor/query_process/nodes/）
# .parent x5 = shopkeeper_brain（knowledge 包的父目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / "knowledge" / ".env")


if __name__ == '__main__':
    setup_logging()

    node = ItemNameConfirmNode()

    # 场景1: 明确商品名 → 期望 confirmed（item_names 有值，answer 为空走搜索）
    print("\n" + "=" * 60)
    print("【场景1】明确商品名")
    state_1 = {
        "original_query": "RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？",
        "session_id": "test_session_confirm_1"
    }
    result_1 = node(state_1)
    print(f"  item_names: {result_1.get('item_names')}")
    print(f"  rewritten_query: {result_1.get('rewritten_query')}")
    print(f"  item_name_options: {result_1.get('item_name_options')}")
    print(f"  answer: {result_1.get('answer')}")

    # 场景3: 无法识别 → 期望 answer 是无法识别提示
    print("\n" + "=" * 60)
    print("【场景3】无法识别商品")
    state_3 = {
        "original_query": "今天天气怎么样？",
        "session_id": "test_session_confirm_3",
        "task_id": "",
        "is_stream": False,
    }
    result_3 = node(state_3)
    print(f"  item_names: {result_3.get('item_names')}")
    print(f"  answer: {result_3.get('answer')}")
    print("=" * 60)
