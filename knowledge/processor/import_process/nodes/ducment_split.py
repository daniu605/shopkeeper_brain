"""
文档切分节点

职责:
    将 Markdown 文档切分为适合向量化的文本块（chunks）

切分流程:
    1. 统一换行符 → 按标题一级切分
    2. 超长块 → 按段落/句子/字符二次切分 + 贪心装箱
    3. 过短块 → 按父标题智能合并
    4. 格式化输出 chunks
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import (
    StateFieldError,
    FileProcessingError,
    DocumentSplitError,
)
from knowledge.processor.import_process.state import ImportGraphState


# ==================== 数据结构 ====================

@dataclass
class Chunk:
    """
    文档切分后的文本块

    Attributes:
        title: 章节标题（一级切分时的标题）
        content: 完整内容（标题 + 正文）
        file_title: 来源文件标题
        parent_title: 父标题（二次切分时产生，标记所属的一级标题）
        part: 片段编号（二次切分时产生，如 "1/3"）
    """
    title: str
    content: str
    file_title: str
    parent_title: str = field(default="")
    part: str = field(default="")
    item_name: str = field(default="")  # 商品/产品名称，由 item_name_recognition 节点回填
    dense_vector: List[float] = field(default_factory=list)  # BGE-M3 稠密向量(1024维)，由 bge_embedding 节点回填
    sparse_vector: Dict[int, float] = field(default_factory=dict)  # BGE-M3 稀疏向量，由 bge_embedding 节点回填
    chunk_id: int = field(default=-1)  # Milvus 自动生成的主键，由 import_milvus 节点回填；-1 表示未入库


# ==================== 类1: 状态校验 ====================

class DocumentSplitValidator:
    """
    文档切分 - 状态校验

    职责:
        - 校验 state 中 md_content 和 file_title 是否存在且有效
        - 不通过则抛出对应异常，标记节点名称和失败原因
    """

    def __init__(self, node_name: str, logger):
        """
        初始化校验器

        :param node_name: 所属节点名称，用于异常信息标识
        :param logger: 日志记录器实例
        """
        self.node_name = node_name
        self.logger = logger

    def validate(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        校验 state 中的必填字段

        :param state: 导入流程状态字典
        :return: (md_content, file_title)
        :raises StateFieldError: md_content 或 file_title 缺失或为空
        """
        md_content = state.get("md_content", "")
        file_title = state.get("file_title", "")

        if not md_content:
            raise StateFieldError(node_name=self.node_name, field_name="md_content")
        if not file_title:
            raise StateFieldError(node_name=self.node_name, field_name="file_title")

        self.logger.info(
            f"[DocumentSplitValidator] 校验通过，"
            f"MD内容长度: {len(md_content)} 字符，文件标题: {file_title}"
        )
        return md_content, file_title


# ==================== 类2: 一级标题切分 ====================

class HeadingSplitter:
    """
    文档切分 - 一级标题切分

    职责:
        - 统一文档中的换行符为 \n
        - 用正则表达式匹配 MD 格式的标题（#, ##, ### 等）
        - 按标题将文档切割为多个章节，每个章节包含标题和正文
    """

    # MD 标题行正则: 匹配 # 开头、后跟空格和标题文字的行
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, logger):
        """
        初始化标题切分器

        :param logger: 日志记录器实例
        """
        self.logger = logger

    def split(self, md_content: str) -> List[Dict]:
        """
        统一入口：统一换行符 → 正则匹配标题 → 按标题切割文档

        :param md_content: Markdown 文档原始文本
        :return: 章节列表，每项包含 title（标题文字）和 content（标题下的正文）
        """
        self.logger.info("[HeadingSplitter] 开始按标题切分")

        # 1. 统一换行符
        normalized = self._normalize_line_endings(md_content)

        # 2. 按标题切分
        sections = self._split_by_heading(normalized)

        self.logger.info(f"[HeadingSplitter] 切分为 {len(sections)} 个章节")
        return sections

    def _normalize_line_endings(self, text: str) -> str:
        """
        统一所有换行符为 \n 格式

        处理 \r\n（Windows）和 \r（旧 Mac）两种换行符

        :param text: 原始文本
        :return: 换行符统一后的文本
        """
        return text.replace('\r\n', '\n').replace('\r', '\n')

    def _split_by_heading(self, content: str) -> List[Dict]:
        """
        按 MD 标题行切割文档为多个章节

        标题前的内容归入 title="" 的根节点（如果存在）

        :param content: 换行符已统一的 MD 文本
        :return: 章节列表，每项为 {"title": str, "content": str}
        """
        matches = list(self.HEADING_PATTERN.finditer(content))

        if not matches:
            # 没有标题，整个文档作为一个章节
            self.logger.info("[HeadingSplitter] 未找到标题，整文档作为一个章节")
            return [{"title": "", "content": content.strip()}]

        sections = []

        # 第一个标题之前的内容 → 归入根节点（title 为空）
        if matches[0].start() > 0:
            pre_text = content[:matches[0].start()].strip()
            if pre_text:
                sections.append({"title": "", "content": pre_text})
                self.logger.info("[HeadingSplitter] 提取根节点（标题前内容）")

        # 逐个标题提取章节
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip()
            sections.append({"title": title, "content": body})

        return sections


# ==================== 类3: 二次切分 + 贪心装箱 ====================

class SecondarySplitter:
    """
    文档切分 - 二次切分与贪心装箱

    职责:
        - 识别超过最大长度限制的章节
        - 按优先级（段落 > 行 > 句号 > 逗号 > 分号 > 字符）切分为原子片段
        - 贪心填充装箱：按顺序将片段塞入箱子，装满后切换下一个箱子
        - 每个箱子生成一个编号片段（part）
    """

    def __init__(self, logger, config: ImportConfig):
        """
        初始化二次切分器

        :param logger: 日志记录器实例
        :param config: 导入流程配置对象，提供 max_content_length 限制
        """
        self.logger = logger
        self.config = config
        self.max_length = config.max_content_length

    def split_long_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        统一入口：遍历所有章节，超长块进行二次切分，正常块直接保留

        :param sections: 一级切分后的章节列表
        :return: 处理后的章节列表（可能包含更多项，超长块被拆分为多个 part）
        """
        self.logger.info("[SecondarySplitter] 开始检查超长块")

        result = []
        split_count = 0

        for section in sections:
            body = section["content"]
            if len(body) <= self.max_length:
                result.append(section)
                continue

            # 超长块 → 二次切分
            self.logger.info(
                f"[SecondarySplitter] 标题 '{section['title']}' 正文 {len(body)} 字符，"
                f"超过上限 {self.max_length}，进行二次切分"
            )
            sub_sections = self._secondary_split(body, section["title"])
            result.extend(sub_sections)
            split_count += 1

        self.logger.info(
            f"[SecondarySplitter] 二次切分完成，"
            f"{split_count} 个块被拆分，总计 {len(result)} 个块"
        )
        return result

    def _secondary_split(self, body: str, title: str) -> List[Dict]:
        """
        对超长正文进行二次切分：拆分为原子片段 → 贪心装箱

        :param body: 超长正文文本
        :param title: 所属章节标题（用于日志和 part 标记）
        :return: 拆分后的子章节列表，每项含 content（仅正文）和 part（编号）
        """
        # 1. 按优先级拆分为原子片段
        pieces = self._split_by_priority(body)
        self.logger.info(f"[SecondarySplitter] 拆分为 {len(pieces)} 个原子片段")

        # 2. 贪心装箱
        bins = self._greedy_bin_pack(pieces)
        self.logger.info(f"[SecondarySplitter] 贪心装箱为 {len(bins)} 个块")

        # 3. 生成带编号的子章节
        return self._build_sub_sections(bins, title)

    def _split_by_priority(self, text: str) -> List[str]:
        """
        按优先级拆分文本为原子片段

        优先级: 段落 > 行 > 句号 > 逗号 > 分号 > 字符
        按第一个能产生多个片段的分隔方式执行

        :param text: 待拆分的正文文本
        :return: 拆分后的文本片段列表
        """
        # 1. 段落（连续换行分隔）
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        # 2. 行（单换行分隔）
        lines = text.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        if len(lines) > 1:
            return lines

        # 3. 句号（中英文句号）
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            return sentences

        # 4. 逗号（中英文逗号）
        clauses = re.split(r'(?<=[，,])\s*', text)
        clauses = [c.strip() for c in clauses if c.strip()]
        if len(clauses) > 1:
            return clauses

        # 5. 分号（中英文分号）
        parts = re.split(r'(?<=[；;])\s*', text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

        # 6. 字符（兜底：逐字符切分）
        return list(text)

    def _greedy_bin_pack(self, pieces: List[str]) -> List[str]:
        """
        贪心装箱：按顺序将片段塞入箱子，装满后切换下一个

        每个箱子的总长度不超过 max_length，片段之间用换行连接。
        如果单个片段本身就超过 max_length，则递归切分它。

        :param pieces: 原子片段列表
        :return: 每个箱子的文本内容列表
        """
        bins = []
        current_parts = []
        current_length = 0

        for piece in pieces:
            # 如果单个片段本身就超限，先递归切分
            if len(piece) > self.max_length:
                sub_pieces = self._split_by_priority(piece)
                # 如果切不出来更小的，就按字符硬切
                if len(sub_pieces) == 1 and len(sub_pieces[0]) > self.max_length:
                    sub_pieces = [
                        sub_pieces[0][i:i + self.max_length]
                        for i in range(0, len(sub_pieces[0]), self.max_length)
                    ]
                # 递归装箱子片段
                for sub in sub_pieces:
                    piece_len = len(sub)
                    needed = piece_len + (1 if current_parts else 0)
                    if current_length + needed > self.max_length and current_parts:
                        bins.append('\n'.join(current_parts))
                        current_parts = []
                        current_length = 0
                    current_parts.append(sub)
                    current_length += piece_len + (1 if len(current_parts) > 1 else 0)
                continue

            piece_len = len(piece)
            # +1 为片段之间的换行符长度
            needed = piece_len + (1 if current_parts else 0)

            if current_length + needed > self.max_length and current_parts:
                # 当前箱子已满，封存并开启新箱子
                bins.append('\n'.join(current_parts))
                current_parts = []
                current_length = 0

            current_parts.append(piece)
            current_length += piece_len + (1 if len(current_parts) > 1 else 0)

        # 最后一个箱子
        if current_parts:
            bins.append('\n'.join(current_parts))

        return bins

    def _build_sub_sections(self, bins: List[str], title: str) -> List[Dict]:
        """
        将装箱结果构建为带编号的子章节

        :param bins: 每个箱子的文本内容列表
        :param title: 所属章节标题（设为 parent_title）
        :return: 子章节列表，每项含 title=""、content、part
        """
        sub_sections = []
        total = len(bins)

        for i, text in enumerate(bins, 1):
            part_label = f"{i}/{total}"
            sub_sections.append({
                "title": "",
                "content": text,
                "parent_title": title,
                "part": part_label,
            })

        return sub_sections


# ==================== 类4: 智能合并 ====================

class MergeSplitter:
    """
    文档切分 - 过短块智能合并

    职责:
        - 识别正文长度小于最小限制的块
        - 按父标题分组，保证同父标题的块共同存在
        - 相邻的短块优先向靠后的块合并
        - 特殊处理：顶层标题为空（根标记）的块可被合并；父块可吸收子块的第一块
    """

    def __init__(self, logger, config: ImportConfig):
        """
        初始化合并器

        :param logger: 日志记录器实例
        :param config: 导入流程配置对象，提供 min_content_length 和 max_content_length
        """
        self.logger = logger
        self.config = config
        self.min_length = config.min_content_length
        self.max_length = config.max_content_length

    def merge_short_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        统一入口：按父标题分组 → 组内贪心合并 → 保持原始顺序

        :param chunks: 切分后的章节字典列表
        :return: 合并后的章节字典列表（短块已合并）
        """
        if not chunks:
            return chunks

        self.logger.info("[MergeSplitter] 开始智能合并过短块")

        # 1. 按 (parent_title, title) 分组，保持原始顺序
        groups = self._group_by_parent(chunks)

        # 2. 组内合并
        merged_groups = []
        for group_key, group_chunks in groups:
            merged = self._merge_group(group_chunks)
            merged_groups.extend(merged)

        self.logger.info(
            f"[MergeSplitter] 合并完成，"
            f"从 {len(chunks)} 个块合并为 {len(merged_groups)} 个块"
        )
        return merged_groups

    def _group_by_parent(self, chunks: List[Dict]) -> List[Tuple[str, List[Dict]]]:
        """
        按 parent_title 分组，保持原始顺序

        同一父标题的块被分入同一组（不论各自 title 是否不同），
        不同组之间按首次出现顺序排列

        :param chunks: 章节字典列表
        :return: [(parent_title, [chunk, ...]), ...] 有序分组列表
        """
        groups: List[Tuple[str, List[Dict]]] = []
        group_map: Dict[str, int] = {}

        for chunk in chunks:
            key = chunk.get("parent_title", "")
            if key not in group_map:
                group_map[key] = len(groups)
                groups.append((key, []))
            groups[group_map[key]][1].append(chunk)

        return groups

    def _merge_group(self, group_chunks: List[Dict]) -> List[Dict]:
        """
        组内贪心合并：遍历每个块，短块合并到前一个块，长块独立保留

        合并策略:
            - 短块 → 合并到前一个块（如果合并后不超限）
            - 合并时，将被合并块的标题拼入内容（保留信息不丢失）
            - 顶层块（parent_title 为空）→ 可被后续短块合并
            - 父块（title 为空但有 parent_title）→ 可吸收子块第一块

        :param group_chunks: 同一 parent_title 组内的章节字典列表
        :return: 合并后的章节字典列表
        """
        if len(group_chunks) <= 1:
            return group_chunks

        merged = []

        for i, chunk in enumerate(group_chunks):
            is_short = self._is_short(chunk)

            # 情况1: 当前块是短块 → 尝试合并到前一个块
            if merged and is_short:
                prev = merged[-1]
                # 将被合并块的标题拼入内容（保留标题信息）
                extra = self._build_merge_content(chunk)
                merged_len = len(prev["content"]) + len(extra) + 1
                if merged_len <= self.max_length:
                    prev["content"] = prev["content"] + '\n' + extra
                    self.logger.info(
                        f"[MergeSplitter] 合并短块 '{chunk.get('title', '')}' "
                        f"→ 前一个块，合并后 {len(prev['content'])} 字符"
                    )
                    continue

            # 情况2: 父块吸收子块第一块
            # 父块: title 为空但有 parent_title
            if (
                merged
                and not chunk.get("title")
                and chunk.get("parent_title")
                and not merged[-1].get("title")
                and merged[-1].get("parent_title") == chunk.get("parent_title")
            ):
                prev = merged[-1]
                merged_len = len(prev["content"]) + len(chunk["content"]) + 1
                if merged_len <= self.max_length:
                    prev["content"] = prev["content"] + '\n' + chunk["content"]
                    self.logger.info(
                        f"[MergeSplitter] 父块吸收子块，合并后 {len(prev['content'])} 字符"
                    )
                    continue

            # 情况3: 无法合并 → 作为新块
            merged.append(dict(chunk))

        return merged

    def _build_merge_content(self, chunk: Dict) -> str:
        """
        构建被合并块的内容（将标题拼入正文，保证信息不丢失）

        :param chunk: 被合并的章节字典
        :return: 拼入标题后的完整内容文本
        """
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        if title:
            return f"# {title}\n\n{content}"
        return content

    def _is_short(self, chunk: Dict) -> bool:
        """
        判断块的正文是否过短（小于 min_content_length）

        :param chunk: 章节字典
        :return: True 表示正文长度小于最小限制
        """
        return len(chunk.get("content", "")) < self.min_length


# ==================== 类5: 格式化输出 ====================

class ChunkFormatter:
    """
    文档切分 - 格式化输出

    职责:
        - 将字典格式的章节转换为 Chunk 数据对象
        - 为每个 chunk 拼接 title + content 作为完整内容
        - 附加 file_title 来源标记
    """

    def __init__(self, logger):
        """
        初始化格式化器

        :param logger: 日志记录器实例
        """
        self.logger = logger

    def format(self, sections: List[Dict], file_title: str) -> List[Chunk]:
        """
        统一入口：将字典列表转换为 Chunk 对象列表

        :param sections: 章节字典列表（含 title, content, parent_title, part）
        :param file_title: 来源文件标题
        :return: Chunk 对象列表
        """
        self.logger.info(f"[ChunkFormatter] 开始格式化 {len(sections)} 个块")

        chunks = []
        for section in sections:
            title = section.get("title", "")
            body = section.get("content", "")
            parent_title = section.get("parent_title", "")
            part = section.get("part", "")

            # content = 标题 + 正文
            if title:
                full_content = f"# {title}\n\n{body}"
            else:
                full_content = body

            chunk = Chunk(
                title=title,
                content=full_content,
                file_title=file_title,
                parent_title=parent_title,
                part=part,
            )
            chunks.append(chunk)

        self.logger.info(f"[ChunkFormatter] 格式化完成，共 {len(chunks)} 个 Chunk")
        return chunks


# ==================== 调度节点 ====================

class Document_Split_Node(BaseNode):
    """
    文档切分节点

    编排 5 个子处理器，完成:
        校验 → 标题切分 → 二次切分 → 智能合并 → 格式化输出
    """
    name = 'document_split'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分主流程

        流程:
            1. 校验 state 中的 md_content 和 file_title
            2. 按 MD 标题一级切分
            3. 超长块按段落/句子/字符二次切分 + 贪心装箱
            4. 过短块按父标题智能合并
            5. 格式化输出 Chunk 列表，写入 state["chunks"]

        :param state: 导入流程状态字典
            - md_content: MD 文档内容（必填）
            - file_title: 文件标题（必填）
        :return: 更新后的 state，新增 chunks 字段（Chunk 列表）
        """
        config = self.config

        # 步骤1: 校验状态
        self.log_step("步骤1", "校验状态字段")
        validator = DocumentSplitValidator(self.name, self.logger)
        md_content, file_title = validator.validate(state)

        # 步骤2: 按标题一级切分
        self.log_step("步骤2", "按标题切分文档")
        heading_splitter = HeadingSplitter(self.logger)
        sections = heading_splitter.split(md_content)

        # 步骤3: 超长块二次切分 + 贪心装箱
        self.log_step("步骤3", "检查超长块并二次切分")
        secondary_splitter = SecondarySplitter(self.logger, config)
        sections = secondary_splitter.split_long_sections(sections)

        # 步骤4: 过短块智能合并
        self.log_step("步骤4", "合并过短块")
        merge_splitter = MergeSplitter(self.logger, config)
        sections = merge_splitter.merge_short_chunks(sections)

        # 步骤5: 格式化输出
        self.log_step("步骤5", "格式化输出 chunks")
        formatter = ChunkFormatter(self.logger)
        chunks = formatter.format(sections, file_title)

        # 更新 state
        state["chunks"] = chunks
        self.log_step("完成", f"文档切分完成，共 {len(chunks)} 个块")
        return state


# ================================================================== #
#                            测试                                     #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    # 1. 构建测试 state
    test_split_state = {
        "file_title": "万用表RS-12的使用",
        "md_content": "",  # 需要填入实际的 MD 内容
    }

    # 读取测试 MD 文件
    test_md_path = r"E:\temp\万用表RS-12的使用\auto\万用表RS-12的使用.md"
    try:
        from pathlib import Path
        test_split_state["md_content"] = Path(test_md_path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"读取测试文件失败: {e}")
        exit(1)

    # 2. 实例节点
    split_node = Document_Split_Node()

    # 3. 调用
    processed_state = split_node(test_split_state)

    # 4. 输出结果
    chunks = processed_state.get("chunks", [])
    print(f"\n切分结果: 共 {len(chunks)} 个块")
    print("-" * 60)
    for i, chunk in enumerate(chunks):
        print(f"\n[块 {i+1}] 标题: {chunk.title}")
        if chunk.parent_title:
            print(f"  父标题: {chunk.parent_title}")
        if chunk.part:
            print(f"  片段: {chunk.part}")
        print(f"  内容长度: {len(chunk.content)} 字符")
        # 打印前100字符预览
        preview = chunk.content[:].replace('\n', ' ')
        print(f"  内容预览: {preview}...")
