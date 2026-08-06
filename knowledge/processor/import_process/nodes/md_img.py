"""
Markdown 图片处理节点

职责:
    1. 校验 MD 文档路径和图片目录路径
    2. 提取图片及其上下文（标题、周围文本）
    3. 通过 VLM 视觉大模型为图片生成描述（含限流和降级策略）
    4. 上传图片到 MinIO 并替换 MD 中的图片链接
"""
import base64
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import (
    StateFieldError,
    FileProcessingError,
    ImageProcessingError,
)
from knowledge.processor.import_process.state import ImportGraphState


# ==================== 数据结构 ====================

@dataclass
class ImageInfo:
    """
    图片信息数据结构

    Attributes:
        rel_path: MD 中的相对路径，如 "images/abc123.jpg"
        abs_path: 本地绝对路径
        context: 图片附近文本上下文（不超过 400 字）
        heading: 图片所在章节标题
        description: VLM 生成的图片描述（默认为空，由 ImageDescriber 填充）
        minio_url: 上传 MinIO 后的 HTTP 链接（默认为空，由上传步骤填充；空表示上传失败）
    """
    rel_path: str
    abs_path: Path
    context: str
    heading: str
    description: str = field(default="")
    minio_url: str = field(default="")


# ==================== 类1: 校验与准备 ====================

class MdImgValidator:
    """
    MD 图片处理 - 校验与准备

    职责:
        - 校验 state 中的必填字段（md_path, file_dir, file_title）
        - 校验 MD 文件是否存在
        - 读取 MD 文档内容
        - 确保 images 目录存在（不存在则创建）
    """

    def __init__(self, node_name: str, logger):
        """
        初始化校验器

        :param node_name: 所属节点名称，用于异常信息标识
        :param logger: 日志记录器实例
        """
        self.node_name = node_name
        self.logger = logger

    def validate_and_prepare(self, state: ImportGraphState) -> Tuple[str, Optional[Path], str]:
        """
        统一入口：校验 state → 校验文件 → 读取 MD 内容 → 检查 images 目录

        :param state: 导入流程状态字典，包含 md_path, file_dir, file_title
        :return: (md_content, images_dir, file_title)
                 - md_content: MD 文档的文本内容
                 - images_dir: images 目录的 Path 对象；目录不存在时为 None
                 - file_title: 文件标题（不含扩展名）
        :raises StateFieldError: state 中必填字段缺失或为空
        :raises FileProcessingError: MD 文件不存在或读取失败
        """
        self.logger.info("[MdImgValidator] 开始校验与准备")

        # 1. 校验 state 字段
        md_path, file_dir, file_title = self._validate_state_fields(state)

        # 2. 校验 MD 文件是否存在
        md_path_obj = self._validate_file_exists(md_path)

        # 3. 读取 MD 内容
        md_content = self._read_md_content(md_path_obj)

        # 4. 检查 images 目录是否存在（不自动创建）
        images_dir = self._check_images_dir(md_path_obj)

        self.logger.info(f"[MdImgValidator] 校验完成，MD内容长度: {len(md_content)} 字符")
        return md_content, images_dir, file_title

    def _validate_state_fields(self, state: ImportGraphState) -> Tuple[str, str, str]:
        """
        校验 state 中的必填字段是否为空

        :param state: 导入流程状态字典
        :return: (md_path, file_dir, file_title) 三个字段的字符串值
        :raises StateFieldError: 任一必填字段缺失或为空
        """
        md_path = state.get("md_path", "")
        file_dir = state.get("file_dir", "")
        file_title = state.get("file_title", "")

        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_path")
        if not file_dir:
            raise StateFieldError(node_name=self.node_name, field_name="file_dir")
        if not file_title:
            raise StateFieldError(node_name=self.node_name, field_name="file_title")

        return md_path, file_dir, file_title

    def _validate_file_exists(self, md_path: str) -> Path:
        """
        校验 MD 文件是否存在于磁盘

        :param md_path: MD 文件的路径字符串
        :return: MD 文件的 Path 对象
        :raises FileProcessingError: 文件不存在
        """
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(
                f"MD 文件不存在: {md_path}",
                self.node_name,
            )
        self.logger.info(f"[MdImgValidator] MD 文件存在: {md_path}")
        return md_path_obj

    def _read_md_content(self, md_path: Path) -> str:
        """
        读取 MD 文件的文本内容

        :param md_path: MD 文件的 Path 对象
        :return: MD 文件的完整文本内容
        :raises FileProcessingError: 文件读取失败
        """
        try:
            content = md_path.read_text(encoding="utf-8")
            self.logger.info(f"[MdImgValidator] 读取 MD 成功，长度: {len(content)} 字符")
            return content
        except Exception as e:
            raise FileProcessingError(
                f"读取 MD 文件失败: {md_path}",
                self.node_name,
                cause=e,
            )

    def _check_images_dir(self, md_path: Path) -> Optional[Path]:
        """
        检查 images 目录是否存在

        MinerU 转换后的目录结构:
            {file_dir}/{file_title}/auto/万用表RS-12的使用.md
            {file_dir}/{file_title}/auto/images/   ← 图片目录

        如果 images 目录不存在，说明文档没有图片，返回 None。

        :param md_path: MD 文件的 Path 对象
        :return: images 目录的 Path 对象；目录不存在时返回 None
        """
        images_dir = md_path.parent / "images"
        if not images_dir.exists():
            self.logger.info(f"[MdImgValidator] images 目录不存在: {images_dir}，文档无图片")
            return None
        self.logger.info(f"[MdImgValidator] images 目录已存在: {images_dir}")
        return images_dir


# ==================== 类2: 图片上下文提取 ====================

class ImageExtractor:
    """
    MD 图片处理 - 图片上下文提取

    职责:
        - 遍历 images 目录中的图片文件
        - 过滤非图片文件（根据后缀判断）
        - 用正则表达式在 MD 内容中定位每张图片的引用位置
        - 截取图片附近的文本上下文（不超过 400 字）
        - 查找图片所在章节的标题
    """

    # MD 图片引用正则: ![alt](path)
    MD_IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    # MD 标题行正则: # 标题文字
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, node_name: str, logger, config: ImportConfig):
        """
        初始化提取器

        :param node_name: 所属节点名称，用于日志标识
        :param logger: 日志记录器实例
        :param config: 导入流程配置对象，提供 image_extensions 等配置
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config

    def extract(self, md_content: str, images_dir: Path) -> List[ImageInfo]:
        """
        统一入口：遍历 images 目录 → 过滤非图片 → 在 MD 中定位 → 截取上下文和标题

        :param md_content: MD 文档的完整文本内容
        :param images_dir: images 目录的 Path 对象
        :return: ImageInfo 列表，每项包含图片路径、上下文、标题
        """
        self.logger.info("[ImageExtractor] 开始提取图片上下文")

        # 1. 列出目录中所有图片文件
        image_files = self._list_image_files(images_dir)
        if not image_files:
            self.logger.warning("[ImageExtractor] images 目录中没有图片文件")
            return []

        self.logger.info(f"[ImageExtractor] 发现 {len(image_files)} 个图片文件")

        # 2. 逐一在 MD 内容中定位并提取上下文
        image_infos = []
        for img_path in image_files:
            info = self._process_single_image(img_path, md_content, images_dir)
            if info is not None:
                image_infos.append(info)

        self.logger.info(f"[ImageExtractor] 成功提取 {len(image_infos)} 张图片的上下文")
        return image_infos

    def _list_image_files(self, images_dir: Path) -> List[Path]:
        """
        列出 images 目录中所有图片文件，过滤非图片后缀

        :param images_dir: images 目录的 Path 对象
        :return: 图片文件的 Path 列表（按文件名排序）
        """
        image_files = []
        for f in sorted(images_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in self.config.image_extensions:
                image_files.append(f)
        return image_files

    def _process_single_image(
        self, img_path: Path, md_content: str, images_dir: Path
    ) -> Optional[ImageInfo]:
        """
        处理单张图片：在 MD 中查找引用 → 提取上下文 → 查找标题

        :param img_path: 图片文件的绝对 Path 对象
        :param md_content: MD 文档的完整文本内容
        :param images_dir: images 目录的 Path 对象，用于计算相对路径
        :return: ImageInfo 对象；如果图片未在 MD 中被引用则返回 None
        """
        # 计算图片在 MD 中的相对路径（如 images/xxx.jpg）
        rel_path = str(img_path.relative_to(images_dir.parent))
        # 统一使用正斜杠（MD 标准格式）
        rel_path = rel_path.replace("\\", "/")

        # 在 MD 内容中查找该图片引用
        match = self._find_image_in_md(md_content, rel_path)
        if match is None:
            self.logger.debug(f"[ImageExtractor] 图片未在 MD 中引用，跳过: {rel_path}")
            return None

        # 提取上下文
        context = self._extract_context(md_content, match)
        # 查找所在标题
        heading = self._find_heading_above(md_content, match.start())

        self.logger.info(f"[ImageExtractor] 提取完成: {rel_path} | 标题: {heading}")
        return ImageInfo(
            rel_path=rel_path,
            abs_path=img_path,
            context=context,
            heading=heading,
        )

    def _find_image_in_md(self, md_content: str, rel_path: str) -> Optional[re.Match]:
        """
        用正则表达式在 MD 内容中查找指定图片引用的位置

        :param md_content: MD 文档的完整文本内容
        :param rel_path: 图片在 MD 中的相对路径，如 "images/abc.jpg"
        :return: re.Match 对象（包含匹配位置信息）；未找到则返回 None
        """
        escaped_path = re.escape(rel_path)
        pattern = re.compile(r'!\[[^\]]*\]\(' + escaped_path + r'\)')
        return pattern.search(md_content)

    def _extract_context(self, md_content: str, match: re.Match, max_chars: int = 400) -> str:
        """
        以图片引用位置为中心，截取前后文本作为上下文，总长度不超过 max_chars

        策略: 前后各取 max_chars/2 字符，然后清理多余空白

        :param md_content: MD 文档的完整文本内容
        :param match: 图片引用的 re.Match 对象
        :param max_chars: 上下文最大字符数，默认 400
        :return: 截取后的上下文文本
        """
        half = max_chars // 2
        center = (match.start() + match.end()) // 2

        # 计算截取范围
        start = max(0, center - half)
        end = min(len(md_content), center + half)

        # 截取并移除图片引用本身（上下文中不需要包含图片语法）
        context = md_content[start:end]
        context = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', context)

        # 清理多余空白
        context = re.sub(r'\s+', ' ', context).strip()

        # 截断到最大长度
        if len(context) > max_chars:
            context = context[:max_chars]

        return context

    def _find_heading_above(self, md_content: str, position: int) -> str:
        """
        从图片位置向前搜索，找到最近的章节标题

        :param md_content: MD 文档的完整文本内容
        :param position: 图片引用在 MD 中的起始字符位置
        :return: 标题文字（不含 # 符号）；未找到则返回 "无标题"
        """
        text_before = md_content[:position]
        matches = list(self.HEADING_PATTERN.finditer(text_before))
        if matches:
            return matches[-1].group(2).strip()
        return "无标题"


# ==================== 类3: VLM 图片描述 ====================

class ImageDescriber:
    """
    MD 图片处理 - VLM 图片描述生成

    职责:
        - 遍历所有图片，调用 VLM 视觉大模型生成描述
        - 滑动时间窗口限制 API 访问速率（默认 60 秒内最多 15 次）
        - 通过 AIClients.get_openai() 获取客户端（内置双重检查锁）
        - 任何步骤失败时使用降级策略返回默认描述
    """

    def __init__(self, node_name: str, logger, config: ImportConfig):
        """
        初始化描述生成器

        :param node_name: 所属节点名称，用于日志标识
        :param logger: 日志记录器实例
        :param config: 导入流程配置对象，提供 vl_model, requests_per_minute 等配置
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config
        self.max_requests = config.requests_per_minute
        self.window_seconds = 60
        self.vl_model = config.vl_model
        self.rate_limiter = _SlidingWindowRateLimiter(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            logger=logger,
        )

    def describe_all(self, image_infos: List[ImageInfo]) -> List[ImageInfo]:
        """
        为所有图片生成 VLM 描述（串行版本，保留以兼容旧调用）

        :param image_infos: ImageInfo 列表（含图片路径、上下文、标题）
        :return: 同一列表，每项的 description 字段已被填充
        """
        self.logger.info(f"[ImageDescriber] 开始为 {len(image_infos)} 张图片生成描述")
        result = list(self.describe_stream(image_infos))
        self.logger.info("[ImageDescriber] 所有图片描述生成完成")
        return result

    def describe_stream(self, image_infos: List[ImageInfo]):
        """
        生成器（生产者）：逐张描述图片，描述完一张立即 yield，不等其它图

        这是流水线的「生产端」：配合上传线程，可让「描述下一张」与「上传上一张」重叠。
        单张失败时降级为默认描述，保证每张图都带 description 流出。

        :param image_infos: ImageInfo 列表
        :yield: 每描述完一张就产出该 ImageInfo（description 已填充）
        """
        total = len(image_infos)
        for i, info in enumerate(image_infos, 1):
            try:
                info.description = self._describe_single(info)
                self.logger.info(
                    f"[ImageDescriber] [{i}/{total}] "
                    f"{info.rel_path} → {info.description[:50]}..."
                )
            except Exception as e:
                # 降级策略：使用默认描述
                info.description = self._get_default_description(info.heading)
                self.logger.warning(
                    f"[ImageDescriber] [{i}/{total}] "
                    f"{info.rel_path} 描述失败，使用降级描述: {e}"
                )
            yield info  # 描述完立即交出，供上传线程消费

    def _describe_single(self, image_info: ImageInfo) -> str:
        """
        为单张图片生成 VLM 描述

        流程: 编码 base64 → 构建 prompt → 限流等待 → 调用 VLM API

        :param image_info: 单张图片的 ImageInfo 对象
        :return: VLM 生成的图片描述文本
        :raises Exception: 编码失败、API 调用失败等任何异常
        """
        # 1. 编码图片为 base64
        base64_img = self._encode_image_base64(image_info.abs_path)

        # 2. 构建 VLM prompt
        prompt = self._build_vlm_prompt(image_info.context, image_info.heading)

        # 3. 限流等待
        self.rate_limiter.wait_if_needed()

        # 4. 调用 VLM API
        description = self._call_vlm(base64_img, prompt, image_info.abs_path.suffix)
        return description.strip()

    def _encode_image_base64(self, image_path: Path) -> str:
        """
        读取图片文件并编码为 base64 字符串

        :param image_path: 图片文件的 Path 对象
        :return: base64 编码的字符串
        :raises FileProcessingError: 图片文件不存在
        """
        if not image_path.exists():
            raise FileProcessingError(
                f"图片文件不存在: {image_path}",
                self.node_name,
            )
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    def _call_vlm(self, base64_img: str, prompt: str, suffix: str) -> str:
        """
        调用 VLM 视觉大模型 API 生成图片描述

        通过 AIClients.get_openai() 获取 OpenAI 兼容客户端（内置双重检查锁），
        以 base64 格式发送图片和 prompt，获取文本描述。

        :param base64_img: 图片的 base64 编码字符串
        :param prompt: 给 VLM 的提示词
        :param suffix: 图片文件后缀（如 .jpg），用于推断 MIME 类型
        :return: VLM 返回的图片描述文本
        :raises Exception: API 调用失败、响应异常等
        """
        from knowledge.utils.client.ai_clients import AIClients

        # 通过双重检查锁获取 OpenAI 客户端
        client = AIClients.get_openai()

        # 推断 MIME 类型
        mime_type = self._get_mime_type(suffix)

        response = client.chat.completions.create(
            model=self.vl_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_img}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=200,
        )

        return response.choices[0].message.content

    def _build_vlm_prompt(self, context: str, heading: str) -> str:
        """
        构建给 VLM 的提示词，引导模型根据图片和上下文生成简洁描述

        :param context: 图片附近的文本上下文
        :param heading: 图片所在章节的标题
        :return: 完整的提示词文本
        """
        return (
            f"请根据图片内容，结合以下上下文信息，为这张图片生成一句简洁准确的中文描述（不超过50字）。\n"
            f"所在章节: {heading}\n"
            f"上下文: {context[:200]}\n"
            f"请直接输出描述文字，不要加任何前缀或解释。"
        )

    @staticmethod
    def _get_mime_type(suffix: str) -> str:
        """
        根据文件后缀推断 MIME 类型

        :param suffix: 文件后缀（如 .jpg, .png）
        :return: MIME 类型字符串（如 image/jpeg）
        """
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        return mime_map.get(suffix.lower(), "image/jpeg")

    @staticmethod
    def _get_default_description(heading: str) -> str:
        """
        降级策略：当 VLM 调用失败时返回默认描述

        :param heading: 图片所在章节的标题
        :return: 默认描述文本
        """
        if heading and heading != "无标题":
            return f"文档插图-{heading}"
        return "文档插图"


class _SlidingWindowRateLimiter:
    """
    滑动时间窗口限流器

    原理:
        维护一个 deque 存储每次 API 调用的时间戳。
        每次调用前清除窗口外的旧记录，如果窗口内调用次数达到上限，
        则 sleep 等待最早的记录过期后再放行。
    """

    def __init__(self, max_requests: int, window_seconds: int, logger):
        """
        初始化限流器

        :param max_requests: 时间窗口内允许的最大请求数
        :param window_seconds: 时间窗口大小（秒）
        :param logger: 日志记录器实例
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.logger = logger
        self._timestamps: deque = deque()
        self._lock = threading.Lock()  # 多线程并发调用时保护 _timestamps

    def wait_if_needed(self):
        """
        检查当前是否需要等待，如果需要则 sleep 直到有可用配额（线程安全）

        逻辑:
            1. 清除超过 window_seconds 的旧记录
            2. 如果窗口内记录数 >= max_requests，等待最早记录过期
            3. 记录本次调用时间戳

        线程安全要点:
            - 检查/修改 _timestamps 全程持锁
            - sleep 放在「锁外」：持锁睡觉会阻塞其它线程，违背限流器并发初衷
        """
        while True:
            with self._lock:
                now = time.time()

                # 1. 清除窗口外的旧记录
                while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                    self._timestamps.popleft()

                # 2. 有配额 → 记账放行
                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                # 3. 无配额 → 算出要睡多久（在锁内算，锁外睡）
                wait_time = self._timestamps[0] + self.window_seconds - now + 0.1

            # 锁外 sleep：释放锁让其它线程也能检查/排队
            self.logger.info(
                f"[RateLimiter] 达到限流上限 ({self.max_requests}/{self.window_seconds}s)，"
                f"等待 {wait_time:.1f}s"
            )
            time.sleep(wait_time)


# ==================== 类4: MinIO 上传 + MD 替换 ====================

class ImageReplacer:
    """
    MD 图片处理 - 图片上传与 MD 链接替换

    职责:
        - 将本地图片上传到 MinIO 实现持久化存储
        - 将 MD 中的本地图片引用替换为 MinIO HTTP 链接 + 图片描述
        - 替换格式: ![](images/xxx.jpg) → ![图片描述](http://minio/.../xxx.jpg)
    """

    # MIME 类型映射
    MIME_MAP = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }

    def __init__(self, node_name: str, logger, config: ImportConfig, file_title: str):
        """
        初始化替换器

        :param node_name: 所属节点名称，用于日志标识
        :param logger: 日志记录器实例
        :param config: 导入流程配置对象，提供 minio_bucket, minio_endpoint 等配置
        :param file_title: 文件标题，用于构建 MinIO 对象路径
        """
        self.node_name = node_name
        self.logger = logger
        self.config = config
        self.file_title = file_title

    def replace_all(self, image_infos: List[ImageInfo], md_content: str) -> str:
        """
        统一入口：逐张图片上传到 MinIO → 替换 MD 中的图片引用

        :param image_infos: ImageInfo 列表（含图片路径、描述）
        :param md_content: MD 文档的原始文本内容
        :return: 替换后的 MD 文本内容（图片链接已更新为 MinIO HTTP 地址）
        """
        self.logger.info(f"[ImageReplacer] 开始上传 {len(image_infos)} 张图片并替换 MD 引用")

        for i, info in enumerate(image_infos, 1):
            try:
                # 构建 MinIO 对象名称
                object_name = f"{self.file_title}/images/{info.abs_path.name}"

                # 上传图片到 MinIO
                minio_url = self._upload_to_minio(info.abs_path, object_name)

                # 替换 MD 中的图片引用
                md_content = self._replace_image_ref(
                    md_content, info.rel_path, info.description, minio_url
                )
                self.logger.info(
                    f"[ImageReplacer] [{i}/{len(image_infos)}] "
                    f"上传成功: {info.rel_path} → {minio_url}"
                )
            except Exception as e:
                # 上传失败时保留原始引用，记录日志
                self.logger.error(
                    f"[ImageReplacer] [{i}/{len(image_infos)}] "
                    f"上传失败: {info.rel_path}，保留原始引用: {e}"
                )

        self.logger.info("[ImageReplacer] 所有图片处理完成")
        return md_content

    def upload_single(self, info: ImageInfo) -> str:
        """
        上传单张图片到 MinIO（流水线的「消费端」单元）

        成功：把 HTTP 链接写入 info.minio_url 并返回；
        失败：降级——info.minio_url 置空、记日志、返回 ""（不影响其它图片）。

        :param info: 已带描述的单张图片 ImageInfo
        :return: MinIO HTTP 链接；失败返回 ""
        """
        try:
            object_name = f"{self.file_title}/images/{info.abs_path.name}"
            minio_url = self._upload_to_minio(info.abs_path, object_name)
            info.minio_url = minio_url
            self.logger.info(f"[ImageReplacer] 上传成功: {info.rel_path} → {minio_url}")
            return minio_url
        except Exception as e:
            info.minio_url = ""
            self.logger.error(
                f"[ImageReplacer] 上传失败: {info.rel_path}，保留原始引用: {e}"
            )
            return ""

    def apply_replacements(self, image_infos: List[ImageInfo], md_content: str) -> str:
        """
        所有图片上传完成后，统一把 MD 里的本地引用替换为「描述 + MinIO 链接」

        只替换上传成功（minio_url 非空）的图片；失败的保留原始引用。
        纯字符串操作，放在流水线末尾串行执行，避免并发改字符串。

        :param image_infos: 已上传（含 minio_url）的 ImageInfo 列表
        :param md_content: 原始 MD 文本
        :return: 替换后的 MD 文本
        """
        for info in image_infos:
            if info.minio_url:
                md_content = self._replace_image_ref(
                    md_content, info.rel_path, info.description, info.minio_url
                )
        return md_content

    def _upload_to_minio(self, image_path: Path, object_name: str) -> str:
        """
        上传单个图片文件到 MinIO

        :param image_path: 本地图片文件的 Path 对象
        :param object_name: MinIO 中的对象名称（路径），如 "万用表RS-12的使用/images/abc.jpg"
        :return: 图片的 HTTP 访问 URL
        :raises ImageProcessingError: 上传失败
        """
        try:
            from knowledge.utils.client.storage_clients import StorageClients

            # 通过双重检查锁获取 MinIO 客户端
            minio_client = StorageClients.get_minio_client()
            bucket_name = self.config.minio_bucket

            # 推断 content_type
            content_type = self.MIME_MAP.get(image_path.suffix.lower(), "application/octet-stream")

            # 上传文件
            with open(image_path, "rb") as f:
                minio_client.put_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    data=f,
                    length=image_path.stat().st_size,
                    content_type=content_type,
                )

            # 构建 HTTP URL
            protocol = "https://" if self.config.minio_secure else "http://"
            url = f"{protocol}{self.config.minio_endpoint}/{bucket_name}/{object_name}"
            return url

        except Exception as e:
            raise ImageProcessingError(
                f"上传图片到 MinIO 失败: {image_path.name}",
                self.node_name,
                cause=e,
            )

    def _replace_image_ref(
        self, md_content: str, rel_path: str, description: str, minio_url: str
    ) -> str:
        """
        用正则替换 MD 中的图片引用为 MinIO 链接 + 描述

        替换前: ![](images/xxx.jpg)
        替换后: ![图片描述](http://minio-url/.../xxx.jpg)

        :param md_content: MD 文档文本内容
        :param rel_path: 图片的原始相对路径
        :param description: VLM 生成的图片描述
        :param minio_url: MinIO 上的图片 HTTP URL
        :return: 替换后的 MD 文本内容
        """
        escaped_path = re.escape(rel_path)
        pattern = r'!\[[^\]]*\]\(' + escaped_path + r'\)'
        replacement = f'![{description}]({minio_url})'
        return re.sub(pattern, replacement, md_content)


# ==================== 类5: 并发流水线协调器 ====================

class ImagePipeline:
    """
    图片处理并发流水线（生产者-消费者模型）

    职责:
        - 把「VLM 描述」（慢、受限流）与「MinIO 上传」（网络 IO）重叠执行，提升效率
        - 描述者（生产者，主线程）：描述完一张立即放入队列，转去描述下一张
        - 上传者（消费者，工作线程）：从队列取出已描述的图片并上传 MinIO
        - 描述下一张时的限流等待时间，正好被用来上传上一张 → 等待不再空转

    原理:
        用线程安全的 queue.Queue 作「传送带」解耦两端；
        全部上传完成拿到 URL 后，再串行统一替换 MD 引用。
    """

    _SENTINEL = None  # 结束信号：消费者收到它就知道没有更多图片了

    def __init__(self, describer: "ImageDescriber", replacer: "ImageReplacer", logger):
        """
        :param describer: 图片描述器（生产端）
        :param replacer: 图片上传/替换器（消费端）
        :param logger: 日志记录器实例
        """
        self.describer = describer
        self.replacer = replacer
        self.logger = logger

    def run(self, image_infos: List[ImageInfo], md_content: str) -> str:
        """
        执行流水线：并发地「描述 + 上传」，最后统一替换 MD 引用

        :param image_infos: 待处理图片列表
        :param md_content: 原始 MD 文本
        :return: 替换后的 MD 文本
        """
        upload_queue: queue.Queue = queue.Queue()

        # 启动上传线程（消费者）
        uploader = threading.Thread(
            target=self._upload_worker, args=(upload_queue,), daemon=True
        )
        uploader.start()

        # 主线程担任描述者（生产者）：描述完一张立即放上传送带
        self.logger.info(f"[ImagePipeline] 流水线启动，共 {len(image_infos)} 张图片（描述/上传并发）")
        for info in self.describer.describe_stream(image_infos):
            upload_queue.put(info)

        # 所有图片描述完毕 → 发送结束信号，等上传线程收尾
        upload_queue.put(self._SENTINEL)
        uploader.join()
        self.logger.info("[ImagePipeline] 所有图片描述与上传完成")

        # 末尾串行替换 MD 引用（此时所有 minio_url 已就绪）
        return self.replacer.apply_replacements(image_infos, md_content)

    def _upload_worker(self, upload_queue: queue.Queue) -> None:
        """
        上传消费者：不断从队列取已描述的图片并上传，直到收到结束信号

        :param upload_queue: 描述者与上传者共享的传送带
        :return: None
        """
        while True:
            info = upload_queue.get()
            if info is self._SENTINEL:
                upload_queue.task_done()
                break
            # upload_single 内部已做降级，不会抛出异常中断消费
            self.replacer.upload_single(info)
            upload_queue.task_done()


# ==================== 调度节点 ====================

class Md_Img_Node(BaseNode):
    """
    Markdown 图片处理节点

    编排 4 个子处理器，完成:
        校验准备 → 图片上下文提取 → VLM 描述生成 → MinIO 上传与 MD 替换
    """
    name = 'md_img'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        MD 图片处理主流程

        流程:
            1. MdImgValidator: 校验 state 和文件，读取 MD 内容，确保 images 目录
            2. ImageExtractor: 遍历图片，提取上下文和标题
            3. ImageDescriber: VLM 生成图片描述（含限流和降级）
            4. ImageReplacer: 上传图片到 MinIO，替换 MD 中的图片链接
            5. 将处理后的 MD 内容写回 state["md_content"]

        :param state: 导入流程状态字典
            - md_path: MD 文件路径（必填）
            - file_dir: 文件目录（必填）
            - file_title: 文件标题（必填）
        :return: 更新后的 state，新增 md_content 字段（处理后的 MD 文本）
        """
        config = self.config

        # 步骤1: 校验与准备
        self.log_step("步骤1", "校验路径与读取MD文档")
        validator = MdImgValidator(self.name, self.logger)
        md_content, images_dir, file_title = validator.validate_and_prepare(state)

        # 如果没有 images 目录，直接返回 MD 内容
        if images_dir is None:
            self.logger.info("[Md_Img_Node] 无 images 目录，跳过图片处理，直接返回 MD 内容")
            state["md_content"] = md_content
            return state

        # 步骤2: 提取图片上下文
        self.log_step("步骤2", "提取图片上下文与标题")
        extractor = ImageExtractor(self.name, self.logger, config)
        image_infos = extractor.extract(md_content, images_dir)

        # 如果没有图片，直接返回
        if not image_infos:
            self.logger.info("[Md_Img_Node] 未发现需要处理的图片，跳过后续步骤")
            state["md_content"] = md_content
            return state

        # 步骤3: 流水线并发处理（描述 ‖ 上传），末尾统一替换 MD 引用
        # 描述下一张的限流等待时间，正好用来上传上一张 → 提升整体效率
        self.log_step("步骤3", f"流水线并发处理 {len(image_infos)} 张图片（描述+上传）")
        describer = ImageDescriber(self.name, self.logger, config)
        replacer = ImageReplacer(self.name, self.logger, config, file_title)
        pipeline = ImagePipeline(describer, replacer, self.logger)
        md_content = pipeline.run(image_infos, md_content)

        # 步骤4: 更新 state + 输出备份文件
        state["md_content"] = md_content
        self._write_output(md_content, state.get("md_path", ""))
        self.log_step("步骤4", "MD 图片处理完成")
        return state

    def _write_output(self, md_content: str, md_path: str):
        """
        将处理后的 MD 内容写入备份文件，方便查看节点处理结果

        输出路径: 原始 MD 同目录，文件名加 _img 后缀
        例: 万用表RS-12的使用.md → 万用表RS-12的使用_img.md

        :param md_content: 处理后的 MD 文本内容
        :param md_path: 原始 MD 文件路径
        """
        if not md_path:
            return
        try:
            original = Path(md_path)
            output_path = original.parent / f"{original.stem}_img{original.suffix}"
            output_path.write_text(md_content, encoding="utf-8")
            self.logger.info(f"[Md_Img_Node] 备份文件已输出: {output_path}")
        except Exception as e:
            self.logger.warning(f"[Md_Img_Node] 输出备份文件失败: {e}")


# ================================================================== #
#                            测试                                     #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    # 1. 构建测试 state（需要先通过 pdf_to_md 节点转换出 MD 文件）
    test_md_img_state = {
        "file_dir": r"E:\temp",
        "import_file_path": r"E:\doc\万用表RS-12的使用.pdf",
        "file_title": "万用表RS-12的使用",
        "md_path": r"E:\temp\万用表RS-12的使用\auto\万用表RS-12的使用.md",
    }

    # 2. 实例 Md_Img_Node 节点
    md_img_node = Md_Img_Node()

    # 3. 调用（通过 __call__ 走 BaseNode 的日志/任务追踪）
    processed_state = md_img_node(test_md_img_state)

    # 4. 输出结果
    print("md_content 长度:", len(processed_state.get("md_content", "")))
