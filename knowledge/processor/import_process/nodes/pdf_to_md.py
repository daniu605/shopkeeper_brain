import os
import shutil
import subprocess
import sys
from os import path
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class Pdf_To_Md_Node(BaseNode):
    name = 'pdf_to_md'

    def _get_md_path(self, import_file_path: Path, file_dir_path: Path)->str:
        file_name = import_file_path.stem
        md_path = file_dir_path / file_name / "auto" / f"{file_name}.md"
        return str(md_path)


    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path) -> int:
        import_file = import_file_path
        file_dir = file_dir_path
        proc = subprocess.Popen(
            # -b pipeline：用传统解析管线，不走 VLM/turbomind（规避 RTX50 Blackwell 的 turbomind 内核不兼容）
            args=[self._get_mineru_command(), "-p", str(import_file), "-o", str(file_dir), "-b", "pipeline"],
            stdout=subprocess.PIPE,  # 把输出接到管道
            stderr=subprocess.STDOUT,  # 错误也并进同一个管道
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,  # 行缓冲
        )

        # 关键：实时逐行读，而不是等结束再读
        for line in proc.stdout:
            self.logger.info(f"[MinerU] {line.rstrip()}")

        processed_code = proc.wait()  # 读完输出后，等进程结束拿返回码
        return processed_code

    def _get_mineru_command(self) -> str:
        """
        定位 mineru 可执行文件，保证不依赖调用方的 PATH 环境

        查找顺序:
            1. shutil.which 在 PATH 中查找（venv 激活时的正常情况）
            2. 当前 Python 解释器所在 venv 的 Scripts 目录（服务未激活 venv 也能找到）

        :return: mineru 命令的完整路径或命令名
        """
        # 1. PATH 查找（含 venv 激活场景）
        found = shutil.which("mineru")
        if found:
            return found

        # 2. 兜底：从当前解释器所在 venv 的 Scripts/bin 目录找
        scripts_dir = Path(sys.executable).parent
        candidate = scripts_dir / ("mineru.exe" if os.name == "nt" else "mineru")
        if candidate.exists():
            self.logger.info(f"从 venv Scripts 目录定位到 mineru: {candidate}")
            return str(candidate)

        # 3. 都找不到，退回裸命令名（最终由 Popen 抛 WinError/FileNotFoundError 暴露）
        return "mineru"

    def _validate_state_import_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        import_file_path = state.get('import_file_path', "")
        file_dir_path = state.get('file_dir', "")
        if not import_file_path:
            raise ValidationError("文件上传路径为空", self.name)

        import_file_path = Path(import_file_path)

        if not import_file_path.exists():
            raise FileProcessingError("文件上传路径不存在", self.name)
        if not file_dir_path:
            file_dir_path = import_file_path.parent  # 输出路径为空就拿取输入路径的父目录
        file_dir_path = Path(file_dir_path)

        return import_file_path, file_dir_path

    # import_file_path: str  # 导入文件路径,pdf所在路径，文件及路径的全路径。
    # file_dir: str  # 导入(出)文件目录，pdf转换为md存放的目录，父目录级别
    # md_path: str  # 转换后Markdown 文件路径，具体的文件及目录全路径。
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        pdf转为md格式节点

        :param state:
        import_file_path: 必须。需要被转换的 PDF 源文件路径。
        file_dir: 可选。指定转换后文件的存放目录。如果不传，代码默认使用源文件所在的父目录。

       :return:
       md_path: 核心产出。转换成功后的 Markdown 文件绝对路径字符串。
        """

        # 1.校验路径参数
        self.logger.info("[Pdf_To_Md_Node] 开始校验路径参数")
        import_file_path_obj, file_dir_path_obj = self._validate_state_import_path(state)

        self.logger.info("[Pdf_To_Md_Node] 开始执行转换pdf转为md")
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_path_obj)
        if processed_code == 0:
            self.logger.info(f"PDF->{import_file_path_obj}转为md->{file_dir_path_obj}成功")
        else:
            self.logger.error("PDF->{import_file_path_obj}转为md->{file_dir_path_obj}失败")
            raise PdfConversionError("pdf转化失败", self.name)

        self.logger.info("[Pdf_To_Md_Node] 获取转化后的md路径")
        md_path = self._get_md_path(import_file_path_obj, file_dir_path_obj)

        self.logger.info("[Pdf_To_Md_Node] 更新state然后返回")
        state["md_path"] = md_path

        return state





if __name__ == '__main__':
    setup_logging()

    # 方式：直接实例该节点对象，调用 process 方法
    # 1. 构建该节点需要的 state
    test_pdf_to_md_state = {
        "file_dir": r"E:\temp",
        "import_file_path": r"E:\doc\万用表RS-12的使用.pdf"
    }
    # 2. 实例 pdf_to_md_node 节点
    pdf_to_md_node = Pdf_To_Md_Node()

    # 3. 调用 process 方法
    processed_state = pdf_to_md_node(test_pdf_to_md_state)




