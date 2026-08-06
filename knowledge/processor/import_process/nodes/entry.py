from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


# 所有节点继承baseNode
class Entry_Node(BaseNode):
    name = 'entry_node'

    # 实现该节点的核心方法
    def process(self, state: ImportGraphState) -> ImportGraphState:

        """
        入口节点：用于判断pdf or md 格式，并且提取文件标题

        :param state:
        import_file_path: 用户上传或指定的文件完整路径

        :return:
        file_title: 文件的纯名称（不含后缀），用于后续命名。
        is_pdf_enabled: 布尔值，标记是否需要走 PDF 转换流程。
        is_md_enabled: 布尔值，标记是否是 Markdown 文件。
        pdf_path 或 md_path: 根据判断结果，将原始路径归档到对应的变量中。
        """

        self.log_step("步骤1", "获取文件路径")
        path = state.get('import_file_path')

        self.log_step("步骤2", "判断路径是否为空")

        if not path:
            raise ValidationError(f"import_path文件路径为空", self.name)

        self.log_step("步骤3", "获取文件后缀")
        import_path = Path(path)
        path_suffix_lower = import_path.suffix.lower()

        self.log_step("步骤4", "判断文件后缀")
        if path_suffix_lower == '.pdf':
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = path
        elif path_suffix_lower == '.md':
            state["is_md_read_enabled"] = True
            state["md_path"] = path
        else:
            raise ValidationError(f"import_path不可取(仅支持pdf或md)", self.name)

        self.log_step("步骤5", "获取文件标题")
        file_title = import_path.stem
        state["file_title"] = file_title

        return state



















#         self.log_step("步骤1", "获取文件路径")
#         # 1.获取文件路径
#         path = state.get('import_file_path')
#         file_dir = state.get('file_dir')
#
#         self.log_step("步骤2", "判断路径是否为空")
#         # 2.检查文件路径是否为空
#         if not path:
#             raise ValidationError(f"import_path文件路径为空", self.name)
#
#         # 3.获取文件后缀
#         import_path = Path(path)
#         suffix = import_path.suffix.lower()
#         if suffix == '.pdf':
#             state["is_pdf_read_enabled"] = True
#             state["pdf_path"] = path
#         elif suffix == '.md':
#             state["is_md_read_enabled"] = True
#             state["md_path"] = path
#         else:
#             raise ValidationError(f"import_path不可取", self.name)
#
#         self.log_step("步骤3", "获取文件标题")
#         # 4.提取文件标题
#         title = import_path.stem
#         state["file_title"] = title
#
#         self.log_step("步骤4", "最后结束end...")
#
#         # 5.返回state
#         return state
#
#     # ================================================================== #
#     #                        测试                                        #
#     # ================================================================== #
#
#
# if __name__ == '__main__':
#     setup_logging()
#
#     # 方式：直接实例该节点对象，调用 process 方法
#     # 1. 构建该节点需要的 state
#     test_entry_state = {
#         "file_dir": r"E:\开发\项目掌柜智库\2.资料\pdf文档\存文档位置",
#         "import_file_path": r"E:\开发\项目掌柜智库\2.资料\pdf文档\doc\万用表RS-12的使用.pdf"
#     }
#     # 2. 实例 EntryNode 节点
#     entry_node = Entry_Node()
#
#     # 3. 调用 process 方法
#     processed_state = entry_node(test_entry_state)
