
import logging

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import  Bge_Embedding_Node
from knowledge.processor.import_process.nodes.ducment_split import  Document_Split_Node
from knowledge.processor.import_process.nodes.entry import  Entry_Node
from knowledge.processor.import_process.nodes.import_milvus import Import_milvus_Node
from knowledge.processor.import_process.nodes.item_name_recognition import Item_Name_Recognition_Node
from knowledge.processor.import_process.nodes.md_img import  Md_Img_Node
from knowledge.processor.import_process.nodes.pdf_to_md import  Pdf_To_Md_Node
from knowledge.processor.import_process.state import ImportGraphState
from langgraph.constants import END

# 主流程日志器（与各节点的 import.xxx 日志器同级，统一格式输出）
logger = logging.getLogger("import.main_graph")


def import_router(state:ImportGraphState):
    if state.get("is_md_read_enabled"):
        return "md"
    if state.get("is_pdf_read_enabled"):
        return "pdf"
    return END

def create_import_graph()->CompiledStateGraph:
    # 1.创建图
    import_graph = StateGraph(ImportGraphState)

    # 2.定义节点
    import_graph.set_entry_point("entry_node")

    # 2.定义节点
    nodes = {
            "entry_node": Entry_Node(),
            "pdf_to_md_node": Pdf_To_Md_Node(),
            "md_img_node": Md_Img_Node(),
            "document_split_node": Document_Split_Node(),
            "item_name_rec_node": Item_Name_Recognition_Node(),
            "bge_embedding_node": Bge_Embedding_Node(),
            "import_milvus_node": Import_milvus_Node()
         }
    for key, value in nodes.items():
        import_graph.add_node(key, value)


    # 3.定义开始条件边
    import_graph.add_conditional_edges(
        "entry_node",import_router,{
        "md" : "md_img_node",
        "pdf" : "pdf_to_md_node",
         END : END
    }
    )

    # 4.输入顺序边
    # PDF 路径：pdf_to_md 只产出 md_path(文件路径)，需经 md_img 读取内容并写入 md_content
    # 两条路径在 md_img_node 汇合，再进入 document_split
    import_graph.add_edge("pdf_to_md_node","md_img_node")
    import_graph.add_edge("md_img_node","document_split_node")
    import_graph.add_edge("document_split_node","item_name_rec_node")
    import_graph.add_edge("item_name_rec_node","bge_embedding_node")
    import_graph.add_edge("bge_embedding_node","import_milvus_node")
    import_graph.add_edge("import_milvus_node",END)

    # 5.编译图
    graph_compiled = import_graph.compile()

    # 6.测试运行
    return graph_compiled


# 下面是运行导入图的流程是否正常
kb_import_graph_app = create_import_graph()  # CompiledStateGraph

def run_import_graph(import_file_path: str, file_dir: str, task_id: str = "") -> dict:
    """
    便捷函数：运行导入流程
    :param import_file_path: PDF/MD 文件路径
    :param file_dir: 输出目录
    :param task_id: 任务 ID（用于前端轮询追踪，写入 state 后 BaseNode 自动调用 task_util）
    :return: 最终状态字典
    """
    state = {
        "task_id": task_id,
        "is_pdf_read_enabled": True,  # 默认值
        "is_md_read_enabled": False,  # 默认值
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    logger.info("=" * 60)
    logger.info(f"导入流程开始 | 文件: {import_file_path} | 任务: {task_id}")
    logger.info("=" * 60)

    final_state = None
    node_count = 0
    for event in kb_import_graph_app.stream(state):
        for node_name, node_state in event.items():
            node_count += 1
            logger.info(f"▶ 节点完成 [{node_count}]: {node_name}")
            final_state = node_state

    logger.info("=" * 60)
    logger.info(f"导入流程结束，共执行 {node_count} 个节点")
    logger.info("=" * 60)
    return final_state





    # for event in kb_import_graph_app.stream(state):
    #     for node_name, state in event.items():  # 注意：这里使用了嵌套循环，因为一个事件可能包含多个节点的更新
    #         print(node_name)
    #         final_state = state  # 循环结束后，final_state 保存的是最后一个节点执行后的状态
    #
    # return final_state  # 返回工作流执行完成后的最终状态。


if __name__ == "__main__":
    # 配置日志：必须先调用，否则各节点的 INFO 级别日志不会显示
    setup_logging()

    import_file_path = r"E:\开发\项目掌柜智库\2.资料\pdf文档\doc\万用表RS-12的使用.pdf"
    file_dir = r"E:\开发\项目掌柜智库\2.资料\pdf文档\存文档位置"

    final_state = run_import_graph(import_file_path, file_dir)
    print(final_state)
    print("-" * 50)
    kb_import_graph_app.get_graph().print_ascii()











