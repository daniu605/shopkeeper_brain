"""
验证商品名是否成功存入 Milvus

用途:
    1. 存在性检查 —— 直接扫描 kb_item_names_v1 集合，列出已存入的商品名
    2. 语义检索回查 —— 用一句自然语言提问，做混合检索(dense+sparse)，
       看能否把对应的商品名召回，验证向量确实可用

集合 schema (kb_item_names_v1):
    pk            VARCHAR (主键, 自增)
    file_title    VARCHAR (来源文件标题)
    item_name     VARCHAR (商品名)
    dense_vector  FLOAT_VECTOR (1024维, BGE-M3 稠密向量)
    sparse_vector SPARSE_FLOAT_VECTOR (BGE-M3 稀疏向量)

运行方式:
    直接运行本文件即可（已自动处理项目路径）。
"""
import sys
from pathlib import Path

# 1. 将项目根目录加入 sys.path
#    本文件位于 .../shopkeeper_brain/knowledge/test/import_process/
#    要能 import knowledge.* ，需把 knowledge 的父目录(shopkeeper_brain)加入路径
#    .parent x4 = shopkeeper_brain
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / "knowledge" / ".env")  # 加载环境变量(连接地址、集合名等)

# 集合名：与 item_name_recognition 节点写入的集合保持一致
COLLECTION_NAME = "kb_item_names_v1"


# ------------------------------------------------------------------
# 检查1: 数据是否真的存进去了（直接扫描集合）
# ------------------------------------------------------------------
def verify_stored(client) -> bool:
    """扫描集合，列出已存入的商品名。返回是否有数据。"""
    print("=" * 60)
    print(f"【检查1】扫描集合 [{COLLECTION_NAME}]，确认数据是否存入")
    print("=" * 60)

    # 先确认集合存在
    if not client.has_collection(COLLECTION_NAME):
        print(f"❌ 集合 [{COLLECTION_NAME}] 不存在！请先运行 item_name_recognition 节点。")
        return False

    # 扫描前 20 条记录（pk 是自增字符串主键，用 != "" 匹配全部）
    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter='pk != ""',
        output_fields=["item_name", "file_title"],
        limit=20,
    )

    if not rows:
        print("⚠️  集合存在但为空，还没有任何商品名存入！")
        return False

    print(f"✅ 集合中共扫描到 {len(rows)} 条记录：")
    for i, row in enumerate(rows, 1):
        print(f"   {i}. 商品名=[{row.get('item_name')}]  来源文件=[{row.get('file_title')}]")
    return True


# ------------------------------------------------------------------
# 检查2: 语义检索回查（混合检索 dense + sparse）
# ------------------------------------------------------------------
def verify_search(client, query_text: str):
    """用自然语言提问，做混合检索，验证向量可用、能召回商品名。"""
    print()
    print("=" * 60)
    print(f"【检查2】语义检索回查")
    print(f"   提问: {query_text}")
    print("=" * 60)

    # 延迟导入重量级依赖
    from knowledge.utils.client.ai_clients import AIClients
    from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
    from knowledge.utils.milvus_util import (
        create_hybrid_search_requests,
        execute_hybrid_search_query,
    )

    # 1. 生成查询向量（is_query=True 表示编码"问题"，走 encode_queries）
    bge_m3 = AIClients.get_bge_m3_client()
    vectors = generate_bge_m3_hybrid_vectors(bge_m3, [query_text], is_query=True)
    dense_vector = vectors["dense"][0]
    sparse_vector = vectors["sparse"][0]

    # 2. 构建混合检索请求 (dense + sparse 两路)
    search_requests = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=3,
    )

    # 3. 执行混合检索
    results = execute_hybrid_search_query(
        milvus_client=client,
        collection_name=COLLECTION_NAME,
        search_requests=search_requests,
        limit=3,
        output_fields=["item_name", "file_title"],
    )

    # 4. 打印结果
    print("\n检索结果：")
    if not results or not results[0]:
        print("⚠️  未召回任何结果！")
        return

    for hit in results[0]:
        entity = hit["entity"]
        print("-" * 50)
        print(f"📌 商品名:   {entity.get('item_name', 'N/A')}")
        print(f"📄 来源文件: {entity.get('file_title', 'N/A')}")
        print(f"📊 融合得分: {hit.get('distance', 0):.4f}")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    from knowledge.utils.client.storage_clients import StorageClients

    # 获取 Milvus 客户端（复用项目的双重检查锁单例）
    try:
        client = StorageClients.get_milvus_client()
    except Exception as e:
        print(f"❌ Milvus 客户端连接失败: {e}")
        print("   请确认 Milvus 服务已启动，且 .env 中 MILVUS_URL 配置正确。")
        return

    # 检查1: 是否存入
    has_data = verify_stored(client)

    # 检查2: 只有确实有数据时才做语义检索
    if has_data:
        verify_search(client, "RS-12 万用表怎么测电压")


if __name__ == "__main__":
    main()
