"""
存储服务连通性测试脚本

用途:
    在正式跑导入流程前，快速确认 MinIO / Milvus / MongoDB 三个存储服务
    是否已正常启动、配置是否正确。

使用方式:
    直接运行本文件，会依次检测每个服务并打印结果，最后给出汇总。

每个服务的"健康"判定:
    - MinIO : 能列出存储桶 (list_buckets)
    - Milvus: 能列出集合 (list_collections)
    - MongoDB: admin 库 ping 成功
"""
import time
import traceback

from dotenv import load_dotenv

load_dotenv()  # 先加载 .env，保证 StorageClients 能读到环境变量


# ------------------------------------------------------------------
# 单个服务的检测结果
# ------------------------------------------------------------------
class CheckResult:
    """封装单个服务的检测结果"""

    def __init__(self, name: str, ok: bool, detail: str, cost: float):
        self.name = name        # 服务名称
        self.ok = ok            # 是否成功
        self.detail = detail    # 详细信息（成功摘要 / 失败原因）
        self.cost = cost        # 耗时(秒)

    def __str__(self):
        icon = "✅ 正常" if self.ok else "❌ 异常"
        return f"{icon}  {self.name:<10} 耗时 {self.cost:.2f}s  |  {self.detail}"


# ------------------------------------------------------------------
# 各服务的检测函数
# ------------------------------------------------------------------
def check_minio() -> CheckResult:
    """检测 MinIO 对象存储是否可用"""
    name = "MinIO"
    start = time.time()
    try:
        from knowledge.utils.client.storage_clients import StorageClients

        client = StorageClients.get_minio_client()
        buckets = client.list_buckets()  # 能列出桶 = 连接 + 鉴权都 OK
        bucket_names = [b.name for b in buckets]
        cost = time.time() - start
        return CheckResult(name, True, f"连接成功，共 {len(bucket_names)} 个存储桶: {bucket_names}", cost)
    except Exception as e:
        cost = time.time() - start
        return CheckResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", cost)


def check_milvus() -> CheckResult:
    """检测 Milvus 向量数据库是否可用"""
    name = "Milvus"
    start = time.time()
    try:
        from knowledge.utils.client.storage_clients import StorageClients

        client = StorageClients.get_milvus_client()
        collections = client.list_collections()  # 能列出集合 = 服务已启动
        cost = time.time() - start
        return CheckResult(name, True, f"连接成功，已有集合: {collections}", cost)
    except Exception as e:
        cost = time.time() - start
        return CheckResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", cost)


def check_mongodb() -> CheckResult:
    """检测 MongoDB 文档数据库是否可用"""
    name = "MongoDB"
    start = time.time()
    try:
        from knowledge.utils.client.storage_clients import StorageClients

        db = StorageClients.get_mongo_db()
        # admin 库 ping 是 MongoDB 官方推荐的健康检查方式
        ping_result = db.client.admin.command("ping")
        cost = time.time() - start
        return CheckResult(name, True, f"连接成功，ping={ping_result.get('ok')}，数据库: {db.name}", cost)
    except Exception as e:
        cost = time.time() - start
        return CheckResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", cost)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("开始检测存储服务连通性...".center(60))
    print("=" * 70)

    # 依次执行所有检测（每个函数内部都已捕获异常，不会中断整体流程）
    checks = [check_minio, check_milvus, check_mongodb]
    results = [check() for check in checks]

    # 打印每个服务的结果
    print()
    for result in results:
        print(result)

    # 汇总
    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    print("-" * 70)
    if ok_count == total:
        print(f"🎉 全部通过：{ok_count}/{total} 个服务正常启动。")
    else:
        failed = [r.name for r in results if not r.ok]
        print(f"⚠️  {ok_count}/{total} 个服务正常，失败的服务: {', '.join(failed)}")
        print("   请检查对应服务是否已启动，以及 .env 中的连接配置是否正确。")
    print("=" * 70)

    # 返回退出码：全部正常为 0，否则为 1（方便 CI / 脚本串联判断）
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    exit(main())
