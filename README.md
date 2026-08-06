# shopkeeper_brain 问器知识库

#### 介绍

**问器知识库**是一个面向商品/设备技术文档的 RAG（检索增强生成）问答系统。
用户上传产品手册（PDF/MD）后，系统自动完成「文档解析 → 切片 → 向量化 → 入库」，
之后用户即可用自然语言提问，系统通过「商品确认 → 多路检索 → 融合重排 → 流式生成」给出基于知识库的答案，
并支持联网搜索兜底与多轮对话上下文理解。

核心特性：

- **导入流水线**：PDF/MD → Markdown → 图片描述（VLM）→ 智能切片 → 商品名识别 → 混合向量 → Milvus 入库
- **查询流水线**：商品名确认 → 三路并行检索（向量 / HyDE / 联网）→ RRF 融合 → Reranker 重排 → SSE 流式答案
- **异步架构**：上传/提问后立即返回任务 ID，后台线程执行，前端轮询/长连接实时看进度
- **优雅降级**：每个环节都有防御性取值与降级策略，单点故障不阻断整条流程
- **可观测性**：全链路日志 + 任务节点级状态追踪

#### 软件架构

```
┌──────────────────────────── 导入流水线（导入即入库） ───────────────────────────┐
│                                                                              │
│  一本书(PDF/MD)                                                               │
│    │                                                                         │
│    ▼                                                                         │
│  entry → pdf_to_md → md_img → document_split → item_name_rec → bge_embedding │
│  前台     扫描室      插图修复    裁切装订          鉴定师         指纹采集      │
│    │                                                                         │
│    ▼                                                                         │
│  import_milvus（入库：拆旧重建/新建/复用 + schema + 两套检索台 + 上架贴号）        │
│    │                                                                         │
│    ▼                                                                         │
│  Milvus kb_chunks_v1（稠密 COSINE + 稀疏 IP 混合检索索引）                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── 查询流水线（提问即回答） ─────────────────────────┐
│                                                                              │
│  用户提问（POST /query）                                                       │
│    │                                                                         │
│    ▼                                                                         │
│  item_name_confirm（历史+LLM 提取商品名 → 混合检索名册 → 评分对齐）               │
│    │ 已确认 ──► multi_search（三路并行）                                        │
│    │            ├─ vector_search（BGE-M3 直接向量检索）                        │
│    │            ├─ hyde_search（LLM 生成假设文档再检索）                        │
│    │            └─ web_search_mcp（MCP 联网搜索兜底）                          │
│    │                    │                                                     │
│    ▼                    ▼                                                     │
│  answer_output ◄── rerank（断崖截断）◄── rrf（倒数排名融合）◄── 三路结果汇合      │
│    │                                                                         │
│    ▼                                                                         │
│  SSE 流式答案 + 对话落库 MongoDB                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**技术栈**

| 组件 | 用途 |
|------|------|
| FastAPI + uvicorn | HTTP API（导入服务 8000 / 查询服务 8001）|
| LangGraph | 导入/查询两条 StateGraph 流水线 |
| Milvus | 向量库（切片集合 + 商品名册，混合检索）|
| MongoDB | 多轮对话历史存储与回填 |
| MinIO | 上传文件与图片的对象存储 |
| BGE-M3 | 稠密 + 稀疏混合向量 |
| BGE-Reranker | 重排序（断崖检测截断）|
| Qwen（DashScope）| LLM 商品识别 / HyDE / 答案生成；VLM 图片描述 |
| MinerU | PDF → Markdown 解析 |

**目录结构**

```
knowledge/
├── api/            # FastAPI 路由层（import_router / query_router）
├── core/           # 依赖注入（缓存单例 + Depends）、路径工具
├── front/          # 前端页面（import.html 导入 / chat.html 对话）
├── processor/
│   ├── import_process/   # 导入流水线（7 个节点 + 状态 + 配置 + 异常）
│   └── query_process/    # 查询流水线（7 个节点 + 状态 + 配置 + 异常）
├── prompt/         # 提示词模板
├── schema/         # Pydantic 数据模型（请求/响应契约）
├── service/        # 业务层（查询服务 / 历史服务）
├── utils/          # 客户端单例、任务追踪、SSE、Milvus/Mongo 工具
└── .env            # 环境配置（密钥、地址、模型路径）
```

#### 安装教程

1. **环境要求**：Python 3.12、NVIDIA 显卡（可选，GPU 加速）、CUDA Toolkit 12.8（MinerU VLM 与向量模型需要）
2. **安装依赖**：
   ```bash
   cd knowledge
   pip install -r requirements.txt
   ```
3. **配置环境变量**：复制 `.env` 并按实际情况填写：
   - `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`：DashScope 密钥
   - `MILVUS_URL` / `MONGO_URL` / `MINIO_ENDPOINT`：存储服务地址
   - `BGE_M3_PATH` / `BGE_RERANKER_LARGE`：本地模型路径
   - `BGE_DEVICE=cuda:0`：向量模型运行设备
4. **准备存储服务**：启动 Milvus、MongoDB、MinIO（可用 `knowledge/test/import_process/测试存储服务器是否启动成功.py` 一键验证）

#### 使用说明

1. **启动导入服务**（端口 8000，前端导入页）：
   ```bash
   cd E:\shopkeeper_brain\shopkeeper_brain
   set PYTHONPATH=E:\shopkeeper_brain\shopkeeper_brain
   knowledge\.venv\Scripts\python.exe knowledge\api\import_router.py
   ```
   浏览器打开 `http://127.0.0.1:8000/import.html`，上传 PDF/MD 即可实时查看节点进度。

2. **启动查询服务**（端口 8001，前端对话页）：
   ```bash
   knowledge\.venv\Scripts\python.exe knowledge\api\query_router.py
   ```
   浏览器打开 `http://127.0.0.1:8001/chat.html`，输入问题即可获得流式答案。

3. **核心 API**：
   | 服务 | 接口 | 说明 |
   |------|------|------|
   | 导入 | `POST /upload` | 上传文件，返回 task_id |
   | 导入 | `GET /status/{task_id}` | 导入节点状态轮询 |
   | 查询 | `POST /query` | 提交问题（流式返回 task_id，非流式返回答案）|
   | 查询 | `GET /stream/{task_id}` | SSE 长连接（progress/delta/final 事件）|
   | 查询 | `GET /history/{session_id}` / `DELETE` | 历史加载 / 清空 |
   | 通用 | `GET /docs` | Swagger 交互文档 |

4. **单独测试节点**：每个节点文件都带 `__main__` 测试块（如 `bge_embedding.py`、`rrf.py`），可直接运行验证。

#### 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request
