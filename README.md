# Recipe Agent 食谱规划智能体

一个基于 LangGraph、GraphRAG 和可验证工具调用的食谱规划 Agent。

项目目标不是做一个聊天式 RAG 演示，而是实现能够理解约束、规划任务、调用工具、校验结果并执行多步骤目标的领域 Agent。

## 已完成能力

- 强类型 Agent 状态与 LangGraph 多步骤循环
- LLM Planner，以及无模型环境下的规则 Planner 降级
- 工具注册表和统一 Pydantic 工具协议
- 过敏原、忌口、制作时间、难度和库存约束校验
- 菜谱检索、完整步骤读取、多日菜单和购物清单工具
- 多轮菜单修改与购物清单修改
- SQLite 用户长期画像和 LangGraph SQLite Checkpointer
- 敏感工具人工确认、暂停恢复和 `thread_id` 任务查询
- FastAPI、任务状态查询和响应式 Web 前端
- HowToCook Markdown 解析、知识图谱构建和 Neo4j 导入
- BGE 向量化、Milvus、BM25 和 RRF 混合检索实现
- 实体词典、可选的 LLM 结构化查询解析和统一 Hybrid/图谱召回
- 确定性可信回答与可选的 LLM 自然表达层
- 覆盖查询解析、规划、工具、校验和回答的结构化执行 trace
- 食材类别与多跳图推理，支持“鸡肉配什么蔬菜”这类关系问题
- 自动化回归测试与真实 Neo4j 场景评测
- Dockerfile 和 Docker Compose 数据服务

当前解析 322 道真实菜谱，并生成 4,382 个图谱节点和 7,162 条关系。

## Agent 流程

```text
plan -> execute -> validate
          ^            |
          |            +-> replan
          |            |
          +------------+-> respond
```

典型菜单任务会依次调用：

```text
search_recipes
  -> create_meal_plan
  -> create_shopping_list
  -> respond
```

过敏原、时间和难度等硬约束会在工具参数和校验节点中再次执行，不能通过提示词删除。
库存表达会区分“只有/仅有”和“现有/手头有”，前者执行严格库存过滤，后者优先按现有食材召回。

## 数据来源

菜谱原始内容来自：

- [HowToCook](https://github.com/Anduin2017/HowToCook)
- License: The Unlicense

本项目独立实现 Markdown 数据解析、知识图谱构建、GraphRAG 检索和 Agent 编排。具体来源边界和原创贡献见 [NOTICE.md](NOTICE.md)。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pip install -e ".[rag]"
```

运行单元测试：

```powershell
pytest
```

单元测试默认使用内存菜谱，不依赖 Neo4j、Milvus 或模型 API。

## 安全保存 API Key

不要把 API Key 写入源码、聊天记录或 Git。Windows 下可以使用 DPAPI，让密钥只能由当前 Windows 用户解密：

```powershell
.\scripts\set_llm_key.cmd
.\scripts\run_dev.ps1
```

加密文件保存在 `.secrets/llm.key`，该目录已经加入 `.gitignore`。

## 数据与索引

```powershell
docker compose up -d
python scripts/parse_howtocook.py
python scripts/build_graph.py
python scripts/import_neo4j.py --reset
python scripts/build_vector_index.py
```

启动开发服务：

```powershell
.\scripts\run_dev.ps1
```

使用 Docker Compose 启动完整服务时，用下面的脚本解密并注入本机 API Key：

```powershell
.\scripts\run_compose.ps1 -Rebuild
```

默认只启用 LLM 查询解析，Planner 继续使用确定性规则。需要同时启用 LLM Planner 时：

```powershell
.\scripts\run_compose.ps1 -Rebuild -UseLLMPlanner
```

启动后访问：

- Web 页面：http://127.0.0.1:8020
- API 文档：http://127.0.0.1:8020/docs
- 健康检查：http://127.0.0.1:8020/healthz

## Docker

```powershell
docker compose build backend
docker compose up -d
```

Docker 内后端监听 `8000`，宿主机映射到 `8020`，避免和本机其他服务冲突。

## 评测

```powershell
python scripts/evaluate_agent.py
```

当前评测集包含 54 条真实 Neo4j 场景。Neo4j 和 Hybrid Retrieval 均完成规则 Planner
回归评测，分别输出任务成功、检索命中、约束满足、引用覆盖和延迟指标。

真实 HTTP Agent 可靠性评测：

```powershell
python scripts/evaluate_reliability.py --output docs/reliability.md
```

该评测覆盖库存同义表达、过敏与忌口、冲突菜拦截、多轮追问、菜单规划、无关问题和提示注入。

## 持续集成

推送代码或创建 Pull Request 时，GitHub Actions 会自动执行：

- `pytest`
- `ruff check`
- `ruff format --check`
- `docker compose config`
- Docker 后端镜像构建

## 文档

- [架构说明](docs/architecture.md)
- [项目介绍](docs/project_story.md)
- [面试问答](docs/interview_guide.md)
- [评测报告](docs/evaluation.md)
- [混合检索评测](docs/evaluation-hybrid.md)
- [可靠性评测](docs/reliability.md)
- [学习笔记](docs/learning-notes.md)

## 后续路线

- Langfuse 或 OpenTelemetry 全链路追踪
- 增加 LLM Planner 自动评测
- 补充演示视频和最终截图
