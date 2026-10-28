# 今晚吃啥 架构说明

## 项目目标

构建一个面向真实任务的食谱规划 Agent，而不是只支持单轮问答的 RAG 演示。

系统需要理解用户目标、过敏原、时间、库存和营养偏好，完成菜谱检索、菜单规划、购物清单生成，并保证答案只使用知识库中的食材。

## 总体架构

```text
Web 前端 / FastAPI
        |
LangGraph Agent
  ├── Planner
  ├── Tool Executor
  ├── Constraint Validator
  ├── SQLite Checkpointer
  └── Final Responder
       ├── 确定性回答
       └── 可选 LLM 自然表达
        |
Tool Layer
  ├── search_recipes
  ├── get_recipe
  ├── recommend_pairings
  ├── get_nutrition
  ├── check_inventory
  ├── create_meal_plan
  ├── create_shopping_list
  ├── revise_meal_plan
  └── revise_shopping_list
        |
GraphRAG Data Layer
  ├── Neo4j
  ├── Milvus
  ├── BGE Embedding
  ├── BM25
  └── RRF
```

## 数据链路

```text
HowToCook Markdown
  -> parse_howtocook.py
  -> recipes.json
  -> build_graph.py
  -> graph.json
  -> import_neo4j.py
  -> Neo4j

recipes.json
  -> build_vector_index.py
  -> BGE Embedding
  -> Milvus
```

当前数据规模：

```text
菜谱：322
图谱节点：4,379
图谱关系：7,184
Milvus 向量：322
```

## Agent 执行流程

```text
START
  -> plan
  -> execute
  -> validate
      -> 校验失败：replan
      -> 需要下一步：plan
      -> 任务完成：respond
  -> END
```

Planner 可以有两种实现：

- `RuleBasedPlanner`：无模型环境下的确定性 fallback
- `LLMRecipePlanner`：通过 ChatOpenAI 输出结构化计划

无论使用哪种 Planner，过敏原、时间和难度参数都会由程序再次覆盖，避免模型删除硬约束。

## 查询理解

自然语言进入 Agent 前会先转换成强类型 `QueryPlan`：

- `query_plan.py` 负责确定性解析和安全兜底
- `lexicon.py` 从真实菜谱数据构建食材名称和别名词典
- `ingredients.py` 统一清理数量、厨具和解析噪声，并生成购物清单规范名
- `query_analyzer.py` 在配置 API Key 时可使用 LLM 理解同义表达
- LLM 结果必须经过原文 grounding 校验，不能凭空生成食材

“现有/手头有/冰箱里有”和“只有/仅有/只剩”会被区分：

- 前者是已有食材，优先使用这些食材召回，但允许补充其他食材
- 后者是严格库存约束，菜谱中的非基础食材必须全部在库存中

过敏原、忌口和严格库存属于硬约束，永远由确定性代码执行，不交给向量模型判断。

任务条件按轮次隔离：

- 过敏原、忌口和长期饮食目标持续生效
- 天数、餐数、菜数、时间和库存只在本轮菜单任务或用户本轮修改时生效
- 新话题不会继承上一轮的“30 分钟”“鸡蛋库存”等任务条件

### 自然表达层

最终回答先生成一份经过约束校验的确定性答案，再由可选的 LLM 表达层改写成自然对话：

- 只能改写已有事实，不能新增菜谱、食材或营养信息
- 发现证据外菜名或显式新增食材会拒绝改写并回退确定性回答
- 菜谱步骤、营养数值、安全冲突等高风险回答保留确定性输出
- 问候、感谢、能力询问等闲聊不触发检索，由 LLM 结合对话历史自然回应
- 可通过 `LLM_SHADOW_COMPARE=true` 额外生成一份不检索的纯模型答案，记录菜名覆盖率和
  证据外事实；该答案只用于离线对照，不参与最终事实裁决

### 执行追踪

每个请求都会生成结构化 trace，包含：

- `query_analyzer`：查询理解和意图识别耗时
- `plan`：Planner 决策
- `execute`：工具名称、执行耗时和结果
- `validate`：约束错误和冲突菜数量
- `replan`：重规划次数
- `respond`：使用确定性回答还是 LLM 自然表达
- `request`：整条请求总耗时

同一套节点通过 OpenTelemetry 生成标准 span，并支持控制台或 OTLP 导出。
开启后可以在 Jaeger 等兼容平台查看请求、查询解析、Planner、工具、
校验和回答的完整父子链路。追踪默认关闭，关闭时不加载 SDK，也不影响测试。

## 工具职责

| 工具 | 职责 |
|---|---|
| `search_recipes` | 内存、Hybrid、图谱或统一多路检索 |
| `get_recipe` | 读取完整食材和制作步骤 |
| `recommend_pairings` | 通过图共现关系推荐主食材搭配 |
| `get_nutrition` | 查询营养标签 |
| `check_inventory` | 比较库存和所需食材 |
| `create_meal_plan` | 生成多日菜单 |
| `create_shopping_list` | 生成缺少的食材清单 |
| `revise_meal_plan` | 按追问修改指定日期的菜单 |
| `revise_shopping_list` | 按追问删除购物清单项 |

所有工具都使用 Pydantic 定义输入，并返回统一 `ToolResult`。

## 检索策略

### Neo4j

- 菜名、分类、难度过滤
- 制作时间过滤
- 食材和描述匹配
- 过敏原过滤
- 完整菜谱和步骤读取
- 食材类别与 `食材 -> 菜谱 -> 食材` 多跳搭配推理

### Hybrid Retrieval

- Milvus 向量召回
- jieba + BM25 关键词召回
- 分类、难度和过敏原过滤
- 制作时间过滤
- RRF 融合

### Unified Retrieval

当 Milvus 和 Neo4j 同时可用时，`search_recipes` 使用统一召回：

```text
Hybrid 候选（向量 + BM25）
             \
              -> RRF 融合 -> 硬约束过滤 -> 排序结果
             /
Neo4j 图谱候选
```

明确包含食材、库存、分类或时间条件的问题会同时查询 Hybrid 和图谱；
开放型推荐问题优先使用 Hybrid，减少图谱精确匹配带来的噪声。

餐别和营养目标会进入确定性过滤与排序。晚餐请求会排除调料、饮料、甜品、
早餐和半成品；低糖请求会过滤明确含糖浆、白砂糖、冰糖、蜂蜜或可乐的菜谱；
命中高蛋白等营养标签的候选会优先排序。

菜系在菜谱解析阶段写入 `cuisine_type`，并同步写入 Milvus metadata 与
Neo4j `Recipe.cuisineType`。当前覆盖川菜、鲁菜、粤菜、淮扬菜、闽菜、浙菜、
湘菜和徽菜的确定性特征词，无法可靠判断的菜谱保持为空，不强行分类。

食材别名（例如“花菜/花椰菜/菜花”）会同时用于实体链接、BM25 扩展和库存过滤，
不依赖某一种固定问法。

菜谱简介中的“分钟、小时、天”等时间表达会在解析阶段提取为
`estimated_minutes`。范围值采用上限，避免低估制作时间；无法可靠提取时保留为空，
回答中显示“时间未标注”，而不是显示 `0 分钟`。

## 用户记忆

SQLite 保存以下长期信息：

- 过敏原
- 忌口
- 营养目标
- 库存

规划前会将用户画像合并到 Agent 约束。

## 持久化

LangGraph 使用 `SqliteSaver` 保存任务状态。

- 每个任务对应一个 `thread_id`
- `/tasks/{thread_id}` 可以查询持久化状态
- 敏感工具在 `confirmation_required=true` 时暂停
- `/tasks/{thread_id}/confirm` 可以批准工具并恢复原任务

## 评测

`scripts/evaluate_agent.py` 使用真实数据执行 54 条场景评测，并支持两种检索模式：

- `neo4j`：Neo4j 图查询
- `hybrid`：Milvus + BM25 + RRF

两种模式当前都使用 RuleBasedPlanner 做确定性回归，指标包括：

- 任务成功率
- 检索命中率
- 相关性命中率
- 约束满足率
- 引用覆盖率
- 平均延迟

LLM Planner 通过 `--planner llm` 使用同一评测集单独运行，不能和确定性规则结果混在一起。
报告会额外记录 Planner 调用次数、LLM 直接规划比例和规则降级比例。
