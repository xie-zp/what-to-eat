"""测试环境配置。"""

from __future__ import annotations

import os

# 单元测试和 API 回归测试使用内存菜谱，避免依赖本机 Neo4j、Milvus 或模型 API。
os.environ["USE_NEO4J"] = "false"
os.environ["USE_HYBRID"] = "false"
os.environ["USE_LLM_PLANNER"] = "false"
