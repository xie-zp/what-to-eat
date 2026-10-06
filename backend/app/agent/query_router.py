"""查询复杂度分析与检索策略路由。

先判断查询需要哪种检索能力，再决定走混合检索、图检索还是两者组合，
避免所有查询都付出图遍历的开销，也避免关系型问题只走向量召回。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStrategy(Enum):
    """可用的检索策略。"""

    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"


@dataclass(frozen=True)
class QueryAnalysis:
    """查询特征分析结果。"""

    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    strategy: SearchStrategy
    confidence: float
    reasoning: str
    signals: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入状态或追踪的结构。"""

        return {
            "query_complexity": self.query_complexity,
            "relationship_intensity": self.relationship_intensity,
            "reasoning_required": self.reasoning_required,
            "entity_count": self.entity_count,
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "signals": list(self.signals),
        }


# 需要跨实体推理、条件组合或因果解释的问法提升复杂度。
COMPLEXITY_KEYWORDS = (
    "为什么",
    "如何",
    "关系",
    "影响",
    "原因",
    "比较",
    "区别",
    "替代",
    "替换",
    "只有",
    "仅有",
    "不同菜系",
    "不同口味",
)
# 出现搭配、组合类词汇说明问题落在实体关系上，图检索更合适。
RELATION_KEYWORDS = (
    "配",
    "搭配",
    "组合",
    "相关",
    "联系",
    "连接",
    "一起",
    "还是",
)
# 多个并列条件会增加过滤维度。
MULTI_CONDITION_MARKERS = (
    "并且",
    "而且",
    "同时",
    "还要",
    "不超过",
    "避开",
)


def route_query(
    goal: str,
    constraints: dict[str, Any] | None = None,
) -> QueryAnalysis:
    """用确定性规则分析查询特征并给出检索策略。"""

    text = goal.strip()
    constraint_values = constraints or {}
    entity_count = _count_entities(constraint_values)

    complexity_hits = [keyword for keyword in COMPLEXITY_KEYWORDS if keyword in text]
    relation_hits = [keyword for keyword in RELATION_KEYWORDS if keyword in text]
    condition_hits = [marker for marker in MULTI_CONDITION_MARKERS if marker in text]

    complexity = min(1.0, 0.4 * len(complexity_hits) + 0.2 * len(condition_hits))
    relationship = min(1.0, 0.5 * len(relation_hits))
    if len(text) >= 20:
        complexity = min(1.0, complexity + 0.15)
    if entity_count >= 3:
        complexity = min(1.0, complexity + 0.15)

    strategy, reasoning = _decide_strategy(complexity, relationship)
    signals = tuple([*complexity_hits, *relation_hits, *condition_hits])
    return QueryAnalysis(
        query_complexity=round(complexity, 2),
        relationship_intensity=round(relationship, 2),
        reasoning_required=complexity >= 0.4 or relationship >= 0.5,
        entity_count=entity_count,
        strategy=strategy,
        confidence=0.6,
        reasoning=reasoning,
        signals=signals,
    )


def _decide_strategy(
    complexity: float,
    relationship: float,
) -> tuple[SearchStrategy, str]:
    """把两个维度的分数映射成检索策略。"""

    if relationship >= 0.5:
        return (
            SearchStrategy.GRAPH_RAG,
            "查询落在实体搭配关系上，优先走图谱多跳检索。",
        )
    if complexity > 0.7:
        return (
            SearchStrategy.GRAPH_RAG,
            "查询包含多条件或跨实体推理，优先走图谱多跳检索。",
        )
    if complexity >= 0.4:
        return (
            SearchStrategy.COMBINED,
            "查询复杂度中等，混合检索与图谱检索同时参与。",
        )
    return (
        SearchStrategy.HYBRID_TRADITIONAL,
        "查询是直接的查找或推荐，走混合检索即可。",
    )


def _count_entities(constraints: dict[str, Any]) -> int:
    """统计查询计划里已经识别的实体数量。"""

    values: set[str] = set()
    for key in ("search_ingredients", "inventory", "entities", "target_entities"):
        for item in constraints.get(key) or []:
            text = str(item).strip()
            if text:
                values.add(text)
    return len(values)


def strategy_sources(strategy: str | None) -> list[str] | None:
    """把策略名映射为统一检索需要的召回通道。"""

    if strategy == SearchStrategy.GRAPH_RAG.value:
        return ["graph"]
    if strategy == SearchStrategy.HYBRID_TRADITIONAL.value:
        return ["hybrid"]
    if strategy == SearchStrategy.COMBINED.value:
        return ["hybrid", "graph"]
    return None
