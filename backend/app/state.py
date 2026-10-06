"""食谱 Agent 使用的强类型状态和领域模型。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from app.domain.ingredients import clean_ingredient_names


class RecipeEvidence(BaseModel):
    """检索或规划工具返回的候选菜谱。"""

    recipe_id: str
    name: str
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    nutrition_tags: list[str] = Field(default_factory=list)
    cuisine_type: str = ""
    category: str = "未知"
    difficulty: int = 0
    estimated_calories: int | None = None
    estimated_minutes: int | None = None
    prep_minutes: int = 0
    cook_minutes: int = 0
    source: str = ""
    score: float = 0.0

    @field_validator("ingredients", mode="before")
    @classmethod
    def clean_ingredients(cls, value):
        """过滤工具、容器和解析噪声，避免进入菜单与购物清单。"""

        if not isinstance(value, list):
            return []
        return clean_ingredient_names([str(item) for item in value])

    @property
    def total_minutes(self) -> int | None:
        """返回预计总制作时间。"""

        if self.estimated_minutes is not None:
            return self.estimated_minutes
        total = self.prep_minutes + self.cook_minutes
        return total if total > 0 else None


def format_recipe_minutes(recipe: RecipeEvidence) -> str:
    """把制作时间格式化为用户可读文本。"""

    total_minutes = recipe.total_minutes
    return f"{total_minutes} 分钟" if total_minutes is not None else "时间未标注"


class ToolCallRecord(BaseModel):
    """一次工具调用的记录。"""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = False
    error: str | None = None
    latency_ms: float = 0.0


class ToolResult(BaseModel):
    """所有工具统一返回的标准结果。"""

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """Agent 执行过程中的一条结构化追踪事件。"""

    name: str
    duration_ms: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)


class PlannedAction(BaseModel):
    """Planner 决定执行的下一步。"""

    action: Literal["tool", "final"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    intent: str = ""
    planner: Literal["rule", "llm", "llm_fallback"] = "rule"


class RecipeAgentState(TypedDict, total=False):
    """LangGraph 节点之间共享的状态。"""

    user_id: str
    conversation_id: str
    confirmation_required: bool
    confirmed_tools: list[str]
    query_plan: dict[str, Any]
    search_strategy: str
    query_analysis: dict[str, Any]
    goal: str
    constraints: dict[str, Any]

    messages: Annotated[list[dict[str, Any]], operator.add]
    history: Annotated[list[dict[str, Any]], operator.add]
    current_task: dict[str, Any] | None

    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    evidence: Annotated[list[RecipeEvidence], operator.add]
    recipe_details: Annotated[list[RecipeEvidence], operator.add]
    pairing_recommendations: Annotated[list[dict[str, Any]], operator.add]
    citations: Annotated[list[str], operator.add]
    blocked_recipe_ids: Annotated[list[str], operator.add]
    trace_events: Annotated[list[TraceEvent], operator.add]
    planner_usage: Annotated[list[str], operator.add]
    nutrition_result: dict[str, Any] | None
    meal_plan: dict[str, Any] | None
    shopping_list: list[dict[str, Any]]
    validation_errors: Annotated[list[str], operator.add]

    step_count: int
    attempt: int
    final_answer: str | None
    planner_done: bool
    status: str


def create_initial_state(
    goal: str,
    constraints: dict[str, Any] | None = None,
    user_id: str = "local-user",
    conversation_id: str = "",
    history: list[dict[str, Any]] | None = None,
    meal_plan: dict[str, Any] | None = None,
    shopping_list: list[dict[str, Any]] | None = None,
    confirmation_required: bool = False,
    query_plan: dict[str, Any] | None = None,
) -> RecipeAgentState:
    """创建完整的 Agent 初始状态。"""

    # 所有入口都必须经过同一套查询解析，避免 API、脚本和测试行为不一致。
    from app.agent.query_plan import build_query_plan, merge_constraints
    from app.agent.query_router import route_query

    if query_plan is None:
        query_plan = build_query_plan(goal, history=history)
    else:
        query_plan = dict(query_plan)
    merged_constraints = merge_constraints(
        constraints or {},
        query_plan.get("constraints", {}),
    )
    query_plan["constraints"] = merged_constraints
    analysis = route_query(goal, merged_constraints)

    return RecipeAgentState(
        user_id=user_id,
        conversation_id=conversation_id,
        confirmation_required=confirmation_required,
        confirmed_tools=[],
        query_plan=query_plan,
        search_strategy=analysis.strategy.value,
        query_analysis=analysis.to_dict(),
        goal=goal,
        constraints=merged_constraints,
        messages=list(history or []),
        history=list(history or []),
        current_task=None,
        tool_calls=[],
        tool_results=[],
        evidence=[],
        recipe_details=[],
        pairing_recommendations=[],
        citations=[],
        blocked_recipe_ids=[],
        trace_events=[],
        planner_usage=[],
        nutrition_result=None,
        meal_plan=meal_plan,
        shopping_list=list(shopping_list or []),
        validation_errors=[],
        step_count=0,
        attempt=0,
        final_answer=None,
        planner_done=False,
        status="new",
    )
