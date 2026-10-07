"""构建食谱规划 LangGraph。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.state import RecipeAgentState
from app.tools.registry import ToolRegistry

from .nodes import (
    make_execute_node,
    make_plan_node,
    replan_node,
    route_after_plan,
    route_after_validation,
    validate_node,
)
from .planner import Planner, RuleBasedPlanner
from .responder import DeterministicResponder, Responder


def build_what_to_eat(
    registry: ToolRegistry,
    max_steps: int = 12,
    planner: Planner | None = None,
    responder: Responder | None = None,
    checkpointer=None,
):
    """编译第一阶段 Agent 工作流。"""

    planner = planner or RuleBasedPlanner()
    responder = responder or DeterministicResponder()
    builder = StateGraph(RecipeAgentState)
    builder.add_node("plan", make_plan_node(planner, max_steps))
    builder.add_node("execute", make_execute_node(registry))
    builder.add_node("validate", validate_node)
    builder.add_node("replan", replan_node)
    builder.add_node("respond", responder.respond)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )
    builder.add_edge("execute", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "replan": "replan",
            "respond": "respond",
            "plan": "plan",
        },
    )
    builder.add_edge("replan", "execute")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)
