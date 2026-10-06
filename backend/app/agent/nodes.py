"""食谱规划 LangGraph 使用的节点。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.domain.inventory import expand_ingredient_aliases, recipe_matches_inventory
from app.observability import trace_node
from app.state import (
    RecipeAgentState,
    RecipeEvidence,
    ToolCallRecord,
    ToolResult,
    TraceEvent,
    format_recipe_minutes,
)
from app.tools.registry import ToolRegistry

from .context import (
    available_ingredients,
    excluded_ingredients,
    get_constraints,
    get_query_plan,
    inventory_is_strict,
    max_difficulty,
    max_minutes,
    requested_recipe_name,
)
from .planner import (
    Planner,
    _effective_dietary_goals,
    _meal_period_excluded_categories,
)

logger = logging.getLogger(__name__)

Node = Callable[[RecipeAgentState], dict[str, Any]]
CONFIRMATION_TOOLS = {
    "create_meal_plan",
    "create_shopping_list",
    "revise_meal_plan",
    "revise_shopping_list",
}


def _elapsed_ms(started: float) -> float:
    """返回从计时点开始的毫秒数。"""

    return round((time.perf_counter() - started) * 1000, 2)


def _tool_span_attributes(state: RecipeAgentState) -> dict[str, Any]:
    task = state.get("current_task") or {}
    return {
        "agent.tool_name": task.get("tool_name"),
        "agent.tool_intent": task.get("intent"),
    }


def make_plan_node(planner: Planner, max_steps: int) -> Node:
    """创建规划节点。"""

    @trace_node(
        "agent.plan",
        lambda state: {"agent.step": state.get("step_count", 0) + 1},
    )
    def plan_node(state: RecipeAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        step_count = state.get("step_count", 0) + 1
        if step_count > max_steps:
            return {
                "step_count": step_count,
                "planner_done": True,
                "status": "max_steps_reached",
                "final_answer": "Agent 达到最大执行步数，任务已停止。",
                "trace_events": [
                    TraceEvent(
                        name="plan",
                        duration_ms=_elapsed_ms(started),
                        data={"status": "max_steps_reached", "step": step_count},
                    )
                ],
            }

        try:
            action = planner.plan(state)
        except Exception as exc:
            logger.exception("Agent 规划失败")
            return {
                "step_count": step_count,
                "current_task": None,
                "planner_done": True,
                "status": "planning_failed",
                "final_answer": f"任务规划失败，已停止执行：{type(exc).__name__}",
                "trace_events": [
                    TraceEvent(
                        name="plan",
                        duration_ms=_elapsed_ms(started),
                        data={"status": "failed", "step": step_count},
                    )
                ],
            }
        if action.action == "final":
            return {
                "step_count": step_count,
                "current_task": None,
                "planner_done": True,
                "status": "ready_to_respond",
                "planner_usage": [action.planner],
                "trace_events": [
                    TraceEvent(
                        name="plan",
                        duration_ms=_elapsed_ms(started),
                        data={
                            "action": "final",
                            "planner": action.planner,
                            "step": step_count,
                        },
                    )
                ],
            }

        confirmed_tools = set(state.get("confirmed_tools", []))
        if (
            state.get("confirmation_required")
            and action.tool_name in CONFIRMATION_TOOLS
            and action.tool_name not in confirmed_tools
        ):
            return {
                "step_count": step_count,
                "current_task": {
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                    "intent": action.intent,
                },
                "planner_done": True,
                "status": "awaiting_confirmation",
                "planner_usage": [action.planner],
                "trace_events": [
                    TraceEvent(
                        name="plan",
                        duration_ms=_elapsed_ms(started),
                        data={
                            "action": "confirmation",
                            "planner": action.planner,
                            "tool_name": action.tool_name,
                            "step": step_count,
                        },
                    )
                ],
            }

        return {
            "step_count": step_count,
            "current_task": {
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "intent": action.intent,
            },
            "planner_done": False,
            "status": "executing",
            "planner_usage": [action.planner],
            "trace_events": [
                TraceEvent(
                    name="plan",
                    duration_ms=_elapsed_ms(started),
                    data={
                        "action": "tool",
                        "planner": action.planner,
                        "tool_name": action.tool_name,
                        "step": step_count,
                    },
                )
            ],
        }

    return plan_node


def make_execute_node(registry: ToolRegistry) -> Node:
    """创建工具执行节点。"""

    @trace_node("agent.execute", _tool_span_attributes)
    def execute_node(state: RecipeAgentState) -> dict[str, Any]:
        task = state.get("current_task") or {}
        tool_name = task.get("tool_name")
        arguments = task.get("arguments") or {}
        if not tool_name:
            return {
                "status": "failed",
                "validation_errors": ["计划中没有可执行工具"],
            }

        started = time.perf_counter()
        try:
            result = registry.execute(tool_name, arguments)
            recipes = [
                RecipeEvidence.model_validate(item) for item in result.data.get("recipes", [])
            ]
            recipe_detail = result.data.get("recipe")
            details = [RecipeEvidence.model_validate(recipe_detail)] if recipe_detail else []
            pairings = result.data.get("pairings")
            if pairings is not None:
                anchor = result.data.get("anchor") or ""
                pairings = [{"anchor": anchor, **pairing} for pairing in pairings]
            citations = [item.source for item in [*recipes, *details] if item.source]
            if pairings:
                citations.extend(
                    example["source"]
                    for pairing in pairings
                    for example in pairing.get("examples", [])
                    if example.get("source")
                )
        except Exception as exc:
            logger.exception("工具 %s 返回了无法处理的结果", tool_name)
            result = ToolResult(
                ok=False,
                error=f"工具 {tool_name} 执行结果无法解析：{type(exc).__name__}",
            )
            recipes = []
            details = []
            pairings = None
            citations = []
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        call = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            ok=result.ok,
            error=result.error,
            latency_ms=latency_ms,
        )
        return {
            "tool_calls": [call],
            "tool_results": [result],
            "trace_events": [
                TraceEvent(
                    name="execute",
                    duration_ms=latency_ms,
                    data={
                        "tool_name": tool_name,
                        "ok": result.ok,
                        "error": result.error,
                    },
                )
            ],
            **(
                {
                    "evidence": recipes,
                    "citations": citations,
                }
                if recipes
                else {}
            ),
            **(
                {
                    "recipe_details": details,
                    "citations": citations,
                }
                if details
                else {}
            ),
            **(
                {
                    "pairing_recommendations": pairings,
                    "citations": citations,
                }
                if pairings is not None
                else {}
            ),
            **({"meal_plan": result.data["meal_plan"]} if result.data.get("meal_plan") else {}),
            **(
                {"shopping_list": result.data["shopping_list"]}
                if result.data.get("shopping_list") is not None
                else {}
            ),
            **({"nutrition_result": result.data} if tool_name == "get_nutrition" else {}),
            "status": "validating",
        }

    return execute_node


@trace_node(
    "agent.validate",
    lambda state: {"agent.evidence_count": len(state.get("evidence", []))},
)
def validate_node(state: RecipeAgentState) -> dict[str, Any]:
    """校验过敏原和总烹饪时间等硬约束。"""

    started = time.perf_counter()
    errors: list[str] = []
    blocked_recipe_ids: list[str] = []
    evidence = state.get("evidence", [])
    current_tool = (state.get("current_task") or {}).get("tool_name")
    constraints = get_constraints(state)
    allergies = constraints.get("allergies") or []
    avoid_ingredients = constraints.get("avoid_ingredients") or []
    time_limit = max_minutes(state)
    available = available_ingredients(state)
    strict_inventory = inventory_is_strict(state)

    if (
        not evidence
        and current_tool not in {"revise_shopping_list", "recommend_pairings"}
        and not any(call.tool_name == "recommend_pairings" for call in state.get("tool_calls", []))
    ):
        errors.append("没有找到符合约束的候选菜谱")

    for recipe in evidence:
        recipe_blocked = False
        for allergy in allergies:
            if any(allergy in ingredient for ingredient in recipe.ingredients):
                errors.append(f"{recipe.name} 含有过敏原 {allergy}")
                recipe_blocked = True
        for ingredient in avoid_ingredients:
            if any(ingredient in item for item in recipe.ingredients):
                errors.append(f"{recipe.name} 含有忌口食材 {ingredient}")
                recipe_blocked = True
        if time_limit is not None:
            total_minutes = recipe.total_minutes
            if total_minutes is None:
                errors.append(f"{recipe.name} 未标注制作时间，无法确认是否在 {time_limit} 分钟内")
                recipe_blocked = True
            elif total_minutes > time_limit:
                errors.append(f"{recipe.name} 总耗时 {total_minutes} 分钟，超过 {time_limit} 分钟")
                recipe_blocked = True
        if strict_inventory and not recipe_matches_inventory(recipe.ingredients, available):
            errors.append(f"{recipe.name} 使用了当前库存之外的食材")
            recipe_blocked = True
        if recipe_blocked:
            blocked_recipe_ids.append(recipe.recipe_id)

    return {
        "validation_errors": errors,
        "blocked_recipe_ids": blocked_recipe_ids,
        "status": "validated" if not errors else "validation_failed",
        "trace_events": [
            TraceEvent(
                name="validate",
                duration_ms=_elapsed_ms(started),
                data={
                    "error_count": len(errors),
                    "blocked_count": len(blocked_recipe_ids),
                },
            )
        ],
    }


@trace_node(
    "agent.replan",
    lambda state: {"agent.attempt": state.get("attempt", 0) + 1},
)
def replan_node(state: RecipeAgentState) -> dict[str, Any]:
    """扩大检索范围，同时保留全部硬约束。"""

    started = time.perf_counter()
    constraints = get_constraints(state)
    query = requested_recipe_name(state) or ""
    return {
        "attempt": state.get("attempt", 0) + 1,
        "current_task": {
            "tool_name": "search_recipes",
            "arguments": {
                "query": query,
                "category": constraints.get("category"),
                "cuisine": constraints.get("cuisine"),
                "max_minutes": max_minutes(state),
                "max_difficulty": max_difficulty(state),
                "exclude_ingredients": excluded_ingredients(state),
                "required_ingredients": constraints.get("search_ingredients") or [],
                "dietary_goals": _effective_dietary_goals(constraints),
                "exclude_categories": _meal_period_excluded_categories(
                    str(constraints.get("meal_period")) if constraints.get("meal_period") else None
                ),
                "available_ingredients": available_ingredients(state),
                "inventory_strict": inventory_is_strict(state),
                "sources": ["hybrid", "graph"],
                "limit": 5,
            },
            "intent": "扩大检索范围，但仍然执行全部硬过滤",
        },
        "planner_done": False,
        "status": "replanning",
        "trace_events": [
            TraceEvent(
                name="replan",
                duration_ms=_elapsed_ms(started),
                data={"attempt": state.get("attempt", 0) + 1},
            )
        ],
    }


@trace_node("agent.respond")
def respond_node(state: RecipeAgentState) -> dict[str, Any]:
    """根据工具证据生成确定性回答。"""

    if state.get("status") == "awaiting_confirmation":
        task = state.get("current_task") or {}
        tool_name = task.get("tool_name") or "写操作"
        return {
            "final_answer": (
                f"执行 `{tool_name}` 前需要人工确认。确认后使用同一个 thread_id 恢复任务。"
            ),
            "status": "awaiting_confirmation",
        }

    query_plan = get_query_plan(state)
    if query_plan.requires_clarification:
        candidates = query_plan.context_candidates
        candidate_text = "、".join(f"《{name}》" for name in candidates) or "多道菜"
        return {
            "final_answer": f"上一轮有多个可选结果：{candidate_text}。请告诉我具体要追问哪一道。",
            "status": "completed",
        }

    intent = query_plan.intent
    if intent == "unsupported":
        return {
            "final_answer": "我只能处理菜谱、食材、库存和菜单规划相关问题。",
            "status": "completed",
        }

    pairings = state.get("pairing_recommendations")
    if intent == "comparison" and pairings is not None:
        if not pairings:
            return {
                "final_answer": "没有找到足够的共现菜谱，暂时无法比较这些食材。",
                "status": "completed",
            }
        anchor = pairings[0].get("anchor") or (
            query_plan.entities[0] if query_plan.entities else "主食材"
        )
        records = {
            str(item.get("ingredient") or ""): item for item in pairings if isinstance(item, dict)
        }
        lines = [f"关于 {anchor} 的共现证据："]
        for target in query_plan.target_entities:
            record = records.get(target)
            if record is None:
                record = next(
                    (
                        item
                        for ingredient, item in records.items()
                        if expand_ingredient_aliases(target) & expand_ingredient_aliases(ingredient)
                    ),
                    None,
                )
            if not record:
                lines.append(f"- {target}：没有找到可靠共现记录")
                continue
            examples = [
                item.get("name", "")
                for item in record.get("examples", [])
                if isinstance(item, dict) and item.get("name")
            ]
            example_text = f"，例如 {'、'.join(examples)}" if examples else ""
            lines.append(f"- {target}：在 {record.get('frequency') or 0} 道菜中出现{example_text}")
        lines.append("以上是比较候选食材在图谱中的共现次数，不按菜谱数量之外的因素推断。")
        return {"final_answer": "\n".join(lines), "status": "completed"}

    if pairings:
        anchor = pairings[0].get("anchor") or "该食材"
        lines = [f"{anchor}适合搭配这些蔬菜："]
        citations = []
        for pairing in pairings:
            if not isinstance(pairing, dict):
                continue
            examples = pairing.get("examples", [])
            example_names = [
                item.get("name", "")
                for item in examples
                if isinstance(item, dict) and item.get("name")
            ]
            citations.extend(
                item.get("source", "")
                for item in examples
                if isinstance(item, dict) and item.get("source")
            )
            example_text = f"，例如 {'、'.join(example_names)}" if example_names else ""
            ingredient = str(pairing.get("ingredient") or "未知食材")
            frequency = pairing.get("frequency") or 0
            lines.append(f"- {ingredient}：在 {frequency} 道菜中出现{example_text}")
        citations = list(dict.fromkeys(citations))
        if citations:
            lines.append("搭配依据：" + "、".join(citations))
        return {"final_answer": "\n".join(lines), "status": "completed"}

    if state.get("pairing_recommendations") == [] and any(
        call.tool_name == "recommend_pairings" for call in state.get("tool_calls", [])
    ):
        return {
            "final_answer": "没有找到足够的共现菜谱，暂时无法给出可靠的蔬菜搭配建议。",
            "status": "completed",
        }

    if state.get("status") == "max_steps_reached":
        return {"final_answer": state.get("final_answer")}

    if state.get("status") == "planning_failed":
        return {
            "final_answer": state.get("final_answer") or "任务规划失败，请调整问题后重试。",
            "status": "completed",
        }

    if intent == "pairing" and not pairings:
        return {
            "final_answer": "请告诉我主食材，例如“鸡肉配什么蔬菜好”。",
            "status": "completed",
        }

    requested_name = requested_recipe_name(state) or ""
    all_evidence = list({recipe.recipe_id: recipe for recipe in state.get("evidence", [])}.values())
    blocked_ids = set(state.get("blocked_recipe_ids", []))
    blocked_recipes = [recipe for recipe in all_evidence if recipe.recipe_id in blocked_ids]
    evidence = [recipe for recipe in all_evidence if recipe.recipe_id not in blocked_ids]

    if intent == "recipe_detail" and not state.get("recipe_details"):
        if not requested_name:
            return {
                "final_answer": "请告诉我具体的菜名，例如“咖喱炒蟹怎么做”。",
                "status": "completed",
            }
        return {
            "final_answer": (
                f"知识库中没有找到与《{requested_name}》匹配的完整菜谱。我不会用其他相似菜谱替代。"
            ),
            "status": "completed",
        }

    if intent == "nutrition_question":
        nutrition = state.get("nutrition_result") or {}
        calories = nutrition.get("estimated_calories")
        tags = nutrition.get("nutrition_tags") or []
        recipe_name = nutrition.get("recipe_name") or requested_name or "该菜谱"
        if calories is None and not tags:
            return {
                "final_answer": (
                    "当前知识库没有保存这道菜可靠的结构化热量或营养标签，我不会凭食材自行推算。"
                ),
                "status": "completed",
            }
        lines = [f"《{recipe_name}》的结构化营养信息："]
        if calories is not None:
            lines.append(f"- 菜谱原文标注热量：约 {calories} 千卡")
        if tags:
            lines.append(f"- 营养标签：{'、'.join(tags)}")
        lines.append("以上信息来自菜谱原文，未使用模型推测。")
        return {"final_answer": "\n".join(lines), "status": "completed"}

    if intent == "yes_no":
        return {
            "final_answer": "当前知识图谱不足以可靠回答这个是否判断，请提供更具体的菜谱条件。",
            "status": "completed",
        }

    if intent == "substitution" and not state.get("meal_plan"):
        return {
            "final_answer": "当前没有已经生成的菜单，无法执行食材替换。",
            "status": "completed",
        }

    if not evidence:
        unknown_terms = get_constraints(state).get("unknown_food_terms") or []
        if unknown_terms:
            terms = "、".join(f"“{term}”" for term in unknown_terms)
            return {
                "final_answer": (
                    f"菜谱库中没有找到与{terms}匹配的食材或菜谱，我不会用无关菜谱替代。"
                ),
                "status": "completed",
            }
        if blocked_recipes:
            blocked_names = "、".join(f"《{item.name}》" for item in blocked_recipes)
            return {
                "final_answer": (
                    f"{blocked_names}不符合你当前的过敏原或忌口条件，我不会把它放进推荐里。"
                    "按现在的要求暂时也没找到合适的替代菜，可以放宽一个条件再试。"
                ),
                "status": "completed",
            }
        return {
            "final_answer": (
                "按这些条件暂时没有找到合适的菜。"
                "可以放宽时间，或者再补一两样现有食材，我重新帮你配。"
            ),
            "status": "completed",
        }

    recipe_details = state.get("recipe_details", [])
    meal_plan = state.get("meal_plan")
    if recipe_details:
        detail = recipe_details[-1]
        lines = [f"《{detail.name}》的制作步骤："]
        lines.extend(f"{index}. {step}" for index, step in enumerate(detail.steps, start=1))
        if detail.ingredients:
            lines.append("所需食材：" + "、".join(detail.ingredients))
        citations = [detail.source] if detail.source else []
    elif meal_plan:
        lines = ["已生成菜单规划："]
        for day in meal_plan.get("days", []):
            if not isinstance(day, dict):
                continue
            meal_names = [
                str(meal.get("name"))
                for meal in day.get("meals", [])
                if isinstance(meal, dict) and meal.get("name")
            ]
            if meal_names:
                lines.append(f"- 第{day.get('day', '?')}天：{'、'.join(meal_names)}")
        selected_ids = {
            meal["recipe_id"]
            for day in meal_plan.get("days", [])
            if isinstance(day, dict)
            for meal in day.get("meals", [])
            if isinstance(meal, dict) and meal.get("recipe_id")
        }
        citations = [
            recipe.source
            for recipe in evidence
            if recipe.recipe_id in selected_ids and recipe.source
        ]
    else:
        lines = []
        if blocked_recipes:
            blocked_names = "、".join(f"《{item.name}》" for item in blocked_recipes)
            lines.append(
                f"{blocked_names}不符合你当前的过敏原或忌口条件，我不会把它放进推荐里。"
                "如果还想吃同类口味，可以看看下面的替代菜："
            )
        lines.append("为你找到以下候选菜谱：")
        constraints = get_constraints(state)
        applied_constraints = []
        if constraints.get("allergies"):
            applied_constraints.append(f"排除过敏原：{'、'.join(constraints['allergies'])}")
        if constraints.get("avoid_ingredients"):
            applied_constraints.append(f"排除忌口：{'、'.join(constraints['avoid_ingredients'])}")
        if constraints.get("max_minutes") is not None:
            applied_constraints.append(f"不超过 {constraints['max_minutes']} 分钟")
        if constraints.get("category"):
            applied_constraints.append(f"分类：{constraints['category']}")
        if applied_constraints:
            lines.append("筛选条件：" + "；".join(applied_constraints))
        for recipe in evidence:
            lines.append(
                f"- {recipe.name}：{format_recipe_minutes(recipe)}，"
                f"食材：{', '.join(recipe.ingredients)}，"
                f"营养标签：{', '.join(recipe.nutrition_tags) or '无'}"
            )
        lines.append("以上候选均来自工具结果，未使用模型补充未检索到的食材。")
        citations = list(dict.fromkeys(recipe.source for recipe in evidence if recipe.source))

    citations = list(dict.fromkeys(citations))
    if citations:
        lines.append("来源：" + "、".join(citations))
    shopping_list = state.get("shopping_list", [])
    if shopping_list:
        shopping_names = [
            str(item.get("name") or "")
            for item in shopping_list
            if isinstance(item, dict) and item.get("name")
        ]
        if shopping_names:
            lines.append("购物清单：" + "、".join(shopping_names))
    return {"final_answer": "\n".join(lines), "status": "completed"}


def route_after_plan(state: RecipeAgentState) -> str:
    """规划后决定回答还是执行工具。"""

    if state.get("status") == "awaiting_confirmation":
        return "respond"
    if state.get("planner_done"):
        return "respond"
    return "execute"


def route_after_validation(state: RecipeAgentState) -> str:
    """在重新规划、继续规划和最终回答之间选择。"""

    if state.get("status") == "max_steps_reached":
        return "respond"
    if state.get("validation_errors") and state.get("attempt", 0) < 1:
        return "replan"
    if not state.get("evidence") and state.get("attempt", 0) >= 1:
        return "respond"
    if state.get("planner_done"):
        return "respond"
    return "plan"
