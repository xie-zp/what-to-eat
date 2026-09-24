"""按对话轮次和意图决定约束是否应该生效。"""

from __future__ import annotations

from typing import Any

from .query_plan import merge_constraints, parse_query_plan

SAFETY_KEYS = (
    "allergies",
    "avoid_ingredients",
    "dietary_goals",
)
TASK_KEYS = (
    "max_minutes",
    "max_difficulty",
    "days",
    "meals_per_day",
    "household_size",
    "inventory",
    "inventory_strict",
)
TASK_INTENTS = {"meal_plan", "shopping", "substitution"}


def build_effective_constraints(
    profile_constraints: dict[str, Any],
    request_constraints: dict[str, Any],
    explicit_keys: set[str] | list[str] | None,
    query_plan: Any,
) -> dict[str, Any]:
    """合并长期安全约束、本轮显式约束和本轮查询抽取约束。"""

    plan = parse_query_plan(query_plan)
    extracted = {
        key: value
        for key, value in plan.constraints.model_dump(exclude_none=True).items()
        if value not in ([], "", False)
    }
    explicit = set(explicit_keys or [])
    effective: dict[str, Any] = {}

    # 安全和长期偏好始终保留。
    for key in SAFETY_KEYS:
        values = [
            *(profile_constraints.get(key) or []),
            *(request_constraints.get(key) or []),
        ]
        if values:
            effective[key] = list(dict.fromkeys(values))

    # 任务条件只在菜单/购物类任务中默认生效。
    task_defaults: dict[str, Any] = {}
    if plan.intent in TASK_INTENTS:
        for key in TASK_KEYS:
            if key in explicit or key in extracted:
                continue
            value = request_constraints.get(key)
            if value in (None, [], ""):
                value = profile_constraints.get(key)
            if value not in (None, [], ""):
                task_defaults[key] = value

    # 当前语句抽取出的条件优先于表单默认值。
    effective = merge_constraints(extracted, {**effective, **task_defaults})

    # 用户本轮真正修改过的条件优先于意图推断和默认值。
    explicit_request = {
        key: request_constraints[key]
        for key in explicit
        if key in request_constraints and request_constraints[key] is not None
    }
    if explicit_request:
        effective = merge_constraints(explicit_request, effective)
    return effective
