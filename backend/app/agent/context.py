"""集中读取 Agent 状态，避免节点各自处理空值和上下文。"""

from __future__ import annotations

from typing import Any

from app.domain.inventory import is_broad_ingredient_category
from app.state import RecipeAgentState

from .lexicon import known_recipe_names
from .query_plan import QueryPlan, parse_query_plan


def get_query_plan(state: RecipeAgentState) -> QueryPlan:
    """安全读取查询计划，旧状态或损坏数据自动降级。"""

    return parse_query_plan(state.get("query_plan"))


def get_intent(state: RecipeAgentState) -> str:
    """返回查询意图。"""

    return get_query_plan(state).intent


def get_entities(state: RecipeAgentState) -> list[str]:
    """返回当前查询中的明确实体。"""

    return list(get_query_plan(state).entities)


def get_primary_entity(state: RecipeAgentState) -> str | None:
    """返回主实体，没有时明确返回 None。"""

    entities = get_entities(state)
    return entities[0] if entities else None


def get_target_entities(state: RecipeAgentState) -> list[str]:
    """返回比较或搭配问题中的目标实体。"""

    return list(get_query_plan(state).target_entities)


def requested_recipe_name(state: RecipeAgentState) -> str | None:
    """解析当前任务要读取或追问的菜名。"""

    entities = get_entities(state)
    recipe_names = [entity for entity in entities if entity in known_recipe_names()]
    entity = recipe_names[0] if recipe_names else (entities[0] if entities else None)
    if entity:
        return entity

    details = state.get("recipe_details") or []
    if len(details) == 1:
        return details[0].name

    evidence = state.get("evidence") or []
    names = list(dict.fromkeys(item.name for item in evidence if item.name))
    if len(names) == 1:
        return names[0]
    return None


def get_constraints(state: RecipeAgentState) -> dict[str, Any]:
    """返回合并后的约束，同时兼容旧的查询计划状态。"""

    constraints = dict(state.get("constraints") or {})
    plan_constraints = get_query_plan(state).constraints.model_dump(exclude_none=True)
    for key, value in plan_constraints.items():
        if value in (None, [], "", {}):
            continue
        if isinstance(value, list) and isinstance(constraints.get(key), list):
            constraints[key] = list(dict.fromkeys([*constraints[key], *value]))
        else:
            constraints.setdefault(key, value)
    return constraints


def excluded_ingredients(state: RecipeAgentState) -> list[str]:
    """合并过敏原和忌口，避免两处过滤只生效其中一个。"""

    constraints = get_constraints(state)
    values = [
        *(constraints.get("allergies") or []),
        *(constraints.get("avoid_ingredients") or []),
    ]
    return _normalize_string_list(values)


def available_ingredients(state: RecipeAgentState) -> list[str]:
    """返回用户当前可用食材。"""

    return _normalize_string_list(get_constraints(state).get("inventory") or [])


def inventory_is_strict(state: RecipeAgentState) -> bool:
    """判断用户是否要求只使用现有食材。"""

    constraints = get_constraints(state)
    if not constraints.get("inventory_strict"):
        return False
    # 宽泛类别无法逐项核对，“只有鸡肉和蔬菜”按优先匹配处理。
    inventory = constraints.get("inventory") or []
    return not any(is_broad_ingredient_category(str(item)) for item in inventory)


def max_minutes(state: RecipeAgentState) -> int | None:
    """安全读取最大制作时间。"""

    value = get_constraints(state).get("max_minutes")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def max_difficulty(state: RecipeAgentState) -> int | None:
    """安全读取最大难度。"""

    value = get_constraints(state).get("max_difficulty")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]
