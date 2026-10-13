"""按对话轮次应用约束的测试。"""

from app.agent.constraint_scope import build_effective_constraints
from app.agent.query_plan import build_query_plan


def test_turn_constraints_do_not_leak_into_general_question() -> None:
    profile = {"allergies": ["花生"]}
    request = {
        "max_minutes": 30,
        "inventory": ["鸡蛋"],
        "allergies": ["花生"],
    }
    plan = build_query_plan("川菜有哪些特色菜？")

    constraints = build_effective_constraints(profile, request, set(), plan)

    assert constraints["allergies"] == ["花生"]
    assert constraints["cuisine"] == "川菜"
    assert "max_minutes" not in constraints
    assert "inventory" not in constraints


def test_explicit_turn_constraint_is_applied() -> None:
    plan = build_query_plan("推荐一道菜")

    constraints = build_effective_constraints(
        {},
        {"max_minutes": 15, "inventory": ["花菜"]},
        {"max_minutes", "inventory"},
        plan,
    )

    assert constraints["max_minutes"] == 15
    assert constraints["inventory"] == ["花菜"]


def test_extracted_constraint_overrides_stale_default() -> None:
    plan = build_query_plan("推荐5分钟内的菜")
    constraints = build_effective_constraints(
        {},
        {"max_minutes": 30},
        set(),
        plan,
    )

    assert constraints["max_minutes"] == 5


def test_meal_plan_keeps_task_defaults() -> None:
    plan = build_query_plan("规划两天晚餐菜单")
    constraints = build_effective_constraints(
        {},
        {"days": 3, "meals_per_day": 1, "max_minutes": 30, "inventory": ["鸡蛋"]},
        set(),
        plan,
    )

    assert constraints["days"] == 2
    assert constraints["meals_per_day"] == 1
    assert constraints["max_minutes"] == 30
    assert constraints["inventory"] == ["鸡蛋"]
