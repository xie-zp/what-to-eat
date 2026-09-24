"""多轮对话中的菜单修改工具。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.state import ToolResult

from .base import BaseTool
from .meal_plan import MealRecipe

DAY_MARKERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

MEAT_KEYWORDS = {
    "鸡肉",
    "鸡胸",
    "鸡腿",
    "鸡翅",
    "鸡爪",
    "鸡丁",
    "鸡汤",
    "鸭肉",
    "鹅肉",
    "鱼肉",
    "鱼片",
    "鱼块",
    "虾",
    "蟹",
    "牛肉",
    "牛腩",
    "牛排",
    "羊肉",
    "猪肉",
    "肉丝",
    "肉片",
    "肉末",
    "排骨",
    "火腿",
    "香肠",
    "培根",
    "腊肠",
    "海鲜",
}


class ReviseMealPlanInput(BaseModel):
    """修改已有菜单所需的上下文。"""

    instruction: str
    meal_plan: dict[str, Any]
    recipes: list[MealRecipe] = Field(default_factory=list)
    available_ingredients: list[str] = Field(default_factory=list)


def _target_days(instruction: str, available_days: list[int]) -> list[int]:
    if any(word in instruction for word in ("每天", "全部", "所有")):
        return available_days

    match = re.search(r"第([一二两三四五六七八九十\d]+)天", instruction)
    if not match:
        return available_days

    raw = match.group(1)
    day = int(raw) if raw.isdigit() else DAY_MARKERS.get(raw)
    return [day] if day in available_days else available_days


def _is_vegetarian(recipe: MealRecipe) -> bool:
    text = recipe.name + "".join(recipe.ingredients)
    return not any(keyword in text for keyword in MEAT_KEYWORDS)


class ReviseMealPlanTool(BaseTool):
    """使用已经检索到的候选菜谱修改指定日期，不凭空生成菜名。"""

    name = "revise_meal_plan"
    description = "修改已有菜单中的指定日期或整份菜单，只能使用检索结果中的菜谱。"
    input_model = ReviseMealPlanInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """替换目标日期的菜品，并重新汇总所需食材。"""

        params = self.input_model.model_validate(arguments)
        days = params.meal_plan.get("days") or []
        if not days:
            return ToolResult(ok=False, error="当前没有可以修改的菜单")
        if not params.recipes:
            return ToolResult(ok=False, error="修改菜单前需要先检索候选菜谱")

        available_days = [int(day["day"]) for day in days if "day" in day]
        target_days = _target_days(params.instruction, available_days)
        used_recipe_ids = {
            meal.get("recipe_id")
            for day in days
            for meal in day.get("meals", [])
            if meal.get("recipe_id")
        }

        candidates = list(params.recipes)
        if any(word in params.instruction for word in ("素菜", "素食", "纯素")):
            candidates = [recipe for recipe in candidates if _is_vegetarian(recipe)]

        alternatives = [recipe for recipe in candidates if recipe.recipe_id not in used_recipe_ids]
        if not alternatives:
            alternatives = candidates
        if not alternatives:
            return ToolResult(
                ok=False,
                error="检索结果中没有满足本次修改要求的菜谱，请调整条件后重试",
            )

        revised_days: list[dict[str, Any]] = []
        replacement_index = 0
        for day in days:
            if int(day["day"]) not in target_days:
                revised_days.append(day)
                continue

            meal_count = len(day.get("meals") or []) or 1
            meals = []
            for _ in range(meal_count):
                recipe = alternatives[replacement_index % len(alternatives)]
                replacement_index += 1
                meals.append(recipe.model_dump())
            revised_days.append({"day": day["day"], "meals": meals})

        required_ingredients = list(
            dict.fromkeys(
                ingredient
                for day in revised_days
                for meal in day.get("meals", [])
                for ingredient in meal.get("ingredients", [])
            )
        )
        available = {
            ingredient.strip() for ingredient in params.available_ingredients if ingredient.strip()
        }
        shopping_list = [
            {
                "name": ingredient,
                "reason": "当前库存中没有",
            }
            for ingredient in required_ingredients
            if ingredient not in available
        ]
        return ToolResult(
            ok=True,
            data={
                "meal_plan": {
                    "days": revised_days,
                    "required_ingredients": required_ingredients,
                },
                "shopping_list": shopping_list,
            },
            evidence_ids=[recipe.recipe_id for recipe in alternatives],
        )
