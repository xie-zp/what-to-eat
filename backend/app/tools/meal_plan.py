"""多日菜单规划工具。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.ingredients import clean_ingredient_names
from app.state import ToolResult

from .base import BaseTool


class MealRecipe(BaseModel):
    """菜单规划使用的一道菜。"""

    recipe_id: str
    name: str
    ingredients: list[str] = Field(default_factory=list)

    @field_validator("ingredients", mode="before")
    @classmethod
    def clean_ingredients(cls, value):
        if not isinstance(value, list):
            return []
        return clean_ingredient_names([str(item) for item in value])


class MealPlanInput(BaseModel):
    """菜单规划输入。"""

    days: int = Field(default=1, ge=1, le=14)
    meals_per_day: int = Field(default=1, ge=1, le=5)
    recipes: list[MealRecipe]


class MealPlanTool(BaseTool):
    """按照天数循环安排候选菜谱。"""

    name = "create_meal_plan"
    description = "使用已经检索到的菜谱生成多日菜单。"
    input_model = MealPlanInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """生成简单可解释的多日菜单。"""

        params = self.input_model.model_validate(arguments)
        if not params.recipes:
            return ToolResult(ok=False, error="菜单规划至少需要一道菜谱")

        days = []
        required_ingredients: list[str] = []
        # 候选还够的时候优先用没排过的菜：既保持结果可复现，也避免一周菜单
        # 前五天把同样的几道菜轮一遍。
        usage = [0] * len(params.recipes)
        for day in range(1, params.days + 1):
            meals = []
            for _ in range(params.meals_per_day):
                index = min(range(len(params.recipes)), key=lambda item: (usage[item], item))
                usage[index] += 1
                recipe = params.recipes[index]
                meals.append(recipe.model_dump())
                required_ingredients.extend(recipe.ingredients)
            days.append({"day": day, "meals": meals})

        plan = {
            "days": days,
            "required_ingredients": list(dict.fromkeys(required_ingredients)),
        }
        return ToolResult(
            ok=True,
            data={"meal_plan": plan},
            evidence_ids=[recipe.recipe_id for recipe in params.recipes],
        )
