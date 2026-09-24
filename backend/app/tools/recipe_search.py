"""菜谱搜索工具。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.domain.inventory import (
    recipe_contains_required_ingredients,
    recipe_matches_inventory,
)
from app.state import ToolResult

from .base import BaseTool
from .catalog import RECIPES


class RecipeSearchInput(BaseModel):
    """菜谱搜索工具的输入参数。"""

    query: str = ""
    category: str | None = None
    max_difficulty: int | None = Field(default=None, ge=1, le=5)
    max_minutes: int | None = None
    exclude_ingredients: list[str] = Field(default_factory=list)
    required_ingredients: list[str] = Field(default_factory=list)
    available_ingredients: list[str] = Field(default_factory=list)
    inventory_strict: bool = False
    sources: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class RecipeSearchTool(BaseTool):
    """使用文本和硬过滤条件搜索菜谱。"""

    name = "search_recipes"
    description = "根据文本、总烹饪时间和排除食材搜索菜谱。"
    input_model = RecipeSearchInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """执行确定性菜谱搜索。"""

        params = self.input_model.model_validate(arguments)
        excluded = {item.strip() for item in params.exclude_ingredients if item.strip()}
        query = params.query.strip().lower()
        query_terms = [item for item in re.split(r"[\s,，、]+", query) if item]

        matches = []
        for recipe in RECIPES:
            if excluded.intersection(recipe.ingredients):
                continue
            if params.category and params.category != recipe.category:
                continue
            if params.max_difficulty is not None and recipe.difficulty > params.max_difficulty:
                continue
            if params.max_minutes is not None:
                total_minutes = recipe.total_minutes
                if total_minutes is None or total_minutes > params.max_minutes:
                    continue
            if params.inventory_strict and not recipe_matches_inventory(
                recipe.ingredients,
                params.available_ingredients,
            ):
                continue
            if not recipe_contains_required_ingredients(
                recipe.ingredients,
                params.required_ingredients,
            ):
                continue

            searchable = " ".join(
                [
                    recipe.name,
                    recipe.category,
                    *recipe.ingredients,
                    *recipe.nutrition_tags,
                ]
            ).lower()
            if query_terms and not any(term in searchable for term in query_terms):
                continue
            matches.append(recipe)

        matches.sort(
            key=lambda item: (
                item.score,
                -(item.total_minutes if item.total_minutes is not None else 10**6),
            ),
            reverse=True,
        )
        selected = matches[: params.limit]
        return ToolResult(
            ok=True,
            data={"recipes": [item.model_dump() for item in selected]},
            evidence_ids=[item.recipe_id for item in selected],
        )
