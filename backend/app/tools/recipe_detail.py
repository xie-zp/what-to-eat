"""完整菜谱读取工具。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.domain.recipe import ParsedRecipe
from app.state import ToolResult

from .base import BaseTool
from .catalog import RECIPES


class RecipeDetailInput(BaseModel):
    """完整菜谱读取参数。"""

    recipe_id: str


class RecipeDetailTool(BaseTool):
    """从可注入的数据源读取完整菜谱。"""

    name = "get_recipe"
    description = "根据菜谱 ID 获取完整食材、制作步骤和来源。"
    input_model = RecipeDetailInput

    def __init__(
        self,
        loader: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        self.loader = loader or self._load_from_memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """读取一道完整菜谱。"""

        params = self.input_model.model_validate(arguments)
        record = self.loader(params.recipe_id)
        if not record:
            return ToolResult(ok=False, error=f"未知菜谱：{params.recipe_id}")
        return ToolResult(
            ok=True,
            data={"recipe": record},
            evidence_ids=[params.recipe_id],
        )

    @staticmethod
    def _load_from_memory(recipe_id: str) -> dict[str, Any] | None:
        recipe = next(
            (item for item in RECIPES if item.recipe_id == recipe_id),
            None,
        )
        return recipe.model_dump() if recipe else None


class Neo4jRecipeDetailTool(RecipeDetailTool):
    """从 Neo4j 读取完整菜谱。"""

    def __init__(self, graph) -> None:
        super().__init__(loader=lambda recipe_id: self._load(graph, recipe_id))

    @staticmethod
    def _load(graph, recipe_id: str) -> dict[str, Any] | None:
        record = graph.get_recipe(recipe_id)
        if not record:
            return None
        record["source"] = f"neo4j:{recipe_id}"
        record["nutrition_tags"] = record.get("nutrition_tags") or []
        record["prep_minutes"] = 0
        record["cook_minutes"] = 0
        record["steps"] = [
            step.get("description", "")
            for step in record.get("steps", [])
            if step.get("description")
        ]
        return record


class FileRecipeDetailTool(RecipeDetailTool):
    """从标准化 recipes.json 读取完整菜谱。"""

    def __init__(self, path: str | Path) -> None:
        records = {}
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data:
            recipe = ParsedRecipe.model_validate(item)
            records[recipe.recipe_id] = {
                "recipe_id": recipe.recipe_id,
                "name": recipe.name,
                "ingredients": [item.name for item in recipe.ingredients],
                "steps": recipe.steps,
                "category": recipe.category,
                "difficulty": recipe.difficulty,
                "estimated_calories": recipe.estimated_calories,
                "estimated_minutes": recipe.estimated_minutes,
                "nutrition_tags": recipe.nutrition_tags,
                "prep_minutes": 0,
                "cook_minutes": 0,
                "source": f"file:{recipe.source_path}",
            }
        super().__init__(loader=records.get)
