"""营养查询工具。"""

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


class NutritionInput(BaseModel):
    """营养查询工具的输入参数。"""

    recipe_id: str


class NutritionTool(BaseTool):
    """返回已知菜谱的热量和营养标签。"""

    name = "get_nutrition"
    description = "返回指定菜谱中保存的热量和营养标签。"
    input_model = NutritionInput

    def __init__(self, loader: Callable[[str], dict[str, Any] | None] | None = None) -> None:
        self.loader = loader or self._load_from_memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """根据菜谱 ID 查询营养信息。"""

        params = self.input_model.model_validate(arguments)
        record = self.loader(params.recipe_id)
        if record is None:
            return ToolResult(ok=False, error=f"未知菜谱：{params.recipe_id}")
        return ToolResult(
            ok=True,
            data={
                "recipe_id": params.recipe_id,
                "recipe_name": record.get("name") or "",
                "estimated_calories": record.get("estimated_calories"),
                "nutrition_tags": record.get("nutrition_tags") or [],
            },
            evidence_ids=[params.recipe_id],
        )

    @staticmethod
    def _load_from_memory(recipe_id: str) -> dict[str, Any] | None:
        recipe = next(
            (item for item in RECIPES if item.recipe_id == recipe_id),
            None,
        )
        return recipe.model_dump() if recipe else None


class Neo4jNutritionTool(NutritionTool):
    """从 Neo4j 菜谱属性读取营养信息。"""

    def __init__(self, graph) -> None:
        super().__init__(loader=lambda recipe_id: graph.get_recipe(recipe_id))


class FileNutritionTool(NutritionTool):
    """从标准化 recipes.json 读取营养信息。"""

    def __init__(self, path: str | Path) -> None:
        records = {}
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data:
            recipe = ParsedRecipe.model_validate(item)
            records[recipe.recipe_id] = recipe.model_dump()
        super().__init__(loader=records.get)
