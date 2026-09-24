"""营养查询工具。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.state import ToolResult

from .base import BaseTool
from .catalog import RECIPES


class NutritionInput(BaseModel):
    """营养查询工具的输入参数。"""

    recipe_id: str


class NutritionTool(BaseTool):
    """返回已知菜谱的营养标签。"""

    name = "get_nutrition"
    description = "返回指定菜谱中保存的营养标签。"
    input_model = NutritionInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """根据菜谱 ID 查询营养信息。"""

        params = self.input_model.model_validate(arguments)
        recipe = next(
            (item for item in RECIPES if item.recipe_id == params.recipe_id),
            None,
        )
        if recipe is None:
            return ToolResult(ok=False, error=f"未知菜谱：{params.recipe_id}")
        return ToolResult(
            ok=True,
            data={"recipe_id": recipe.recipe_id, "nutrition_tags": recipe.nutrition_tags},
            evidence_ids=[recipe.recipe_id],
        )
