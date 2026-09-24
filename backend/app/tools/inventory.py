"""库存比较工具。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.ingredients import canonical_shopping_name, clean_ingredient_names
from app.state import ToolResult

from .base import BaseTool


class InventoryInput(BaseModel):
    """库存比较工具的输入参数。"""

    available: list[str] = Field(default_factory=list)
    required: list[str] = Field(default_factory=list)

    @field_validator("available", "required", mode="before")
    @classmethod
    def clean_ingredients(cls, value):
        if not isinstance(value, list):
            return []
        return clean_ingredient_names([str(item) for item in value])


class InventoryTool(BaseTool):
    """不使用 LLM，直接计算缺少的食材。"""

    name = "check_inventory"
    description = "比较所需食材和用户当前库存。"
    input_model = InventoryInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """返回已有食材和缺少食材。"""

        params = self.input_model.model_validate(arguments)
        available = {
            canonical for item in params.available if (canonical := canonical_shopping_name(item))
        }
        required = {
            canonical for item in params.required if (canonical := canonical_shopping_name(item))
        }
        missing = sorted(required - available)
        return ToolResult(
            ok=True,
            data={
                "available": sorted(available & required),
                "missing": missing,
            },
        )
