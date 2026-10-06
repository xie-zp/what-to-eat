"""购物清单生成工具。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.ingredients import canonical_shopping_name, clean_ingredient_names
from app.state import ToolResult

from .base import BaseTool

SHOPPING_EXCLUDED_INGREDIENTS = {
    "水",
    "清水",
    "开水",
    "热水",
    "温水",
    "凉白开水",
}


class ShoppingListInput(BaseModel):
    """购物清单输入。"""

    required_ingredients: list[str] = Field(default_factory=list)
    available_ingredients: list[str] = Field(default_factory=list)

    @field_validator("required_ingredients", "available_ingredients", mode="before")
    @classmethod
    def clean_ingredients(cls, value):
        if not isinstance(value, list):
            return []
        return clean_ingredient_names([str(item) for item in value])


class ShoppingListTool(BaseTool):
    """计算并与库存比较后生成食材购物清单。"""

    name = "create_shopping_list"
    description = "比较菜单需要和现有库存，生成不包含厨具和水的食材购物清单。"
    input_model = ShoppingListInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """生成缺少的食材清单。"""

        params = self.input_model.model_validate(arguments)
        required = list(
            dict.fromkeys(
                canonical
                for ingredient in params.required_ingredients
                if (canonical := canonical_shopping_name(ingredient))
            )
        )
        available = {
            canonical
            for ingredient in params.available_ingredients
            if (canonical := canonical_shopping_name(ingredient))
        }
        shopping_list = [
            {
                "name": ingredient,
                "reason": "当前库存中没有",
            }
            for ingredient in required
            if ingredient not in available and ingredient not in SHOPPING_EXCLUDED_INGREDIENTS
        ]
        return ToolResult(
            ok=True,
            data={"shopping_list": shopping_list},
        )
