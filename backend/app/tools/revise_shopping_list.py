"""多轮对话中的购物清单修改工具。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.state import ToolResult

from .base import BaseTool

REMOVE_MARKERS = ("去掉", "删除", "移除", "不要", "不买")


class ReviseShoppingListInput(BaseModel):
    """修改购物清单所需的上下文。"""

    instruction: str
    shopping_list: list[dict[str, Any]] = Field(default_factory=list)


def _removed_items(instruction: str) -> list[str]:
    marker_pattern = "|".join(REMOVE_MARKERS)
    match = re.search(rf"(?:{marker_pattern})(.+)", instruction)
    if not match:
        return []
    values = re.split(r"[，,、和及以及\s]+", match.group(1))
    return [value.strip() for value in values if value.strip()]


class ReviseShoppingListTool(BaseTool):
    """按用户指令删除购物清单项，不修改菜单。"""

    name = "revise_shopping_list"
    description = "从已有购物清单中删除用户指定食材。"
    input_model = ReviseShoppingListInput

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """删除明确点名的购物项。"""

        params = self.input_model.model_validate(arguments)
        if not params.shopping_list:
            return ToolResult(ok=False, error="当前没有可以修改的购物清单")

        removed_items = _removed_items(params.instruction)
        if not removed_items:
            return ToolResult(ok=False, error="没有识别出需要删除的购物项")

        revised = [
            item
            for item in params.shopping_list
            if not any(
                removed in str(item.get("name", "")) or str(item.get("name", "")) in removed
                for removed in removed_items
            )
        ]
        if len(revised) == len(params.shopping_list):
            return ToolResult(ok=False, error="购物清单中没有找到指定食材")

        return ToolResult(ok=True, data={"shopping_list": revised})
