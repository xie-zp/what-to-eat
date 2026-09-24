"""Agent 执行器使用的工具注册表。"""

from __future__ import annotations

from typing import Any

from app.state import ToolResult

from .base import BaseTool


class ToolRegistry:
    """集中注册工具，并提供统一执行入口。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """使用唯一名称注册工具。"""

        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """返回已注册工具，不存在时抛出明确异常。"""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未知工具：{name}") from exc

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """执行工具，并将未预期异常转换为 ToolResult。"""

        try:
            return self.get(name).run(arguments)
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def schemas(self) -> list[dict[str, Any]]:
        """返回兼容 OpenAI Tool Calling 的工具 Schema。"""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def names(self) -> list[str]:
        """返回全部已注册工具名称。"""

        return list(self._tools)
