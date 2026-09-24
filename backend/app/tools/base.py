"""Agent 工具的基础协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from app.state import ToolResult


class BaseTool(ABC):
    """带强类型输入契约的可执行工具。"""

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """校验参数并执行工具。"""

    def input_schema(self) -> dict[str, Any]:
        """返回用于模型 Tool Calling 的 JSON Schema。"""

        return self.input_model.model_json_schema()
