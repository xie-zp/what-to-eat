"""基于 Neo4j 知识图谱的菜谱搜索工具。"""

from __future__ import annotations

from typing import Any

from app.domain.inventory import (
    recipe_contains_required_ingredients,
    recipe_matches_inventory,
)
from app.rag import Neo4jKnowledgeGraph
from app.state import ToolResult

from .base import BaseTool
from .recipe_search import RecipeSearchInput


class Neo4jRecipeSearchTool(BaseTool):
    """使用 Neo4j 查询真实菜谱知识图谱。"""

    name = "search_recipes"
    description = "根据文本、分类、难度、时间和排除食材搜索菜谱。"
    input_model = RecipeSearchInput

    def __init__(self, graph: Neo4jKnowledgeGraph) -> None:
        self.graph = graph

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """执行 Neo4j 菜谱查询。"""

        params = self.input_model.model_validate(arguments)
        records = self.graph.find_recipes(
            query=params.query,
            category=params.category,
            max_difficulty=params.max_difficulty,
            max_minutes=params.max_minutes,
            exclude_ingredients=params.exclude_ingredients,
            available_ingredients=params.available_ingredients,
            inventory_strict=params.inventory_strict,
            limit=params.limit,
        )
        recipes = []
        for rank, record in enumerate(records, start=1):
            if params.inventory_strict and not recipe_matches_inventory(
                record.get("ingredients") or [],
                params.available_ingredients,
            ):
                continue
            if not recipe_contains_required_ingredients(
                record.get("ingredients") or [],
                params.required_ingredients,
            ):
                continue
            recipes.append(
                {
                    "recipe_id": record["recipe_id"],
                    "name": record["name"],
                    "ingredients": record.get("ingredients") or [],
                    "nutrition_tags": [],
                    "category": (record.get("categories") or ["未知"])[0],
                    "difficulty": int(record.get("difficulty") or 0),
                    "estimated_minutes": record.get("estimated_minutes"),
                    "prep_minutes": 0,
                    "cook_minutes": 0,
                    "source": f"neo4j:{record['recipe_id']}",
                    "score": float(record.get("search_score") or 1.0 / rank),
                }
            )
        selected = recipes[: params.limit]
        return ToolResult(
            ok=True,
            data={"recipes": selected},
            evidence_ids=[item["recipe_id"] for item in selected],
        )
