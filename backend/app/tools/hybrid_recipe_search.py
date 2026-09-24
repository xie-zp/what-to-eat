"""基于混合检索的菜谱搜索工具。"""

from __future__ import annotations

from typing import Any

from app.domain.inventory import recipe_contains_required_ingredients
from app.rag.hybrid import HybridRecipeSearcher
from app.state import ToolResult

from .base import BaseTool
from .recipe_search import RecipeSearchInput


class HybridRecipeSearchTool(BaseTool):
    """使用向量检索、BM25 和 RRF 搜索菜谱。"""

    name = "search_recipes"
    description = "根据文本、分类、难度、时间和排除食材搜索菜谱。"
    input_model = RecipeSearchInput

    def __init__(self, searcher: HybridRecipeSearcher) -> None:
        self.searcher = searcher

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """执行混合搜索。"""

        params = self.input_model.model_validate(arguments)
        records = self.searcher.search(
            query=params.query,
            limit=params.limit,
            category=params.category,
            max_difficulty=params.max_difficulty,
            max_minutes=params.max_minutes,
            exclude_ingredients=params.exclude_ingredients,
            available_ingredients=params.available_ingredients,
            inventory_strict=params.inventory_strict,
        )
        recipes = []
        for record in records:
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
                    "category": record.get("category") or "未知",
                    "difficulty": int(record.get("difficulty") or 0),
                    "estimated_minutes": record.get("estimated_minutes"),
                    "prep_minutes": 0,
                    "cook_minutes": 0,
                    "source": f"hybrid:{record['recipe_id']}",
                    "score": float(record.get("rrf_score") or 0.0),
                }
            )
        return ToolResult(
            ok=True,
            data={"recipes": recipes},
            evidence_ids=[item["recipe_id"] for item in recipes],
        )
